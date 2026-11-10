"""Data parallelism and ZeRO stages 1, 2, 3 (+ CPU offload) on the virtual GPUs.

Every virtual GPU runs `_rank_main` (SPMD, like a torch.distributed script). The four arrangements share
one training loop and differ only in *what each rank keeps* and *which collective it calls*:

              keeps in full            keeps 1/N of            gradients leave as            weights come back by
  DDP         W, G, master, m, v       -                       reduce-scatter + all-gather   local cast
  ZeRO-1      W, G                     master, m, v            reduce-scatter                all-gather (bf16)
  ZeRO-2      W                        G, master, m, v         reduce-scatter, per bucket    all-gather (bf16)
  ZeRO-3      -                        W, G, master, m, v      reduce-scatter, per bucket    nothing: all-gather
                                                                                             per layer, in fwd+bwd
Numbers: W/G are bf16 (2 B), master/m/v are fp32 (4 B each), which is the lesson's 2+2+4+8 = 16 B.
On this CPU the matmuls run on fp32 copies of the bf16 weights (no bf16 units): bf16 in, fp32 accumulate,
which is what a tensor core does too. The ledger counts what a GPU would store: the bf16 tensors.
"""
import math
import time
from dataclasses import asdict, dataclass, field

import torch
from torch.utils.flop_counter import FlopCounterMode

from .hw import calibrated, with_
from .model import GPTCfg, build, unit_names, CharData
from .vgpu import Cluster, CATS, STATE, nbytes

NAMES = {0: 'DDP', 1: 'ZeRO-1', 2: 'ZeRO-2', 3: 'ZeRO-3'}
DT = {'bf16': torch.bfloat16, 'fp32': torch.float32, 'fp64': torch.float64}
ALIGN = 16        # elements; every slice is a whole number of 16-wide fp32 SIMD vectors (see Layout)
ADAM_BYTES = 28   # per element: read g (2), master r/w (8), m r/w (8), v r/w (8), write bf16 weight (2)


@dataclass
class RunCfg:
    stage: int = 0
    world: int = 32
    gpus_per_node: int = 8
    steps: int = 60
    micro_bs: int = 4                # sequences per GPU per step
    model: GPTCfg = field(default_factory=GPTCfg)
    lr: float = 3e-3
    warmup: int = 10
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    wd: float = 0.1
    clip: float = 1.0
    dtype: str = 'bf16'              # weights, gradients and everything on the wire
    offload: bool = False            # optimizer state + update on the CPU (ZeRO-Offload)
    overlap: bool = True             # collectives on their own streams; False = every collective blocks
    prefetch: bool = True            # ZeRO-3: gather unit i+1 while unit i computes
    card: str = 'H100'
    latency: float = 0.0
    record_step: int = 1             # the step whose memory trace and timeline are kept
    seed: int = 0
    data_seed: int = 0
    comm_mode: str = 'rendezvous'    # or 'mailbox': every ring hop a real message between threads

    @property
    def name(self):
        n = NAMES[self.stage] + (' + offload' if self.offload else '')
        return n if self.dtype == 'bf16' else f'{n} [{self.dtype}]'


class Layout:
    """One unit's parameters as one flat buffer, padded so N ranks get equal contiguous slices.

    Each slice is also padded to a multiple of ALIGN elements. Without it, the last few elements of an odd-sized
    slice go through the CPU's scalar tail loop instead of the fused-multiply-add vector loop, and the fp32
    master copy differs from DDP's in the last bit for those elements (I measured 3 of 818,241, at 1e-9).
    DeepSpeed and FSDP pad their shards to an alignment boundary for the same family of reasons."""

    def __init__(self, unit, N, align=ALIGN):
        self.params = list(unit.parameters())
        self.shapes = [p.shape for p in self.params]
        self.numels = [p.numel() for p in self.params]
        self.numel = sum(self.numels)
        self.padded = self.numel + (-self.numel) % (N * align)
        self.chunk = self.padded // N

    def flatten(self, tensors, dtype):
        out = torch.zeros(self.padded, dtype=dtype)
        o = 0
        for t, n in zip(tensors, self.numels):
            out[o:o + n] = t.reshape(-1)
            o += n
        return out

    def assign(self, flat, cdt):
        """Point the module's parameters at a full (gathered) flat buffer, as fp32 working copies."""
        o = 0
        for p, n, s in zip(self.params, self.numels, self.shapes):
            p.data = flat[o:o + n].view(s).to(cdt, copy=True)
            o += n

    def release(self):
        """Free the full parameters (DeepSpeed's ZeRO-3 does exactly this: param.data = torch.empty(0))."""
        for p in self.params:
            p.data = torch.empty(0, dtype=p.dtype)


def lr_at(cfg, step):
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    q = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * q)))


def adamw_(p, m, v, g, lr, t, b1, b2, eps, wd):
    """torch.optim.AdamW(foreach=False), line for line, on flat (possibly sharded) tensors."""
    p.mul_(1 - lr * wd)
    m.lerp_(g, 1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    denom = (v.sqrt() / math.sqrt(1 - b2 ** t)).add_(eps)
    p.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))


def unit_flops(cfg: RunCfg):
    """Measured (not estimated) forward and backward FLOPs of each unit for one GPU's micro-batch."""
    torch.manual_seed(0)
    model = build(cfg.model)
    x = torch.randint(cfg.model.vocab, (cfg.micro_bs, cfg.model.seq))
    out, h = [], x
    for i, u in enumerate(model.units):
        inp = h if i == 0 else h.detach().requires_grad_()
        with FlopCounterMode(display=False) as f:
            o = u(inp) if i < len(model.units) - 1 else u(inp, x)
        fwd = f.get_total_flops()
        with FlopCounterMode(display=False) as f:
            o.backward(torch.ones_like(o))
        out.append({'fwd': fwd, 'bwd': f.get_total_flops()})
        h = o
    return out


class _ActCounter:
    """Counts bytes autograd saves for backward (activations), excluding the weights themselves."""

    def __init__(self, params):
        self.skip = {p.untyped_storage().data_ptr() for p in params if p.numel()}
        self.seen, self.bytes = set(), 0

    def pack(self, t):
        if t.numel():
            ptr = t.untyped_storage().data_ptr()
            if ptr not in self.skip and ptr not in self.seen:
                self.seen.add(ptr)
                self.bytes += t.untyped_storage().nbytes()
        return t


def _rank_main(g, cfg: RunCfg, data: CharData, flops):
    torch.set_num_threads(1)
    N, r, hw, st = g.world, g.rank, g.hw, cfg.stage
    dt = DT[cfg.dtype]
    cdt = torch.float64 if cfg.dtype == 'fp64' else torch.float32     # compute + master precision
    names = unit_names(cfg.model)
    last = len(names) - 1
    sl = lambda L: slice(r * L.chunk, (r + 1) * L.chunk)
    opt_mem = g.host if cfg.offload else g.mem
    W_DIM = 'weights'

    # ------------------------------------------------------------------ set-up: what this rank keeps
    model = build(cfg.model, cfg.seed, cdt)
    units = list(model.units)
    lay = [Layout(u, N) for u in units]
    S = []
    for i, (u, L) in enumerate(zip(units, lay)):
        full = L.flatten([p.data for p in L.params], cdt)
        own = full.clone() if st == 0 else full[sl(L)].clone()
        s = {'master': own, 'm': torch.zeros_like(own), 'v': torch.zeros_like(own)}
        s['W'] = full.to(dt, copy=True) if st < 3 else own.to(dt, copy=True)
        s['G'] = torch.zeros(L.padded if st < 2 else L.chunk, dtype=dt)
        for k, cat in (('master', 'master'), ('m', 'adam_m'), ('v', 'adam_v')):
            opt_mem.add(f'{k}{i}', cat, s[k])
        g.mem.add(f'W{i}', W_DIM, s['W'])
        g.mem.add(f'G{i}', 'grads', s['G'])
        if st < 3:
            L.assign(s['W'], cdt)
        else:
            L.release()
        S.append(s)
    del full

    def gather(i, issue, label):
        """ZeRO-3: all-gather unit i's bf16 shards into the full unit, register it, point the module at it."""
        full, done = g.all_gather(S[i]['W'], label, 'ag', issue)
        g.mem.add(f'gath{i}', 'gathered', full)
        lay[i].assign(full, cdt)
        return done

    def free(i):
        lay[i].release()
        g.mem.drop(f'gath{i}')

    losses, steady, host_steady, wall = [], None, None, {'fwd': 0.0, 'bwd': 0.0, 'opt': 0.0}
    pending_w = [0.0] * len(units)        # when last step's weight all-gather of unit i lands (ZeRO-1/2)
    step_start, rec = [], {}
    act_bytes = [0] * len(units)
    gnorms = []
    live = []                              # GPU bytes still allocated at the end of every step (leak check)
    opt_elems = 0

    for step in range(cfg.steps):
        recording = step == cfg.record_step
        g.begin_step(recording)
        step_start.append(g.t)
        lr = lr_at(cfg, step)
        x, y = data.batch(step, N * cfg.micro_bs, cfg.model.seq, cfg.data_seed)
        x, y = x[r * cfg.micro_bs:(r + 1) * cfg.micro_bs], y[r * cfg.micro_bs:(r + 1) * cfg.micro_bs]

        # -------------------------------------------------------------- forward
        acts, h, t_unit = [], x, []
        issue_next = g.t
        for i, u in enumerate(units):
            g.mem.phase = f'fwd {names[i]}'
            if st == 3:
                done = gather(i, issue_next if cfg.prefetch else None, 'weight all-gather (fwd)')
                g.wait(done, 'wait: weights')
            elif st in (1, 2):
                g.wait(pending_w[i], 'wait: weights')
            issue_next = g.t
            t0 = time.perf_counter()
            inp = h if i == 0 else h.detach().requires_grad_()
            ac = _ActCounter(lay[i].params)
            with torch.autograd.graph.saved_tensors_hooks(ac.pack, lambda t: t):
                out = u(inp) if i < last else u(inp, y)
            wall['fwd'] += time.perf_counter() - t0
            g.compute(flops[i]['fwd'] / hw.flops, f'fwd {names[i]}')
            act_bytes[i] = ac.bytes
            g.mem.add(f'act{i}', 'activations', ac.bytes)
            if st == 3:
                free(i)
            acts.append((inp, out))
            h = out
        losses.append(out.item())

        # -------------------------------------------------------------- backward (unit by unit, last first)
        gout, pend_g = None, []
        issue_next = g.t
        for i in reversed(range(len(units))):
            g.mem.phase = f'bwd {names[i]}'
            if st == 3:
                done = gather(i, issue_next if cfg.prefetch else None, 'weight all-gather (bwd)')
                g.wait(done, 'wait: weights')
            issue_next = g.t
            inp, out = acts[i]
            t0 = time.perf_counter()
            out.backward(gout) if gout is not None else out.backward()
            wall['bwd'] += time.perf_counter() - t0
            g.compute(flops[i]['bwd'] / hw.flops, f'bwd {names[i]}')
            gout = inp.grad if i > 0 else None
            acts[i] = None
            g.mem.drop(f'act{i}')
            gflat = lay[i].flatten([p.grad for p in lay[i].params], dt)
            for p in lay[i].params:
                p.grad = None
            if st == 3:
                free(i)
            G = S[i]['G']
            if st == 0:      # all-reduce, as its two halves, the moment this bucket is ready
                G.copy_(gflat)
                shard, d = g.reduce_scatter(G, 'grad reduce-scatter', 'rs')
                full, d = g.all_gather(shard, 'grad all-gather', 'rs', d)
                G.copy_(full)
            elif st == 1:    # reduce-scatter; keep the reduced slice inside the (still full) gradient buffer
                G.copy_(gflat)
                shard, d = g.reduce_scatter(G, 'grad reduce-scatter', 'rs')
                G[sl(lay[i])] = shard
            else:            # the full gradient exists only as a transient bucket, then only my slice stays
                g.mem.add(f'bucket{i}', 'grad_bucket', gflat)
                shard, d = g.reduce_scatter(gflat, 'grad reduce-scatter', 'rs')
                G.copy_(shard)
                g.mem.drop(f'bucket{i}')
            del gflat
            pend_g.append(d)
            if not cfg.overlap:
                g.wait(d, 'wait: gradients')

        g.mem.phase = 'optimizer'
        for d in pend_g:
            g.wait(d, 'wait: gradients')

        # -------------------------------------------------------------- gradient clipping, sharded
        # The global norm needs every slice. Each rank sums the squares of its own slice; the N partial
        # sums are all-gathered (N floats) and added in rank order, so every arrangement gets the same
        # bits. DDP has every slice locally and does the same sum without communicating.
        def my_grad(i):
            G, L = S[i]['G'], lay[i]
            return G if st >= 2 else G[sl(L)] if st == 1 else G

        if st == 0:
            parts = []
            for c in range(N):
                acc = torch.zeros((), dtype=cdt)
                for i, L in enumerate(lay):
                    acc = acc + S[i]['G'][c * L.chunk:(c + 1) * L.chunk].to(cdt).pow(2).sum()
                parts.append(acc)
            parts = torch.stack(parts)
        else:
            acc = torch.zeros((), dtype=cdt)
            for i in range(len(lay)):
                acc = acc + my_grad(i).to(cdt).pow(2).sum()
            parts, d = g.all_gather(acc.reshape(1), 'grad-norm all-gather', 'rs')
            g.wait(d, 'wait: grad norm')
        total = parts[0]
        for c in range(1, N):
            total = total + parts[c]
        norm = math.sqrt(total.item()) / N        # the slices hold SUMS over N ranks; the gradient is their mean
        coef = min(1.0, cfg.clip / (norm + 1e-6))
        gnorms.append(norm)

        # -------------------------------------------------------------- optimizer on what I own
        t0 = time.perf_counter()
        elems = 0
        for i, L in enumerate(lay):
            s = S[i]
            gr = my_grad(i)
            if cfg.offload:
                g.wait(g.pcie(nbytes(gr), 'grads GPU->CPU'), 'wait: PCIe')
            gg = gr.to(cdt) * (coef / N)
            adamw_(s['master'], s['m'], s['v'], gg, lr, step + 1, *cfg.betas, cfg.eps, cfg.wd)
            elems += s['master'].numel()
        wall['opt'] += time.perf_counter() - t0
        opt_elems = elems
        g.compute(elems * ADAM_BYTES / (hw.host_bw if cfg.offload else hw.hbm),
                  'optimizer (CPU)' if cfg.offload else 'optimizer')

        # -------------------------------------------------------------- the new weights
        g.mem.phase = 'weights'
        for i, L in enumerate(lay):
            s = S[i]
            new = s['master'].to(dt)
            if cfg.offload:
                d = g.pcie(nbytes(new), 'weights CPU->GPU')
                g.wait(d, 'wait: PCIe')
            if st == 0 or st == 3:
                s['W'].copy_(new)
            else:
                full, d = g.all_gather(new, 'weight all-gather', 'ag')
                s['W'].copy_(full)
                pending_w[i] = d
                if not cfg.overlap:
                    g.wait(d, 'wait: weights')
            if st < 3:
                L.assign(s['W'], cdt)

        g.mem.phase = 'idle'
        live.append(g.mem.total)
        if recording:
            rec = {'sent': {f'{k[0]}|{k[1]}': v for k, v in g.sent.items()}, 'calls': dict(g.calls),
                   'events': list(g.events), 'trace': list(g.mem.trace),
                   'peak': g.mem.peak, 'peak_cat': g.mem.peak_cat, 'peak_phase': g.mem.peak_phase,
                   'host_peak': g.host.peak}
            steady, host_steady = g.mem.snapshot(), g.host.snapshot()
            g.recording = g.mem.tracing = g.host.tracing = False
    step_start.append(max(g.t, *[g.clock[k] for k in list(g.clock)]))

    out = {'rank': r, 'losses': losses, 'gnorms': gnorms, 'steady': steady, 'host_steady': host_steady,
           'act_bytes': act_bytes, 'live': live, 'opt_elems': opt_elems, 'wall': wall, 'step_start': step_start,
           'end_live': g.mem.snapshot(), **rec}
    # what this rank owns at the end, for the "are all arrangements identical?" checks
    out['_master'] = [s['master'] for s in S]
    out['_W'] = [s['W'] for s in S]
    out['_layout'] = [(L.numel, L.padded, L.chunk) for L in lay]
    return out


class Result:
    def __init__(self, cfg: RunCfg, hw, ranks, flops, wall_s):
        self.cfg, self.hw, self.ranks, self.flops, self.wall_s = cfg, hw, ranks, flops, wall_s
        self.layout = ranks[0]['_layout']
        self.psi = sum(n for n, _, _ in self.layout)          # true parameter count
        self.psi_padded = sum(p for _, p, _ in self.layout)

    # ---------------- training
    @property
    def losses(self):
        return [sum(r['losses'][k] for r in self.ranks) / len(self.ranks) for k in range(self.cfg.steps)]

    def full(self, key='_master'):
        """Reassemble the full flat tensor of every unit from the ranks that own its slices."""
        st, outs = self.cfg.stage, []
        own_full = (key == '_master' and st == 0) or (key == '_W' and st < 3)
        for i, (n, padded, chunk) in enumerate(self.layout):
            t = self.ranks[0][key][i] if own_full else torch.cat([r[key][i] for r in self.ranks])
            outs.append(t[:n])
        return torch.cat(outs)

    def drop_weights(self):
        """Free the per-rank weight copies kept for the equality checks (DDP at N=32 keeps ~160 MB of them)."""
        for r in self.ranks:
            r.pop('_master', None)
            r.pop('_W', None)
        return self

    # ---------------- memory
    def mem(self, rank=0, which='steady'):
        return self.ranks[rank][which]

    def bytes_per_param(self, rank=0, which='steady', cats=STATE, padded=False):
        """Bytes this GPU holds per model parameter. padded=True divides by the stored (padded) element count,
        which is what the lesson's formulas describe exactly; the default divides by the true parameter count."""
        m = self.ranks[rank][which]
        return sum(m[c] for c in cats) / (self.psi_padded if padded else self.psi)

    def gpu_bytes(self, rank=0, which='steady'):
        return sum(self.ranks[rank][which].values())

    # ---------------- communication
    def sent(self, rank=0):
        """Bytes this rank put on the wire in one step, by (collective, link)."""
        return self.ranks[rank]['sent']

    def sent_total(self, rank=0, links=('nvlink', 'ib')):
        return sum(v for k, v in self.ranks[rank]['sent'].items() if k.split('|')[1] in links)

    def P(self, padded=False):
        """The lesson's P: one full copy of the weights in bf16, in bytes."""
        return (self.psi_padded if padded else self.psi) * torch.tensor([], dtype=DT[self.cfg.dtype]).element_size()

    # ---------------- time
    def step_time(self, rank=None):
        """Simulated seconds for the recorded step (start of step k to start of step k+1)."""
        k = self.cfg.record_step
        rs = self.ranks if rank is None else [self.ranks[rank]]
        return max(r['step_start'][k + 1] - r['step_start'][k] for r in rs)

    def compute_time(self):
        return sum(f['fwd'] + f['bwd'] for f in self.flops) / self.hw.flops

    def summary(self):
        m = self.ranks[0]
        return {
            'name': self.cfg.name, 'stage': self.cfg.stage, 'world': self.cfg.world, 'offload': self.cfg.offload,
            'overlap': self.cfg.overlap, 'dtype': self.cfg.dtype, 'psi': self.psi, 'psi_padded': self.psi_padded,
            'bytes_per_param': self.bytes_per_param(), 'steady': m['steady'], 'host_steady': m['host_steady'],
            'peak': m['peak'], 'peak_cat': m['peak_cat'], 'peak_phase': m['peak_phase'], 'host_peak': m['host_peak'],
            'act_bytes': sum(m['act_bytes']), 'sent': m['sent'], 'calls': m['calls'],
            'sent_total': self.sent_total(), 'P': self.P(), 'opt_elems': m['opt_elems'],
            'fwd_bwd_flops': sum(f['fwd'] + f['bwd'] for f in self.flops),
            'step_time': self.step_time(), 'compute_time': self.compute_time(),
            'final_loss': self.losses[-1], 'wall_s': self.wall_s, 'wall': m['wall'],
        }


def run(cfg: RunCfg, data: CharData = None, quiet=False, keep_weights=True) -> Result:
    import sys
    sys.setswitchinterval(float(__import__("os").environ.get("S12_SWITCH", "2e-3")))   # measured: 1-5 ms is ~2x faster than 0.1 ms or 20 ms for 32 threads
    assert not (cfg.offload and cfg.stage == 0), 'offload partitions the optimizer state: stage >= 1'
    data = data or CharData()
    torch.set_num_threads(1)          # 32 threads already; one core each
    flops = unit_flops(cfg)
    tok = cfg.micro_bs * cfg.model.seq
    hw = calibrated(cfg.card, tok, gpus_per_node=cfg.gpus_per_node, latency=cfg.latency)
    # The lesson prices a step at 6 * params * tokens FLOPs. My tiny model really does more than that per token
    # (attention is a big share at width 128), so the card is sped up by the same factor: one toy step then
    # takes exactly as long, relative to its own 2P transfer, as the lesson's 30B step does to its 2P.
    psi = sum(p.numel() for p in build(cfg.model).parameters())
    hw = with_(hw, flops=hw.flops * sum(f['fwd'] + f['bwd'] for f in flops) / (6 * psi * tok))
    t0 = time.perf_counter()
    ranks = Cluster(cfg.world, hw, mode=cfg.comm_mode).run(_rank_main, cfg, data, flops)
    res = Result(cfg, hw, ranks, flops, time.perf_counter() - t0)
    if not keep_weights:
        res.drop_weights()
    if not quiet:
        print(f'{cfg.name:<22} N={cfg.world:<3} {res.bytes_per_param():6.3f} B/param  '
              f'sent {res.sent_total() / res.P():.3f} P  loss {res.losses[-1]:.4f}  ({res.wall_s:.0f}s)')
    return res
