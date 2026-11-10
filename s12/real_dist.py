"""The reality check: the same model, data and optimizer on 32 *real* processes with torch.distributed (gloo, CPU),
using PyTorch's own implementations instead of mine:

    ddp          DistributedDataParallel + AdamW                                  (data parallelism)
    zero1        DDP + torch.distributed.optim.ZeroRedundancyOptimizer(AdamW)    (PyTorch's ZeRO-1)
    fsdp2_zero2  fully_shard(..., reshard_after_forward=False)                   (FSDP's ZeRO-2 analogue)
    fsdp2_zero3  fully_shard(..., reshard_after_forward=True)                    (ZeRO-3, the lesson's FSDP2)

Everything is fp32 here (gloo on CPU), so the training state is 4 + 4 + 8 = 16 B/param, not the mixed-precision
2 + 2 + 4 + 8. What is measured on every rank: bytes of the parameters, gradients and optimizer state it holds
(local shards only), the payload of every collective it calls, and the loss at every step.

    python -m s12.real_dist --world 4 --strategy ddp          # locally
    modal run tools/modal_app.py                              # 32 ranks x 4 strategies on Modal
"""
import argparse
import json
import math
import os
import socket
import time
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

STRATEGIES = ['ddp', 'zero1', 'fsdp2_zero2', 'fsdp2_zero3']
LR, WARMUP, BETAS, EPS, WD, CLIP, MICRO_BS = 3e-3, 10, (0.9, 0.95), 1e-8, 0.1, 1.0, 4


def lr_at(step, steps):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    q = (step - WARMUP) / max(1, steps - WARMUP)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * q)))


class CommCounter:
    """Wraps the torch.distributed collectives so every call's payload is recorded (bytes, per op).
    Payload = the full tensor for all-reduce/broadcast, the gathered output for all-gather, the input for
    reduce-scatter. (FSDP2 in recent PyTorch calls all_gather_single / reduce_scatter_single.)"""
    # op -> (positional index, keyword names) of the tensor whose size is the payload
    OPS = {'all_reduce': (0, ('tensor',)), 'broadcast': (0, ('tensor',)),
           'all_gather_into_tensor': (0, ('output_tensor',)), 'reduce_scatter_tensor': (1, ('input',)),
           'all_gather_single': (0, ('output_tensor', 'output')), 'reduce_scatter_single': (1, ('input', 'input_tensor'))}

    def __init__(self):
        self.bytes, self.calls, self.on = defaultdict(int), defaultdict(int), True
        for op in self.OPS:
            if hasattr(dist, op):
                self._wrap(op)

    def _wrap(self, op):
        fn, (pos, keys) = getattr(dist, op), self.OPS[op]

        def wrapped(*a, **k):
            if self.on:
                t = a[pos] if len(a) > pos else next(k[n] for n in keys if n in k)
                self.bytes[op] += t.numel() * t.element_size()
                self.calls[op] += 1
            return fn(*a, **k)

        setattr(dist, op, wrapped)

    @staticmethod
    def hook():
        """DDP talks to the process group from C++. Routing its buckets through the stock Python all-reduce hook
        makes them pass through the wrapped dist.all_reduce, so they are counted like everything else."""
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        return lambda state, bucket: default_hooks.allreduce_hook(None, bucket)


def local(t):
    return t.to_local() if hasattr(t, 'to_local') else t


def nb(t):
    t = local(t)
    return t.numel() * t.element_size()


def worker(rank, world, strategy, steps, port, out_path):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), USE_LIBUV='0')
    dist.init_process_group('gloo', rank=rank, world_size=world)
    torch.set_num_threads(1)
    from torch.nn.parallel import DistributedDataParallel as DDP
    from s12.model import CharData, GPTCfg, build
    cfg, data = GPTCfg(), CharData()
    cc = CommCounter()
    model = build(cfg, seed=0)
    names = {id(p): n for n, p in model.named_parameters()}
    psi = sum(p.numel() for p in model.parameters())
    kw = dict(lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, foreach=False)

    if strategy in ('ddp', 'zero1'):
        net = DDP(model)
        net.register_comm_hook(None, cc.hook())
        if strategy == 'ddp':
            opt = torch.optim.AdamW(net.parameters(), **kw)
        else:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            opt = ZeroRedundancyOptimizer(net.parameters(), optimizer_class=torch.optim.AdamW, **kw)
    else:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        mesh = init_device_mesh('cpu', (world,))
        resh = strategy == 'fsdp2_zero3'
        for u in model.units:
            fully_shard(u, mesh=mesh, reshard_after_forward=resh)
        fully_shard(model, mesh=mesh, reshard_after_forward=resh)
        net = model
        opt = torch.optim.AdamW(net.parameters(), **kw)

    def opt_state_bytes():
        st = opt.optim.state if strategy == 'zero1' else opt.state
        return sum(nb(v) for s in st.values() for v in s.values() if torch.is_tensor(v) and v.numel() > 1)

    losses, rec, t0 = [], {}, time.perf_counter()
    for step in range(steps):
        cc.bytes.clear(), cc.calls.clear()
        x, y = data.batch(step, world * MICRO_BS, cfg.seq)
        x, y = x[rank * MICRO_BS:(rank + 1) * MICRO_BS], y[rank * MICRO_BS:(rank + 1) * MICRO_BS]
        loss = net(x, y)
        loss.backward()
        grads = sum(nb(p.grad) for p in net.parameters() if p.grad is not None)
        cc.on = False                  # clipping's norm reduction is bookkeeping, not ZeRO traffic
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        cc.on = True
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps)
        opt.step()
        opt.zero_grad(set_to_none=True)
        cc.on = False
        l = loss.detach().clone()
        dist.all_reduce(l)
        cc.on = True
        losses.append(l.item() / world)
        if step == 1:
            rec = {'params': sum(nb(p) for p in net.parameters()), 'grads': grads, 'opt': opt_state_bytes(),
                   'comm_bytes': dict(cc.bytes), 'comm_calls': dict(cc.calls),
                   'local_numel': {n: local(p).numel() for n, p in net.named_parameters()}}
    rec['wall_s'] = time.perf_counter() - t0
    every = [None] * world
    dist.all_gather_object(every, rec)
    if rank == 0:
        res = {'strategy': strategy, 'world': world, 'steps': steps, 'psi': psi, 'torch': torch.__version__,
               'losses': losses, 'ranks': every}
        with open(out_path, 'w') as f:
            json.dump(res, f)
    dist.barrier()
    dist.destroy_process_group()


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


def launch(strategy, world=32, steps=30, out_path=None):
    out_path = out_path or f'/tmp/real_{strategy}_{world}.json'
    mp.start_processes(worker, args=(world, strategy, steps, free_port(), out_path), nprocs=world, join=True,
                       start_method='spawn')
    with open(out_path) as f:
        return json.load(f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', type=int, default=4)
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--strategy', default='ddp')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    import tempfile
    r = launch(a.strategy, a.world, a.steps, a.out or os.path.join(tempfile.gettempdir(), f'real_{a.strategy}.json'))
    r0 = r['ranks'][0]
    print(a.strategy, 'loss', [round(v, 4) for v in r['losses']])
    print('  B/param  params %.3f grads %.3f opt %.3f' % (r0['params'] / r['psi'], r0['grads'] / r['psi'], r0['opt'] / r['psi']))
    print('  comm', r0['comm_bytes'], r0['comm_calls'], 'psi', r['psi'])
