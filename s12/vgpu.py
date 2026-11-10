"""32 virtual GPUs = 32 Python threads that can only talk to each other by passing messages.

Every virtual GPU has
  * a memory ledger: every tensor it keeps is registered with a category and its real byte size,
    so "how much memory does rank 7 hold right now" is a sum, not a formula;
  * a mailbox: the only way data reaches it is `send()` from another rank (the ring collectives below
    send only to the right-hand neighbour, exactly like NCCL's ring);
  * clocks, one per stream (compute, all-gather, reduce-scatter, PCIe), like CUDA streams. A transfer
    takes bytes / bandwidth of the link it crosses (NVLink inside a node, InfiniBand between nodes), and
    a receiver cannot use a message before it arrives. That gives a simulated step time with real
    overlap and real waiting, on top of the real arithmetic.
"""
import queue
import threading
import traceback
from collections import defaultdict

import torch

CATS = ['weights', 'grads', 'master', 'adam_m', 'adam_v', 'gathered', 'grad_bucket', 'activations']
STATE = CATS[:5]
LABEL = {'weights': 'weights (bf16)', 'grads': 'gradients (bf16)', 'master': 'master copy (fp32)',
         'adam_m': 'Adam m (fp32)', 'adam_v': 'Adam v (fp32)', 'gathered': 'gathered layer (transient)',
         'grad_bucket': 'gradient bucket (transient)', 'activations': 'activations'}
# the lesson's ZeRO widget palette, so my figures read like its figures
COLOR = {'weights': '#0891b2', 'grads': '#d97706', 'master': '#7c3aed', 'adam_m': '#159447',
         'adam_v': '#6cc48a', 'gathered': '#8fd3e8', 'grad_bucket': '#f3c47e', 'activations': '#a3abb5'}


def nbytes(t) -> int:
    return t if isinstance(t, int) else t.numel() * t.element_size()


class Ledger:
    """Live bytes per category, the peak, and (optionally) a trace of every allocation and free."""

    def __init__(self):
        self.live, self.cat = {}, dict.fromkeys(CATS, 0)
        self.total = self.peak = 0
        self.peak_cat, self.peak_phase = dict(self.cat), ''
        self.phase, self.tracing, self.trace = '', False, []

    def add(self, key, cat, t):
        n = nbytes(t)
        assert key not in self.live, f'{key} allocated twice'
        self.live[key] = (cat, n)
        self.cat[cat] += n
        self.total += n
        if self.total > self.peak:
            self.peak, self.peak_cat, self.peak_phase = self.total, dict(self.cat), self.phase
        self._log('+', key)

    def drop(self, key):
        cat, n = self.live.pop(key)
        self.cat[cat] -= n
        self.total -= n
        self._log('-', key)

    def reset_peak(self):
        self.peak, self.peak_cat, self.peak_phase = self.total, dict(self.cat), self.phase

    def snapshot(self):
        return dict(self.cat)

    def _log(self, op, key):
        if self.tracing:
            self.trace.append({'phase': self.phase, 'ev': op + key, **self.cat})


class Cluster:
    """mode='mailbox': every hop of every ring is a real message between threads (slow, literal).
    mode='rendezvous': the ranks meet at a barrier and one thread plays the same ring, hop by hop, for all of
    them: the same chunks, in the same order, added in the same order, with the same byte and clock
    bookkeeping. tests/test_core.py checks the two give bit-identical results; rendezvous is ~50x faster
    because Python threads hand off a GIL, not a GPU's worth of bandwidth."""

    def __init__(self, world: int, hw, timeout: float = 300.0, mode: str = 'rendezvous'):
        self.world, self.hw, self.timeout, self.mode = world, hw, timeout, mode
        self.boxes = [[queue.Queue() for _src in range(world)] for _dst in range(world)]   # [to][from]
        self.abort = threading.Event()
        self.slots, self.outs = [None] * world, [None] * world
        self.barrier = threading.Barrier(world, action=self._play, timeout=timeout)
        self.gpus = [VirtualGPU(self, r) for r in range(world)]

    # ------------------------------------------------------------ rendezvous collectives
    def meet(self, rank, **slot):
        self.slots[rank] = slot
        self.barrier.wait()
        return self.outs[rank]

    def _play(self):
        sl = self.slots
        kind, label, stream = sl[0]['kind'], sl[0]['label'], sl[0]['stream']
        assert all(x['kind'] == kind and x['label'] == label for x in sl), [(x['kind'], x['label']) for x in sl]
        N = self.world
        ar = torch.arange(N)
        gpus = self.gpus
        t0 = [g._begin(stream, x['issue'], label) for g, x in zip(gpus, sl)]
        if kind == 'rs':
            X = torch.stack([x['t'].view(N, -1) for x in sl])           # [rank, chunk, elems]
            buf = X[ar, (ar - 1) % N]
            for s in range(N - 1):
                self._hop(buf[0], stream, label, t0)
                buf = buf.roll(1, 0) + X[ar, (ar - s - 2) % N]            # receive from the left, add my copy
            res = [buf[r] for r in range(N)]
            hop_bytes = nbytes(X[0, 0])
        else:  # 'ag'
            S = torch.stack([x['t'] for x in sl])                        # [rank, *shard]
            out = torch.empty((N, N) + tuple(S.shape[1:]), dtype=S.dtype)
            out[ar, ar] = S
            buf = S
            for s in range(N - 1):
                self._hop(buf[0], stream, label, t0)
                buf = buf.roll(1, 0)
                out[ar, (ar - s - 1) % N] = buf
            res = [out[r].reshape(-1) for r in range(N)]
            hop_bytes = nbytes(S[0])
        for r, g in enumerate(gpus):
            self.outs[r] = (res[r], g._end(stream, label, t0[r], (N - 1) * hop_bytes))

    def _hop(self, sample, stream, label, entry):
        """One ring step for the clocks and byte counters: rank r sends to r+1 at the same time as everyone.
        A send cannot start before the receiver has entered the collective (NCCL writes into a buffer the
        receiver posts), nor before the link has finished the previous chunk."""
        N, b, g = self.world, nbytes(sample), self.gpus
        arrive = [0.0] * N
        for r in range(N):
            dst = (r + 1) % N
            start = max(g[r].clock[stream], g[r].link_free[(dst, stream)], entry[dst])
            arrive[dst] = start + self.hw.latency + b / self.hw.link_bw(r, dst)
            g[r].link_free[(dst, stream)] = arrive[dst]
            g[r].sent[(label, self.hw.link_kind(r, dst))] += b
        for r in range(N):
            g[r].clock[stream] = max(g[r].clock[stream], arrive[r])

    def run(self, fn, *args):
        """Run fn(gpu, *args) on every virtual GPU at once (one thread each), SPMD style."""
        out, errs = [None] * self.world, []

        def body(g):
            try:
                out[g.rank] = fn(g, *args)
            except BaseException:  # noqa: BLE001 - surface the first failure, unblock the others
                errs.append((g.rank, traceback.format_exc()))
                self.abort.set()
                self.barrier.abort()

        ts = [threading.Thread(target=body, args=(g,), name=f'vgpu{g.rank}') for g in self.gpus]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errs:
            real = [e for e in errs if 'BrokenBarrierError' not in e[1] and 'another virtual GPU' not in e[1]]
            r, tb = sorted(real or errs)[0]
            raise RuntimeError(f'virtual GPU {r} failed:\n{tb}')
        return out


class VirtualGPU:
    def __init__(self, cluster: Cluster, rank: int):
        self.cl, self.rank, self.world, self.hw = cluster, rank, cluster.world, cluster.hw
        self.node = rank // self.hw.gpus_per_node
        self.mem, self.host = Ledger(), Ledger()        # GPU memory, and the node's CPU memory (offload)
        self.clock = defaultdict(float)                  # stream -> time (s)
        self.link_free = defaultdict(float)              # (dst, stream) -> time the link is next free
        self.sent = defaultdict(int)                     # (label, link) -> bytes sent this step
        self.calls = defaultdict(int)                    # label -> collective calls this step
        self.events, self.recording = [], False
        self._peer_ready = 0.0

    # ---------------------------------------------------------------- time
    @property
    def t(self):
        return self.clock['compute']

    def compute(self, seconds, label):
        t0 = self.clock['compute']
        self.clock['compute'] = t0 + seconds
        self._ev('compute', label, t0, t0 + seconds)

    def wait(self, done, label='wait'):
        """The compute stream needs a result that a transfer finishes at `done`."""
        if done > self.clock['compute'] + 1e-15:
            self._ev('idle', label, self.clock['compute'], done)
            self.clock['compute'] = done

    def _ev(self, stream, label, t0, t1, b=0):
        if self.recording:
            self.events.append((stream, label, t0, t1, b))

    # ---------------------------------------------------------------- point to point
    def send(self, dst, t, stream, label):
        b = nbytes(t)
        start = max(self.clock[stream], self.link_free[(dst, stream)], self._peer_ready)
        arrive = start + self.hw.latency + b / self.hw.link_bw(self.rank, dst)
        self.link_free[(dst, stream)] = arrive
        self.sent[(label, self.hw.link_kind(self.rank, dst))] += b
        self.cl.boxes[dst][self.rank].put((self.rank, label, t.clone(), arrive))

    def recv(self, src, stream, label):
        waited = 0.0
        while True:
            try:
                s, lab, t, arrive = self.cl.boxes[self.rank][src].get(timeout=0.5)
                break
            except queue.Empty:
                waited += 0.5
                if self.cl.abort.is_set():
                    raise RuntimeError('another virtual GPU failed')
                if waited > self.cl.timeout:
                    raise TimeoutError(f'rank {self.rank} waited {waited}s for {label} from {src}')
        assert s == src and lab == label, f'rank {self.rank} expected {label} from {src}, got {lab} from {s}'
        self.clock[stream] = max(self.clock[stream], arrive)
        return t

    # ---------------------------------------------------------------- ring collectives
    def _begin(self, stream, issue, label):
        self.clock[stream] = max(self.clock[stream], self.t if issue is None else issue)
        self.calls[label] += 1
        return self.clock[stream]

    def _handshake(self, stream, label, t0):
        """Mailbox mode: tell my left neighbour when I entered the collective (it sends to me), and learn when my
        right neighbour did (I send to it). Zero bytes: this is NCCL's 'buffer posted' signal, not data."""
        N, r = self.world, self.rank
        self.cl.boxes[(r - 1) % N][r].put((r, label + '/ready', None, t0))
        _, lab, _, t_right = self.cl.boxes[r][(r + 1) % N].get(timeout=self.cl.timeout)
        assert lab == label + '/ready', lab
        self._peer_ready = t_right

    def _end(self, stream, label, t0, b):
        self._ev(stream, label, t0, self.clock[stream], b)
        return self.clock[stream]

    def reduce_scatter(self, flat, label, stream='rs', issue=None):
        """Ring reduce-scatter of a flat tensor of length N*chunk. Rank r ends with the SUM of chunk r.

        Step s: send the chunk you just accumulated to the right, receive one from the left, add your own
        copy of it. After N-1 steps every rank holds one fully-summed chunk. Each rank sends (N-1)/N of
        the tensor. The summation order is fixed by the ring, not by thread timing, so it is deterministic.
        """
        N, r = self.world, self.rank
        if N == 1:
            self.calls[label] += 1
            return flat.clone(), self.t
        if self.cl.mode == 'rendezvous':
            return self.cl.meet(r, kind='rs', t=flat, stream=stream, issue=issue, label=label)
        ch = flat.view(N, -1)
        t0 = self._begin(stream, issue, label)
        self._handshake(stream, label, t0)
        buf = ch[(r - 1) % N]
        for s in range(N - 1):
            self.send((r + 1) % N, buf, stream, label)
            got = self.recv((r - 1) % N, stream, label)
            buf = got + ch[(r - s - 2) % N]
        return buf, self._end(stream, label, t0, (N - 1) * nbytes(ch[0]))

    def all_gather(self, shard, label, stream='ag', issue=None):
        """Ring all-gather: rank r starts with chunk r and ends with all N chunks, in rank order."""
        N, r = self.world, self.rank
        if N == 1:
            self.calls[label] += 1
            return shard.clone(), self.t
        if self.cl.mode == 'rendezvous':
            return self.cl.meet(r, kind='ag', t=shard, stream=stream, issue=issue, label=label)
        out = torch.empty((N,) + tuple(shard.shape), dtype=shard.dtype)
        out[r] = shard
        t0 = self._begin(stream, issue, label)
        self._handshake(stream, label, t0)
        buf = shard
        for s in range(N - 1):
            self.send((r + 1) % N, buf, stream, label)
            buf = self.recv((r - 1) % N, stream, label)
            out[(r - s - 1) % N] = buf
        return out.reshape(-1), self._end(stream, label, t0, (N - 1) * nbytes(shard))

    def all_reduce(self, flat, label, stream='rs', issue=None):
        """All-reduce = reduce-scatter, then all-gather of the reduced chunks (Section 4 of the lesson)."""
        shard, d = self.reduce_scatter(flat, label, stream, issue)
        return self.all_gather(shard, label, stream, d)

    def pcie(self, b, label, issue=None):
        """A copy between this GPU and its own CPU (offload). Bytes only; there is no other rank involved."""
        t0 = max(self.clock['pcie'], self.t if issue is None else issue)
        self.clock['pcie'] = t0 + b / self.hw.bw_pcie
        self.sent[(label, 'pcie')] += b
        self.calls[label] += 1
        self._ev('pcie', label, t0, self.clock['pcie'], b)
        return self.clock['pcie']

    # ---------------------------------------------------------------- bookkeeping per step
    def begin_step(self, record: bool):
        self.sent.clear()
        self.calls.clear()
        self.recording = record
        self.mem.tracing = self.host.tracing = record
        if record:
            self.events.clear()
            self.mem.trace.clear()
            self.mem.reset_peak()
            self.host.reset_peak()
