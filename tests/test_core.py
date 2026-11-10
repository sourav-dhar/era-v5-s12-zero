"""python -m pytest -q tests   (about a minute on a laptop CPU)"""
import pytest
import torch

from s12 import theory as T
from s12.hw import Hardware
from s12.model import CharData
from s12.vgpu import Cluster, STATE
from s12.zero import RunCfg, run

DATA = CharData()


def _collect(N, mode, fn):
    cl = Cluster(N, Hardware(), mode=mode)
    return cl.run(fn), cl


@pytest.mark.parametrize('mode', ['mailbox', 'rendezvous'])
@pytest.mark.parametrize('N', [2, 5, 8])
def test_reduce_scatter_then_all_gather_is_all_reduce(N, mode):
    torch.manual_seed(N)
    xs = [torch.randn(N * 6, dtype=torch.float64) for _ in range(N)]
    exact = torch.stack(xs).sum(0)

    def fn(g):
        shard, _ = g.reduce_scatter(xs[g.rank], 'rs')
        full, _ = g.all_gather(shard, 'ag')
        return shard, full, dict(g.sent)

    out, _ = _collect(N, mode, fn)
    for r, (shard, full, sent) in enumerate(out):
        assert torch.allclose(shard, exact.view(N, -1)[r])            # rank r owns the sum of chunk r
        assert torch.equal(full, out[0][1])                            # everyone ends with the same bits
        assert torch.allclose(full, exact)
        per = xs[0].numel() * 8
        assert sum(v for k, v in sent.items() if k[0] == 'rs') == (N - 1) * per // N   # (N-1)/N of the data
        assert sum(v for k, v in sent.items() if k[0] == 'ag') == (N - 1) * per // N


def test_mailbox_and_rendezvous_are_bit_identical():
    for st in range(4):
        a = run(RunCfg(stage=st, world=4, steps=2, comm_mode='mailbox'), DATA, quiet=True)
        b = run(RunCfg(stage=st, world=4, steps=2), DATA, quiet=True)
        assert a.losses == b.losses and torch.equal(a.full(), b.full())
        assert a.ranks[1]['events'] == b.ranks[1]['events']


@pytest.fixture(scope='module')
def stages():
    return {st: run(RunCfg(stage=st, world=8, steps=3), DATA, quiet=True) for st in range(4)} | \
           {f'off{st}': run(RunCfg(stage=st, world=8, steps=3, offload=True), DATA, quiet=True) for st in (1, 2, 3)}


def test_every_arrangement_trains_the_same_model(stages):
    ref = stages[0]
    for k, r in stages.items():
        assert r.losses == ref.losses, k
        assert torch.equal(r.full('_master'), ref.full('_master')), k
        assert torch.equal(r.full('_W'), ref.full('_W')), k


def test_memory_matches_the_lesson_formula_per_category(stages):
    for k, r in stages.items():
        st, off = r.cfg.stage, r.cfg.offload
        want = T.bytes_per_param(st, 8, by_cat=True, offload=off)
        got = {c: r.ranks[0]['steady'][c] / r.psi_padded for c in STATE}
        assert got == pytest.approx(want, abs=1e-12), (k, got, want)


def test_communication_is_2P_2P_2P_3P(stages):
    for st in range(4):
        r = stages[st]
        wire = sum(v for k, v in r.sent(0).items() if k.split('|')[0] != 'grad-norm all-gather' and k.split('|')[1] != 'pcie')
        assert wire / r.P(padded=True) == pytest.approx(T.comm_P(st, 8), abs=1e-12)


def test_data_parallel_equals_one_big_gpu_in_fp64():
    big = run(RunCfg(stage=0, world=1, micro_bs=16, steps=3, dtype='fp64'), DATA, quiet=True)
    z3 = run(RunCfg(stage=3, world=4, micro_bs=4, steps=3, dtype='fp64'), DATA, quiet=True)
    assert (big.full() - z3.full()).abs().max().item() < 1e-13
