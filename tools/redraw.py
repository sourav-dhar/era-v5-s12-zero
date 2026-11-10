"""Redraw the figures that only need saved results (results/site.<profile>.json), without re-running the notebook."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s12 import plots as PL  # noqa: E402
from s12.vgpu import CATS  # noqa: E402


class Saved:
    """Just enough of s12.zero.Result, rebuilt from the exported JSON, for the figures that need nothing else."""

    def __init__(self, st, world, gpn, losses):
        self.cfg = NS(world=world, gpus_per_node=gpn, stage=st['stage'], offload=st['offload'], name=st['name'])
        self.ranks = [{'steady': dict(zip(CATS, v)) if isinstance(v, list) else v} for v in st['steady']]
        self._bpp, self.losses = st['bpp'], losses

    def bytes_per_param(self, padded=True, **_):
        return self._bpp


def main(profile='full'):
    T = json.loads((ROOT / 'results' / f'site.{profile}.json').read_text(encoding='utf-8'))
    R = {int(k): Saved(v, T['world'], T['gpus_per_node'], T['losses']) for k, v in T['stages'].items() if k.isdigit()}
    PL.hero(R, name='00_hero')
    PL.losses(R, NS(losses=T['one_gpu_losses']), name='03_same_model')
    PL.ladder({int(s): {int(n): v for n, v in m.items()} for s, m in T['ladder'].items()})
    print('redrew 00_hero, 03_same_model, 11_ladder_30B')


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    main(*sys.argv[1:])
