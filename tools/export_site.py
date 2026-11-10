"""results/site.<profile>.json -> docs/data.js (what the companion page reads). Compresses the memory traces."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATS = ['weights', 'grads', 'master', 'adam_m', 'adam_v', 'gathered', 'grad_bucket', 'activations']


def main(profile='full'):
    site = json.loads((ROOT / 'results' / f'site.{profile}.json').read_text())
    summ = json.loads((ROOT / 'results' / f'summary.{profile}.json').read_text())
    for st in site['stages'].values():
        st['trace'] = [{'p': e['phase'], 'e': e['ev'], 'v': [e[c] for c in CATS]} for e in st['trace']]
        st['steady'] = [[s[c] for c in CATS] for s in st['steady']]
        st['host'] = [st['host'][c] for c in CATS]
        t0 = st['step_start']
        st['events'] = [[s, l, round((a - t0) * 1e6, 4), round((b - t0) * 1e6, 4)] for s, l, a, b, _ in st['events']]
    site['cats'] = CATS
    site['summary'] = {k: summ.get(k) for k in ('psi', 'psi_padded', 'final_loss', 'fp64_max_weight_diff', 'bf16_one_gpu_loss_diff_max',
                                                 'memory', 'comm', 'work', 'topology', 'real', 'steps', 'world', 'torch',
                                                 'collectives', 'offload', 'same_model')}
    out = ROOT / 'docs' / 'data.js'
    out.parent.mkdir(exist_ok=True)
    out.write_text('window.S12 = ' + json.dumps(site, separators=(',', ':')) + ';\n', encoding='utf-8')
    print(f'wrote {out} ({out.stat().st_size / 1024:.0f} KiB)')


if __name__ == '__main__':
    main(*sys.argv[1:])
