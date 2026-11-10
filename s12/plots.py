"""Figures. Each function takes measured results and draws one figure; the notebook decides what to draw."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from . import theory as T
from .hw import CARD_GIB, GIB
from .style import (CAT_COLOR, CAT_LABEL, CATS, GRID, INK, INK2, MUTED, OFFLOAD_COLOR, RED, STAGE_COLOR,
                    STAGE_NAME, SURF, fmt_bytes, label_end, save, suptitle, title)
from .vgpu import STATE

SHORT = {'weights': 'W', 'grads': 'G', 'master': 'fp32', 'adam_m': 'm', 'adam_v': 'v'}


def _color(res):
    return OFFLOAD_COLOR if res.cfg.offload else STAGE_COLOR[res.cfg.stage]


# --------------------------------------------------------------------------- 00 hero: 32 GPUs x 4 arrangements
def hero(results, name='00_hero'):
    """Every GPU of the 32 as a column of what it stores, for each arrangement, on one shared scale."""
    stages = [results[s] for s in range(4)]
    N, gpn = stages[0].cfg.world, stages[0].cfg.gpus_per_node
    top = max(sum(r.ranks[k]['steady'][c] for c in STATE) for r in stages for k in range(N))
    fig, axes = plt.subplots(4, 1, figsize=(13, 7.4), sharex=True)
    xs = np.array([k + (k // gpn) * 1.6 for k in range(N)])
    for ax, r in zip(axes, stages):
        for k in range(N):
            m, y = r.ranks[k]['steady'], 0
            for c in STATE:
                h = m[c] / top
                ax.bar(xs[k], h, bottom=y, width=0.82, color=CAT_COLOR[c], linewidth=0)
                y += h
        st = r.cfg.stage
        bpp = r.bytes_per_param(padded=True)     # per stored element: what the lesson's formula describes
        ax.text(-2.2, 0.5, STAGE_NAME[st], ha='right', va='center', fontsize=12, fontweight='bold', color=STAGE_COLOR[st])
        ax.text(xs[-1] + 1.4, 0.62, f'{bpp:.2f} B/param', ha='left', va='center', fontsize=10.5, fontweight='bold', color=INK)
        ax.text(xs[-1] + 1.4, 0.32, f'{fmt_bytes(sum(r.ranks[0]["steady"][c] for c in STATE))} per GPU', ha='left',
                va='center', fontsize=8.5, color=INK2)
        ax.set_ylim(0, 1.04)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines['bottom'].set_visible(False)
    for n in range(N // gpn):
        x0, x1 = xs[n * gpn] - 0.5, xs[n * gpn + gpn - 1] + 0.5
        axes[-1].annotate('', (x0, -0.12), (x1, -0.12), xycoords=('data', 'axes fraction'),
                          arrowprops=dict(arrowstyle='-', color=MUTED, lw=1))
        axes[-1].text((x0 + x1) / 2, -0.28, f'node {n}  ·  GPUs {n * gpn}–{n * gpn + gpn - 1}', transform=axes[-1].get_xaxis_transform(),
                      ha='center', fontsize=8.5, color=INK2)
    axes[-1].set_xticks([])
    handles = [Patch(color=CAT_COLOR[c], label=CAT_LABEL[c]) for c in STATE]
    fig.legend(handles=handles, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.86, top=0.9, hspace=0.18)
    suptitle(fig, f'What each of the {N} virtual GPUs stores, measured', 'one column per GPU, steady state between steps, '
             'same vertical scale in every row · ZeRO deletes duplicates, it does not delete information')
    save(fig, name)


# --------------------------------------------------------------------------- 01 the cluster
def cluster(world=32, gpn=8, name='01_cluster'):
    fig, ax = plt.subplots(figsize=(13, 2.9))
    ax.set_xlim(-1, world * 1.0 + (world // gpn) * 0.9 + 0.5)
    ax.set_ylim(-1.9, 1.7)
    ax.axis('off')
    pos = {}
    for k in range(world):
        n = k // gpn
        x = k + n * 0.9
        pos[k] = x
        ax.add_patch(FancyBboxPatch((x - 0.36, -0.36), 0.72, 0.72, boxstyle='round,pad=0.02,rounding_size=0.08',
                                    fc='#e8f5f9', ec='#0891b2', lw=1.1))
        ax.text(x, 0, str(k), ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0e5f75')
    for n in range(world // gpn):
        x0 = pos[n * gpn] - 0.6
        x1 = pos[n * gpn + gpn - 1] + 0.6
        ax.add_patch(FancyBboxPatch((x0, -0.62), x1 - x0, 1.5, boxstyle='round,pad=0.02,rounding_size=0.15',
                                    fc='none', ec=MUTED, lw=1, ls=(0, (3, 2))))
        ax.text((x0 + x1) / 2, 1.05, f'node {n}  (8 GPUs, NVLink 450 GB/s)', ha='center', fontsize=8.5, color=INK2)
    for k in range(world):
        dst = (k + 1) % world
        inter = k // gpn != dst // gpn
        if dst == 0:
            arr = FancyArrowPatch((pos[k], -0.4), (pos[0], -0.4), connectionstyle='arc3,rad=0.12',
                                  arrowstyle='-|>', mutation_scale=9, color=RED, lw=1.6)
        else:
            arr = FancyArrowPatch((pos[k] + 0.38, 0), (pos[dst] - 0.38, 0), arrowstyle='-|>', mutation_scale=8,
                                  color=RED if inter else '#0891b2', lw=1.8 if inter else 1.0)
        ax.add_patch(arr)
    ax.text(pos[world - 1] / 2, -1.75, 'the ring: every GPU sends only to its right-hand neighbour · red hops cross '
            'InfiniBand (50 GB/s), nine times slower, and a ring runs at the pace of its slowest hop',
            ha='center', fontsize=9, color=INK2)
    suptitle(fig, f'{world} virtual GPUs = {world // gpn} nodes × {gpn}, one Python thread each')
    save(fig, name)


# --------------------------------------------------------------------------- 03 same model
def losses(results, single=None, name='03_same_model'):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4), gridspec_kw={'width_ratios': [1.35, 1]})
    for s in range(4):
        r = results[s]
        a.plot(r.losses, color=STAGE_COLOR[s], lw=5.5 - s * 1.3, alpha=0.9, label=STAGE_NAME[s])
    a.set_xlabel('step')
    a.set_ylabel('training loss (mean over 32 GPUs)')
    a.legend(loc='upper right')
    title(a, 'Four arrangements, one loss curve', 'the lines are drawn on top of each other: thick to thin')
    ref = results[0]
    for s in range(1, 4):
        d = np.abs(np.array(results[s].losses) - np.array(ref.losses))
        b.plot(np.maximum(d, 1e-18), color=STAGE_COLOR[s], lw=1.8, label=f'{STAGE_NAME[s]} − DDP')
    if single is not None:
        d = np.abs(np.array(single.losses) - np.array(ref.losses))
        b.plot(d, color=MUTED, lw=1.6, ls='--', label='1 GPU, 32× batch − DDP')
    b.set_yscale('log')
    b.set_ylim(1e-18, 1)
    b.axhline(1e-18, color=STAGE_COLOR[3], lw=0)
    b.text(0.02, 0.05, 'ZeRO-1/2/3 vs DDP: exactly 0 at every step\n(drawn at the floor of the axis)', transform=b.transAxes,
           fontsize=8.5, color=INK2)
    b.set_xlabel('step')
    b.set_ylabel('|loss difference|')
    b.legend(loc='center right', fontsize=8)
    title(b, 'Difference from DDP', 'bitwise identical, not just "close"')
    fig.tight_layout()
    save(fig, name)


# --------------------------------------------------------------------------- 04 memory per GPU: steady vs peak
def memory_bars(runs, name='04_memory_per_gpu'):
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    labels, x = [], 0
    for r in runs:
        for which, dx in (('steady', -0.21), ('peak_cat', 0.21)):
            m, y = r.ranks[0][which], 0
            for c in CATS:
                v = m[c] / 2 ** 20
                if v:
                    ax.bar(x + dx, v, bottom=y, width=0.38, color=CAT_COLOR[c], linewidth=0,
                           hatch='///' if which == 'peak_cat' and c in ('gathered', 'grad_bucket', 'activations') else None,
                           edgecolor='white')
                y += v
            ax.text(x + dx, y + 0.15, 'between\nsteps' if which == 'steady' else 'peak', ha='center', va='bottom',
                    fontsize=7.2, color=MUTED, linespacing=0.95)
            if which == 'steady':
                ax.text(x + dx, y + 1.25, f'{r.bytes_per_param():.2f} B/p', ha='center', fontsize=8.5,
                        fontweight='bold', color=_color(r))
        labels.append(r.cfg.name.replace(' + ', '\n+ '))
        x += 1
    ax.set_xticks(range(len(runs)), labels)
    ax.set_ylabel('MiB on GPU 0')
    handles = [Patch(color=CAT_COLOR[c], label=CAT_LABEL[c]) for c in CATS]
    ax.legend(handles=handles, ncol=4, loc='upper right', fontsize=8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    title(ax, 'GPU 0, between steps and at its worst moment',
          'left bar: training state that lives forever · right bar: the peak inside a step, transients hatched')
    save(fig, name)


# --------------------------------------------------------------------------- 05 memory timeline in one step
def memory_timeline(results, rank=0, name='05_memory_timeline'):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1), sharey=True)
    ymax = 0
    for ax, s in zip(axes, range(4)):
        tr = results[s].ranks[rank]['trace']
        xs = np.arange(len(tr))
        ys = np.array([[e[c] / 2 ** 20 for e in tr] for c in CATS])
        ax.stackplot(xs, ys, colors=[CAT_COLOR[c] for c in CATS], step='post', linewidth=0)
        ymax = max(ymax, ys.sum(0).max())
        # phase bands
        phases, last = [], None
        for i, e in enumerate(tr):
            p = e['phase'].split(' ')[0]
            if p != last:
                phases.append((i, p))
                last = p
        for i, p in phases:
            ax.axvline(i, color='white', lw=0.8)
            ax.text(i + 0.4, 0.985, {'fwd': 'fwd', 'bwd': 'bwd', 'optimizer': 'opt', 'weights': 'AG', 'idle': ''}.get(p, p),
                    transform=ax.get_xaxis_transform(), fontsize=7.2, color=INK2, va='top', fontweight='bold')
        pk = results[s].ranks[rank]
        title(ax, STAGE_NAME[s], f'peak {fmt_bytes(pk["peak"])} during {pk["peak_phase"]}')
        ax.set_xlabel('allocation / free events in step 2 →')
        ax.title.set_color(STAGE_COLOR[s])
    axes[0].set_ylabel(f'MiB held by GPU {rank}')
    axes[0].set_ylim(0, ymax * 1.14)
    handles = [Patch(color=CAT_COLOR[c], label=CAT_LABEL[c]) for c in CATS]
    fig.legend(handles=handles, ncol=8, loc='upper center', bbox_to_anchor=(0.5, 0.0), fontsize=8)
    suptitle(fig, 'Memory inside one training step, every allocation and free', 'ZeRO-2 and ZeRO-3 trade a flat floor '
             'for brief spikes: a gradient bucket, or one gathered layer, alive only while it is needed')
    fig.tight_layout()
    save(fig, name)


# --------------------------------------------------------------------------- 06 bytes/param vs N
def bytes_vs_n(sweep, name='06_bytes_vs_N'):
    """sweep: {(stage, N): Result}"""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    Ns = sorted({n for _, n in sweep})
    nn = np.logspace(0, np.log2(max(Ns)) * np.log10(2), 100)
    for s in range(4):
        a.plot(nn, [T.bytes_per_param(s, n) for n in nn], color=STAGE_COLOR[s], lw=1.4, alpha=0.55)
        meas = [sweep[(s, n)].bytes_per_param(padded=True) for n in Ns]
        a.scatter(Ns, meas, color=STAGE_COLOR[s], s=36, zorder=3, edgecolor='white', linewidth=0.8)
        label_end(a, Ns[-1], meas[-1], f'{STAGE_NAME[s]}  {meas[-1]:.3f}', STAGE_COLOR[s], dx=8)
        sent = [sweep[(s, n)].sent_total() / sweep[(s, n)].P(padded=True) for n in Ns]
        b.plot(nn, [T.comm_P(s, n) for n in nn], color=STAGE_COLOR[s], lw=1.4, alpha=0.55)
        b.scatter(Ns, sent, color=STAGE_COLOR[s], s=36, zorder=3, edgecolor='white', linewidth=0.8)
    for ax in (a, b):
        ax.set_xscale('log', base=2)
        ax.set_xticks(Ns, [str(n) for n in Ns])
        ax.set_xlabel('number of GPUs N')
        ax.set_xlim(0.8, max(Ns) * 3)
    a.set_yscale('log')
    a.set_ylabel('bytes per parameter on one GPU')
    title(a, 'Memory: measured (dots) on the lesson formula (lines)', '16  ·  4 + 12/N  ·  2 + 14/N  ·  16/N')
    b.set_ylabel('bytes sent per GPU per step  /  P')
    b.text(Ns[-1] * 1.15, 3 * (Ns[-1] - 1) / Ns[-1], 'ZeRO-3\n3(N−1)/N', color=STAGE_COLOR[3], fontsize=8.5, va='center', fontweight='bold')
    b.text(Ns[-1] * 1.15, 2 * (Ns[-1] - 1) / Ns[-1], 'DDP, ZeRO-1, ZeRO-2\n2(N−1)/N', color=INK2, fontsize=8.5, va='center', fontweight='bold')
    title(b, 'Communication: measured (dots) on 2(N−1)/N and 3(N−1)/N', 'the lesson rounds (N−1)/N to 1: "2P, 2P, 2P, 3P"')
    fig.tight_layout()
    save(fig, name)


# --------------------------------------------------------------------------- 07 what goes on the wire
def comm_breakdown(results, name='07_comm'):
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    order = ['grad reduce-scatter', 'grad all-gather', 'weight all-gather', 'weight all-gather (fwd)',
             'weight all-gather (bwd)', 'grad-norm all-gather']
    col = {'grad reduce-scatter': '#d97706', 'grad all-gather': '#f3c47e', 'weight all-gather': '#0891b2',
           'weight all-gather (fwd)': '#0891b2', 'weight all-gather (bwd)': '#67c4dd', 'grad-norm all-gather': INK}
    seen = set()
    for i, s in enumerate(range(4)):
        r = results[s]
        per = {}
        for k, v in r.sent(0).items():
            lab, link = k.split('|')
            per[lab] = per.get(lab, 0) + v
        x = 0
        for lab in order:
            if lab in per:
                w = per[lab] / r.P()
                ax.barh(i, w, left=x, color=col[lab], height=0.6, label=lab if lab not in seen else None,
                        edgecolor='white', linewidth=1)
                seen.add(lab)
                if w > 0.2:
                    ax.text(x + w / 2, i, f'{w:.2f}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
                x += w
        calls = sum(r.ranks[0]['calls'].values())
        ax.text(x + 0.04, i, f'{x:.3f} P   ·  {calls} collective calls', va='center', fontsize=8.5, color=INK2)
    ax.set_yticks(range(4), [STAGE_NAME[s] for s in range(4)])
    ax.invert_yaxis()
    ax.set_xlim(0, 3.9)
    ax.set_xlabel('bytes GPU 0 sends in one step, in units of P (one bf16 copy of the model)')
    ax.legend(ncol=3, loc='lower right', fontsize=8)
    title(ax, 'What crosses the wire, per GPU per step', 'DDP\'s all-reduce is a reduce-scatter plus an all-gather; '
          'ZeRO-1/2 keep the same two halves and move the all-gather to the weights; ZeRO-3 gathers weights twice')
    save(fig, name)


# --------------------------------------------------------------------------- 08 one step on the clock
LANE = {'compute': 2, 'idle': 2, 'ag': 1, 'rs': 0, 'pcie': -1}


def _ev_color(stream, label):
    if stream == 'idle':
        return '#e7e6e2'
    if stream == 'compute':
        return '#7c3aed' if label.startswith('optimizer') else ('#2a78d6' if label.startswith('fwd') else '#5f97de')
    if stream == 'pcie':
        return OFFLOAD_COLOR
    return '#0891b2' if stream == 'ag' else '#d97706'


def timelines(rows, name='08_timeline', unit=1e-6, sub=None):
    """rows: list of (label, Result, rank)"""
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 1.25 * len(rows) + 1.0), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (lab, r, rank) in zip(axes, rows):
        k = r.cfg.record_step
        t0 = r.ranks[rank]['step_start'][k]
        for stream, label, a, b, _ in r.ranks[rank]['events']:
            lane = LANE[stream]
            ax.barh(lane, (b - a) / unit, left=(a - t0) / unit, height=0.72, color=_ev_color(stream, label),
                    hatch='////' if stream == 'idle' else None, edgecolor='white' if stream != 'idle' else '#c9c8c3',
                    linewidth=0.4)
        st = r.step_time()
        ax.axvline(st / unit, color=INK, lw=1)
        ax.text(st / unit, 2.75, f' {st / unit:.0f} µs', fontsize=8.5, fontweight='bold', va='center')
        ax.set_yticks([2, 1, 0], ['compute', 'all-gather', 'reduce-\nscatter'], fontsize=7.5)
        ax.set_ylim(-0.6, 3.1)
        ax.text(-0.075, 0.5, lab, transform=ax.transAxes, ha='right', va='center', fontsize=10, fontweight='bold',
                color=_color(r), linespacing=1.0)
        ax.grid(axis='y', visible=False)
    axes[-1].set_xlabel('µs since the start of the step (simulated clock)')
    handles = [Patch(color='#2a78d6', label='forward'), Patch(color='#5f97de', label='backward'),
               Patch(color='#7c3aed', label='optimizer'), Patch(facecolor='#e7e6e2', hatch='////', edgecolor='#c9c8c3', label='waiting on the network'),
               Patch(color='#0891b2', label='all-gather'), Patch(color='#d97706', label='reduce-scatter')]
    fig.legend(handles=handles, ncol=6, loc='upper center', bbox_to_anchor=(0.55, 0.0), fontsize=8.5)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.08, hspace=0.3)
    suptitle(fig, 'One training step on the simulated clock', sub)
    save(fig, name)


# --------------------------------------------------------------------------- 09 step time
def step_time_bars(table, name='09_step_time'):
    """table: list of dicts {label, compute, optimizer, exposed, total, color}"""
    fig, ax = plt.subplots(figsize=(12.5, 0.42 * len(table) + 1.4))
    for i, row in enumerate(table):
        x = 0
        for key, c in (('compute', '#2a78d6'), ('optimizer', '#7c3aed'), ('exposed', '#e34948')):
            ax.barh(i, row[key], left=x, color=c, height=0.66, edgecolor='white', linewidth=0.6)
            x += row[key]
        ax.text(x + 4, i, f"{row['total']:.0f} µs  ·  network {row['exposed'] / row['compute'] * 100:.0f}% of compute",
                va='center', fontsize=8.3, color=INK2)
    ax.set_yticks(range(len(table)), [r['label'] for r in table], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel('simulated µs per step, GPU 0')
    ax.set_xlim(0, max(r['total'] for r in table) * 1.42)
    ax.legend(handles=[Patch(color='#2a78d6', label='forward + backward'), Patch(color='#7c3aed', label='optimizer'),
                       Patch(color='#e34948', label='time the network adds (not hidden)')], loc='lower right', fontsize=8.5)
    title(ax, 'Where a step\'s time goes', 'virtual cards calibrated to the lesson\'s 64-GPU step: same compute-to-network '
          'ratio, so DDP without overlap should land near 34% (H100) and 77% (B200)')
    save(fig, name)


# --------------------------------------------------------------------------- 11 the V5 ladder
def ladder(measured_bpp, name='11_ladder_30B', params=30e9):
    """measured_bpp: {stage: {N: bytes_per_param measured}}"""
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for s in range(4):
        th = [T.gib_per_gpu(s, n, params) for n in Ns]
        ax.plot(Ns, th, color=STAGE_COLOR[s], lw=1.6, alpha=0.6)
        mN = sorted(measured_bpp[s])
        ax.scatter(mN, [measured_bpp[s][n] * params / GIB for n in mN], color=STAGE_COLOR[s], s=40, zorder=3,
                   edgecolor='white', linewidth=0.8)
        ff = T.first_fit(s, params)
        label_end(ax, Ns[-1], th[-1], f'{STAGE_NAME[s]}  ' + (f'fits from {ff}' if ff else 'never fits'),
                  STAGE_COLOR[s], dx=8)
    ax.axhline(CARD_GIB, color=RED, lw=1.4, ls='--')
    ax.text(1.05, CARD_GIB * 0.86, f'one 80 GB card = {CARD_GIB:.1f} GiB', color=RED, fontsize=8.5, fontweight='bold', va='top')
    ax.axhline(params * 4 / GIB, color=MUTED, lw=0.8, ls=':')
    ax.text(24, params * 4 / GIB * 1.08, f'4 B/param floor of ZeRO-1 = {params * 4 / GIB:.1f} GiB', color=MUTED, fontsize=8, va='bottom')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.set_xlim(0.85, 1400)
    ax.set_xlabel('GPUs')
    ax.set_ylabel(f'GiB per GPU for a {params / 1e9:.0f}B model')
    title(ax, f'My measured bytes/param, scaled to V5 ({params / 1e9:.0f}B parameters)',
          'dots: measured on the toy model at N = 1…32, times 30B · lines: the lesson\'s formula')
    save(fig, name)


# --------------------------------------------------------------------------- 13 the real thing
def real_vs_sim(real, sim_fp32, name='13_real_pytorch'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={'width_ratios': [1.15, 1, 1.05]})
    a, b, c = axes
    cols = {'ddp': STAGE_COLOR[0], 'zero1': STAGE_COLOR[1], 'fsdp2_zero2': STAGE_COLOR[2], 'fsdp2_zero3': STAGE_COLOR[3]}
    labs = {'ddp': 'DDP', 'zero1': 'ZeroRedundancyOptimizer', 'fsdp2_zero2': 'FSDP2, no reshard', 'fsdp2_zero3': 'FSDP2 (ZeRO-3)'}
    sl = np.array(sim_fp32.losses[:len(real['ddp']['losses'])])
    for k, r in real.items():
        a.plot(np.abs(np.array(r['losses']) - sl) + 1e-12, color=cols[k], lw=1.6, label=labs[k])
    a.set_yscale('log')
    a.set_xlabel('step')
    a.set_ylabel('|loss − my simulator\'s loss|')
    a.legend(fontsize=8)
    title(a, 'PyTorch vs my simulator, fp32', 'same init, same data, same AdamW: they agree to fp32 rounding')
    keys = list(real)
    x = np.arange(len(keys))
    psi = real['ddp']['psi']
    bot = np.zeros(len(keys))
    for part, colr in (('params', CAT_COLOR['weights']), ('grads', CAT_COLOR['grads']), ('opt', CAT_COLOR['adam_m'])):
        v = np.array([real[k]['ranks'][0][part] / psi for k in keys])
        b.bar(x, v, bottom=bot, color=colr, width=0.6, label={'params': 'parameters', 'grads': 'gradients', 'opt': 'AdamW state'}[part])
        bot += v
    for i, k in enumerate(keys):
        b.text(i, bot[i] + 0.2, f'{bot[i]:.2f}', ha='center', fontsize=9, fontweight='bold')
    b.set_xticks(x, [labs[k].replace(' (', '\n(').replace(', ', ',\n') for k in keys], fontsize=7.6)
    b.set_ylabel('bytes per parameter on rank 0 (fp32)')
    b.legend(fontsize=8)
    title(b, 'What rank 0 holds, measured from the tensors', f'fp32 everywhere: 4 + 4 + 8, sharded over {real["ddp"]["world"]}')
    W = real['ddp']['world']
    P = psi * 4
    for i, k in enumerate(keys):
        cb = real[k]['ranks'][0]['comm_bytes']
        ring = 0
        for op, v in cb.items():
            ring += v * (2 if op == 'all_reduce' else 1) * (W - 1) / W
        c.bar(i, ring / P, color=cols[k], width=0.6)
        c.text(i, ring / P + 0.05, f'{ring / P:.2f} P', ha='center', fontsize=9, fontweight='bold')
    c.set_xticks(x, [labs[k].replace(' (', '\n(').replace(', ', ',\n') for k in keys], fontsize=7.6)
    c.set_ylabel('ring bytes per rank per step / P')
    title(c, 'What each rank sends, from the collective calls', 'payloads counted by wrapping torch.distributed')
    fig.tight_layout()
    save(fig, name)


def tok_rows(local_numel):
    """Rows of the 65 x 128 token embedding this rank holds. It is the model's first parameter, so position 0 works
    even where sharding replaced the parameter objects and their names were lost."""
    k = next((n for n in local_numel if n.endswith('tok.weight')), next(iter(local_numel)))
    return local_numel[k] // 128


def fsdp_imbalance(real, name='14_fsdp2_imbalance'):
    r = real['fsdp2_zero3']
    W = r['world']
    opt = np.array([x['opt'] for x in r['ranks']])
    loc = [x['local_numel'] for x in r['ranks']]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={'width_ratios': [1.3, 1]})
    a.bar(range(W), opt / 1024, color=[STAGE_COLOR[3] if v == opt.max() else '#9fd4b1' for v in opt], width=0.8)
    a.axhline(opt.mean() / 1024, color=INK, lw=0.8, ls='--')
    a.text(W - 0.5, opt.mean() / 1024 * 1.02, f'mean {opt.mean() / 1024:.1f} KiB', ha='right', va='bottom', fontsize=8.5)
    a.set_xlabel('rank')
    a.set_ylabel('AdamW state held (KiB)')
    title(a, 'FSDP2 does not shard evenly', f'rank {int(opt.argmax())} holds {opt.max() / opt.min() - 1:.1%} more optimizer '
          f'state than rank {int(opt.argmin())}: the busiest rank is the one that runs out first')
    rows = [tok_rows(l) * 128 for l in loc]
    b.bar(range(W), np.array(rows) / 128, color='#0891b2', width=0.8)
    b.set_xlabel('rank')
    b.set_ylabel('rows of the 65 × 128 token embedding')
    title(b, 'Why: every tensor is cut along dim 0', '65 rows over 32 ranks = chunks of ⌈65/32⌉ = 3: '
          'ranks 0–20 get 3, rank 21 gets 2, ranks 22–31 get none')
    fig.tight_layout()
    save(fig, name)
