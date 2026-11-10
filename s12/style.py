"""One visual language for every figure: the Session 10/11 chrome, and the Session 12 lesson's own colours for the
four arrangements and the five kinds of training state, so a figure here reads like the lesson's widgets."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from .theory import COLOR as STAGE_COLOR, NAME as STAGE_NAME
from .vgpu import COLOR as CAT_COLOR, LABEL as CAT_LABEL, CATS

FIG = Path(__file__).resolve().parents[1] / 'figures'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e7e6e2', '#fcfcfb'
RED, CARD = '#e34948', '#e34948'
OFFLOAD_COLOR = '#b45309'


def apply():
    mpl.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
        'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.labelcolor': INK2, 'axes.labelsize': 9.5, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'xtick.major.size': 0, 'ytick.major.size': 0,
        'xtick.minor.size': 0, 'ytick.minor.size': 0,
        'text.color': INK, 'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'axes.titlepad': 24, 'legend.frameon': False, 'legend.fontsize': 8.5,
        'lines.linewidth': 2.0, 'lines.solid_capstyle': 'round', 'figure.dpi': 110,
        'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'axes.prop_cycle': mpl.cycler(color=[STAGE_COLOR[k] for k in range(4)]),
    })


def title(ax, head, sub=None):
    ax.set_title(head)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.8, color=INK2, va='bottom')


def suptitle(fig, head, sub=None):
    """Title + subtitle above the figure, spaced in inches so they never collide whatever the figure height."""
    h = fig.get_size_inches()[1]
    fig.text(0.01, 1 + (0.62 if sub else 0.3) / h, head, ha='left', va='bottom', fontsize=13, fontweight='bold')
    if sub:
        fig.text(0.01, 1 + 0.3 / h, sub, ha='left', va='bottom', fontsize=9.2, color=INK2)


def save(fig, name, show=True):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f'{name}.png')
    if show:
        plt.show()
    else:
        plt.close(fig)


def label_end(ax, x, y, text, color, dx=4, dy=0, **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points', color=color, fontsize=8.5,
                va='center', fontweight='bold', **kw)


def fmt_bytes(b):
    for unit, k in (('GiB', 2 ** 30), ('MiB', 2 ** 20), ('KiB', 2 ** 10)):
        if abs(b) >= k:
            return f'{b / k:.2f} {unit}'
    return f'{b:.0f} B'


apply()
