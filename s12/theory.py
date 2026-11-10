"""What the lesson says should happen. Everything measured in the notebook is checked against these."""
from .hw import GIB, CARD_GIB, LESSON_PARAMS, NVLINK, INFINIBAND

STAGES = [0, 1, 2, 3]
NAME = {0: 'DDP', 1: 'ZeRO-1', 2: 'ZeRO-2', 3: 'ZeRO-3'}
COLOR = {0: '#7c3aed', 1: '#d97706', 2: '#0891b2', 3: '#159447'}   # the lesson's memory-ladder colours

# bytes per weight, split the way the lesson's widget splits them (weights, grads, master, Adam m, Adam v)
SPLIT = {'weights': 2, 'grads': 2, 'master': 4, 'adam_m': 4, 'adam_v': 4}
SHARDED = {0: set(), 1: {'master', 'adam_m', 'adam_v'}, 2: {'grads', 'master', 'adam_m', 'adam_v'},
           3: {'weights', 'grads', 'master', 'adam_m', 'adam_v'}}


def bytes_per_param(stage: int, n: int, by_cat=False, offload=False):
    """16, 4 + 12/N, 2 + 14/N, 16/N (Sections 6 and 7). Offload moves the fp32 state off the GPU."""
    d = {c: (b / n if c in SHARDED[stage] else b) for c, b in SPLIT.items()}
    if offload:
        d.update(master=0.0, adam_m=0.0, adam_v=0.0)
    return d if by_cat else sum(d.values())


def comm_P(stage: int, n: int = None):
    """Bytes each GPU sends per step, in multiples of P. The lesson rounds (N-1)/N to 1: 2P, 2P, 2P, 3P."""
    k = 3 if stage == 3 else 2
    return k if n is None else k * (n - 1) / n


def gib_per_gpu(stage: int, n: int, params: float = LESSON_PARAMS, offload=False):
    return params * bytes_per_param(stage, n, offload=offload) / GIB


def first_fit(stage: int, params: float = LESSON_PARAMS, card_gib: float = CARD_GIB, max_k: int = 12):
    for k in range(max_k + 1):
        if gib_per_gpu(stage, 2 ** k, params) <= card_gib:
            return 2 ** k
    return None


def replicated_floor_params(card_gib: float = CARD_GIB):
    """ZeRO-1 keeps 4 B/weight (bf16 weights + grads) on every card regardless of N: the model size at which
    that alone fills the card. 74.5 GiB / 4 B = 20.0B parameters (Section 7)."""
    return card_gib * GIB / 4


def comm_seconds(stage: int, params: float = LESSON_PARAMS, bw: float = INFINIBAND, bytes_per=2):
    return comm_P(stage) * params * bytes_per / bw


LINKS = {'NVLink': NVLINK, 'InfiniBand': INFINIBAND}
