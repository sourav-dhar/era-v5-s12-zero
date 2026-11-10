"""The hardware the 32 virtual GPUs pretend to be, using the lesson's own constants (widgets 4, 6, 7).

Bandwidths are the real ones (NVLink 450 GB/s, InfiniBand 50 GB/s, PCIe 60 GB/s). Compute is the one
thing that has to be scaled: my toy step pushes 256 tokens through each GPU, the lesson's step pushes
1M tokens through 64 GPUs (15,625 each). The ratio comm/compute = (2 * 2 bytes * Psi / BW) / (6 * Psi * tok / F)
does not depend on the model size Psi at all, only on tokens-per-GPU and F/BW. So a virtual card whose
effective FLOP/s is the real card's x (256 / 15,625) gives my toy step the *same* communication-to-compute
ratio as the lesson's 30B step: 34% on H100, 77% on B200.
"""
from dataclasses import dataclass, replace

GIB = 1024 ** 3
CARD_BYTES = 80e9                       # an 80 GB card ...
CARD_GIB = CARD_BYTES / GIB             # ... is 74.5 GiB
LESSON_PARAMS = 30e9                    # V5, every calculation in the lesson uses 30B
LESSON_TOKENS, LESSON_GPUS, MFU = 1e6, 64, 0.40
PEAK = {'H100': 990e12, 'B200': 990e12 * 2.28}
HBM = {'H100': 3.35e12, 'B200': 8.0e12}  # bytes/s, sets how long the (memory-bound) Adam update takes
NVLINK, INFINIBAND, PCIE = 450e9, 50e9, 60e9


@dataclass(frozen=True)
class Hardware:
    name: str = 'H100'
    flops: float = PEAK['H100'] * MFU   # effective FLOP/s of one virtual GPU
    hbm: float = HBM['H100']            # bytes/s for elementwise optimizer work on the GPU
    host_bw: float = 100e9              # bytes/s for the optimizer when it runs on the CPU (offload)
    bw_intra: float = NVLINK            # GPU <-> GPU inside a node
    bw_inter: float = INFINIBAND        # GPU <-> GPU across nodes
    bw_pcie: float = PCIE               # GPU <-> its own CPU
    latency: float = 0.0                # fixed cost per message (s); the lesson's arithmetic uses 0
    gpus_per_node: int = 8

    def link_bw(self, src: int, dst: int) -> float:
        same = src // self.gpus_per_node == dst // self.gpus_per_node
        return self.bw_intra if same else self.bw_inter

    def link_kind(self, src: int, dst: int) -> str:
        return 'nvlink' if src // self.gpus_per_node == dst // self.gpus_per_node else 'ib'


def calibrated(card: str = 'H100', tokens_per_gpu: int = 256, **kw) -> Hardware:
    """A virtual `card` that gives a `tokens_per_gpu` step the lesson's compute-to-communication ratio."""
    scale = tokens_per_gpu / (LESSON_TOKENS / LESSON_GPUS)
    return Hardware(name=card, flops=PEAK[card] * MFU * scale, hbm=HBM[card], **kw)


def lesson_step_seconds(card: str = 'H100') -> float:
    """Section 5: 6 * params * tokens / (GPUs * peak * MFU). 7.10 s on H100, 3.12 s on B200."""
    return 6 * LESSON_PARAMS * LESSON_TOKENS / (LESSON_GPUS * PEAK[card] * MFU)


def with_(hw: Hardware, **kw) -> Hardware:
    return replace(hw, **kw)
