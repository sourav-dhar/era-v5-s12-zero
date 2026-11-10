"""The demo model: the Session 10/11 character-level GPT, shrunk, and cut into *units*.

A unit is what ZeRO-3 / FSDP gathers and frees as one piece (FSDP calls it a FlatParameter; in FSDP2 it is
whatever you call `fully_shard()` on). Here: the embeddings, each transformer block, and the head.
"""
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).resolve().parents[1] / 'data' / 'tinyshakespeare.txt'
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


@dataclass(frozen=True)
class GPTCfg:
    vocab: int = 65
    seq: int = 64
    d: int = 128
    n_layer: int = 4
    n_head: int = 4


class Embed(nn.Module):
    def __init__(self, c: GPTCfg):
        super().__init__()
        self.tok, self.pos = nn.Embedding(c.vocab, c.d), nn.Embedding(c.seq, c.d)

    def forward(self, idx):
        return self.tok(idx) + self.pos(torch.arange(idx.shape[1]))


class Block(nn.Module):
    def __init__(self, c: GPTCfg):
        super().__init__()
        self.h = c.n_head
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv, self.proj = nn.Linear(c.d, 3 * c.d), nn.Linear(c.d, c.d)
        self.fc, self.out = nn.Linear(c.d, 4 * c.d), nn.Linear(4 * c.d, c.d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class Head(nn.Module):
    def __init__(self, c: GPTCfg):
        super().__init__()
        self.ln, self.lm = nn.LayerNorm(c.d), nn.Linear(c.d, c.vocab)

    def forward(self, x, targets):
        logits = self.lm(self.ln(x))
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


class GPT(nn.Module):
    def __init__(self, c: GPTCfg = GPTCfg()):
        super().__init__()
        self.cfg = c
        self.units = nn.ModuleList([Embed(c), *[Block(c) for _ in range(c.n_layer)], Head(c)])
        for name, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.ones_(p) if '.ln' in name and name.endswith('weight') else nn.init.zeros_(p)
            else:
                std = 0.02 / math.sqrt(2 * c.n_layer) if name.endswith(('proj.weight', 'out.weight')) else 0.02
                nn.init.normal_(p, 0.0, std)

    @property
    def names(self):
        return unit_names(self.cfg)

    def forward(self, idx, targets):
        h = idx
        for u in self.units[:-1]:
            h = u(h)
        return self.units[-1](h, targets)


def unit_names(c: GPTCfg):
    return ['embed', *[f'block{i}' for i in range(c.n_layer)], 'head']


def build(c: GPTCfg = GPTCfg(), seed: int = 0, dtype=torch.float32) -> GPT:
    """Every rank calls this with the same seed, so every rank starts from identical weights
    (what DDP gets by broadcasting rank 0's weights at start-up). The global RNG is shared by all 32
    threads, so seeding + drawing has to happen under a lock or the ranks would race for random numbers."""
    with _INIT_LOCK:
        torch.manual_seed(seed)
        return GPT(c).to(dtype)


_INIT_LOCK = threading.Lock()


class CharData:
    """TinyShakespeare as characters. batch(step) is a pure function of (seed, step): every rank, and the
    single-GPU reference, can draw the same global batch and take its own slice of it."""

    def __init__(self, path: Path = DATA):
        if not path.exists():
            import urllib.request
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL, path)
        text = path.read_text(encoding='utf-8')
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def batch(self, step: int, n_seq: int, seq: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed * 1_000_003 + step)
        ix = torch.randint(len(self.data) - seq - 1, (n_seq,), generator=g)
        x = torch.stack([self.data[i:i + seq] for i in ix])
        y = torch.stack([self.data[i + 1:i + seq + 1] for i in ix])
        return x, y
