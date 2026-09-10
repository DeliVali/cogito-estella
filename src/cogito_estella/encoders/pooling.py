"""Learned attention pooling over frozen encoder token states.

`AttnPool` is the module trained in exp058 (copied verbatim); `load_pool` restores it
from that experiment's checkpoint layout `{"pool": state_dict, "arch": {...}, ...}`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

DIM, HEADS, QUERIES, HIDDEN = 1024, 8, 1, 1024
_ARCH = {"d": DIM, "heads": HEADS, "queries": QUERIES, "hidden": HIDDEN}


class PoolWeightsError(RuntimeError):
    """Pooling weights are absent, unreadable or of the wrong shape."""


class AttnPool(nn.Module):
    """`queries` learned queries attend over masked token states -> one d-dim vector."""

    def __init__(self, d: int = DIM, heads: int = HEADS, queries: int = QUERIES,
                 hidden: int = HIDDEN):
        super().__init__()
        self.d, self.queries = d, queries
        self.q = nn.Parameter(torch.randn(queries, d) * d ** -0.5)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d * queries, hidden), nn.GELU(),
                                nn.Linear(hidden, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h [B,T,d] float, mask [B,T] bool (True = real token)."""
        q = self.q.unsqueeze(0).expand(h.shape[0], -1, -1)
        o, _ = self.attn(q, h, h, key_padding_mask=~mask, need_weights=False)
        o = o.reshape(h.shape[0], -1)
        return self.norm(o + self.ff(o))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pool(path: str | Path, device: str = "cpu") -> AttnPool:
    """Frozen AttnPool from a pool checkpoint. Architecture follows the stored `arch`."""
    path = Path(path)
    if not path.is_file():
        raise PoolWeightsError(f"pooling weights not found: {path}")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:                       # a corrupt file is not a missing one
        raise PoolWeightsError(f"pooling weights unreadable: {path}: {exc}") from exc
    if not isinstance(ck, dict) or "pool" not in ck:
        keys = sorted(ck) if isinstance(ck, dict) else type(ck).__name__
        raise PoolWeightsError(f"{path} carries no 'pool' state dict (found: {keys})")
    arch = {**_ARCH, **(ck.get("arch") or {})}
    pool = AttnPool(d=arch["d"], heads=arch["heads"], queries=arch["queries"],
                    hidden=arch["hidden"])
    try:
        pool.load_state_dict(ck["pool"])
    except RuntimeError as exc:
        raise PoolWeightsError(f"{path} does not fit AttnPool{arch}: {exc}") from exc
    pool.eval().requires_grad_(False)
    return pool.to(device)
