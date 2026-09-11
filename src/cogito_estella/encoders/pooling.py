"""Learned attention pooling over frozen encoder token states.

`AttnPool` is the module trained in exp058 (copied verbatim); `load_pool` restores it
from that experiment's checkpoint layout `{"pool": state_dict, "arch": {...}, ...}` or
from the released `pool.safetensors` + `pool.json` pair.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

DIM, HEADS, QUERIES, HIDDEN = 1024, 8, 1, 1024
_ARCH = {"d": DIM, "heads": HEADS, "queries": QUERIES, "hidden": HIDDEN}
SAFETENSORS_META = "cogito"          # header key carrying the metadata as JSON


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


def _read_pool(path: Path) -> tuple[dict, dict]:
    """(state dict, arch) from `pool.safetensors` + `pool.json`, or from a `.pt`."""
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
        sidecar = path.with_suffix(".json")
        if sidecar.is_file():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        else:
            with safe_open(str(path), framework="pt") as fh:
                meta = json.loads((fh.metadata() or {}).get(SAFETENSORS_META) or "{}")
        return state, meta.get("arch") or {}
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or "pool" not in ck:
        keys = sorted(ck) if isinstance(ck, dict) else type(ck).__name__
        raise PoolWeightsError(f"{path} carries no 'pool' state dict (found: {keys})")
    return ck["pool"], ck.get("arch") or {}


def load_pool(path: str | Path, device: str = "cpu") -> AttnPool:
    """Frozen AttnPool from a pool checkpoint. Architecture follows the stored `arch`."""
    path = Path(path)
    if not path.is_file():
        raise PoolWeightsError(f"pooling weights not found: {path}")
    try:
        state, stored_arch = _read_pool(path)
    except PoolWeightsError:
        raise
    except Exception as exc:                       # a corrupt file is not a missing one
        raise PoolWeightsError(f"pooling weights unreadable: {path}: {exc}") from exc
    arch = {**_ARCH, **stored_arch}
    pool = AttnPool(d=arch["d"], heads=arch["heads"], queries=arch["queries"],
                    hidden=arch["hidden"])
    try:
        pool.load_state_dict(state)
    except RuntimeError as exc:
        raise PoolWeightsError(f"{path} does not fit AttnPool{arch}: {exc}") from exc
    pool.eval().requires_grad_(False)
    return pool.to(device)
