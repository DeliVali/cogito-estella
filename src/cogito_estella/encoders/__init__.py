"""Pluggable text encoders behind one contract.

`encode(texts, lang, batch_size, normalize) -> float32 [N, 1024]`. `normalize=None`
means the encoder's native contract: SONAR raw (the published heads were trained on
raw SONAR vectors), BGE-M3 unit-norm.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

DEFAULT_ENCODER = "sonar"
LEGACY_ENCODER = "sonar"     # checkpoints predating the encoder key; never follows the default
DIM = 1024
CANARY_PATH = Path(__file__).with_name("canary.json")
CANARY_TOLERANCE = 0.02
CANARY_PAIRS = (
    ("The encoder maps text to a vector.", "The decoder was trained on tokens."),
    ("SONAR decodes the concepts.", "The concepts are decoded by SONAR."),
)


class EncoderMismatch(RuntimeError):
    """Active encoder disagrees with a checkpoint, another checkpoint or the canary."""


@runtime_checkable
class TextEncoder(Protocol):
    name: str
    revision: str
    dim: int
    native_normalized: bool

    def encode(self, texts: list[str], lang: str = "eng_Latn", batch_size: int = 64,
               normalize: bool | None = None) -> np.ndarray: ...


def _sonar(device: str | None = None, download: bool = True) -> TextEncoder:
    from cogito_estella.encoders.sonar import SonarEncoder

    return SonarEncoder(device=device)


def _bge_m3(device: str | None = None, download: bool = True) -> TextEncoder:
    from cogito_estella.encoders.bge_m3 import BgeM3Encoder

    return BgeM3Encoder(device=device, download=download)


# Factories, not classes: importing the registry must not import torch backends.
ENCODERS: dict[str, Callable[..., TextEncoder]] = {"sonar": _sonar, "bge-m3": _bge_m3}


def get_encoder(name: str, device: str | None = None, download: bool = True) -> TextEncoder:
    try:
        factory = ENCODERS[name]
    except KeyError:
        raise ValueError(
            f"unknown encoder {name!r}; available: {', '.join(ENCODERS)}") from None
    return factory(device=device, download=download)


def resolve_encoder_name(explicit: str | None, checkpoint_names: list[str | None],
                         env: dict | None = None) -> str:
    """explicit > agreeing checkpoint metadata > env COGITO_ENCODER > DEFAULT_ENCODER.

    A checkpoint set that disagrees with itself, or an env request that disagrees with
    the checkpoints, is an error rather than a silently overridden preference.
    """
    names = sorted({n for n in checkpoint_names if n})
    if len(names) > 1:
        raise EncoderMismatch(f"checkpoints disagree on encoder: {', '.join(names)}")
    if explicit:
        return explicit
    wanted = (env or {}).get("COGITO_ENCODER") or None
    if names:
        if wanted and wanted != names[0]:
            raise EncoderMismatch(
                f"COGITO_ENCODER={wanted} but the checkpoints were trained on {names[0]}")
        return names[0]
    return wanted or DEFAULT_ENCODER


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-12)


def check_contract(enc: TextEncoder, texts: list[str], *, normalize: bool | None = True,
                   batch_size: int = 64) -> dict:
    """Booleans + the deviations behind them. `normalize=True` is the comparable
    setting across encoders (SONAR is raw natively)."""
    texts = list(texts)
    if not texts:
        raise ValueError("check_contract needs at least one text")
    mat = enc.encode(texts, batch_size=batch_size, normalize=normalize)
    again = enc.encode(texts, batch_size=batch_size, normalize=normalize)
    singles = np.concatenate(
        [enc.encode([t], batch_size=batch_size, normalize=normalize) for t in texts], axis=0)
    norm_dev = float(np.max(np.abs(np.linalg.norm(mat, axis=1) - 1.0)))
    repeat_dev = float(np.max(np.abs(mat.astype(np.float64) - again.astype(np.float64))))
    min_cos = float(np.min(_cosine_rows(mat, singles)))
    return {
        "dim": mat.shape == (len(texts), enc.dim),
        "dtype": mat.dtype == np.float32,
        "unit_norm": norm_dev < 1e-3,
        "deterministic": repeat_dev < 1e-5,
        "batch_invariant": min_cos >= 0.999,
        "max_norm_dev": norm_dev,
        "max_repeat_dev": repeat_dev,
        "min_batch_cos": min_cos,
    }


def load_canary() -> dict:
    return json.loads(CANARY_PATH.read_text(encoding="utf-8"))


def canary_cosines(enc: TextEncoder) -> list[float]:
    """Cosine of each canary pair under `normalize=True`."""
    flat = [s for pair in CANARY_PAIRS for s in pair]
    emb = enc.encode(flat, normalize=True)
    return [float(_cosine_rows(emb[i:i + 1], emb[i + 1:i + 2])[0])
            for i in range(0, len(flat), 2)]


def write_canary(enc: TextEncoder, cosines: list[float]) -> None:
    """Re-baseline the tracked reference; callers must gate it behind an explicit opt-in."""
    data = load_canary() if CANARY_PATH.exists() else {}
    data["pairs"] = [list(p) for p in CANARY_PAIRS]
    data["tolerance"] = CANARY_TOLERANCE
    data[enc.name] = {"revision": enc.revision, "dim": enc.dim,
                      "cosines": [round(float(c), 4) for c in cosines]}
    CANARY_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
