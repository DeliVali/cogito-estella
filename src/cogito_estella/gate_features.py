"""Cheap surface features for the adaptive-resolution gate.

Round-trip failure risk is predictable from cheap signals (cf. BLT entropy patching);
this is the cheapest possible predictor (no SONAR forward).
"""
import numpy as np

FEATURE_NAMES = [
    "length",          # log char length
    "digit_ratio",
    "punct_ratio",
    "brace_ratio",
    "bracket_ratio",
    "quote_ratio",
    "upper_ratio",
    "whitespace_ratio",
    "nonascii_ratio",
]


def surface_features(text: str) -> dict[str, float]:
    n = len(text)
    if n == 0:
        return {name: 0.0 for name in FEATURE_NAMES}
    digits = sum(c.isdigit() for c in text)
    braces = sum(c in "{}" for c in text)
    brackets = sum(c in "[]" for c in text)
    quotes = sum(c in "\"'" for c in text)
    uppers = sum(c.isupper() for c in text)
    spaces = sum(c.isspace() for c in text)
    nonascii = sum(ord(c) > 127 for c in text)
    punct = sum((not c.isalnum()) and (not c.isspace()) for c in text)
    return {
        "length": float(np.log1p(n)),
        "digit_ratio": digits / n,
        "punct_ratio": punct / n,
        "brace_ratio": braces / n,
        "bracket_ratio": brackets / n,
        "quote_ratio": quotes / n,
        "upper_ratio": uppers / n,
        "whitespace_ratio": spaces / n,
        "nonascii_ratio": nonascii / n,
    }


def surface_vector(text: str) -> np.ndarray:
    feats = surface_features(text)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)
