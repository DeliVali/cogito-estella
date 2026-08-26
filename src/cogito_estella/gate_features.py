"""Features de superficie baratas para el gate de resolución adaptativa.

Hipótesis (idea de Jeffrey, respaldada por el patching por entropía de BLT): el
riesgo de que una unidad round-trippee mal por SONAR es predecible desde señales
baratas. Estas features son el predictor más barato posible (sin tocar SONAR);
exp002 las compara contra un probe sobre el propio embedding SONAR.
"""
import numpy as np

FEATURE_NAMES = [
    "length",          # log-longitud en caracteres (normaliza escala)
    "digit_ratio",     # fracción de dígitos
    "punct_ratio",     # fracción de puntuación/símbolos no alfanuméricos ni espacio
    "brace_ratio",     # fracción de llaves { }
    "bracket_ratio",   # fracción de corchetes [ ]
    "quote_ratio",     # fracción de comillas " '
    "upper_ratio",     # fracción de mayúsculas
    "whitespace_ratio",  # fracción de espacios en blanco
    "nonascii_ratio",  # fracción de caracteres fuera de ASCII
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
