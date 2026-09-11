"""Supervised re-ranking of `ask`'s sentence candidates: scale-free features, a
logistic model fitted offline, and the weights file the query path loads."""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cogito_estella.mcp.rank import rank, tokenize

FEATURES = ("lex_norm", "lex_rank", "bm25_norm", "q_token_cov", "q_bigram_cov", "ent_hits",
            "provenance", "dense_cos", "dense_missing", "len_log", "rel_pos", "numeric_match")
ENT_CAP = 3                     # more mentions of the same question carry no extra signal
LEN_CENTER = 25                 # sentence length the log feature is centered on
BM25_K1 = 1.2
BM25_B = 0.75
MIN_SCALE = 1e-6                # below this a column is constant: standardizing would blow up
WEIGHTS_NAME = "rerank_weights.json"
WEIGHTS_ENV = "COGITO_RERANK_WEIGHTS"
_QUANTITY = ("how many", "how much", "what percentage", "what size", "what number", "how long")
_DIGIT = re.compile(r"\d")
_warned: set[str] = set()


def is_quantity_question(question: str) -> bool:
    """Whether the question asks for a number: digits in a sentence then mean something."""
    flat = " ".join(str(question).lower().split())
    return any(p in flat for p in _QUANTITY)


def bigrams(tokens: Sequence[str]) -> list[tuple[str, str]]:
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def _idf_fn(idf) -> Callable[[str], float]:
    if isinstance(idf, Mapping):
        return lambda t: float(idf.get(t, 0.0))
    return lambda t: float(idf(t))


def bm25_scores(store_sentences_tokens: Sequence[Sequence[str]], idf, q_tokens: Sequence[str],
                k1: float = BM25_K1, b: float = BM25_B) -> np.ndarray:
    """Okapi BM25 of one query against tokenized sentences; length normalization uses the
    mean length of the sentences given, so the caller fixes the universe."""
    n = len(store_sentences_tokens)
    out = np.zeros(n, dtype=float)
    q = list(dict.fromkeys(q_tokens))               # a repeated query token weighs once
    if not n or not q:
        return out
    lengths = np.asarray([len(s) for s in store_sentences_tokens], dtype=float)
    avgdl = float(lengths.mean()) or 1.0
    weight = _idf_fn(idf)
    for i, toks in enumerate(store_sentences_tokens):
        if not toks:
            continue
        counts = Counter(toks)
        norm = k1 * (1.0 - b + b * len(toks) / avgdl)
        total = 0.0
        for t in q:
            f = counts.get(t, 0)
            if f:
                total += weight(t) * f * (k1 + 1.0) / (f + norm)
        out[i] = total
    return out


@dataclass
class FeatureContext:
    """Per-candidate signals the store already holds; everything else is derived here.
    Scalar values broadcast over the candidates."""

    lex_scores: Sequence[float]
    bm25: Sequence[float]
    q_tokens: Sequence[str]
    q_bigrams: Sequence[tuple[str, str]]
    q_ents: Sequence[str]
    prov_flags: Sequence[bool]
    dense_cos: Sequence[float] | float = 0.0
    dense_available: Sequence[bool] | bool = False
    sent_len: Sequence[int] | None = None
    rel_pos: Sequence[float] | float = 0.0


def _column(value, n: int, dtype=float) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    if arr.size == 1 and n != 1:
        arr = np.repeat(arr, n)
    if arr.size != n:
        raise ValueError(f"context column has {arr.size} values for {n} candidates")
    return arr


def _normalized(values: np.ndarray) -> np.ndarray:
    """Share of the best candidate: the absolute scale of a scorer must not reach the model."""
    best = float(values.max()) if values.size else 0.0
    return values / best if best > 0 else np.zeros_like(values)


def _reciprocal_rank(values: Sequence[float]) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    for place, i in enumerate(rank(values), start=1):
        out[i] = 1.0 / place
    return out


def featurize(question: str, cands: Sequence[str], ctx: FeatureContext) -> np.ndarray:
    """[n, 12] feature matrix in `FEATURES` order for one question and its candidates."""
    n = len(cands)
    X = np.zeros((n, len(FEATURES)), dtype=float)
    lex = _column(ctx.lex_scores, n)
    bm = _column(ctx.bm25, n)
    prov = _column(ctx.prov_flags, n)
    cos = _column(ctx.dense_cos, n)
    avail = _column(ctx.dense_available, n, dtype=bool)
    pos = _column(ctx.rel_pos, n)
    lengths = None if ctx.sent_len is None else _column(ctx.sent_len, n)
    if n == 0:
        return X
    q_tokens = list(dict.fromkeys(ctx.q_tokens))
    q_bi = {tuple(pair) for pair in ctx.q_bigrams}
    ents = [set(tokenize(e)) for e in ctx.q_ents]
    quantity = is_quantity_question(question)
    X[:, 0] = _normalized(lex)
    X[:, 1] = _reciprocal_rank(lex)
    X[:, 2] = _normalized(bm)
    X[:, 6] = prov
    X[:, 7] = np.where(avail, cos, 0.0)
    X[:, 8] = np.where(avail, 0.0, 1.0)
    X[:, 10] = pos
    for i, text in enumerate(cands):
        toks = tokenize(text)
        tset = set(toks)
        if q_tokens:
            X[i, 3] = sum(1 for t in q_tokens if t in tset) / len(q_tokens)
        if q_bi:
            seen = set(bigrams(toks))
            X[i, 4] = sum(1 for pair in q_bi if pair in seen) / len(q_bi)
        X[i, 5] = min(sum(1 for e in ents if e and e <= tset), ENT_CAP)
        size = len(toks) if lengths is None else int(lengths[i])
        X[i, 9] = math.log(max(size, 1)) - math.log(LEN_CENTER)
        X[i, 11] = 1.0 if quantity and _DIGIT.search(text) else 0.0
    return X


def _plain(value):
    """Array scalars and arrays reach the report block; json only knows builtins."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not serializable")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0                                    # exp of a large positive z overflows
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


@dataclass
class RelevanceScorer:
    """Logistic regression on standardized features; the ordering is the logit's."""

    w: np.ndarray | None = None
    b: float = 0.0
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    features: tuple[str, ...] = FEATURES
    trained_on: str = ""
    cv: dict = field(default_factory=dict)

    def fit(self, X, y, l2: float = 1.0, steps: int = 2000, lr: float = 0.1):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != len(FEATURES):
            raise ValueError(f"expected {len(FEATURES)} features, got {X.shape}")
        if X.shape[0] != y.size:
            raise ValueError(f"{X.shape[0]} rows against {y.size} labels")
        self.features = FEATURES
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < MIN_SCALE] = 1.0                  # a constant column contributes nothing
        self.std = std
        Z = (X - self.mean) / self.std
        n = max(X.shape[0], 1)
        w = np.zeros(X.shape[1], dtype=float)
        b = 0.0
        for _ in range(int(steps)):
            err = _sigmoid(Z @ w + b) - y
            w -= lr * (Z.T @ err / n + l2 * w / n)
            b -= lr * float(err.mean())
        self.w, self.b = w, float(b)
        return self

    def score(self, X) -> np.ndarray:
        if self.w is None or self.mean is None or self.std is None:
            raise ValueError("scorer is not fitted and has no weights")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(self.w):
            raise ValueError(f"expected {len(self.w)} features, got {X.shape}")
        return ((X - self.mean) / self.std) @ self.w + self.b

    def predict_proba(self, X) -> np.ndarray:
        return _sigmoid(self.score(X))

    def validate(self):
        """Every defect that would silently mis-rank is fatal here instead."""
        if list(self.features) != list(FEATURES):
            raise ValueError("weights carry other feature names than the current contract")
        vectors = {"w": self.w, "mean": self.mean, "std": self.std}
        for name, v in vectors.items():
            if v is None or np.asarray(v).reshape(-1).size != len(FEATURES):
                raise ValueError(f"{name} has the wrong length for {len(FEATURES)} features")
        if not all(np.isfinite(np.asarray(v, dtype=float)).all() for v in vectors.values()) \
                or not math.isfinite(float(self.b)):
            raise ValueError("weights hold a non-finite value")
        if float(np.asarray(self.std, dtype=float).min()) < MIN_SCALE:
            raise ValueError("a feature scale is zero; standardizing would not be defined")
        return self

    def to_json(self, path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"features": list(self.features), "mean": np.asarray(self.mean).tolist(),
                "std": np.asarray(self.std).tolist(), "w": np.asarray(self.w).tolist(),
                "b": float(self.b), "trained_on": self.trained_on, "cv": self.cv}
        path.write_text(json.dumps(blob, indent=2, default=_plain) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path) -> RelevanceScorer:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        sc = cls(w=np.asarray(blob.get("w", []), dtype=float),
                 b=float(blob.get("b", 0.0)),
                 mean=np.asarray(blob.get("mean", []), dtype=float),
                 std=np.asarray(blob.get("std", []), dtype=float),
                 features=tuple(blob.get("features", ())),
                 trained_on=str(blob.get("trained_on", "")),
                 cv=blob.get("cv") or {})
        return sc.validate()


def default_weights_path() -> Path:
    """The file shipped beside this module, unless an operator points elsewhere."""
    override = os.environ.get(WEIGHTS_ENV)
    return Path(override) if override else Path(__file__).with_name(WEIGHTS_NAME)


def load_default() -> RelevanceScorer | None:
    """The shipped scorer, or None when it is absent or unusable: the caller falls back."""
    path = default_weights_path()
    if not path.is_file():
        return None
    try:
        return RelevanceScorer.from_json(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        key = str(path)
        if key not in _warned:                      # one line per process, not per question
            _warned.add(key)
            print(f"cogito-mcp: rerank weights unusable ({exc}); ranking stays lexical",
                  file=sys.stderr)
        return None
