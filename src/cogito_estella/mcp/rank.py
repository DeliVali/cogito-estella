"""Sentence scorers for `ask`: lexical IDF overlap and SONAR cosine, both in [0, 1]
plus a provenance bonus for sentences that back a retrieved fact."""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

import numpy as np

PROVENANCE_BONUS = 0.3
MIN_SINGULAR = 3        # strip a trailing plural `s` only above this length

_FUNCTION_WORDS = """
a about above after again against all also am among an and any are as at be because been
before being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers him his how i if in into is it its itself
just may me might more most must my no nor not now of off on once only or other others ought
our out over own per same shall she should so some such than that the their them then there
these they this those through to too under until up us very via was we were what when where
which while who whom why will with within without would you your
"""
STOPWORDS = frozenset(_FUNCTION_WORDS.split())

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")


def tokenize(text: str) -> list[str]:
    """Lowercase content words, singularized; function words dropped."""
    out: list[str] = []
    for raw in _WORD.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        tok = raw[:-1] if len(raw) > MIN_SINGULAR and raw.endswith("s") else raw
        if tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def rank(scores: Sequence[float], k: int | None = None) -> list[int]:
    """Indices by score descending; ties keep sentence order."""
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return order if k is None else order[:k]


def _boost(scores: list[float], indices: Iterable[int] | None) -> list[float]:
    for i in set(indices or ()):
        if 0 <= i < len(scores):
            scores[i] += PROVENANCE_BONUS
    return scores


class LexicalScorer:
    """IDF-weighted question/sentence token overlap, normalized by the question's own IDF mass."""

    def __init__(self, sentences: Sequence[str]) -> None:
        self.sets = [set(tokenize(s)) for s in sentences]
        self.n = len(self.sets)
        df: dict[str, int] = {}
        for toks in self.sets:
            for t in toks:
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.default_idf = math.log(self.n + 1) + 1        # df = 0
        self.idf = {t: math.log((self.n + 1) / (d + 1)) + 1 for t, d in df.items()}

    def idf_of(self, token: str) -> float:
        return self.idf.get(token, self.default_idf)

    def score(self, question: str, boost: Iterable[int] | None = None) -> list[float]:
        q = list(dict.fromkeys(tokenize(question)))        # a repeated token weighs once
        denom = sum(self.idf_of(t) for t in q)
        if denom <= 0:
            return _boost([0.0] * self.n, boost)
        return _boost([sum(self.idf_of(t) for t in q if t in s) / denom for s in self.sets], boost)


class SonarScorer:
    """Cosine between the question embedding and each sentence embedding, mapped to [0, 1].
    `embeddings` is [N, D], L2-normalized, in sentence order; `encode(texts) -> ndarray`."""

    def __init__(self, embeddings, encode) -> None:
        self.emb = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        self.encode = encode
        self.n = 0 if self.emb.size == 0 else self.emb.shape[0]

    def score(self, question: str, boost: Iterable[int] | None = None) -> list[float]:
        if self.n == 0:
            return _boost([], boost)
        q = np.asarray(self.encode([question]), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm:
            q /= norm                                      # a stray scale must not skew cosine
        cos = np.clip(self.emb @ q, -1.0, 1.0)             # float16 storage overshoots +-1
        return _boost(((cos + 1.0) / 2.0).tolist(), boost)
