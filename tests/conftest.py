"""Shared fixtures. spaCy is optional: tests that need it skip cleanly when absent."""
import hashlib

import numpy as np
import pytest


@pytest.fixture(scope="session")
def nlp():
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        pytest.skip("en_core_web_sm not installed")


@pytest.fixture
def fake_extractor(nlp):
    """Real CogitoGraphExtractor methods over a fake head: `extract`/`extract_batch`
    return fixed triples per sentence; spans and lexicalization run for real."""
    from cogito_estella.integrations.llamaindex_connector import CogitoGraphExtractor

    def make(triples_by_sentence, ent2id=None):
        ex = object.__new__(CogitoGraphExtractor)
        ex.ent2id = ent2id or {"concept": 1, "sonar": 2, "encoder": 3, "text": 4,
                               "vector": 5, "budget": 6, "committee": 7, "decoder": 8}
        ex._nlp = nlp
        ex.device, ex.rels = "cpu", []
        ex.extract = lambda text, candidates=None, lang="eng_Latn", return_scores=False: \
            list(triples_by_sentence.get(text, []))
        ex.extract_batch = lambda texts, candidates=None, lang="eng_Latn": \
            [list(triples_by_sentence.get(t, [])) for t in texts]
        return ex
    return make


class FakeEncoder:
    """Offline TextEncoder: deterministic unit-norm hash vectors, records every call.
    `name` is settable so a fake can stand in for the encoder a checkpoint names."""

    revision = "fake-1"
    dim = 1024
    native_normalized = True

    def __init__(self, name: str = "fake"):
        self.name = name
        self.calls: list[dict] = []

    def encode(self, texts, lang="eng_Latn", batch_size=64, normalize=None):
        self.calls.append({"lang": lang, "batch_size": batch_size, "normalize": normalize})
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
            vec = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            rows.append(vec / np.linalg.norm(vec))
        return np.stack(rows).astype(np.float32)


class WrongDimEncoder(FakeEncoder):
    def encode(self, texts, lang="eng_Latn", batch_size=64, normalize=None):
        return super().encode(texts, lang, batch_size, normalize)[:, :512]


class UnnormalizedEncoder(FakeEncoder):
    def encode(self, texts, lang="eng_Latn", batch_size=64, normalize=None):
        return super().encode(texts, lang, batch_size, normalize) * 3.0


@pytest.fixture
def fake_encoder():
    """The class, not an instance: callers pick the name (`fake_encoder("sonar")`)."""
    return FakeEncoder


@pytest.fixture
def wrong_dim_encoder():
    return WrongDimEncoder


@pytest.fixture
def unnormalized_encoder():
    return UnnormalizedEncoder
