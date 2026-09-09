"""Encoder registry, precedence and contract. Real adapters run under -m integration."""
import hashlib
import json

import numpy as np
import pytest

from cogito_estella.encoders import (
    CANARY_PAIRS,
    CANARY_PATH,
    CANARY_TOLERANCE,
    DEFAULT_ENCODER,
    ENCODERS,
    EncoderMismatch,
    check_contract,
    get_encoder,
    load_canary,
    resolve_encoder_name,
)
from cogito_estella.encoders.bge_m3 import BGE_M3_REVISION, BgeM3Encoder
from cogito_estella.encoders.sonar import SonarEncoder

TEXTS = [
    "The encoder maps text to a vector.",
    "The committee approved the new budget.",
    "SONAR decodes the concepts.",
    "A graph stores entities and relations.",
    "Vectors are compared with cosine similarity.",
]


class FakeEncoder:
    """Deterministic unit-norm hash encoder; records the kwargs it was called with."""

    name = "fake"
    revision = "fake-1"
    dim = 1024
    native_normalized = True

    def __init__(self):
        self.calls = []

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


# --- registry -------------------------------------------------------------------

def test_registry_keys_are_exactly_sonar_and_bge_m3():
    assert set(ENCODERS) == {"sonar", "bge-m3"}
    assert DEFAULT_ENCODER == "sonar"


def test_get_encoder_unknown_name_lists_the_available_ones():
    with pytest.raises(ValueError) as exc:
        get_encoder("nope")
    assert "sonar" in str(exc.value) and "bge-m3" in str(exc.value)


# --- precedence -----------------------------------------------------------------

def test_explicit_wins_over_checkpoint_and_env():
    assert resolve_encoder_name("bge-m3", ["sonar"], {"COGITO_ENCODER": "sonar"}) == "bge-m3"


def test_agreeing_checkpoints_decide():
    assert resolve_encoder_name(None, ["bge-m3", "bge-m3", None], {}) == "bge-m3"


def test_disagreeing_checkpoints_raise_with_both_names():
    with pytest.raises(EncoderMismatch) as exc:
        resolve_encoder_name(None, ["sonar", "bge-m3"], {})
    assert "sonar" in str(exc.value) and "bge-m3" in str(exc.value)


def test_env_decides_when_no_explicit_and_no_checkpoint_metadata():
    assert resolve_encoder_name(None, [None], {"COGITO_ENCODER": "bge-m3"}) == "bge-m3"


def test_env_disagreeing_with_the_checkpoint_raises():
    with pytest.raises(EncoderMismatch) as exc:
        resolve_encoder_name(None, ["sonar"], {"COGITO_ENCODER": "bge-m3"})
    assert "sonar" in str(exc.value) and "bge-m3" in str(exc.value)


def test_default_when_nothing_is_specified():
    assert resolve_encoder_name(None, [], None) == DEFAULT_ENCODER
    assert resolve_encoder_name(None, [None, None], {}) == DEFAULT_ENCODER


# --- contract checker -----------------------------------------------------------

def test_check_contract_all_true_for_a_conforming_encoder():
    report = check_contract(FakeEncoder(), TEXTS)
    for key in ("dim", "dtype", "unit_norm", "deterministic", "batch_invariant"):
        assert report[key] is True, report
    assert report["max_norm_dev"] < 1e-3


def test_check_contract_flags_a_wrong_width():
    assert check_contract(WrongDimEncoder(), TEXTS)["dim"] is False


def test_check_contract_flags_a_non_unit_norm():
    assert check_contract(UnnormalizedEncoder(), TEXTS)["unit_norm"] is False


def test_check_contract_rejects_an_empty_text_list():
    with pytest.raises(ValueError):
        check_contract(FakeEncoder(), [])


def test_check_contract_forwards_batch_size_and_normalize():
    enc = FakeEncoder()
    check_contract(enc, TEXTS, batch_size=8)
    assert enc.calls[0] == {"lang": "eng_Latn", "batch_size": 8, "normalize": True}


# --- adapter metadata, lazily (no weights touched) ------------------------------

def test_adapters_expose_metadata_without_loading_weights():
    sonar = SonarEncoder(device="cpu")
    assert (sonar.name, sonar.dim, sonar.native_normalized) == ("sonar", 1024, False)
    assert sonar.revision == "text_sonar_basic_encoder"
    assert sonar._pipe is None

    bge = BgeM3Encoder(device="cpu", download=False)
    assert (bge.name, bge.dim, bge.native_normalized) == ("bge-m3", 1024, True)
    assert bge.revision == BGE_M3_REVISION and len(BGE_M3_REVISION) == 40
    assert bge._model is None


@pytest.mark.parametrize("adapter", [
    lambda: SonarEncoder(device="cpu"),
    lambda: BgeM3Encoder(device="cpu", download=False),
])
def test_empty_input_returns_an_empty_matrix_without_loading_weights(adapter):
    out = adapter().encode([])
    assert out.shape == (0, 1024) and out.dtype == np.float32


# --- canary file ----------------------------------------------------------------

def test_canary_file_is_shipped_and_covers_both_encoders():
    assert CANARY_PATH.exists()
    data = load_canary()
    assert data["pairs"] == [list(p) for p in CANARY_PAIRS]
    assert len(CANARY_PAIRS) == 2 and all(len(p) == 2 for p in CANARY_PAIRS)
    for name in ("sonar", "bge-m3"):
        cosines = data[name]["cosines"]
        assert len(cosines) == len(CANARY_PAIRS)
        assert all(-1.0 <= c <= 1.0 for c in cosines)
    assert 0 < CANARY_TOLERANCE <= 0.05
    assert json.loads(CANARY_PATH.read_text(encoding="utf-8")) == data


# --- real adapters --------------------------------------------------------------

@pytest.fixture(scope="module", params=["sonar", "bge-m3"])
def real_encoder(request):
    """One real encoder at a time; VRAM is released before the next parameter."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    if request.param == "sonar":
        pytest.importorskip("sonar")
    enc = get_encoder(request.param, device="cuda", download=False)
    try:
        enc.encode(["warm up"], normalize=True)
    except (OSError, RuntimeError) as exc:  # weights not on disk
        pytest.skip(f"{request.param} unavailable: {exc}")
    yield enc
    del enc
    torch.cuda.empty_cache()


@pytest.mark.integration
def test_real_encoder_meets_the_contract(real_encoder):
    report = check_contract(real_encoder, TEXTS)
    for key in ("dim", "dtype", "unit_norm", "deterministic", "batch_invariant"):
        assert report[key] is True, (real_encoder.name, report)


@pytest.mark.integration
def test_real_encoder_native_normalization_matches_its_flag(real_encoder):
    raw = real_encoder.encode(TEXTS)
    norms = np.linalg.norm(raw, axis=1)
    unit = bool(np.all(np.abs(norms - 1.0) < 1e-3))
    assert unit is real_encoder.native_normalized, (real_encoder.name, norms.tolist())


@pytest.mark.integration
def test_canary_cosines_are_stable_and_refreshed(real_encoder):
    """Compare against the shipped reference, refresh it, then re-verify from disk."""
    from cogito_estella.encoders import canary_cosines, write_canary

    fresh = canary_cosines(real_encoder)
    stored = load_canary().get(real_encoder.name, {}).get("cosines")
    if stored is not None:
        dev = np.max(np.abs(np.array(stored) - np.array(fresh)))
        assert dev <= CANARY_TOLERANCE, (real_encoder.name, stored, fresh, float(dev))
    write_canary(real_encoder, fresh)
    reread = load_canary()[real_encoder.name]["cosines"]
    again = canary_cosines(real_encoder)
    assert np.max(np.abs(np.array(reread) - np.array(again))) <= CANARY_TOLERANCE
    assert load_canary()[real_encoder.name]["revision"] == real_encoder.revision
