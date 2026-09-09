"""Encoder contract at the extractor boundary: checkpoint metadata, precedence, canary.
Offline: fake encoders over a real decoder state dict; no model weights are touched."""
import json

import numpy as np
import pytest
import torch

import cogito_estella.integrations.llamaindex_connector as lc
from cogito_estella.encoders import EncoderMismatch, canary_cosines
from cogito_estella.integrations.llamaindex_connector import CogitoGraphExtractor
from cogito_estella.model.candidate_decoder import (
    CandidateDecoderConfig,
    CandidateGraphDecoder,
)

VOCAB = {"ent2id": {"encoder": 1, "text": 2, "vector": 3}, "rel2id": {"give": 0, "map": 1}}


@pytest.fixture(scope="session")
def dec_state():
    """One real state dict behind every checkpoint here; fp16 halves the 127 MB payload
    that `load_state_dict` casts back on load."""
    dec = CandidateGraphDecoder(CandidateDecoderConfig(n_relations=len(VOCAB["rel2id"])))
    return {k: (v.half() if v.is_floating_point() else v)
            for k, v in dec.state_dict().items()}


@pytest.fixture(scope="session")
def ck(dec_state, tmp_path_factory):
    """`ck(**metadata)` -> checkpoint path, written once per distinct metadata."""
    root = tmp_path_factory.mktemp("ckpts")

    def make(**meta):
        key = "-".join(f"{k}_{v}" for k, v in sorted(meta.items())) or "bare"
        path = root / f"{key}.pt"
        if not path.exists():
            torch.save({"dec": dec_state, **meta}, path)
        return str(path)
    return make


@pytest.fixture
def vocab_file(tmp_path):
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(VOCAB))
    return str(p)


@pytest.fixture(autouse=True)
def _no_env_encoder(monkeypatch):
    monkeypatch.delenv("COGITO_ENCODER", raising=False)


def _built(ck, vocab_file, enc, **meta):
    return CogitoGraphExtractor(ck(**meta), vocab_file, device="cpu", encoder=enc)


# --- checkpoint contract --------------------------------------------------------

def test_a_sonar_era_checkpoint_is_rejected_under_bge_m3(ck, vocab_file):
    """The dimension collision the contract exists for: 1024 == 1024, wrong space."""
    with pytest.raises(EncoderMismatch) as exc:
        CogitoGraphExtractor(ck(), vocab_file, device="cpu", encoder="bge-m3")
    assert "sonar" in str(exc.value) and "bge-m3" in str(exc.value)


def test_checkpoint_metadata_decides_the_encoder_without_loading_a_model(ck, vocab_file):
    ex = CogitoGraphExtractor(ck(encoder="bge-m3"), vocab_file, device="cpu")
    assert (ex.encoder_name, ex.dim, ex.head_normalize) == ("bge-m3", 1024, False)
    assert ex._encoder is None                     # the property was never touched


def test_env_disagreeing_with_the_checkpoint_stops_the_load(ck, vocab_file, monkeypatch):
    monkeypatch.setenv("COGITO_ENCODER", "bge-m3")
    with pytest.raises(EncoderMismatch) as exc:
        CogitoGraphExtractor(ck(encoder="sonar"), vocab_file, device="cpu")
    assert "sonar" in str(exc.value) and "bge-m3" in str(exc.value)


def test_an_explicit_encoder_overrides_the_env(ck, vocab_file, monkeypatch):
    monkeypatch.setenv("COGITO_ENCODER", "bge-m3")
    ex = CogitoGraphExtractor(ck(encoder="sonar"), vocab_file, device="cpu", encoder="sonar")
    assert ex.encoder_name == "sonar"


def test_checkpoints_that_disagree_with_each_other_stop_the_load(ck, vocab_file):
    with pytest.raises(EncoderMismatch) as exc:
        CogitoGraphExtractor([ck(encoder="sonar"), ck(encoder="bge-m3")], vocab_file,
                             device="cpu")
    assert "sonar" in str(exc.value) and "bge-m3" in str(exc.value)


def test_a_checkpoint_of_another_width_is_rejected(ck, vocab_file):
    with pytest.raises(EncoderMismatch) as exc:
        CogitoGraphExtractor(ck(dim=512, encoder="sonar"), vocab_file, device="cpu")
    assert "512" in str(exc.value)


def test_checkpoints_that_disagree_on_normalization_stop_the_load(ck, vocab_file):
    with pytest.raises(EncoderMismatch) as exc:
        CogitoGraphExtractor([ck(encoder="sonar"), ck(encoder="sonar", normalize=True)],
                             vocab_file, device="cpu")
    assert "normalize" in str(exc.value)


# --- encoder wiring -------------------------------------------------------------

def test_the_encoder_is_built_from_the_registry_on_first_use(ck, vocab_file, fake_encoder,
                                                             monkeypatch):
    made = []

    def fake_get(name, device=None, download=True):
        made.append((name, device, download))
        return fake_encoder(name)

    monkeypatch.setattr(lc, "get_encoder", fake_get)
    ex = CogitoGraphExtractor(ck(encoder="sonar"), vocab_file, device="cpu", download=False)
    assert made == []                              # constructing an extractor loads no weights
    assert ex.encoder.name == "sonar" and ex.encoder is ex.encoder
    assert made == [("sonar", "cpu", False)]


def test_a_raw_checkpoint_asks_the_encoder_for_its_native_vectors(ck, vocab_file,
                                                                  fake_encoder):
    """SONAR-era heads were trained on raw vectors: `_encode` must not normalize."""
    enc = fake_encoder("sonar")
    ex = _built(ck, vocab_file, enc, encoder="sonar")
    out = ex._encode(["a", "b"], "spa_Latn")
    assert enc.calls[-1] == {"lang": "spa_Latn", "batch_size": 64, "normalize": None}
    assert isinstance(out, torch.Tensor) and out.shape == (2, 1024)


def test_a_normalized_checkpoint_asks_for_unit_vectors(ck, vocab_file, fake_encoder):
    enc = fake_encoder("sonar")
    ex = _built(ck, vocab_file, enc, encoder="sonar", normalize=True)
    assert ex.head_normalize is True
    ex._encode(["a"], "eng_Latn")
    assert enc.calls[-1] == {"lang": "eng_Latn", "batch_size": 64, "normalize": True}


def test_encode_batch_is_float16_unit_norm_in_chunks(ck, vocab_file, fake_encoder):
    enc = fake_encoder("sonar")
    ex = _built(ck, vocab_file, enc, encoder="sonar")
    out = ex.encode_batch([f"sentence {i}" for i in range(70)])
    assert out.shape == (70, 1024) and out.dtype == np.float16
    assert np.allclose(np.linalg.norm(out.astype(np.float32), axis=1), 1.0, atol=1e-2)
    assert [c["normalize"] for c in enc.calls] == [True, True]      # 64 + 6
    assert ex.encode_batch([]).shape == (0, 1024)


# --- canary ---------------------------------------------------------------------

def _reference(enc, offset):
    return {enc.name: {"revision": enc.revision, "dim": enc.dim,
                       "cosines": [c + offset for c in canary_cosines(enc)]}}


def test_check_canary_passes_inside_the_tolerance(ck, vocab_file, fake_encoder, monkeypatch):
    enc = fake_encoder("sonar")
    ex = _built(ck, vocab_file, enc, encoder="sonar")
    monkeypatch.setattr(lc, "load_canary", lambda: _reference(enc, 0.01))
    ex.check_canary()


def test_check_canary_fails_when_the_encoder_drifted(ck, vocab_file, fake_encoder,
                                                     monkeypatch):
    enc = fake_encoder("sonar")
    ex = _built(ck, vocab_file, enc, encoder="sonar")
    monkeypatch.setattr(lc, "load_canary", lambda: _reference(enc, 0.5))
    with pytest.raises(EncoderMismatch, match="canary"):
        ex.check_canary()


def test_check_canary_fails_when_the_reference_is_missing(ck, vocab_file, fake_encoder,
                                                          monkeypatch):
    enc = fake_encoder("sonar")
    ex = _built(ck, vocab_file, enc, encoder="sonar")
    monkeypatch.setattr(lc, "load_canary", dict)
    with pytest.raises(EncoderMismatch, match="canary"):
        ex.check_canary()
