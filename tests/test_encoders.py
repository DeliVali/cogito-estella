"""Encoder registry, precedence and contract. Real adapters run under -m integration."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cogito_estella.encoders import (
    CANARY_PAIRS,
    CANARY_PATH,
    CANARY_TOLERANCE,
    DEFAULT_ENCODER,
    ENCODERS,
    EncoderMismatch,
    check_contract,
    encoder_revision,
    get_encoder,
    load_canary,
    resolve_encoder_name,
)
from cogito_estella.encoders.bge_m3 import BGE_M3_REVISION, BgeM3Encoder
from cogito_estella.encoders.m2m100_pool import (
    LANG_TO_M2M,
    M2M100_REVISION,
    M2m100PoolEncoder,
    lang_to_m2m,
)
from cogito_estella.encoders.pooling import AttnPool, PoolWeightsError, load_pool, sha256_file
from cogito_estella.encoders.sonar import SonarEncoder

TEXTS = [
    "The encoder maps text to a vector.",
    "The committee approved the new budget.",
    "SONAR decodes the concepts.",
    "A graph stores entities and relations.",
    "Vectors are compared with cosine similarity.",
]


@pytest.fixture(scope="module")
def pool_file(tmp_path_factory):
    """A real 1024-d AttnPool saved in the exp058 checkpoint layout (no weights loaded)."""
    path = tmp_path_factory.mktemp("pool") / "pool.pt"
    torch.save({"pool": AttnPool().state_dict(),
                "arch": {"d": 1024, "heads": 8, "queries": 1, "hidden": 1024},
                "epoch": 5}, path)
    return path


# --- registry -------------------------------------------------------------------

def test_registry_keys_are_exactly_the_three_adapters():
    assert set(ENCODERS) == {"sonar", "bge-m3", "m2m100-pool"}
    assert DEFAULT_ENCODER == "m2m100-pool"


def test_get_encoder_unknown_name_lists_the_available_ones():
    with pytest.raises(ValueError) as exc:
        get_encoder("nope")
    for name in ENCODERS:
        assert name in str(exc.value)


def test_get_encoder_threads_the_pool_path_to_the_factory(pool_file, monkeypatch):
    monkeypatch.delenv("COGITO_POOL", raising=False)
    enc = get_encoder("m2m100-pool", device="cpu", download=False, pool_path=pool_file)
    assert enc.name == "m2m100-pool" and str(enc.pool_path) == str(pool_file)


@pytest.mark.parametrize("name", ["sonar", "bge-m3"])
def test_pool_path_is_ignored_by_the_encoders_without_pooling(name, pool_file):
    assert get_encoder(name, device="cpu", download=False, pool_path=pool_file).name == name


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

def test_check_contract_all_true_for_a_conforming_encoder(fake_encoder):
    report = check_contract(fake_encoder(), TEXTS)
    for key in ("dim", "dtype", "unit_norm", "deterministic", "batch_invariant"):
        assert report[key] is True, report
    assert report["max_norm_dev"] < 1e-3


def test_check_contract_flags_a_wrong_width(wrong_dim_encoder):
    assert check_contract(wrong_dim_encoder(), TEXTS)["dim"] is False


def test_check_contract_flags_a_non_unit_norm(unnormalized_encoder):
    assert check_contract(unnormalized_encoder(), TEXTS)["unit_norm"] is False


def test_check_contract_rejects_an_empty_text_list(fake_encoder):
    with pytest.raises(ValueError):
        check_contract(fake_encoder(), [])


def test_check_contract_forwards_batch_size_and_normalize(fake_encoder):
    enc = fake_encoder()
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


def test_m2m100_pool_exposes_metadata_without_loading_weights(pool_file):
    enc = M2m100PoolEncoder(device="cpu", download=False, pool_path=pool_file)
    assert (enc.name, enc.dim, enc.native_normalized) == ("m2m100-pool", 1024, False)
    assert enc.pool_sha256 == sha256_file(pool_file)
    assert enc.revision == f"{M2M100_REVISION}+pool:{enc.pool_sha256[:12]}"
    assert len(M2M100_REVISION) == 40
    assert enc._model is None and enc._pool is None


@pytest.mark.parametrize("name", ["sonar", "bge-m3", "m2m100-pool"])
def test_empty_input_returns_an_empty_matrix_without_loading_weights(name, pool_file):
    enc = get_encoder(name, device="cpu", download=False, pool_path=pool_file)
    out = enc.encode([])
    assert out.shape == (0, 1024) and out.dtype == np.float32


# --- encode bodies over fake backends (no weights, no GPU) ----------------------

class _FakePipe:
    """Stands in for TextToEmbeddingModelPipeline; records the predict kwargs."""

    def __init__(self, mat):
        self.mat, self.calls = mat, []

    def predict(self, texts, source_lang, batch_size):
        self.calls.append({"n": len(texts), "source_lang": source_lang,
                           "batch_size": batch_size})
        return self.mat


def _sonar_over(mat):
    enc = SonarEncoder(device="cpu")
    enc._pipe = _FakePipe(mat)
    return enc


@pytest.mark.parametrize("normalize", [None, False])
def test_sonar_encode_returns_raw_vectors_unless_asked(normalize):
    """The load-bearing invariant: the published heads receive unnormalized SONAR."""
    enc = _sonar_over(torch.full((2, 1024), 0.005))
    out = enc.encode(["a", "b"], batch_size=7, normalize=normalize)
    assert enc._pipe.calls == [{"n": 2, "source_lang": "eng_Latn", "batch_size": 7}]
    assert out.dtype == np.float32 and out.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(out, axis=1), 0.16, atol=1e-4)


def test_sonar_encode_normalizes_on_request_and_survives_a_zero_row():
    raw = torch.full((2, 1024), 0.005)
    raw[1] = 0.0
    enc = _sonar_over(raw)
    out = enc.encode(["a", "b"], lang="spa_Latn", normalize=True)
    assert enc._pipe.calls[0]["source_lang"] == "spa_Latn"
    assert abs(float(np.linalg.norm(out[0])) - 1.0) < 1e-6
    assert np.all(np.isfinite(out)) and float(np.linalg.norm(out[1])) == 0.0


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self):
        self.batches = []

    def __call__(self, texts, **kwargs):
        self.batches.append(list(texts))
        return _FakeBatch(ids=torch.tensor([int(t) for t in texts]))


class _FakeModel:
    """One-hot CLS at column `id`: L2 normalization preserves the row identity."""

    def __call__(self, ids):
        hidden = torch.zeros(len(ids), 2, 1024)
        hidden[torch.arange(len(ids)), 0, ids] = 1.0
        return SimpleNamespace(last_hidden_state=hidden)


def test_bge_m3_encode_chunks_by_batch_size_and_keeps_row_order():
    enc = BgeM3Encoder(device="cpu", download=False)
    enc._tok, enc._model = _FakeTokenizer(), _FakeModel()
    out = enc.encode([str(i) for i in range(7)], batch_size=3)
    assert [len(b) for b in enc._tok.batches] == [3, 3, 1]
    assert out.shape == (7, 1024) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    assert out.argmax(axis=1).tolist() == list(range(7))


# --- learned pooling ------------------------------------------------------------

def test_load_pool_round_trips_a_saved_attnpool(tmp_path):
    src = AttnPool(d=16, heads=2, queries=1, hidden=8).eval()
    path = tmp_path / "tiny.pt"
    torch.save({"pool": src.state_dict(), "heads": {}, "epoch": 1,
                "arch": {"d": 16, "heads": 2, "queries": 1, "hidden": 8}}, path)
    back = load_pool(path, "cpu")
    h, mask = torch.randn(3, 5, 16), torch.ones(3, 5, dtype=torch.bool)
    mask[1, 3:] = False
    with torch.inference_mode():
        assert torch.allclose(src(h, mask), back(h, mask), atol=1e-6)
    assert back.training is False


def test_load_pool_reports_a_missing_file(tmp_path):
    with pytest.raises(PoolWeightsError) as exc:
        load_pool(tmp_path / "absent.pt", "cpu")
    assert "absent.pt" in str(exc.value)


def test_load_pool_reports_a_checkpoint_without_pool_weights(tmp_path):
    path = tmp_path / "heads_only.pt"
    torch.save({"heads": {}, "epoch": 1}, path)
    with pytest.raises(PoolWeightsError) as exc:
        load_pool(path, "cpu")
    assert "pool" in str(exc.value)


def _save_pool_safetensors(path, pool, arch, sidecar=True):
    from safetensors.torch import save_file

    meta = None if sidecar else {"cogito": json.dumps({"arch": arch})}
    save_file({k: v.contiguous() for k, v in pool.state_dict().items()}, path, metadata=meta)
    if sidecar:
        path.with_suffix(".json").write_text(json.dumps({"arch": arch}))


def test_load_pool_round_trips_a_safetensors_pool_with_a_sidecar(tmp_path):
    arch = {"d": 16, "heads": 2, "queries": 1, "hidden": 8}
    src = AttnPool(**arch).eval()
    path = tmp_path / "tiny.safetensors"
    _save_pool_safetensors(path, src, arch)
    back = load_pool(path, "cpu")
    h, mask = torch.randn(3, 5, 16), torch.ones(3, 5, dtype=torch.bool)
    mask[1, 3:] = False
    with torch.inference_mode():
        assert torch.allclose(src(h, mask), back(h, mask), atol=1e-6)
    assert back.d == 16 and back.training is False


def test_load_pool_reads_the_arch_from_the_safetensors_header(tmp_path):
    arch = {"d": 16, "heads": 2, "queries": 1, "hidden": 8}
    path = tmp_path / "header.safetensors"
    _save_pool_safetensors(path, AttnPool(**arch).eval(), arch, sidecar=False)
    assert load_pool(path, "cpu").d == 16


def test_load_pool_rejects_a_safetensors_pool_that_does_not_fit_its_arch(tmp_path):
    """A default arch over 16-d weights: the mismatch must name the file, not crash."""
    path = tmp_path / "wrong.safetensors"
    _save_pool_safetensors(path, AttnPool(d=16, heads=2, queries=1, hidden=8).eval(), {})
    with pytest.raises(PoolWeightsError) as exc:
        load_pool(path, "cpu")
    assert "wrong.safetensors" in str(exc.value)


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "blob.bin"
    payload = b"cogito" * 5000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


# --- m2m100-pool language codes -------------------------------------------------

def test_sonar_language_codes_map_to_m2m100_codes():
    assert lang_to_m2m("eng_Latn") == "en"
    assert lang_to_m2m("spa_Latn") == "es"
    assert lang_to_m2m("zho_Hans") == "zh"
    assert LANG_TO_M2M["arb_Arab"] == "ar" and LANG_TO_M2M["jpn_Jpan"] == "ja"
    assert len(set(LANG_TO_M2M.values())) == len(LANG_TO_M2M)


def test_an_unknown_language_falls_back_to_english_and_warns_once(capsys, monkeypatch):
    import cogito_estella.encoders.m2m100_pool as mod

    monkeypatch.setattr(mod, "_WARNED", set())
    assert mod.lang_to_m2m("kli_Piqd") == "en"
    assert mod.lang_to_m2m("kli_Piqd") == "en"
    err = capsys.readouterr().err
    assert err.count("kli_Piqd") == 1 and "en" in err


# --- m2m100-pool adapter --------------------------------------------------------

def test_m2m100_pool_without_any_pool_path_says_how_to_supply_one(monkeypatch):
    monkeypatch.delenv("COGITO_POOL", raising=False)
    with pytest.raises(PoolWeightsError) as exc:
        M2m100PoolEncoder(device="cpu", download=False)
    assert "COGITO_POOL" in str(exc.value) and "pool" in str(exc.value)


def test_m2m100_pool_reads_the_pool_path_from_the_environment(pool_file, monkeypatch):
    monkeypatch.setenv("COGITO_POOL", str(pool_file))
    enc = M2m100PoolEncoder(device="cpu", download=False)
    assert str(enc.pool_path) == str(pool_file)


def test_an_explicit_pool_path_wins_over_the_environment(pool_file, tmp_path, monkeypatch):
    monkeypatch.setenv("COGITO_POOL", str(tmp_path / "never.pt"))
    enc = M2m100PoolEncoder(device="cpu", download=False, pool_path=pool_file)
    assert str(enc.pool_path) == str(pool_file)


def test_a_pool_of_another_width_is_rejected_against_the_encoder_contract(tmp_path):
    path = tmp_path / "narrow.pt"
    torch.save({"pool": AttnPool(d=16, heads=2, queries=1, hidden=8).state_dict(),
                "arch": {"d": 16, "heads": 2, "queries": 1, "hidden": 8}}, path)
    enc = M2m100PoolEncoder(device="cpu", download=False, pool_path=path)
    with pytest.raises(PoolWeightsError) as exc:
        enc._load_pool()
    assert "16" in str(exc.value) and "1024" in str(exc.value)


class _FakeM2mBatch(dict):
    def to(self, device):
        return self


class _FakeM2mTokenizer:
    """Records batches and the src_lang each was tokenized under."""

    def __init__(self):
        self.batches, self.kwargs = [], []
        self.src_lang = "en"

    def __call__(self, texts, **kwargs):
        self.batches.append((list(texts), self.src_lang))
        self.kwargs.append(kwargs)
        ids = torch.tensor([[int(t)] for t in texts])
        return _FakeM2mBatch(input_ids=ids,
                             attention_mask=torch.ones_like(ids))


class _FakeM2mModel:
    """One-hot token state at column `id`, so pooling preserves the row identity."""

    def __call__(self, input_ids, attention_mask):
        hidden = torch.zeros(len(input_ids), 1, 1024)
        hidden[torch.arange(len(input_ids)), 0, input_ids[:, 0]] = 3.0
        return SimpleNamespace(last_hidden_state=hidden)


def _fake_m2m(pool_file, monkeypatch):
    monkeypatch.delenv("COGITO_POOL", raising=False)
    enc = M2m100PoolEncoder(device="cpu", download=False, pool_path=pool_file)
    enc._tok, enc._model = _FakeM2mTokenizer(), _FakeM2mModel()
    enc._pool = lambda h, mask: h[:, 0]          # masked pooling stands in as first token
    return enc


def test_m2m100_pool_encode_chunks_by_batch_size_and_keeps_row_order(pool_file, monkeypatch):
    enc = _fake_m2m(pool_file, monkeypatch)
    out = enc.encode([str(i) for i in range(7)], batch_size=3)
    assert [len(b) for b, _ in enc._tok.batches] == [3, 3, 1]
    assert out.shape == (7, 1024) and out.dtype == np.float32
    assert out.argmax(axis=1).tolist() == list(range(7))
    assert enc._tok.kwargs[0]["max_length"] == 128 and enc._tok.kwargs[0]["truncation"] is True


@pytest.mark.parametrize("normalize", [None, False])
def test_m2m100_pool_is_raw_natively_because_the_trunk_was_trained_on_raw(normalize,
                                                                          pool_file,
                                                                          monkeypatch):
    enc = _fake_m2m(pool_file, monkeypatch)
    out = enc.encode(["1", "2"], normalize=normalize)
    assert np.allclose(np.linalg.norm(out, axis=1), 3.0, atol=1e-5)


def test_m2m100_pool_normalizes_on_request(pool_file, monkeypatch):
    enc = _fake_m2m(pool_file, monkeypatch)
    out = enc.encode(["1", "2"], normalize=True)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_m2m100_pool_passes_the_mapped_language_to_the_tokenizer(pool_file, monkeypatch):
    enc = _fake_m2m(pool_file, monkeypatch)
    enc.encode(["1"], lang="spa_Latn")
    assert enc._tok.batches[-1][1] == "es"


# --- canary file ----------------------------------------------------------------

def test_canary_file_is_shipped_and_covers_every_registered_encoder():
    assert CANARY_PATH.exists()
    data = load_canary()
    assert data["pairs"] == [list(p) for p in CANARY_PAIRS]
    assert len(CANARY_PAIRS) == 2 and all(len(p) == 2 for p in CANARY_PAIRS)
    for name in ENCODERS:
        cosines = data[name]["cosines"]
        assert len(cosines) == len(CANARY_PAIRS)
        assert all(-1.0 <= c <= 1.0 for c in cosines)
    assert 0 < CANARY_TOLERANCE <= 0.05
    assert json.loads(CANARY_PATH.read_text(encoding="utf-8")) == data


def test_encoder_revision_comes_from_the_shipped_canary():
    data = load_canary()
    for name in ENCODERS:
        assert encoder_revision(name) == data[name]["revision"]
    assert encoder_revision("never-published") == "unknown"


def test_write_canary_round_trips_without_touching_the_shipped_file(tmp_path, monkeypatch,
                                                                       fake_encoder):
    import cogito_estella.encoders as enc_mod

    target = tmp_path / "canary.json"
    monkeypatch.setattr(enc_mod, "CANARY_PATH", target)
    enc_mod.write_canary(fake_encoder(), [0.123456, -0.5])
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["fake"] == {"revision": "fake-1", "dim": 1024, "cosines": [0.1235, -0.5]}
    assert data["tolerance"] == CANARY_TOLERANCE and data["pairs"] == [list(p) for p in CANARY_PAIRS]


# --- real adapters --------------------------------------------------------------

def _missing_weight_errors() -> tuple[type[BaseException], ...]:
    """Weights absent from disk. Any other failure is an adapter defect, not a skip."""
    errors: list[type[BaseException]] = [OSError]
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError

        errors.append(LocalEntryNotFoundError)
    except ImportError:
        pass
    try:
        from fairseq2.assets import AssetNotFoundError

        errors.append(AssetNotFoundError)
    except ImportError:
        pass
    return tuple(errors)


def _refresh_requested() -> bool:
    """Re-baselining the tracked canary file is opt-in, never a suite side effect."""
    return bool(os.environ.get("COGITO_REFRESH_CANARY"))


def test_only_absent_weights_may_skip_the_integration_fixture():
    errors = _missing_weight_errors()
    assert OSError in errors
    for defect in (RuntimeError, TypeError, ValueError, torch.OutOfMemoryError):
        assert not any(issubclass(defect, err) for err in errors), defect
    pytest.importorskip("fairseq2.assets")
    from fairseq2.assets import AssetNotFoundError

    assert AssetNotFoundError in errors


def test_canary_refresh_is_opt_in(monkeypatch):
    monkeypatch.delenv("COGITO_REFRESH_CANARY", raising=False)
    assert _refresh_requested() is False
    monkeypatch.setenv("COGITO_REFRESH_CANARY", "1")
    assert _refresh_requested() is True


@pytest.fixture(scope="module", params=["sonar", "bge-m3", "m2m100-pool"])
def real_encoder(request):
    """One real encoder at a time; VRAM is released before the next parameter."""
    if not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    if request.param == "sonar":
        pytest.importorskip("sonar")
    if request.param == "m2m100-pool":
        pool = os.environ.get("COGITO_POOL")
        if not pool or not Path(pool).exists():
            pytest.skip("COGITO_POOL does not point at pooling weights")
    enc = get_encoder(request.param, device="cuda", download=False)
    try:
        enc.encode(["warm up"], normalize=True)
    except _missing_weight_errors() as exc:
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
def test_canary_cosines_match_the_shipped_reference(real_encoder):
    """Read-only: COGITO_REFRESH_CANARY=1 is the only way to re-baseline the tracked file."""
    from cogito_estella.encoders import canary_cosines, write_canary

    fresh = canary_cosines(real_encoder)
    if real_encoder.name not in load_canary():
        write_canary(real_encoder, fresh)          # first baseline for a new encoder
    stored = load_canary()[real_encoder.name]
    assert stored["revision"] == real_encoder.revision and stored["dim"] == real_encoder.dim
    dev = float(np.max(np.abs(np.array(stored["cosines"]) - np.array(fresh))))
    assert dev <= CANARY_TOLERANCE, (real_encoder.name, stored["cosines"], fresh, dev)
    if not _refresh_requested():
        return
    write_canary(real_encoder, fresh)
    assert load_canary()[real_encoder.name]["cosines"] == [round(c, 4) for c in fresh]
