"""Encoder-aware weight resolution: explicit paths, the published manifest, offline misses."""
import json

import pytest

from cogito_estella.encoders import DEFAULT_ENCODER, ENSEMBLE_OPERATING_POINT
from cogito_estella.mcp import weights as w


def _fake_hub(tmp_path, manifest=None, absent=()):
    """A downloader over `tmp_path`: every requested file appears unless listed absent."""
    calls = []

    def dl(repo_id, filename, local_files_only=False, subfolder=None):
        calls.append((repo_id, filename, subfolder, local_files_only))
        if filename in absent:
            raise FileNotFoundError(filename)
        if filename == w.MANIFEST:
            if manifest is None:
                raise FileNotFoundError(filename)
            p = tmp_path / filename
            p.write_text(json.dumps(manifest))
            return str(p)
        p = tmp_path / (subfolder or "") / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)
    return dl, calls


# --- explicit paths -------------------------------------------------------------

def test_explicit_paths_are_used_and_must_exist(tmp_path):
    ck, vocab = tmp_path / "a.pt", tmp_path / "v.json"
    ck.write_bytes(b"x"); vocab.write_text("{}")
    got = w.resolve([str(ck)], str(vocab), download=False)
    assert (got.checkpoints, got.vocab, got.pool) == ([ck], vocab, None)
    with pytest.raises(w.WeightsError, match="missing"):
        w.resolve([str(tmp_path / "nope.pt")], str(vocab), download=False)


def test_explicit_paths_never_touch_the_hub(tmp_path):
    ck, vocab = tmp_path / "a.pt", tmp_path / "v.json"
    ck.write_bytes(b"x"); vocab.write_text("{}")

    def dl(*a, **k):
        raise AssertionError("explicit paths must not download")
    got = w.resolve([str(ck)], str(vocab), download=True, downloader=dl)
    assert got.checkpoints == [ck]


def test_explicit_paths_without_a_flag_leave_the_encoder_unset(tmp_path):
    """The checkpoints decide; stamping the process default would announce a space the
    extractor will not use (legacy `.pt` files resolve to sonar)."""
    ck, vocab = tmp_path / "a.pt", tmp_path / "v.json"
    ck.write_bytes(b"x"); vocab.write_text("{}")
    assert w.resolve([str(ck)], str(vocab), download=False).encoder is None
    assert w.resolve([str(ck)], str(vocab), download=False).operating_point is None
    flagged = w.resolve([str(ck)], str(vocab), download=False, encoder="sonar")
    assert flagged.encoder == "sonar"


def test_partial_explicit_is_rejected(tmp_path):
    with pytest.raises(w.WeightsError, match="both"):
        w.resolve([str(tmp_path / "a.pt")], None, download=False)


def test_an_explicit_pool_is_carried_through_and_must_exist(tmp_path):
    ck, vocab, pool = tmp_path / "a.pt", tmp_path / "v.json", tmp_path / "pool.pt"
    ck.write_bytes(b"x"); vocab.write_text("{}"); pool.write_bytes(b"x")
    assert w.resolve([str(ck)], str(vocab), download=False, pool=str(pool)).pool == pool
    with pytest.raises(w.WeightsError, match="pooling"):
        w.resolve([str(ck)], str(vocab), download=False, pool=str(tmp_path / "gone.pt"))


def test_errors_name_the_encoder_only_once_one_is_settled(tmp_path):
    """The published branch can name its default; explicit paths cannot, since nothing
    has read the checkpoints yet."""
    ck, vocab = tmp_path / "a.pt", tmp_path / "v.json"
    ck.write_bytes(b"x"); vocab.write_text("{}")
    gone = str(tmp_path / "gone.pt")
    with pytest.raises(w.WeightsError) as explicit:
        w.resolve([str(ck)], str(vocab), download=False, pool=gone)
    with pytest.raises(w.WeightsError) as published:
        w.resolve(None, None, download=False, pool=gone)
    assert "for encoder" not in str(explicit.value)
    assert f"for encoder {DEFAULT_ENCODER}" in str(published.value)


# --- published assets per encoder -----------------------------------------------

def test_the_default_encoder_resolves_to_the_m2m100_pool_assets(tmp_path):
    dl, calls = _fake_hub(tmp_path)
    got = w.resolve(None, None, download=True, downloader=dl)
    assets = w.ENCODER_ASSETS["m2m100-pool"]
    assert got.encoder == DEFAULT_ENCODER == "m2m100-pool"
    assert [c.name for c in got.checkpoints] == list(assets["checkpoints"])
    assert got.vocab.name == assets["vocab"] and got.pool.name == assets["pool"]
    assert {sub for _, _, sub, _ in calls if sub} == {assets["subfolder"]}


def test_sonar_resolves_to_the_repository_root_files(tmp_path):
    dl, calls = _fake_hub(tmp_path)
    got = w.resolve(None, None, download=True, downloader=dl, encoder="sonar")
    assert [c.name for c in got.checkpoints] == list(w.DEFAULT_CHECKPOINTS)
    assert got.vocab.name == w.DEFAULT_VOCAB and got.pool is None
    assert all(sub is None for _, _, sub, _ in calls)


def test_an_explicit_pool_replaces_the_published_one(tmp_path):
    pool = tmp_path / "local-pool.pt"
    pool.write_bytes(b"x")
    dl, calls = _fake_hub(tmp_path)
    got = w.resolve(None, None, download=True, downloader=dl, pool=str(pool))
    assert got.pool == pool
    assert w.ENCODER_ASSETS["m2m100-pool"]["pool"] not in [f for _, f, _, _ in calls]


def test_the_published_operating_point_is_carried_through(tmp_path):
    dl, _ = _fake_hub(tmp_path)
    assert w.resolve(None, None, download=True, downloader=dl).operating_point == (0.1, 0.7)
    got = w.resolve(None, None, download=True, downloader=dl, encoder="sonar")
    assert got.operating_point == (0.1, 0.8)


def test_the_manifest_operating_point_wins_over_the_built_in_one(tmp_path):
    manifest = {"m2m100-pool": {"operating_point": [0.2, 0.6]}}
    dl, _ = _fake_hub(tmp_path, manifest=manifest)
    assert w.resolve(None, None, download=True, downloader=dl).operating_point == (0.2, 0.6)


def test_a_manifest_operating_point_that_is_not_a_pair_is_rejected(tmp_path):
    dl, _ = _fake_hub(tmp_path, manifest={"m2m100-pool": {"operating_point": [0.2]}})
    with pytest.raises(w.WeightsError, match="operating_point"):
        w.resolve(None, None, download=True, downloader=dl)


def test_an_unpublished_encoder_lists_the_published_ones(tmp_path):
    dl, _ = _fake_hub(tmp_path)
    with pytest.raises(w.WeightsError) as ei:
        w.resolve(None, None, download=True, downloader=dl, encoder="bge-m3")
    assert "bge-m3" in str(ei.value) and "m2m100-pool" in str(ei.value)


def test_offline_miss_names_the_encoder_and_the_missing_files(tmp_path):
    dl, _ = _fake_hub(tmp_path, absent=w.ENCODER_ASSETS["m2m100-pool"]["checkpoints"])
    with pytest.raises(w.WeightsError) as ei:
        w.resolve(None, None, download=False, downloader=dl)
    msg = str(ei.value)
    assert "m2m100-pool" in msg and "cogito-prose-ontology-m2mpool-s2.safetensors" in msg
    assert "--checkpoint" in msg and "huggingface" in msg.lower()


def test_resolve_does_not_swallow_programming_errors(tmp_path):
    def dl(repo_id, filename, local_files_only=False, subfolder=None):
        raise TypeError("bad downloader")
    with pytest.raises(TypeError):
        w.resolve(None, None, download=True, downloader=dl)


# --- manifest -------------------------------------------------------------------

def test_the_manifest_overrides_the_built_in_table(tmp_path):
    manifest = {"m2m100-pool": {"checkpoints": ["new.safetensors"], "vocab": "new.json",
                                "pooling": "new-pool.safetensors", "subfolder": "v2"}}
    dl, calls = _fake_hub(tmp_path, manifest=manifest)
    got = w.resolve(None, None, download=True, downloader=dl)
    assert [c.name for c in got.checkpoints] == ["new.safetensors"]
    assert got.vocab.name == "new.json" and got.pool.name == "new-pool.safetensors"
    assert {sub for _, _, sub, _ in calls if sub} == {"v2"}


def test_a_manifest_only_encoder_is_resolvable(tmp_path):
    manifest = {"future": {"checkpoints": ["f.safetensors"], "vocab": "f.json",
                           "subfolder": "future"}}
    dl, _ = _fake_hub(tmp_path, manifest=manifest)
    got = w.resolve(None, None, download=True, downloader=dl, encoder="future")
    assert got.encoder == "future" and got.pool is None and got.operating_point is None


def test_an_absent_manifest_falls_back_to_the_built_in_table(tmp_path):
    dl, _ = _fake_hub(tmp_path, manifest=None)
    assert w.resolve(None, None, download=True, downloader=dl).encoder == "m2m100-pool"


def test_a_corrupt_manifest_is_a_defect_not_an_absence(tmp_path):
    def dl(repo_id, filename, local_files_only=False, subfolder=None):
        p = tmp_path / filename
        p.write_text("{not json")
        return str(p)
    with pytest.raises(w.WeightsError, match="unreadable"):
        w.load_manifest(download=True, downloader=dl)


def test_a_manifest_entry_without_weights_is_rejected(tmp_path):
    dl, _ = _fake_hub(tmp_path, manifest={"m2m100-pool": {"checkpoints": []}})
    with pytest.raises(w.WeightsError, match="checkpoints"):
        w.resolve(None, None, download=True, downloader=dl)


def test_the_resolved_operating_point_comes_from_the_encoder_module(tmp_path):
    """One source of truth: with no manifest every published encoder resolves to the
    thresholds the encoder module names."""
    dl, _ = _fake_hub(tmp_path, manifest=None)
    for name in w.ENCODER_ASSETS:
        got = w.resolve(None, None, download=True, downloader=dl, encoder=name)
        assert got.operating_point == ENSEMBLE_OPERATING_POINT[name]


# --- spaCy ----------------------------------------------------------------------

def test_ensure_spacy_model_noop_when_present(monkeypatch):
    import spacy.util
    monkeypatch.setattr(spacy.util, "is_package", lambda name: True)
    w.ensure_spacy_model(download=False)


def test_ensure_spacy_model_raises_with_command_when_absent(monkeypatch):
    import spacy.util
    monkeypatch.setattr(spacy.util, "is_package", lambda name: False)
    with pytest.raises(w.WeightsError, match="python -m spacy download en_core_web_sm"):
        w.ensure_spacy_model(download=False)
