"""Weight resolution: explicit paths, Hugging Face defaults, offline misses."""
import pytest

from cogito_estella.mcp import weights as w


def test_explicit_paths_are_used_and_must_exist(tmp_path):
    ck, vocab = tmp_path / "a.pt", tmp_path / "v.json"
    ck.write_bytes(b"x"); vocab.write_text("{}")
    assert w.resolve([str(ck)], str(vocab), download=False) == ([ck], vocab)
    with pytest.raises(w.WeightsError, match="missing"):
        w.resolve([str(tmp_path / "nope.pt")], str(vocab), download=False)


def test_defaults_go_through_the_downloader(tmp_path):
    calls = []

    def dl(repo_id, filename, local_files_only=False):
        calls.append((repo_id, filename, local_files_only))
        p = tmp_path / filename
        p.write_bytes(b"x")
        return str(p)
    cks, vocab = w.resolve(None, None, download=True, downloader=dl)
    assert [c.name for c in cks] == list(w.DEFAULT_CHECKPOINTS) and vocab.name == w.DEFAULT_VOCAB
    assert all(r == w.HF_REPO and not lfo for r, _, lfo in calls) and len(calls) == 4


def test_offline_miss_lists_missing_files():
    def dl(repo_id, filename, local_files_only=False):
        raise FileNotFoundError(filename)
    with pytest.raises(w.WeightsError) as ei:
        w.resolve(None, None, download=False, downloader=dl)
    msg = str(ei.value)
    assert "cogito-prose-ontology-s2.pt" in msg and "vocab-onto.json" in msg
    assert "--checkpoint" in msg and "huggingface" in msg.lower()


def test_partial_explicit_is_rejected(tmp_path):
    with pytest.raises(w.WeightsError, match="both"):
        w.resolve([str(tmp_path / "a.pt")], None, download=False)


def test_ensure_spacy_model_noop_when_present(monkeypatch):
    import spacy.util
    monkeypatch.setattr(spacy.util, "is_package", lambda name: True)
    w.ensure_spacy_model(download=False)


def test_ensure_spacy_model_raises_with_command_when_absent(monkeypatch):
    import spacy.util
    monkeypatch.setattr(spacy.util, "is_package", lambda name: False)
    with pytest.raises(w.WeightsError, match="python -m spacy download en_core_web_sm"):
        w.ensure_spacy_model(download=False)
