import numpy as np

import cogito_estella.mcp.vectors as vec
from cogito_estella.mcp.server import parse_args
from cogito_estella.mcp.store import GraphStore, TextOnlyExtractor

DOC = ("The castle was built in 1205 by the duke. The river floods every spring. "
       "Bread is baked in the village oven. " + " ".join(
           f"Market stall {i} sells wool and copper pots." for i in range(12)))
TOPIC = {"castle": 0, "castillo": 0, "river": 1, "río": 1, "bread": 2, "pan": 2}


class FakeVectors(TextOnlyExtractor):
    """Vectors by topic word, so a Spanish question lands on its English sentence."""
    vectors = True
    encoder_name = "fake-vectors"
    dim = 3

    def _encode(self, texts):
        out = np.full((len(texts), 3), 0.01, dtype=np.float32)
        for row, t in zip(out, texts):
            for w, k in TOPIC.items():
                if w in t.lower():
                    row[k] = 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    encode_batch = encode_query = _encode


def _store(ex):
    s = GraphStore(extractor=ex)
    s.ingest_text(DOC, "doc.txt")
    return s


def _first(reply):
    return reply.split("\n")[1]


def test_a_question_in_another_language_finds_its_sentence():
    reply = _store(FakeVectors()).ask_compact("¿Cuándo se construyó el castillo?", 40, "learned")
    assert "1205" in _first(reply)


def test_without_vectors_the_same_question_does_not():
    reply = _store(TextOnlyExtractor()).ask_compact("¿Cuándo se construyó el castillo?", 40,
                                                    "learned")
    assert "1205" not in reply


def test_same_language_questions_still_rank_lexically_first():
    reply = _store(FakeVectors()).ask_compact("Where is bread baked?", 40, "learned")
    assert "oven" in _first(reply)


def test_an_index_from_before_vectors_gets_them_on_the_next_ingest(tmp_path):
    path = tmp_path / "graph.json"
    old = GraphStore(extractor=TextOnlyExtractor(), path=path)
    old.ingest_text(DOC, "doc.txt")
    new = GraphStore(extractor=FakeVectors(), path=path)
    new.load()
    assert new.ingest_text(DOC, "doc.txt").status == "unchanged"
    assert "doc.txt" in new.emb


def test_a_model_that_cannot_load_falls_back_to_lite(monkeypatch, capsys):
    def broken(download=True):
        raise OSError("offline")
    monkeypatch.setattr(vec, "VectorExtractor", broken)
    ex = vec.lite_factory(True)()
    assert type(ex) is TextOnlyExtractor
    assert "without them" in capsys.readouterr().err


def test_vectors_are_turned_off_by_flag_or_environment(monkeypatch):
    assert vec.lite_factory(False) is TextOnlyExtractor
    monkeypatch.setenv("COGITO_VECTORS", "0")
    assert vec.lite_factory() is TextOnlyExtractor
    monkeypatch.setenv("COGITO_VECTORS", "1")
    assert vec.lite_factory() is not TextOnlyExtractor and vec.lite_factory().lite
    assert parse_args(["--no-vectors", "docs"]).vectors is False
    assert parse_args(["docs"]).vectors is None
