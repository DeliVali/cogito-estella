"""Lite to full without a restart: the graph model loads once and stays."""
from __future__ import annotations

import numpy as np

from cogito_estella.mcp.server import Tools
from cogito_estella.mcp.store import GraphStore, TextOnlyExtractor

DOC = "The generated concepts are decoded by SONAR. The encoder maps text to a vector."


class _Full:
    lite = False
    encoder_name = "fake"
    dim = 4

    def __init__(self):
        self.extracted = 0

    def _scanner(self):
        return TextOnlyExtractor()._scanner()

    def extract_batch_with_provenance(self, texts, doc_offsets=None):
        self.extracted += len(texts)
        return [[{"s": "sonar", "r": "decodes", "o": "concept", "r_lex": False,
                  "span": [0, 1]}] for _ in texts]

    def encode_batch(self, texts):
        return np.ones((len(texts), self.dim), dtype=np.float32) / 2


def _store(tmp_path, loads, fail=False):
    def full():
        loads.append(1)
        if fail:
            raise ImportError("No module named 'torch'")
        return _Full()
    return GraphStore(extractor_factory=TextOnlyExtractor, full_factory=full,
                      path=tmp_path / "graph.json")


def test_deep_adds_facts_to_lite_documents_and_loads_once(tmp_path, nlp):
    loads = []
    store = _store(tmp_path, loads)
    store.ingest_text(DOC, "a.md")
    assert store.is_lite and not store.edges and store.docs["a.md"]["lite"]
    assert store.use_full() is None
    assert not store.is_lite and store.edges and "lite" not in store.docs["a.md"]
    assert "a.md" in store.emb
    store.use_full()
    store.ingest_text("Another sentence here.", "b.md")
    assert loads == [1]                                  # resident: no second load
    assert "lite" not in store.docs["b.md"]              # new documents use the full model


def test_deep_keeps_the_index_across_a_restart(tmp_path, nlp):
    store = _store(tmp_path, [])
    store.ingest_text(DOC, "a.md")
    store.use_full()
    again = _store(tmp_path, [])
    again.load()
    assert again.edges and "lite" not in again.docs["a.md"]


def test_deep_reports_a_missing_model_instead_of_stopping_the_server(tmp_path, nlp):
    store = _store(tmp_path, [], fail=True)
    store.ingest_text(DOC, "a.md")
    assert "could not be loaded" in store.use_full()
    assert store.is_lite and store.docs["a.md"]["lite"]
    bare = GraphStore(extractor_factory=TextOnlyExtractor)
    assert "cogito-estella[full]" in bare.use_full()


def test_ask_deep_answers_and_switches_the_store(tmp_path, nlp):
    loads = []
    store = _store(tmp_path, loads)
    store.ingest_text(DOC, "a.md")
    t = Tools(store)
    assert "SONAR" in t.ask("What decodes the concepts?")
    assert loads == []                                   # a plain ask never loads it
    assert "SONAR" in t.ask("What decodes the concepts?", deep=True)
    assert loads == [1] and not store.is_lite
    t.ask("What maps text?", deep=True)
    assert loads == [1]


def test_deep_reads_files_again_so_the_full_scanner_splits_them(tmp_path, nlp):
    class Scanner(_Full):
        def _scanner(self):                       # a splitter that differs from the lite one
            class Doc:
                def __init__(self, text):
                    self.sents = [type("S", (), {"text": t})() for t in text.split(" | ")]
            return Doc
    path = tmp_path / "a.txt"
    path.write_text("one thing | two things | three things")
    loads = []
    store = GraphStore(extractor_factory=TextOnlyExtractor,
                       full_factory=lambda: (loads.append(1), Scanner())[1],
                       path=tmp_path / "g.json")
    store.ingest_path(path)
    assert store.docs[str(path)]["sentences"] == 1        # lite sentencizer: no full stop
    store.use_full()
    assert store.docs[str(path)]["sentences"] == 3        # the full scanner's split
    assert "lite" not in store.docs[str(path)]


def test_rest_routes_answer_like_the_tools(tmp_path, nlp):
    from starlette.testclient import TestClient

    from cogito_estella.mcp.server import build_server, register_rest
    store = GraphStore(extractor_factory=TextOnlyExtractor)
    store.ingest_text(DOC, "a.md")
    t = Tools(store, "compact")
    server = build_server(store, t)
    register_rest(server, t)
    client = TestClient(server.streamable_http_app())
    q = "What decodes the concepts?"
    assert client.post("/ask", json={"question": q}).text == t.ask(q)
    assert client.post("/context", json={"question": q}).text == t.context(q)
    assert client.post("/ask", json={}).status_code == 400
    assert client.get("/health").text == "ok"


def test_http_bind_spec():
    import pytest

    from cogito_estella.mcp.server import parse_bind
    assert parse_bind("8765") == ("127.0.0.1", 8765)
    assert parse_bind("0.0.0.0:80") == ("0.0.0.0", 80)
    with pytest.raises(ValueError):
        parse_bind("host:notaport")
