"""GraphStore ingestion: stable ids, dedup by hash, replacement, adjacency."""
import json
import threading

import pytest

from cogito_estella.mcp.store import GraphStore

DOC = ("The generated concepts are decoded by SONAR. "
       "The encoder maps text to a vector. "
       "Because the budget grew, the committee met early.")
TRIPLES = {
    "The generated concepts are decoded by SONAR.": [("concept", "improve", "sonar")],
    "The encoder maps text to a vector.": [("encoder", "give", "text")],
    "Because the budget grew, the committee met early.": [("budget", "do", "committee")],
}


@pytest.fixture
def store(fake_extractor):
    return GraphStore(extractor=fake_extractor(TRIPLES))


def test_ingest_builds_records_without_sentence_field(store):
    r = store.ingest_text(DOC, "t")
    assert r.status == "ingested" and r.sentences == 3 and r.triples == 3
    e = store.edges[0]
    assert (e["s"], e["r"], e["o"]) == ("sonar", "decode", "concept")
    assert e["r_lex"] == "decode" and e["r_class"] == "improve" and e["swapped"] is True
    assert "sentence" not in e and store.sentence_of(e).startswith("The generated concepts")
    assert e["source"] == "t" and e["sent_idx"] == 0
    assert set(store.adj) == {"sonar", "concept", "encoder", "text", "budget", "committee"}


def test_ingest_line_format(store):
    line = store.ingest_text(DOC, "t").line()
    for key in ("source=t", "status=ingested", "sentences=3", "triples=3", "raw_tokens=",
                "graph_tokens=", "compression=", "seconds="):
        assert key in line


def test_same_content_is_unchanged(store):
    store.ingest_text(DOC, "t")
    r = store.ingest_text(DOC, "t")
    assert r.status == "unchanged" and r.triples == 0 and len(store.edges) == 3


def test_changed_content_replaces_and_retires_ids(store):
    store.ingest_text(DOC, "t")
    r = store.ingest_text("The encoder maps text to a vector.", "t")
    assert r.status == "replaced" and r.triples == 1
    assert set(store.edges) == {3}                 # ids 0-2 retired, never reused
    assert store.next_id == 4 and store.adj["encoder"] == [3] and "sonar" not in store.adj


def test_second_source_gets_fresh_ids(store):
    store.ingest_text(DOC, "a")
    store.ingest_text("The encoder maps text to a vector.", "b")
    assert sorted(store.edges) == [0, 1, 2, 3] and store.edges[3]["source"] == "b"


def test_split_keeps_absolute_offsets(store):
    sents = store.split(DOC)
    assert [s for s, _ in sents][1] == "The encoder maps text to a vector."
    assert DOC[sents[1][1]:].startswith("The encoder")


def test_ledger_counts_raw_tokens(store):
    store.ingest_text(DOC, "t")
    assert store.ledger.raw_tokens >= 15


# -- finding 1: concurrent ingests must not corrupt ids or the persisted file -----------

def test_concurrent_ingests_keep_ids_unique_and_file_valid(tmp_path, fake_extractor):
    n = 4
    triples = {f"The encoder maps text to vector {i}.": [("encoder", "give", "text")]
               for i in range(n)}
    path = tmp_path / ".cogito" / "graph.json"
    st = GraphStore(extractor=fake_extractor(triples), path=path)
    errors = []

    def worker(i):
        try:
            st.ingest_text(f"The encoder maps text to vector {i}.", f"src{i}")
        except Exception as exc:                       # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(st.edges) == n
    assert len(set(st.edges)) == len(st.edges)         # ids unique

    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["edges"]) == n

    reloaded = GraphStore(extractor=fake_extractor({}), path=path)
    reloaded.load()
    assert len(reloaded.edges) == n

    assert list(path.parent.glob("*.tmp")) == []


# -- finding 6: raw_tokens is derived from the currently stored documents ---------------

def test_raw_tokens_reflects_current_docs_after_replace(store):
    store.ingest_text(DOC, "t")
    store.ingest_text("The encoder maps text to a vector.", "t")
    expected = sum(d["tokens"] for d in store.docs.values())
    assert store.ledger.raw_tokens == expected
    assert f"raw_tokens_ingested={expected}" in store.stats()


# -- finding 8: next_id self-heals if the persisted value undercounts ------------------

def test_load_repairs_next_id_lower_than_max_edge_id(tmp_path, fake_extractor):
    path = tmp_path / "graph.json"
    st = GraphStore(extractor=fake_extractor(TRIPLES), path=path)
    st.ingest_text(DOC, "t")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["next_id"] = 0
    path.write_text(json.dumps(data), encoding="utf-8")

    reloaded = GraphStore(extractor=fake_extractor({}), path=path)
    reloaded.load()
    assert reloaded.next_id == max(reloaded.edges) + 1


# -- finding 9: neighborhood() must not insert empty adjacency entries -----------------

def test_neighborhood_of_unknown_entity_does_not_pollute_adjacency(fake_extractor):
    empty = GraphStore(extractor=fake_extractor({}))
    assert empty.neighborhood("ghost") == []
    assert empty.entities() == "(empty graph)"


# -- finding 17: ingest_path saves once for a whole directory --------------------------

def test_ingest_path_saves_once_for_a_multi_file_directory(tmp_path, fake_extractor, monkeypatch):
    (tmp_path / "a.txt").write_text("The encoder maps text to a vector.")
    (tmp_path / "b.txt").write_text("Because the budget grew, the committee met early.")
    (tmp_path / "c.txt").write_text("The generated concepts are decoded by SONAR.")
    st = GraphStore(extractor=fake_extractor(TRIPLES), path=tmp_path / "out" / "graph.json")
    calls = []
    monkeypatch.setattr(st, "save", lambda: calls.append(1))
    st.ingest_path(tmp_path)
    assert calls == [1]


# -- finding 20: nested directory with mixed suffixes ingests supported docs in order --

def test_ingest_path_nested_directory_sorted_order(tmp_path, fake_extractor):
    (tmp_path / "a.txt").write_text("The encoder maps text to a vector.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("The encoder maps text to a vector.")
    (tmp_path / "sub" / "c.py").write_text("print(1)")
    st = GraphStore(extractor=fake_extractor(
        {"The encoder maps text to a vector.": [("encoder", "give", "text")]}))
    out = st.ingest_path(tmp_path)
    lines = [o if isinstance(o, str) else o.line() for o in out]
    assert len(lines) == 2
    assert lines[0].startswith("source=a.txt")
    assert lines[1].startswith("source=sub/b.md")
