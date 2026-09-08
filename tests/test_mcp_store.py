"""GraphStore ingestion: stable ids, dedup by hash, replacement, adjacency."""
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
