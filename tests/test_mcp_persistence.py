"""graph.json round trip: atomic write, reload, corrupt-file recovery."""
import json
import re

from cogito_estella.mcp.store import GraphStore

DOC = "The generated concepts are decoded by SONAR. The encoder maps text to a vector."
TRIPLES = {
    "The generated concepts are decoded by SONAR.": [("concept", "improve", "sonar")],
    "The encoder maps text to a vector.": [("encoder", "give", "text")],
}


def test_save_and_load_reproduce_query_output(tmp_path, fake_extractor):
    path = tmp_path / ".cogito" / "graph.json"
    a = GraphStore(extractor=fake_extractor(TRIPLES), path=path)
    a.ingest_text(DOC, "t")
    assert path.exists() and not path.with_suffix(".json.tmp").exists()
    b = GraphStore(extractor=fake_extractor({}), path=path)
    b.load()
    assert b.query("sonar") == a.query("sonar")
    assert b.provenance([0]) == a.provenance([0])
    assert b.next_id == a.next_id and b.ledger.raw_tokens == a.ledger.raw_tokens
    assert b.persisted_at is not None and "graph_file=" in b.stats()


def test_file_schema(tmp_path, fake_extractor):
    path = tmp_path / "graph.json"
    st = GraphStore(extractor=fake_extractor(TRIPLES), path=path)
    st.ingest_text(DOC, "t")
    data = json.loads(path.read_text())
    assert data["version"] == 1 and set(data) == {"version", "next_id", "raw_tokens", "docs", "edges"}
    assert "sentence" not in data["edges"]["0"] and data["docs"]["t"]["sha256"]


def test_unchanged_after_reload(tmp_path, fake_extractor):
    path = tmp_path / "graph.json"
    GraphStore(extractor=fake_extractor(TRIPLES), path=path).ingest_text(DOC, "t")
    b = GraphStore(extractor=fake_extractor(TRIPLES), path=path)
    b.load()
    assert b.ingest_text(DOC, "t").status == "unchanged"


def test_corrupt_file_is_renamed_and_store_starts_empty(tmp_path, fake_extractor, capsys):
    path = tmp_path / "graph.json"
    path.write_text("{not json")
    st = GraphStore(extractor=fake_extractor({}), path=path)
    st.load()
    assert not path.exists() and list(tmp_path.glob("graph.json.corrupt-*"))
    assert st.edges == {} and "corrupt" in capsys.readouterr().err


def test_non_object_json_is_treated_as_corrupt(tmp_path, fake_extractor, capsys):
    path = tmp_path / "graph.json"
    path.write_text("[]")
    st = GraphStore(extractor=fake_extractor({}), path=path)
    st.load()
    assert not path.exists() and list(tmp_path.glob("graph.json.corrupt-*"))
    assert st.edges == {} and "corrupt" in capsys.readouterr().err


def test_no_path_means_no_file(tmp_path, fake_extractor):
    st = GraphStore(extractor=fake_extractor(TRIPLES))
    st.ingest_text(DOC, "t")
    assert not list(tmp_path.iterdir())


# -- finding 2: persistence round-trips non-ASCII text through UTF-8 -------------------

def test_graph_file_is_utf8(tmp_path, fake_extractor):
    path = tmp_path / "graph.json"
    text = "The encoder maps text to a vector — café edition."
    triples = {text: [("encoder", "give", "text")]}
    a = GraphStore(extractor=fake_extractor(triples), path=path)
    a.ingest_text(text, "t")
    parsed = json.loads(path.read_bytes().decode("utf-8"))
    assert parsed["docs"]["t"]["sents"][0][0] == text
    b = GraphStore(extractor=fake_extractor({}), path=path)
    b.load()
    assert b.query("encoder") == a.query("encoder")


# -- finding 3: load() recovers from malformed sections without half-loading ------------

def test_edges_as_list_is_treated_as_corrupt(tmp_path, fake_extractor, capsys):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"version": 1, "next_id": 0, "raw_tokens": 0,
                                "docs": {}, "edges": []}), encoding="utf-8")
    st = GraphStore(extractor=fake_extractor({}), path=path)
    st.load()
    assert not path.exists() and list(tmp_path.glob("graph.json.corrupt-*"))
    assert st.edges == {} and "corrupt" in capsys.readouterr().err


def test_edge_missing_subject_is_treated_as_corrupt(tmp_path, fake_extractor, capsys):
    path = tmp_path / "graph.json"
    bad_edge = {"r": "x", "o": "y", "r_lex": None, "source": "t", "sent_idx": 0,
               "s_span": None, "o_span": None}
    path.write_text(json.dumps({
        "version": 1, "next_id": 1, "raw_tokens": 0,
        "docs": {"t": {"sha256": "x", "tokens": 1, "sentences": 1, "sents": [["s", 0]]}},
        "edges": {"0": bad_edge},
    }), encoding="utf-8")
    st = GraphStore(extractor=fake_extractor({}), path=path)
    st.load()
    assert not path.exists() and list(tmp_path.glob("graph.json.corrupt-*"))
    assert st.edges == {} and "corrupt" in capsys.readouterr().err


# -- finding 18: version mismatch names both versions in the stderr message ------------

def test_version_mismatch_message_names_both_versions(tmp_path, fake_extractor, capsys):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"version": 2, "next_id": 0, "raw_tokens": 0,
                                "docs": {}, "edges": {}}), encoding="utf-8")
    st = GraphStore(extractor=fake_extractor({}), path=path)
    st.load()
    err = capsys.readouterr().err
    assert "unsupported graph file version 2" in err and "version 1" in err
    assert not path.exists() and list(tmp_path.glob("graph.json.corrupt-*"))


# -- finding 20: reload reports the same ledger and edge counts ------------------------

def test_stats_after_reload_reports_same_raw_tokens_and_edges(tmp_path, fake_extractor):
    path = tmp_path / "graph.json"
    a = GraphStore(extractor=fake_extractor(TRIPLES), path=path)
    a.ingest_text(DOC, "t")
    b = GraphStore(extractor=fake_extractor({}), path=path)
    b.load()
    a_raw = re.search(r"raw_tokens_ingested=(\d+)", a.stats()).group(1)
    b_raw = re.search(r"raw_tokens_ingested=(\d+)", b.stats()).group(1)
    a_edges = re.search(r"edges=(\d+)", a.stats()).group(1)
    b_edges = re.search(r"edges=(\d+)", b.stats()).group(1)
    assert a_raw == b_raw and a_edges == b_edges
