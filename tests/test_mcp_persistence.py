"""graph.json round trip: atomic write, reload, corrupt-file recovery."""
import json

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
