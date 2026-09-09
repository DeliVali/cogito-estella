"""GraphStore ingestion, embeddings and the `ask` route: ids, dedup, replacement, ranking."""
import io
import json
import re
import sys
import threading

import numpy as np
import pytest

from cogito_estella.mcp.store import DIVIDER, GraphStore
from cogito_estella.mcp.tokens import ntok

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


# -- ask: hybrid graph -> sentence retrieval -------------------------------------

ASK_DOC = ("The encoder maps text to a vector. "
           "The decoder maps text to a vector. "
           "Because the budget grew, the committee met early.")
ASK_TRIPLES = {
    "The encoder maps text to a vector.": [("encoder", "give", "text")],
    "The decoder maps text to a vector.": [("decoder", "give", "text")],
    "Because the budget grew, the committee met early.": [("budget", "do", "committee")],
}
Q = "What does the encoder do with text?"


def unit_vectors(texts, dim=8):
    """Deterministic normalized bag-of-words vectors: a stand-in for the SONAR encoder."""
    rows = []
    for t in texts:
        v = np.zeros(dim, dtype=np.float32)
        for w in re.findall(r"[a-z]+", t.lower()):
            v[sum(map(ord, w)) % dim] += 1.0
        n = float(np.linalg.norm(v))
        rows.append(v / n if n else v)
    return np.asarray(rows, dtype=np.float16)


def keyed_vectors(dim=4, keys=("alpha", "beta", "gamma", "delta")):
    """One-hot per keyword: exact control over cosine, unlike a bag of words."""
    def encode(texts, lang="eng_Latn"):
        rows = []
        for t in texts:
            v = np.zeros(dim, dtype=np.float32)
            for i, k in enumerate(keys):
                if k in t.lower():
                    v[i] = 1.0
            n = float(np.linalg.norm(v))
            rows.append(v / n if n else v)
        return np.asarray(rows, dtype=np.float16)
    return encode


@pytest.fixture
def ask_store(fake_extractor):
    st = GraphStore(extractor=fake_extractor(ASK_TRIPLES))
    st.ingest_text(ASK_DOC, "t")
    return st


@pytest.fixture
def emb_store(fake_extractor):
    def make(path=None):
        ex = fake_extractor(ASK_TRIPLES)
        ex.encode_batch = lambda texts, lang="eng_Latn": unit_vectors(texts)
        st = GraphStore(extractor=ex, path=path)
        st.ingest_text(ASK_DOC, "t")
        return st
    return make


def test_ask_layout_is_header_facts_divider_then_sentences(ask_store):
    lines = ask_store.ask(Q).split("\n")
    assert lines[0] == "entities: encoder, text · scorer=lexical"
    assert lines[1] == "encoder map text #0"
    assert lines[2] == DIVIDER
    assert lines[3] == "decoder give text #1"
    assert lines[4] == "--"
    assert lines[5] == 't s0: "The encoder maps text to a vector."'
    assert all(x.startswith("t s") for x in lines[5:])
    assert not any("committee" in x for x in lines[5:])   # zero lexical overlap, not served


def test_ask_orders_entities_by_ascending_degree(ask_store):
    assert len(ask_store.adj["text"]) > len(ask_store.adj["encoder"])
    assert ask_store.ask("text encoder").startswith("entities: encoder, text ")


def test_ask_keeps_at_most_three_entities(ask_store):
    head = ask_store.ask("encoder decoder budget committee text").split("\n")[0]
    assert head.startswith("entities: encoder, decoder, budget ·")


def test_ask_ignores_generic_question_words(fake_extractor):
    st = GraphStore(extractor=fake_extractor(
        {"The model maps text to a vector.": [("model", "give", "text")]}))
    st.ingest_text("The model maps text to a vector.", "t")
    assert "model" in st.adj
    assert st.ask("Which models does the paper use?").startswith("entities: (none) ·")


def test_question_entities_resolve_es_and_ies_plurals(fake_extractor):
    st = GraphStore(extractor=fake_extractor({}))
    st.adj.update({"policy": [0], "patch": [1], "class": [2]})
    assert st.question_entities("Which policies split the patches into classes?") == [
        "policy", "patch", "class"]


def test_ask_caps_the_facts_block_at_forty_percent_of_budget(ask_store):
    budget = 40
    out = ask_store.ask(Q, budget=budget)
    facts = out.split("\n--\n")[0]
    assert ntok(facts) <= int(budget * 0.4)
    assert "--" in out                                    # sentences still get their share


def test_ask_never_exceeds_the_budget(ask_store):
    for budget in (20, 40, 80, 200, 600):
        assert ntok(ask_store.ask(Q, budget=budget)) <= budget


def test_ask_without_entities_returns_sentences_only(ask_store):
    lines = ask_store.ask("What happened early?").split("\n")
    assert lines[0] == "entities: (none) · scorer=lexical"
    assert lines[1] == "--"
    assert DIVIDER not in lines
    assert 'committee met early' in lines[2]


def test_ask_reports_no_material_when_nothing_matches(ask_store):
    assert ask_store.ask("quantum chromodynamics") == "no material for 'quantum chromodynamics'"


def test_ask_on_an_empty_graph(fake_extractor):
    assert GraphStore(extractor=fake_extractor({})).ask("anything") == "no material for 'anything'"


# -- sentence embeddings --------------------------------------------------------------

def test_ingest_without_an_encoder_stores_no_embeddings(ask_store):
    assert ask_store.emb == {}
    assert "embedded_docs=0/1" in ask_store.stats()


def test_ingest_stores_normalized_float16_embeddings(emb_store):
    st = emb_store()
    assert set(st.emb) == {"t"}
    assert st.emb["t"].shape == (3, 8) and st.emb["t"].dtype == np.float16
    assert "embedded_docs=1/1" in st.stats()


def test_ask_uses_sonar_when_embeddings_are_present(emb_store):
    st = emb_store()
    lines = st.ask("The decoder maps text to a vector.").split("\n")
    assert "· scorer=sonar" in lines[0]
    first = lines[lines.index("--") + 1]
    assert first == 't s1: "The decoder maps text to a vector."'


def test_ask_header_names_sonar_alone_when_every_document_is_embedded(emb_store):
    assert emb_store().ask(Q).split("\n")[0].endswith("scorer=sonar")


def test_ask_keeps_a_lexical_hit_above_unrelated_embedded_sentences(fake_extractor):
    ex = fake_extractor(ASK_TRIPLES)
    ex.encode_batch = keyed_vectors()
    st = GraphStore(extractor=ex)
    st.ingest_text("The gamma route is unrelated. The delta route is unrelated too.", "embedded")
    st.ingest_text("The encoder maps text to a vector.", "plain")
    st.emb.pop("plain")                        # e.g. a sidecar block rejected at load
    lines = st.ask("alpha: to a vector?").split("\n")
    assert lines[0].endswith("scorer=sonar (1/2 docs)")     # the mix is visible
    assert lines[lines.index("--") + 1].startswith("plain s0:")


def test_ask_abstains_when_no_embedded_sentence_clears_the_cosine_floor(fake_extractor):
    ex = fake_extractor({})
    ex.encode_batch = keyed_vectors()
    st = GraphStore(extractor=ex)
    st.ingest_text("The gamma route is unrelated.", "embedded")
    assert st.emb["embedded"].shape[0] == 1
    assert st.ask("which route?") == "no material for 'which route?'"   # lexical would serve it
    assert "route" in st.ask("which route?", scorer="lexical")


def test_ask_scorer_lexical_ignores_the_embeddings(emb_store):
    assert "· scorer=lexical" in emb_store().ask(Q, scorer="lexical").split("\n")[0]


def test_ask_scorer_sonar_falls_back_with_a_note(ask_store):
    assert ask_store.ask(Q, scorer="sonar").startswith(
        "entities: encoder, text · scorer=lexical (sonar unavailable)")


def test_ask_treats_an_unknown_scorer_as_auto(emb_store):
    assert "· scorer=sonar" in emb_store().ask(Q, scorer="SONAR-v2")


def test_ask_falls_back_to_lexical_when_the_question_cannot_be_encoded(emb_store):
    st = emb_store()

    def boom(texts, lang="eng_Latn"):
        raise RuntimeError("no weights")

    st._ex.encode_batch = boom
    assert "· scorer=lexical (sonar unavailable)" in st.ask(Q, scorer="sonar")


def test_ingest_tolerates_an_encoder_that_raises(fake_extractor):
    ex = fake_extractor(ASK_TRIPLES)

    def boom(texts, lang="eng_Latn"):
        raise RuntimeError("cuda is busy")

    ex.encode_batch = boom
    st = GraphStore(extractor=ex)
    assert st.ingest_text(ASK_DOC, "t").triples == 3
    assert st.emb == {}


def test_embeddings_survive_save_and_load(tmp_path, fake_extractor, emb_store):
    path = tmp_path / ".cogito" / "graph.json"
    st = emb_store(path)
    side = path.with_suffix(".emb.npz")
    assert side.exists()
    assert list(path.parent.glob("*.tmp")) == []

    back = GraphStore(extractor=fake_extractor(ASK_TRIPLES), path=path)
    back.load()
    assert set(back.emb) == {"t"}
    assert back.emb["t"].dtype == np.float16
    assert np.array_equal(back.emb["t"], st.emb["t"])


def test_drop_source_drops_its_embeddings_and_the_sidecar(tmp_path, emb_store):
    path = tmp_path / "graph.json"
    st = emb_store(path)
    st.drop_source("t")
    st.save()
    assert st.emb == {}
    assert not path.with_suffix(".emb.npz").exists()


def test_missing_sidecar_leaves_the_store_lexical(tmp_path, fake_extractor, emb_store):
    path = tmp_path / "graph.json"
    emb_store(path)
    path.with_suffix(".emb.npz").unlink()
    back = GraphStore(extractor=fake_extractor(ASK_TRIPLES), path=path)
    back.load()
    assert back.emb == {}
    assert "· scorer=lexical" in back.ask(Q)


def test_corrupt_sidecar_is_tolerated(tmp_path, fake_extractor, emb_store):
    path = tmp_path / "graph.json"
    emb_store(path)
    path.with_suffix(".emb.npz").write_bytes(b"not an npz file")
    back = GraphStore(extractor=fake_extractor(ASK_TRIPLES), path=path)
    back.load()
    assert back.emb == {}
    assert len(back.edges) == 3


def test_sidecar_with_a_stale_sentence_count_is_ignored(tmp_path, fake_extractor, emb_store):
    path = tmp_path / "graph.json"
    emb_store(path)
    side = path.with_suffix(".emb.npz")
    with side.open("wb") as fh:
        np.savez(fh, sources=np.array(["t"]), counts=np.array([2]),
                 emb=np.zeros((2, 8), dtype=np.float16))
    back = GraphStore(extractor=fake_extractor(ASK_TRIPLES), path=path)
    back.load()
    assert back.emb == {}


def test_sidecar_with_a_one_dimensional_matrix_is_rejected(tmp_path, fake_extractor, emb_store):
    path = tmp_path / "graph.json"
    st = emb_store(path)
    sha = st.docs["t"]["sha256"]
    with path.with_suffix(".emb.npz").open("wb") as fh:
        np.savez(fh, sources=np.array(["t"]), counts=np.array([3]), shas=np.array([sha]),
                 emb=np.zeros(3, dtype=np.float16))    # right row count, wrong rank
    back = GraphStore(extractor=fake_extractor(ASK_TRIPLES), path=path)
    back.load()
    assert back.emb == {}
    assert "· scorer=lexical" in back.ask(Q)           # no IndexError out of _embedding_matrix


def test_sidecar_with_a_non_float_matrix_is_rejected(tmp_path, fake_extractor, emb_store):
    path = tmp_path / "graph.json"
    st = emb_store(path)
    with path.with_suffix(".emb.npz").open("wb") as fh:
        np.savez(fh, sources=np.array(["t"]), counts=np.array([3]),
                 shas=np.array([st.docs["t"]["sha256"]]), emb=np.zeros((3, 8), dtype=np.int16))
    back = GraphStore(extractor=fake_extractor(ASK_TRIPLES), path=path)
    back.load()
    assert back.emb == {}


def test_save_survives_a_failing_sidecar_write(tmp_path, emb_store, monkeypatch):
    path = tmp_path / "graph.json"
    st = emb_store(path)

    def boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(np, "savez", boom)
    st.ingest_text("The encoder maps text to a byte.", "u")     # ingest -> save -> sidecar
    assert json.loads(path.read_text(encoding="utf-8"))["docs"].keys() >= {"t", "u"}
    assert list(path.parent.glob("*.tmp")) == []


def test_a_rejected_sidecar_is_kept_for_the_next_reader(tmp_path, fake_extractor, emb_store):
    path = tmp_path / "graph.json"
    st = emb_store(path)
    side = path.with_suffix(".emb.npz")
    with side.open("wb") as fh:                    # pre-0.15.0 layout: every block rejected
        np.savez(fh, sources=np.array(["t"]), counts=np.array([3]), emb=st.emb["t"])
    before = side.read_bytes()
    back = GraphStore(extractor=fake_extractor(ASK_TRIPLES), path=path)
    back.load()
    assert back.emb == {} and set(back.docs) == {"t"}
    back.save()
    assert side.read_bytes() == before             # a save must not destroy what it cannot read


def test_concurrent_ingests_keep_embeddings_aligned(tmp_path, fake_extractor):
    n = 4
    triples = {f"The encoder maps text to vector {i}.": [("encoder", "give", "text")]
               for i in range(n)}
    ex = fake_extractor(triples)
    ex.encode_batch = lambda texts, lang="eng_Latn": unit_vectors(texts)
    st = GraphStore(extractor=ex, path=tmp_path / "graph.json")
    threads = [threading.Thread(target=st.ingest_text,
                                args=(f"The encoder maps text to vector {i}.", f"src{i}"))
               for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(st.emb) == n
    assert all(st.emb[s].shape[0] == st.docs[s]["sentences"] for s in st.docs)


def test_fit_skips_an_oversize_line_and_keeps_the_shorter_ones_behind_it():
    lines = ["hub relate node #" + ",".join(str(i) for i in range(400)), "a b c #1", "d e f #2"]
    assert GraphStore._fit(["head"], lines, 40) == ["a b c #1", "d e f #2"]


def test_fit_stops_after_a_run_of_oversize_lines():
    lines = ["x " * 300] * 40 + ["a b c #1"]
    assert GraphStore._fit(["head"], lines, 40) == []


def test_ask_caps_the_id_list_of_one_grouped_fact_line(fake_extractor):
    sents = [f"The encoder maps text to a vector number {i}." for i in range(12)]
    st = GraphStore(extractor=fake_extractor({t: [("encoder", "give", "text")] for t in sents}))
    st.ingest_text(" ".join(sents), "t")
    line = next(x for x in st.ask("encoder", budget=4000).split("\n") if x.startswith("encoder "))
    assert line.endswith(" …") and line.count(",") == 7
    assert "11" not in line                       # the ellipsis carries no readable edge id


def test_ask_builds_the_lexical_scorer_once_per_corpus(ask_store, monkeypatch):
    import cogito_estella.mcp.store as store_mod
    built = []
    real = store_mod.LexicalScorer
    monkeypatch.setattr(store_mod, "LexicalScorer",
                        lambda sentences: built.append(len(sentences)) or real(sentences))
    ask_store.ask(Q)
    ask_store.ask("what about the committee?")
    assert built == [3]                            # one IDF pass, not one per question
    ask_store.ingest_text("The decoder is new here.", "u")
    ask_store.ask(Q)
    assert built == [3, 4]                         # a new document invalidates it


def test_ask_releases_the_lock_before_encoding_the_question(fake_extractor):
    ex = fake_extractor(ASK_TRIPLES)
    encoding, resume = threading.Event(), threading.Event()

    def slow_encode(texts, lang="eng_Latn"):
        if list(texts) == [Q]:                     # the question, never a document
            encoding.set()
            resume.wait(5)
        return unit_vectors(texts)

    ex.encode_batch = slow_encode
    st = GraphStore(extractor=ex)
    st.ingest_text(ASK_DOC, "t")
    reader = threading.Thread(target=st.ask, args=(Q,))
    reader.start()
    assert encoding.wait(5)
    writer = threading.Thread(target=st.ingest_text, args=("The decoder is elsewhere.", "u"))
    writer.start()
    writer.join(5)
    blocked = writer.is_alive()
    resume.set()
    reader.join(5)
    writer.join(5)
    assert not blocked                             # a model load must not serialize the store


def test_a_redirect_in_one_thread_never_hands_the_wire_to_another(fake_extractor, monkeypatch):
    """Overlapping stdout redirects: the block still running must not print on the MCP
    wire when the other one leaves, and the wire must come back at the end."""
    marker = "encoder load noise"
    reader_in, writer_in, reader_out = threading.Event(), threading.Event(), threading.Event()
    ex = fake_extractor(ASK_TRIPLES)

    def encode(texts, lang="eng_Latn"):
        if list(texts) == [Q]:                     # inside `ask`'s redirect
            reader_in.set()
            writer_in.wait(5)
        return unit_vectors(texts)

    def extract(texts, doc_offsets=None, candidates=None, lang="eng_Latn"):
        writer_in.set()                            # inside `ingest_text`'s redirect
        reader_out.wait(5)
        print(marker)
        return [[] for _ in texts]

    ex.encode_batch = encode
    st = GraphStore(extractor=ex)
    st.ingest_text(ASK_DOC, "t")
    ex.extract_batch_with_provenance = extract

    wire = io.StringIO()                           # stands in for the MCP stdio wire
    monkeypatch.setattr(sys, "stdout", wire)
    reader = threading.Thread(target=st.ask, args=(Q,))
    reader.start()
    assert reader_in.wait(5)
    writer = threading.Thread(target=st.ingest_text, args=("The decoder is elsewhere.", "u"))
    writer.start()
    assert writer_in.wait(5)
    reader.join(5)                                 # leaves while the writer is still inside
    reader_out.set()
    writer.join(5)
    assert not reader.is_alive() and not writer.is_alive()
    assert wire.getvalue() == ""                   # no load noise on the JSON-RPC framing
    assert sys.stdout is wire                      # and the wire is back for the next reply


def test_a_first_ask_from_two_threads_loads_the_extractor_once(tmp_path, fake_extractor):
    path = tmp_path / "graph.json"
    ex = fake_extractor(ASK_TRIPLES)
    ex.encode_batch = lambda texts, lang="eng_Latn": unit_vectors(texts)
    GraphStore(extractor=ex, path=path).ingest_text(ASK_DOC, "t")
    calls, loading, release = [], threading.Event(), threading.Event()

    def factory():
        calls.append(1)
        loading.set()
        release.wait(5)
        return ex

    st = GraphStore(extractor_factory=factory, path=path)
    st.load()
    first = threading.Thread(target=st.ask, args=(Q,))
    first.start()
    assert loading.wait(5)
    second = threading.Thread(target=st.ask, args=(Q,))
    second.start()
    for _ in range(100):                           # let a second factory call surface, if any
        if len(calls) > 1:
            break
        second.join(0.01)
    release.set()
    first.join(5)
    second.join(5)
    assert calls == [1]                            # one encoder load, not one per caller


def test_ask_with_a_budget_too_small_for_any_line_still_names_what_it_found(ask_store):
    assert ask_store.ask(Q, budget=12) == "entities: encoder, text · scorer=lexical"


def test_ask_skips_a_sentence_too_long_for_the_budget_and_serves_a_shorter_one(fake_extractor):
    long_sent = "The encoder maps text to a vector " + "alpha beta gamma delta " * 160 + "end."
    short_sent = "The encoder is fast."
    st = GraphStore(extractor=fake_extractor({}))
    st.ingest_text(f"{long_sent} {short_sent}", "t")
    assert ntok(long_sent) > 600
    out = st.ask("encoder vector", budget=600)
    assert "--" in out
    assert f'{short_sent}"' in out
    assert "alpha beta" not in out
    assert ntok(out) <= 600


def test_sidecar_without_hashes_is_ignored(tmp_path, fake_extractor, emb_store):
    path = tmp_path / "graph.json"
    st = emb_store(path)
    with path.with_suffix(".emb.npz").open("wb") as fh:      # pre-0.15.0 layout
        np.savez(fh, sources=np.array(["t"]), counts=np.array([3]), emb=st.emb["t"])
    back = GraphStore(extractor=fake_extractor(ASK_TRIPLES), path=path)
    back.load()
    assert back.emb == {}
    assert len(back.edges) == 3


def test_sidecar_is_dropped_when_the_document_text_changed_under_it(
        tmp_path, fake_extractor, emb_store, monkeypatch):
    path = tmp_path / "graph.json"
    st = emb_store(path)
    other = ("The decoder writes text into a vector. "
             "The encoder reads text from a vector. "
             "Because the budget shrank, the committee met late.")
    monkeypatch.setattr(st, "_save_embeddings", lambda: None)   # interrupted sidecar write
    st.ingest_text(other, "t")
    assert st.docs["t"]["sentences"] == 3                       # same count, different text

    back = GraphStore(extractor=fake_extractor(ASK_TRIPLES), path=path)
    back.load()
    assert back.emb == {}
    assert "· scorer=lexical" in back.ask(Q)
