"""Tool renderers: what the agent pays for."""
import pytest

from cogito_estella.mcp.store import DIVIDER, GraphStore

DOC = ("The generated concepts are decoded by SONAR. "
       "The encoder maps text to a vector. "
       "Because the budget grew, the committee met early. "
       "Because the decoder decided, SONAR replied quickly.")
TRIPLES = {
    "The generated concepts are decoded by SONAR.": [("concept", "improve", "sonar")],
    "The encoder maps text to a vector.": [("encoder", "give", "text")],
    "Because the budget grew, the committee met early.": [("budget", "do", "committee")],
    "Because the decoder decided, SONAR replied quickly.": [("decoder", "do", "sonar")],
}


@pytest.fixture
def store(fake_extractor):
    st = GraphStore(extractor=fake_extractor(TRIPLES))
    st.ingest_text(DOC, "t")
    return st


def test_query_lists_lexical_facts_then_divider_then_fallback(store):
    out = store.query("sonar")
    lines = out.split("\n")
    assert lines[0].startswith("sonar (") and "hops=1" in lines[0]
    body = lines[1:]
    assert body[0] == "sonar decode concept #0"
    assert DIVIDER in body
    idx = body.index(DIVIDER)
    assert all("#" in ln for ln in body[:idx]) and all("#" in ln for ln in body[idx + 1:])
    assert any(ln.startswith(("decoder do sonar", "sonar see decoder"))
               for ln in body[idx + 1:])


def test_query_without_fallback_has_no_divider(store):
    out = store.query("encoder")
    assert DIVIDER not in out and "encoder map text #1" in out


def test_query_limit_counts_facts_not_divider(store):
    out = store.query("sonar", limit=1)
    lines = out.split("\n")
    assert lines[1] == "sonar decode concept #0" and DIVIDER not in out
    assert lines[-1].startswith("+") and "more" in lines[-1]


def test_query_unknown_entity_lists_known(store):
    out = store.query("zzz")
    assert out.startswith("no entity matches 'zzz'") and "sonar" in out


def test_query_resolves_plural_and_prefix(store):
    assert store.query("concepts").startswith("concept (")
    assert store.query("enc").startswith("encoder (")


def test_provenance_prints_sentence_and_spans(store):
    out = store.provenance([0])
    assert out.startswith("#0 sonar decode concept  t s0 [")
    assert '"The generated concepts are decoded by SONAR."' in out


def test_provenance_retired_and_unknown(store):
    store.ingest_text("The encoder maps text to a vector.", "t")
    out = store.provenance([0, 999])
    assert "#0: retired edge (document replaced)" in out
    assert "#999: unknown edge" in out


def test_search_and_entities_and_stats(store):
    assert 'The encoder maps text to a vector.' in store.search("encoder")
    assert store.search("qqq") == "no sentence mentions 'qqq'"


def test_search_matches_words_that_co_occur_out_of_order(store):
    """A caller composes a natural multi-word phrase; the sentence rarely repeats it
    verbatim (measured: an agent's own multi-word search misses a sentence its own
    single-word retry of the same search finds). Every word of the query must occur in
    the sentence, not the whole query as one contiguous substring."""
    assert 'The encoder maps text to a vector.' in store.search("vector encoder")
    assert store.search("encoder rocket") == "no sentence mentions 'encoder rocket'"
    ents = store.entities()
    assert ents.startswith("sonar(") or "sonar(" in ents
    st = store.stats()
    assert "docs=1" in st and "edges=" in st and "graph_file=none" in st


# -- finding 7: empty or one-letter queries never resolve to the hub -------------------

def test_query_rejects_empty_and_single_letter_names(store):
    assert store.query("").startswith("no entity matches ''")
    assert store.query("s").startswith("no entity matches 's'")
    assert store.query("sonar").startswith("sonar (")


# -- finding 20: hops=2 expands across a neighbour and terminates ----------------------

def test_neighborhood_hops_two_reaches_second_degree_neighbor(store):
    one_hop = {e["id"] for e in store.neighborhood("concept", hops=1)}
    two_hop = {e["id"] for e in store.neighborhood("concept", hops=2)}
    assert one_hop < two_hop
