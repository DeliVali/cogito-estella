"""Span-level provenance: each edge can point back to the exact source sentence and the
character spans of the surface mentions that produced its nodes."""
import torch

from cogito_estella.integrations.llamaindex_connector import (
    CogitoGraphExtractor, candidate_spans, provenance_records)


def test_candidate_spans_point_at_exact_surface_mentions():
    text = "The committees approved two budgets for the hospitals."
    spans = candidate_spans(text, {"committee": 1, "budget": 2, "hospital": 3})
    by_lemma = {s[0]: s for s in spans}
    for lemma, start, end in spans:
        surface = text[start:end]
        assert surface.lower().startswith(lemma[:6]), \
            f"span must cover the surface mention: {lemma} vs '{surface}'"
    assert "committee" in by_lemma and "budget" in by_lemma
    lemma, start, end = by_lemma["committee"]
    assert text[start:end] == "committees"      # surface form, not the lemma


def test_provenance_records_carry_sentence_spans_and_doc_offset():
    sentence = "The committee approved the budget."
    span_map = {"committee": (4, 13), "budget": (28, 34)}
    recs = provenance_records([("committee", "support", "budget")], span_map,
                              sentence, doc_offset=100)
    r = recs[0]
    assert r["s"] == "committee" and r["o"] == "budget" and r["r"] == "support"
    assert r["sentence"] == sentence
    assert r["s_span"] == [104, 113] and r["o_span"] == [128, 134]


def test_memory_candidates_without_mention_get_null_span():
    recs = provenance_records([("clinic", "have", "budget")],
                              {"budget": (0, 6)}, "budget note", doc_offset=0)
    assert recs[0]["s_span"] is None and recs[0]["o_span"] == [0, 6]


def test_to_neo4j_accepts_provenance_records():
    class FakeSession:
        def __init__(self): self.calls = []
        def run(self, q, **p): self.calls.append((q, p))
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class FakeDriver:
        def __init__(self): self.s = FakeSession()
        def session(self, database=None): return self.s

    drv = FakeDriver()
    ex = object.__new__(CogitoGraphExtractor)
    recs = [{"s": "committee", "r": "support", "o": "budget",
             "sentence": "The committee approved the budget.",
             "s_span": [4, 13], "o_span": [27, 33]}]
    CogitoGraphExtractor.to_neo4j(ex, drv, recs, source="doc@v3")
    q, p = drv.s.calls[0]
    assert "sentence" in q and "s_span" in q
    assert p["sentence"].startswith("The committee") and p["s_span"] == [4, 13] \
        and p["src"] == "doc@v3"
