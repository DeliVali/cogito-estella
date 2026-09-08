"""Span-level provenance: each edge can point back to the exact source sentence and the
character spans of the surface mentions that produced its nodes."""

import pytest

from cogito_estella.integrations.llamaindex_connector import (
    CogitoGraphExtractor,
    candidate_spans,
    provenance_records,
    spans_from_doc,
)


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
             "s_span": [4, 13], "o_span": [27, 33],
             "r_class": "improve", "pattern": 2, "swapped": False}]
    CogitoGraphExtractor.to_neo4j(ex, drv, recs, source="doc@v3")
    q, p = drv.s.calls[0]
    assert "sentence" in q and "s_span" in q
    assert "r_class" in q and "pattern" in q and "swapped" in q
    assert p["sentence"].startswith("The committee") and p["s_span"] == [4, 13] \
        and p["src"] == "doc@v3"
    assert p["r_class"] == "improve" and p["pattern"] == 2 and p["swapped"] is False


def test_to_cypher_accepts_provenance_record_dicts():
    ex = object.__new__(CogitoGraphExtractor)
    recs = [{"s": "committee", "r": "support", "o": "budget",
             "sentence": "The committee approved the budget.",
             "s_span": [4, 13], "o_span": [27, 33]}]
    pairs = CogitoGraphExtractor.to_cypher(ex, recs, source="doc@v3")
    _, params = pairs[0]
    assert params == {"s": "committee", "r": "support", "o": "budget", "src": "doc@v3"}


def test_spans_from_doc_matches_first_in_vocab_noun_mentions(nlp):
    doc = nlp("The encoder maps text to a vector.")
    spans = spans_from_doc(doc, {"encoder": 1, "text": 2, "vector": 3})
    assert spans == {"encoder": (4, 11), "text": (17, 21), "vector": (27, 33)}


def test_extract_with_provenance_adds_lexical_label(monkeypatch, nlp):
    """Passive sentence: the record is reoriented and labeled from syntax."""
    from cogito_estella.integrations import llamaindex_connector as lc
    ex = object.__new__(lc.CogitoGraphExtractor)
    ex.ent2id = {"concept": 1, "sonar": 2}
    ex._nlp = nlp
    monkeypatch.setattr(ex, "extract", lambda text, candidates=None, lang="eng_Latn",
                        return_scores=False: [("concept", "improve", "sonar")])
    recs = ex.extract_with_provenance("The concepts are decoded by SONAR.")
    r = recs[0]
    assert (r["s"], r["r"], r["o"]) == ("sonar", "decode", "concept")
    assert r["r_lex"] == "decode" and r["r_class"] == "improve" and r["swapped"] is True


def test_extract_with_provenance_exchanges_spans_on_swap(monkeypatch, nlp):
    """Reorientation must carry the char spans along with s/o, not just the labels."""
    from cogito_estella.integrations import llamaindex_connector as lc
    text = "The concepts are decoded by SONAR."
    ex = object.__new__(lc.CogitoGraphExtractor)
    ex.ent2id = {"concept": 1, "sonar": 2}
    ex._nlp = nlp
    monkeypatch.setattr(ex, "extract", lambda t, candidates=None, lang="eng_Latn",
                        return_scores=False: [("concept", "improve", "sonar")])
    recs = ex.extract_with_provenance(text)
    r = recs[0]
    assert r["swapped"] is True
    assert text[r["s_span"][0]:r["s_span"][1]] == "SONAR"
    assert text[r["o_span"][0]:r["o_span"][1]] == "concepts"


def test_extract_with_provenance_keeps_class_label_without_spacy(monkeypatch):
    """No parser available: the record keeps the class label untouched."""
    from cogito_estella.integrations import llamaindex_connector as lc
    ex = object.__new__(lc.CogitoGraphExtractor)
    ex.ent2id = {"concept": 1, "sonar": 2}
    ex._nlp = False
    monkeypatch.setattr(ex, "extract", lambda text, candidates=None, lang="eng_Latn",
                        return_scores=False: [("concept", "improve", "sonar")])
    recs = ex.extract_with_provenance("The concepts are decoded by SONAR.")
    r = recs[0]
    assert r["r"] == r["r_class"] == "improve"
    assert r["r_lex"] is None and r["swapped"] is False


def test_extract_with_provenance_null_span_when_subject_has_no_mention(monkeypatch, nlp):
    """A memory-supplied subject absent from the sentence keeps the class label and a
    null s_span — the lexicalizer never runs without both surface mentions."""
    from cogito_estella.integrations import llamaindex_connector as lc
    ex = object.__new__(lc.CogitoGraphExtractor)
    ex.ent2id = {"clinic": 1, "budget": 2}
    ex._nlp = nlp
    monkeypatch.setattr(ex, "extract", lambda text, candidates=None, lang="eng_Latn",
                        return_scores=False: [("clinic", "have", "budget")])
    recs = ex.extract_with_provenance("The budget grew.")
    r = recs[0]
    assert r["r"] == r["r_class"] == "have"
    assert r["s_span"] is None and r["o_span"] is not None


def test_extract_batch_with_provenance_matches_single_path(monkeypatch, nlp):
    """Batch path yields the same records as the single-sentence path, offsets applied."""
    from cogito_estella.integrations import llamaindex_connector as lc
    ex = object.__new__(lc.CogitoGraphExtractor)
    ex.ent2id = {"concept": 1, "sonar": 2, "encoder": 3, "text": 4}
    ex._nlp = nlp
    texts = ["The concepts are decoded by SONAR.", "The encoder maps text."]
    gold = {texts[0]: [("concept", "improve", "sonar")], texts[1]: [("encoder", "give", "text")]}
    monkeypatch.setattr(ex, "extract", lambda text, candidates=None, lang="eng_Latn",
                        return_scores=False: gold[text])
    monkeypatch.setattr(ex, "extract_batch", lambda ts, candidates=None, lang="eng_Latn":
                        [gold[t] for t in ts])
    batch = ex.extract_batch_with_provenance(texts, doc_offsets=[0, 100])
    single = [ex.extract_with_provenance(texts[0], doc_offset=0),
              ex.extract_with_provenance(texts[1], doc_offset=100)]
    assert batch == single
    assert (batch[0][0]["s"], batch[0][0]["r"], batch[0][0]["o"]) == ("sonar", "decode", "concept")
    # "encoder" at 4..11, offset 100; the parser reorients this sentence (swapped=True),
    # so "encoder" lands in o_span rather than s_span — asserted via `single` equality above.
    assert batch[1][0]["o_span"] == [104, 111]


# -- finding 14: doc_offsets must align 1:1 with texts ---------------------------------

def test_extract_batch_with_provenance_rejects_misaligned_offsets():
    from cogito_estella.integrations import llamaindex_connector as lc
    ex = object.__new__(lc.CogitoGraphExtractor)
    ex.ent2id, ex._nlp = {}, False
    with pytest.raises(ValueError, match="doc_offsets"):
        ex.extract_batch_with_provenance(["a", "b"], doc_offsets=[0])
