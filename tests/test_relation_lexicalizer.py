"""Pattern table of the relation lexicalizer (spec 2026-09-06, section Components)."""
import pytest

from cogito_estella.relation_lexicalizer import Lex, between_spans, head_token, lexicalize


def span_of(sentence, word):
    i = sentence.index(word)
    return (i, i + len(word))


def test_head_token_returns_the_token_covering_the_span(nlp):
    s = "The encoder of SONAR maps text to a fixed-size vector."
    doc = nlp(s)
    assert head_token(doc, span_of(s, "encoder")).text == "encoder"
    assert head_token(doc, span_of(s, "SONAR")).text == "SONAR"
    assert head_token(doc, (999, 1005)) is None


def test_row2_active_verb_nsubj_dobj(nlp):
    s = "The encoder of SONAR maps text to a fixed-size vector."
    lex = lexicalize(nlp(s), span_of(s, "encoder"), span_of(s, "text"))
    assert lex == Lex(label="map", pattern=2, swapped=False)


def test_row2_sets_direction_from_syntax_when_head_order_is_reversed(nlp):
    s = "The encoder of SONAR maps text to a fixed-size vector."
    lex = lexicalize(nlp(s), span_of(s, "text"), span_of(s, "encoder"))
    assert lex == Lex(label="map", pattern=2, swapped=True)


def test_same_span_is_not_a_relation(nlp):
    s = "The encoder maps text."
    lex = lexicalize(nlp(s), span_of(s, "encoder"), span_of(s, "encoder"))
    assert lex is None


def test_row1_passive_agent_inverts_to_agent_patient(nlp):
    s = "Finally, the generated concepts are decoded by SONAR into a sequence of subwords."
    doc = nlp(s)
    # head proposed (concept, sonar); syntax says SONAR decodes concepts
    lex = lexicalize(doc, span_of(s, "concepts"), span_of(s, "SONAR"))
    assert lex == Lex(label="decode", pattern=1, swapped=True)
    lex = lexicalize(doc, span_of(s, "SONAR"), span_of(s, "concepts"))
    assert lex == Lex(label="decode", pattern=1, swapped=False)


def test_row4_passive_prep_keeps_patient_as_subject(nlp):
    s = "The model is trained on 2.7 billion tokens with a diffusion objective."
    lex = lexicalize(nlp(s), span_of(s, "model"), span_of(s, "tokens"))
    assert lex == Lex(label="train_on", pattern=4, swapped=False)


def test_row4_reduced_relative_acl(nlp):
    s = "We evaluate the model trained on tokens."
    lex = lexicalize(nlp(s), span_of(s, "model"), span_of(s, "tokens"))
    assert lex == Lex(label="train_on", pattern=4, swapped=False)


def test_row4_labels_the_prep_object_of_a_passive_with_an_agent(nlp):
    s = "Finally, the generated concepts are decoded by SONAR into a sequence of subwords."
    lex = lexicalize(nlp(s), span_of(s, "concepts"), span_of(s, "sequence"))
    assert lex == Lex(label="decode_into", pattern=4, swapped=False)


def test_row3_active_verb_prep(nlp):
    s = "In practice, a concept would often correspond to a sentence in a text document."
    lex = lexicalize(nlp(s), span_of(s, "concept"), span_of(s, "sentence"))
    assert lex == Lex(label="correspond_to", pattern=3, swapped=False)


def test_row5_copula_is_a(nlp):
    s = "SONAR is a multilingual embedding space."
    lex = lexicalize(nlp(s), span_of(s, "SONAR"), span_of(s, "space"))
    assert lex == Lex(label="is_a", pattern=5, swapped=False)


def test_row6_nominal_prep(nlp):
    s = "The encoder of SONAR maps text to a fixed-size vector."
    lex = lexicalize(nlp(s), span_of(s, "encoder"), span_of(s, "SONAR"))
    assert lex == Lex(label="of", pattern=6, swapped=False)


def test_row7_compound(nlp):
    s = "We use a diffusion scheduler for the denoising step."
    lex = lexicalize(nlp(s), span_of(s, "diffusion"), span_of(s, "scheduler"))
    assert lex == Lex(label="compound", pattern=7, swapped=True)


def test_negation_prefix(nlp):
    s = "The committee did not approve the budget."
    lex = lexicalize(nlp(s), span_of(s, "committee"), span_of(s, "budget"))
    assert lex == Lex(label="not_approve", pattern=2, swapped=False)


def test_row5_copula_negation_is_not_a(nlp):
    s = "SONAR is not a decoder."
    lex = lexicalize(nlp(s), span_of(s, "SONAR"), span_of(s, "decoder"))
    assert lex == Lex(label="not_is_a", pattern=5, swapped=False)


def test_punctuation_only_preposition_yields_no_label(nlp):
    # governing preposition token is punctuation ("@") -> row 6 would clean to "";
    # the empty label must not reach the caller.
    s = "We report accuracy @ the token level."
    lex = lexicalize(nlp(s), span_of(s, "accuracy"), span_of(s, "level"))
    assert lex is None


def test_between_spans_returns_none_for_punctuation_only_span():
    s = "Code-switching."
    lex = between_spans(s, span_of(s, "Code"), span_of(s, "switching"))
    assert lex is None


def test_particle_is_attached(nlp):
    s = "The scheduler sets up the noise levels."
    lex = lexicalize(nlp(s), span_of(s, "scheduler"), span_of(s, "levels"))
    assert lex == Lex(label="set_up", pattern=2, swapped=False)


def test_fallback_when_spans_are_in_different_sentences(nlp):
    s = "The encoder maps text. The decoder emits tokens."
    lex = lexicalize(nlp(s), span_of(s, "encoder"), span_of(s, "tokens"))
    assert lex is None


def test_fallback_when_no_pattern_links_the_spans(nlp):
    s = "Because the budget grew, the committee met early."
    lex = lexicalize(nlp(s), span_of(s, "budget"), span_of(s, "committee"))
    assert lex is None


def test_fallback_on_out_of_range_span(nlp):
    s = "The encoder maps text."
    assert lexicalize(nlp(s), (0, 3), (500, 504)) is None


def test_fallback_on_doc_without_sentence_boundaries():
    spacy = pytest.importorskip("spacy")
    blank = spacy.blank("en")
    s = "The encoder maps text."
    doc = blank(s)
    assert lexicalize(doc, span_of(s, "encoder"), span_of(s, "text")) is None


def test_control_between_spans_keeps_content_words_and_preps():
    s = "In practice, a concept would often correspond to a sentence in a text document."
    lex = between_spans(s, span_of(s, "concept"), span_of(s, "sentence"))
    assert lex == Lex(label="correspond_to", pattern=0, swapped=False)


def test_control_between_spans_by_swaps_direction():
    s = "Finally, the generated concepts are decoded by SONAR into a sequence of subwords."
    lex = between_spans(s, span_of(s, "concepts"), span_of(s, "SONAR"))
    assert lex == Lex(label="decoded_by", pattern=0, swapped=True)


def test_control_between_spans_truncates_to_three_and_handles_adjacent():
    s = "The model quickly and reliably learns to predict the next embedding vector."
    lex = between_spans(s, span_of(s, "model"), span_of(s, "vector"))
    assert lex == Lex(label="learns_to_predict", pattern=0, swapped=False)
    assert between_spans(s, span_of(s, "embedding"), span_of(s, "vector")) is None


# Measure 2 of the spec: direction on active/passive pairs must be exact.
# Inlined from experiments/exp053_relation_lexicalizer/passives.py (repo tests are
# self-contained; the experiments/ tree is git-ignored and not importable here).
FACTS = [
    ("SONAR", "decode", "concepts", "SONAR decodes the concepts.",
     "The concepts are decoded by SONAR."),
    ("encoder", "map", "text", "The encoder maps the text.", "The text is mapped by the encoder."),
    ("model", "predict", "embedding", "The model predicts the embedding.",
     "The embedding is predicted by the model."),
    ("decoder", "emit", "tokens", "The decoder emits the tokens.",
     "The tokens are emitted by the decoder."),
    ("committee", "approve", "budget", "The committee approved the budget.",
     "The budget was approved by the committee."),
    ("committee", "reject", "budget", "The committee rejected the budget.",
     "The budget was rejected by the committee."),
    ("scheduler", "control", "noise", "The scheduler controls the noise.",
     "The noise is controlled by the scheduler."),
    ("tokenizer", "split", "sentence", "The tokenizer splits the sentence.",
     "The sentence is split by the tokenizer."),
    ("segmenter", "produce", "segments", "The segmenter produces the segments.",
     "The segments are produced by the segmenter."),
    ("network", "compute", "loss", "The network computes the loss.",
     "The loss is computed by the network."),
    ("adapter", "reduce", "error", "The adapter reduces the error.",
     "The error is reduced by the adapter."),
    ("author", "propose", "architecture", "The author proposes the architecture.",
     "The architecture is proposed by the author."),
    ("pipeline", "encode", "documents", "The pipeline encodes the documents.",
     "The documents are encoded by the pipeline."),
    ("gate", "filter", "candidates", "The gate filters the candidates.",
     "The candidates are filtered by the gate."),
    ("head", "score", "pairs", "The head scores the pairs.", "The pairs are scored by the head."),
    ("teacher", "train", "student", "The teacher trains the student.",
     "The student is trained by the teacher."),
    ("agent", "query", "graph", "The agent queries the graph.", "The graph is queried by the agent."),
    ("server", "serve", "facts", "The server serves the facts.", "The facts are served by the server."),
    ("benchmark", "track", "cost", "The benchmark tracks the cost.",
     "The cost is tracked by the benchmark."),
    ("ledger", "record", "tokens", "The ledger records the tokens.",
     "The tokens are recorded by the ledger."),
]

PASSIVES = []
for subj, label, obj, active, passive in FACTS:
    PASSIVES.append((active, (subj, label, obj)))
    PASSIVES.append((passive, (subj, label, obj)))


def probe(nlp):
    """(n_correct, misses). A probe is correct when label and printed direction match."""
    correct, misses = 0, []
    for sentence, (subj, label, obj) in PASSIVES:
        doc = nlp(sentence)
        # feed spans in text order, like the store does when the head proposes them
        first, second = sorted((subj, obj), key=sentence.index)
        lex = lexicalize(doc, span_of(sentence, first), span_of(sentence, second))
        printed = None
        if lex is not None:
            printed = (second, lex.label, first) if lex.swapped else (first, lex.label, second)
        if printed == (subj, label, obj):
            correct += 1
        else:
            misses.append({"sentence": sentence, "expected": [subj, label, obj],
                           "got": list(printed) if printed else None})
    return correct, misses


def test_probe_set_has_forty_entries_in_active_passive_pairs():
    assert len(PASSIVES) == 40


def test_passive_direction_is_exact(nlp):
    correct, misses = probe(nlp)
    assert correct == 40, misses
