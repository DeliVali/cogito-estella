"""Sentence scorers for the `ask` route: tokenizer, lexical IDF, SONAR cosine, ranking."""
import math

import numpy as np
import pytest

from cogito_estella.mcp.rank import (
    PROVENANCE_BONUS,
    STOPWORDS,
    LexicalScorer,
    SonarScorer,
    rank,
    singular,
    singular_forms,
    tokenize,
)

CORPUS = [
    "The encoder maps byte to patch.",
    "The decoder maps patch to byte.",
    "Attention layer uses SwiGLU activation.",
]


# -- tokenize -------------------------------------------------------------------
def test_tokenize_lowercases_keeps_hyphens_and_digits():
    assert tokenize("Byte-level BLT-1 encoders, and the 12 layers!") == [
        "byte-level", "blt-1", "encoder", "12", "layer"]


def test_tokenize_drops_function_words():
    assert "the" in STOPWORDS and "of" in STOPWORDS
    assert tokenize("the of and to in is are") == []


def test_tokenize_strips_one_trailing_s_only_above_three_chars():
    assert tokenize("gas layers") == ["gas", "layer"]


def test_tokenize_singularizes_es_and_ies_plurals():
    assert tokenize("patches boxes losses policies") == ["patch", "box", "loss", "policy"]


def test_tokenize_never_strips_a_double_s():
    assert tokenize("loss class") == ["loss", "class"]


def test_tokenize_maps_a_plural_question_onto_its_singular_sentence():
    assert set(tokenize("How many patches?")) & set(tokenize("Each patch is a byte group.")) \
        == {"patch"}


def test_singular_forms_offer_the_blind_strips_after_the_canonical_one():
    assert singular_forms("patches") == ["patches", "patch", "patche"]
    assert singular_forms("layers") == ["layers", "layer"]
    assert singular_forms("gas") == ["gas"]
    assert singular_forms("class") == ["class"]
    assert singular("series") == "sery"          # documented cost of a purely syntactic rule


def test_tokenize_ignores_punctuation_and_keeps_duplicates():
    assert tokenize("patch, patch; patch.") == ["patch", "patch", "patch"]


def test_tokenize_empty_text():
    assert tokenize("") == []


# -- lexical scorer -------------------------------------------------------------
def test_lexical_idf_formula():
    sc = LexicalScorer(CORPUS)
    assert sc.n == 3
    assert sc.df["map"] == 2 and sc.df["encoder"] == 1
    assert sc.idf["map"] == pytest.approx(math.log(4 / 3) + 1)
    assert sc.idf["encoder"] == pytest.approx(math.log(4 / 2) + 1)
    assert sc.idf_of("quantum") == pytest.approx(math.log(4 / 1) + 1)


def test_lexical_score_is_one_when_every_question_token_is_present():
    sc = LexicalScorer(CORPUS)
    assert sc.score("encoder") == pytest.approx([1.0, 0.0, 0.0])


def test_lexical_score_splits_over_question_tokens():
    sc = LexicalScorer(CORPUS)
    assert sc.score("encoder and decoder") == pytest.approx([0.5, 0.5, 0.0])


def test_lexical_score_counts_absent_question_tokens_in_the_denominator():
    sc = LexicalScorer(CORPUS)
    hit, miss = math.log(4 / 2) + 1, math.log(4 / 1) + 1
    assert sc.score("encoder quantum")[0] == pytest.approx(hit / (hit + miss))


def test_lexical_score_stays_within_unit_range():
    sc = LexicalScorer(CORPUS)
    assert all(0.0 <= s <= 1.0 for s in sc.score("encoder maps byte to patch"))


def test_lexical_score_zero_without_a_shared_token():
    assert LexicalScorer(CORPUS).score("quantum entanglement") == pytest.approx([0.0] * 3)


def test_lexical_score_of_a_stopword_only_question_is_zero():
    assert LexicalScorer(CORPUS).score("what is the") == pytest.approx([0.0] * 3)


def test_lexical_repeated_question_token_is_not_double_counted():
    sc = LexicalScorer(CORPUS)
    assert sc.score("encoder encoder") == pytest.approx(sc.score("encoder"))


def test_lexical_scorer_on_an_empty_corpus():
    assert LexicalScorer([]).score("encoder") == []


# -- provenance bonus and ranking -----------------------------------------------
def test_provenance_bonus_value_and_application():
    assert PROVENANCE_BONUS == 0.3
    sc = LexicalScorer(CORPUS)
    boosted = sc.score("encoder", boost={1})
    assert boosted == pytest.approx([1.0, PROVENANCE_BONUS, 0.0])


def test_provenance_bonus_reorders_a_weak_match():
    sc = LexicalScorer(CORPUS)
    q = "decoder byte swiglu quantum"
    assert rank(sc.score(q)) == [1, 2, 0]
    assert rank(sc.score(q, boost={2})) == [2, 1, 0]


def test_bonus_ignores_out_of_range_indices():
    sc = LexicalScorer(CORPUS)
    assert sc.score("encoder", boost={99, -1}) == pytest.approx([1.0, 0.0, 0.0])


def test_rank_orders_by_score_descending_and_breaks_ties_by_index():
    assert rank([0.1, 0.9, 0.9, 0.5]) == [1, 2, 3, 0]


def test_rank_honours_k_and_empty_scores():
    assert rank([0.1, 0.9, 0.5], k=2) == [1, 2]
    assert rank([]) == []


# -- sonar scorer ---------------------------------------------------------------
def _fake_encode(vec):
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return np.asarray([vec], dtype=np.float32)
    encode.calls = calls
    return encode


def test_sonar_maps_cosine_into_unit_range():
    emb = np.asarray([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float16)
    sc = SonarScorer(emb, _fake_encode([1.0, 0.0, 0.0]))
    assert sc.score("anything") == pytest.approx([1.0, 0.5, 0.0])


def test_sonar_encodes_the_question_as_a_single_batch():
    enc = _fake_encode([1.0, 0.0, 0.0])
    SonarScorer(np.asarray([[1, 0, 0]], dtype=np.float16), enc).score("why bytes?")
    assert enc.calls == [["why bytes?"]]


def test_sonar_ranks_the_closest_sentence_first():
    emb = np.asarray([[0.6, 0.8, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float16)
    sc = SonarScorer(emb, _fake_encode([1.0, 0.0, 0.0]))
    assert rank(sc.score("q")) == [1, 0, 2]


def test_sonar_applies_the_provenance_bonus():
    emb = np.asarray([[1, 0, 0], [0, 0, 1]], dtype=np.float16)
    sc = SonarScorer(emb, _fake_encode([1.0, 0.0, 0.0]))
    assert sc.score("q", boost={1}) == pytest.approx([1.0, 0.5 + PROVENANCE_BONUS])


def test_sonar_accepts_an_unnormalized_question_vector():
    emb = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float16)
    sc = SonarScorer(emb, _fake_encode([7.0, 0.0, 0.0]))
    assert sc.score("q") == pytest.approx([1.0, 0.5])


def test_sonar_scorer_on_an_empty_corpus():
    sc = SonarScorer(np.zeros((0, 3), dtype=np.float16), _fake_encode([1.0, 0.0, 0.0]))
    assert sc.score("q") == []


def test_sonar_scores_never_leave_the_unit_range():
    emb = np.asarray([[1.001, 0.0], [-1.001, 0.0]], dtype=np.float32)
    sc = SonarScorer(emb, _fake_encode([1.0, 0.0]))
    assert sc.score("q") == pytest.approx([1.0, 0.0])
