"""Learned relevance scorer: feature extraction, BM25, logistic model, weights file,
and the `ask` path that re-orders with it."""
import json
import math
import re

import numpy as np
import pytest

from cogito_estella.mcp.rank import tokenize
from cogito_estella.mcp.rerank import (
    ENT_CAP,
    FEATURES,
    LEN_CENTER,
    WEIGHTS_ENV,
    FeatureContext,
    RelevanceScorer,
    bigrams,
    bm25_scores,
    corpus_avgdl,
    featurize,
    is_quantity_question,
    l2_for_shrink,
    load_default,
    penalty_shrink,
)
from cogito_estella.mcp.store import ASK_CANDIDATES, ASK_SCORERS, GraphStore
from cogito_estella.mcp.tokens import ntok

CANDS = [
    "The encoder maps 12 bytes to one patch.",
    "The decoder maps a patch back to bytes.",
    "Attention uses SwiGLU activation everywhere.",
]


def ctx_for(question, cands=CANDS, **over):
    """A context whose scalar fields the store would fill; tests override one at a time."""
    q = tokenize(question)
    fields = {
        "lex_scores": [1.0, 0.5, 0.0][:len(cands)],
        "bm25": [4.0, 2.0, 0.0][:len(cands)],
        "q_tokens": q,
        "q_bigrams": bigrams(q),
        "q_ents": [],
        "prov_flags": [False] * len(cands),
    }
    fields.update(over)
    return FeatureContext(**fields)


# -- feature contract -----------------------------------------------------------
def test_feature_names_and_order():
    assert list(FEATURES) == [
        "lex_norm", "lex_rank", "bm25_norm", "q_token_cov", "q_bigram_cov", "ent_hits",
        "provenance", "dense_cos", "dense_missing", "len_log", "rel_pos", "numeric_match"]
    assert len(FEATURES) == 12


def test_featurize_returns_one_row_per_candidate_and_finite_values():
    X = featurize("how many bytes per patch?", CANDS, ctx_for("how many bytes per patch?"))
    assert X.shape == (3, 12)
    assert np.isfinite(X).all()


def test_featurize_on_no_candidates_keeps_the_column_count():
    assert featurize("q", [], ctx_for("q", cands=[])).shape == (0, 12)


def test_featurize_rejects_a_context_of_a_different_length():
    with pytest.raises(ValueError, match="candidates"):
        featurize("q", CANDS, ctx_for("q", lex_scores=[1.0, 0.5]))


# -- individual features --------------------------------------------------------
def test_lex_norm_divides_by_the_best_candidate():
    X = featurize("q", CANDS, ctx_for("q", lex_scores=[0.4, 0.8, 0.2]))
    assert X[:, 0] == pytest.approx([0.5, 1.0, 0.25])


def test_lex_norm_is_zero_when_no_candidate_scores():
    X = featurize("q", CANDS, ctx_for("q", lex_scores=[0.0, 0.0, 0.0]))
    assert X[:, 0] == pytest.approx([0.0, 0.0, 0.0])


def test_lex_rank_is_the_reciprocal_rank_with_ties_by_position():
    X = featurize("q", CANDS, ctx_for("q", lex_scores=[0.2, 0.9, 0.9]))
    assert X[:, 1] == pytest.approx([1 / 3, 1.0, 0.5])


def test_bm25_norm_divides_by_the_best_candidate():
    X = featurize("q", CANDS, ctx_for("q", bm25=[1.0, 4.0, 0.0]))
    assert X[:, 2] == pytest.approx([0.25, 1.0, 0.0])


def test_q_token_cov_is_the_fraction_of_question_tokens_in_the_sentence():
    q = "encoder patch quantum"                     # two of three tokens in candidate 0
    X = featurize(q, CANDS, ctx_for(q))
    assert X[0, 3] == pytest.approx(2 / 3)
    assert X[2, 3] == pytest.approx(0.0)


def test_q_token_cov_ignores_a_repeated_question_token():
    q = "patch patch quantum"
    X = featurize(q, CANDS, ctx_for(q))
    assert X[0, 3] == pytest.approx(0.5)


def test_q_token_cov_is_zero_for_a_stopword_only_question():
    q = "what is the"
    X = featurize(q, CANDS, ctx_for(q))
    assert X[:, 3] == pytest.approx([0.0] * 3)


def test_q_bigram_cov_counts_adjacent_pairs():
    q = "encoder maps bytes"                         # (encoder, map) adjacent, (map, byte) not
    X = featurize(q, CANDS, ctx_for(q))
    assert X[0, 4] == pytest.approx(0.5)
    assert X[1, 4] == pytest.approx(0.0)


def test_q_bigram_cov_is_zero_for_a_single_token_question():
    X = featurize("encoder", CANDS, ctx_for("encoder"))
    assert X[:, 4] == pytest.approx([0.0] * 3)


def test_ent_hits_counts_mentioned_entities_and_caps_them():
    ents = ["encoder", "byte", "patch", "map"]
    X = featurize("q", CANDS, ctx_for("q", q_ents=ents))
    assert ENT_CAP == 3
    assert X[0, 5] == pytest.approx(float(ENT_CAP))
    assert X[2, 5] == pytest.approx(0.0)


def test_ent_hits_matches_a_plural_mention_of_a_singular_entity():
    X = featurize("q", CANDS, ctx_for("q", q_ents=["byte"]))
    assert X[0, 5] == pytest.approx(1.0)             # the sentence says "bytes"


def test_ent_hits_needs_every_token_of_a_multiword_entity():
    X = featurize("q", CANDS, ctx_for("q", q_ents=["swiglu activation", "swiglu decoder"]))
    assert X[2, 5] == pytest.approx(1.0)


def test_provenance_flag_is_passed_through():
    X = featurize("q", CANDS, ctx_for("q", prov_flags=[False, True, False]))
    assert X[:, 6] == pytest.approx([0.0, 1.0, 0.0])


def test_dense_columns_when_the_embeddings_are_present():
    X = featurize("q", CANDS, ctx_for("q", dense_cos=[0.9, 0.1, -0.2],
                                      dense_available=[True, True, True]))
    assert X[:, 7] == pytest.approx([0.9, 0.1, -0.2])
    assert X[:, 8] == pytest.approx([0.0, 0.0, 0.0])


def test_dense_cos_is_zeroed_where_the_embedding_is_missing():
    X = featurize("q", CANDS, ctx_for("q", dense_cos=[0.9, 0.1, 0.5],
                                      dense_available=[True, False, False]))
    assert X[:, 7] == pytest.approx([0.9, 0.0, 0.0])
    assert X[:, 8] == pytest.approx([0.0, 1.0, 1.0])


def test_dense_defaults_to_missing_for_every_candidate():
    X = featurize("q", CANDS, ctx_for("q"))
    assert X[:, 7] == pytest.approx([0.0] * 3)
    assert X[:, 8] == pytest.approx([1.0] * 3)


def test_len_log_is_centered_and_signed():
    short, long = "one two three.", " ".join(f"w{i}" for i in range(60))
    cands = [short, long]
    X = featurize("q", cands, ctx_for("q", cands=cands))
    assert LEN_CENTER == 25
    assert X[0, 9] == pytest.approx(math.log(3) - math.log(25))
    assert X[1, 9] == pytest.approx(math.log(60) - math.log(25))


def test_len_log_uses_the_supplied_sentence_lengths_when_given():
    X = featurize("q", CANDS, ctx_for("q", sent_len=[25, 25, 25]))
    assert X[:, 9] == pytest.approx([0.0] * 3)


def test_len_log_survives_an_empty_sentence():
    X = featurize("q", [""], ctx_for("q", cands=[""], lex_scores=[0.0], bm25=[0.0],
                                     prov_flags=[False]))
    assert np.isfinite(X[0, 9])


def test_rel_pos_is_passed_through_and_defaults_to_zero():
    assert featurize("q", CANDS, ctx_for("q", rel_pos=[0.0, 0.5, 1.0]))[:, 10] \
        == pytest.approx([0.0, 0.5, 1.0])
    assert featurize("q", CANDS, ctx_for("q"))[:, 10] == pytest.approx([0.0] * 3)


def test_numeric_match_needs_both_a_quantity_question_and_a_digit():
    q = "how many bytes per patch?"
    X = featurize(q, CANDS, ctx_for(q))
    assert X[:, 11] == pytest.approx([1.0, 0.0, 0.0])


def test_numeric_match_is_zero_when_the_question_asks_no_quantity():
    q = "which module maps bytes?"
    X = featurize(q, CANDS, ctx_for(q))
    assert X[:, 11] == pytest.approx([0.0] * 3)


# -- quantity questions ---------------------------------------------------------
@pytest.mark.parametrize("q", ["How many layers?", "how much memory does it use",
                               "What percentage of tokens?", "What size is the patch?",
                               "what number of heads", "How long is the context?"])
def test_is_quantity_question_accepts_the_quantity_forms(q):
    assert is_quantity_question(q)


@pytest.mark.parametrize("q", ["Which encoder maps bytes?", "howmany layers",
                               "Why is the decoder slow?", ""])
def test_is_quantity_question_rejects_the_rest(q):
    assert not is_quantity_question(q)


def test_is_quantity_question_ignores_extra_whitespace():
    assert is_quantity_question("How   many\nlayers?")


# -- bm25 -----------------------------------------------------------------------
def test_bm25_matches_the_formula_on_a_hand_computed_corpus():
    docs = [["a", "b"], ["a", "a", "c"]]
    idf = {"a": 1.0, "b": 2.0, "c": 3.0}
    got = bm25_scores(docs, idf, ["a"])
    # avgdl 2.5; denominators 1 + 1.2 * (0.25 + 0.75 * len / 2.5)
    assert got == pytest.approx([2.2 / 2.02, 4.4 / 3.38])


def test_bm25_saturates_with_term_frequency():
    docs = [["a"], ["a"] * 2, ["a"] * 8]
    got = bm25_scores(docs, {"a": 1.0}, ["a"])
    assert got[0] < got[1] < got[2]
    assert got[2] - got[1] < got[1] - got[0]


def test_bm25_penalizes_a_longer_document_at_equal_term_frequency():
    docs = [["a", "b"], ["a", *[f"w{i}" for i in range(20)]]]
    got = bm25_scores(docs, {"a": 1.0}, ["a"])
    assert got[0] > got[1]


def test_bm25_ignores_query_tokens_absent_from_the_corpus():
    docs = [["a", "b"], ["c"]]
    assert bm25_scores(docs, {"a": 1.0}, ["a"]) \
        == pytest.approx(bm25_scores(docs, {"a": 1.0}, ["a", "zzz"]))


def test_bm25_counts_a_repeated_query_token_once():
    docs = [["a", "b"], ["c"]]
    assert bm25_scores(docs, {"a": 1.0}, ["a", "a"]) \
        == pytest.approx(bm25_scores(docs, {"a": 1.0}, ["a"]))


def test_bm25_accepts_a_callable_idf():
    docs = [["a", "b"], ["c"]]
    assert bm25_scores(docs, lambda t: 1.0, ["a"]) \
        == pytest.approx(bm25_scores(docs, {"a": 1.0}, ["a"]))


def test_bm25_on_an_empty_query_and_an_empty_corpus():
    assert bm25_scores([["a"]], {"a": 1.0}, []) == pytest.approx([0.0])
    assert bm25_scores([], {"a": 1.0}, ["a"]).shape == (0,)


def test_bm25_survives_an_empty_document():
    assert np.isfinite(bm25_scores([[], ["a"]], {"a": 1.0}, ["a"])).all()


def test_corpus_avgdl_is_the_mean_token_length():
    assert corpus_avgdl([["a"], ["a", "b", "c"]]) == pytest.approx(2.0)
    assert corpus_avgdl([]) == pytest.approx(1.0)


def test_bm25_defaults_to_the_mean_length_of_the_sentences_given():
    docs = [["a"], ["a", "b", "c", "d"]]
    assert bm25_scores(docs, {"a": 1.0}, ["a"]) \
        == pytest.approx(bm25_scores(docs, {"a": 1.0}, ["a"], avgdl=corpus_avgdl(docs)))


def test_bm25_length_normalizer_is_the_scale_of_the_column():
    docs = [["a"], ["a", "b", "c", "d"]]
    short = bm25_scores(docs, {"a": 1.0}, ["a"], avgdl=2.0)
    long = bm25_scores(docs, {"a": 1.0}, ["a"], avgdl=50.0)
    assert short != pytest.approx(long)


def test_bm25_on_a_subset_matches_the_universe_when_the_normalizer_is_pinned():
    """The defect the parameter closes: scoring candidates alone silently rescales the
    column unless the universe's normalizer travels with the query."""
    universe = [["a"], ["a", "b"], ["a"] * 9, ["c"] * 40, ["d"] * 30]
    idf, q = {"a": 1.0}, ["a"]
    full = bm25_scores(universe, idf, q)
    subset = universe[:3]
    drifted = bm25_scores(subset, idf, q)
    pinned = bm25_scores(subset, idf, q, avgdl=corpus_avgdl(universe))
    assert pinned == pytest.approx(full[:3])
    assert drifted != pytest.approx(full[:3])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_bm25_rejects_a_normalizer_that_is_not_a_length(bad):
    with pytest.raises(ValueError, match="avgdl"):
        bm25_scores([["a"]], {"a": 1.0}, ["a"], avgdl=bad)


# -- bigrams --------------------------------------------------------------------
def test_bigrams_are_adjacent_token_pairs():
    assert bigrams(["a", "b", "c"]) == [("a", "b"), ("b", "c")]
    assert bigrams(["a"]) == [] and bigrams([]) == []


# -- model ----------------------------------------------------------------------
def separable(n=200, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 2] > 0).astype(float)
    return X, y


def test_fit_separates_the_two_classes():
    X, y = separable()
    sc = RelevanceScorer().fit(X, y)
    s = sc.score(X)
    assert s[y == 1].mean() > s[y == 0].mean()
    assert (s[y == 1].min() > s[y == 0].max()) or np.mean((s > 0) == (y > 0)) > 0.9


def test_fit_learns_the_sign_of_the_informative_features():
    X, y = separable()
    sc = RelevanceScorer().fit(X, y)
    assert sc.w[0] > 0 and sc.w[2] > 0
    assert abs(sc.w[0]) > abs(sc.w[5])


def test_fit_standardizes_with_the_training_statistics():
    X, y = separable()
    X[:, 3] = X[:, 3] * 100 + 500
    sc = RelevanceScorer().fit(X, y)
    assert sc.mean[3] == pytest.approx(X[:, 3].mean())
    assert sc.std[3] == pytest.approx(X[:, 3].std())
    assert np.isfinite(sc.score(X)).all()


def test_fit_keeps_a_constant_feature_finite():
    X, y = separable()
    X[:, 4] = 7.0
    sc = RelevanceScorer().fit(X, y)
    assert sc.std[4] > 0 and np.isfinite(sc.score(X)).all()


def test_stronger_l2_shrinks_the_weights():
    X, y = separable()
    weak = RelevanceScorer().fit(X, y, l2=1.0)
    strong = RelevanceScorer().fit(X, y, l2=1000.0)
    assert np.abs(strong.w).sum() < np.abs(weak.w).sum()


def test_fit_records_the_penalty_it_applied():
    X, y = separable()
    assert RelevanceScorer().fit(X, y).l2 == pytest.approx(1.0)
    assert RelevanceScorer().fit(X, y, l2=7.5).l2 == pytest.approx(7.5)


def test_l2_is_graded_against_the_row_count():
    """Loss and penalty are both summed over the rows, so doubling the rows halves the
    penalty's share: the same fit needs twice the l2. Pinning this keeps the number in
    the weights file readable as one convention rather than two."""
    X, y = separable()
    doubled = np.vstack([X, X])
    labels = np.concatenate([y, y])
    one = RelevanceScorer().fit(X, y, l2=200.0)
    two = RelevanceScorer().fit(doubled, labels, l2=400.0)
    assert two.w == pytest.approx(one.w, abs=2e-3)
    unmatched = RelevanceScorer().fit(doubled, labels, l2=200.0)
    assert np.abs(unmatched.w).sum() > np.abs(one.w).sum()


def test_fit_refuses_a_penalty_that_would_diverge():
    X, y = separable()
    with pytest.raises(ValueError, match="diverge"):
        RelevanceScorer().fit(X, y, l2=1e6)


@pytest.mark.parametrize("bad", [-1.0, float("nan")])
def test_fit_rejects_a_penalty_that_is_not_a_penalty(bad):
    X, y = separable()
    with pytest.raises(ValueError, match="l2"):
        RelevanceScorer().fit(X, y, l2=bad)


def test_fit_rejects_a_feature_count_that_is_not_the_contract():
    with pytest.raises(ValueError, match="features"):
        RelevanceScorer().fit(np.zeros((4, 3)), np.zeros(4))


def test_fit_rejects_mismatched_labels():
    with pytest.raises(ValueError, match="rows"):
        RelevanceScorer().fit(np.zeros((4, 12)), np.zeros(3))


def test_score_and_proba_agree_on_the_order():
    X, y = separable()
    sc = RelevanceScorer().fit(X, y)
    s, p = sc.score(X), sc.predict_proba(X)
    assert ((p > 0) & (p < 1)).all()
    assert np.argsort(s).tolist() == np.argsort(p).tolist()


def test_score_accepts_a_single_row():
    X, y = separable()
    sc = RelevanceScorer().fit(X, y)
    assert sc.score(X[:1]).shape == (1,)


def test_score_before_fit_fails_loudly():
    with pytest.raises(ValueError, match="not fitted"):
        RelevanceScorer().score(np.zeros((2, 12)))


# -- weights file ---------------------------------------------------------------
AVGDL = 24.0


def fitted(X, y, **kw):
    """A fit that also pins what a stored weights file has to carry."""
    return RelevanceScorer().fit(X, y, **kw).pin_universe(AVGDL, "fixture sentences")


def test_json_round_trip_preserves_the_scores(tmp_path):
    X, y = separable()
    sc = fitted(X, y)
    sc.trained_on = "fixture"
    sc.cv = {"learned_recall_at_budget": 0.9, "lexical_recall_at_budget": 0.8, "folds": []}
    path = tmp_path / "w.json"
    sc.to_json(path)
    back = RelevanceScorer.from_json(path)
    assert back.score(X) == pytest.approx(sc.score(X))
    assert back.trained_on == "fixture" and back.cv["folds"] == []


def test_to_json_serializes_array_scalars_left_in_the_report(tmp_path):
    X, y = separable()
    sc = fitted(X, y)
    sc.cv = {"learned_recall_at_budget": np.float64(0.9),
             "folds": [{"fold": np.int64(2), "learned": np.float32(0.5)}]}
    path = tmp_path / "w.json"
    sc.to_json(path)
    blob = json.loads(path.read_text())
    assert blob["cv"]["learned_recall_at_budget"] == pytest.approx(0.9)
    assert blob["cv"]["folds"][0]["fold"] == 2


def test_weights_file_holds_the_documented_keys(tmp_path):
    X, y = separable()
    path = tmp_path / "w.json"
    fitted(X, y).to_json(path)
    blob = json.loads(path.read_text())
    assert set(blob) >= {"features", "mean", "std", "w", "b", "trained_on", "cv",
                         "l2", "bm25_avgdl", "bm25_universe"}
    assert blob["features"] == list(FEATURES)
    assert len(blob["mean"]) == len(blob["std"]) == len(blob["w"]) == 12


def test_weights_file_records_the_penalty_and_the_ranking_universe(tmp_path):
    X, y = separable()
    path = tmp_path / "w.json"
    fitted(X, y, l2=3.0).to_json(path)
    blob = json.loads(path.read_text())
    assert blob["l2"] == pytest.approx(3.0)
    assert blob["bm25_avgdl"] == pytest.approx(AVGDL)
    assert blob["bm25_universe"] == "fixture sentences"
    back = RelevanceScorer.from_json(path)
    assert back.l2 == pytest.approx(3.0) and back.bm25_avgdl == pytest.approx(AVGDL)
    assert back.bm25_universe == "fixture sentences"


def test_to_json_refuses_weights_that_do_not_pin_the_universe(tmp_path):
    X, y = separable()
    with pytest.raises(ValueError, match="normalizer"):
        RelevanceScorer().fit(X, y).to_json(tmp_path / "w.json")


@pytest.mark.parametrize("key, message", [("bm25_avgdl", "normalizer"), ("l2", "penalty")])
def test_from_json_rejects_a_file_that_pins_neither_convention(tmp_path, key, message):
    X, y = separable()
    path = tmp_path / "w.json"
    fitted(X, y).to_json(path)
    blob = json.loads(path.read_text())
    del blob[key]
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match=message):
        RelevanceScorer.from_json(path)


def test_from_json_rejects_a_normalizer_that_is_not_a_length(tmp_path):
    X, y = separable()
    path = tmp_path / "w.json"
    fitted(X, y).to_json(path)
    blob = json.loads(path.read_text())
    blob["bm25_avgdl"] = 0.0
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="normalizer"):
        RelevanceScorer.from_json(path)


@pytest.mark.parametrize("bad", [0.0, -3.0, float("nan")])
def test_pin_universe_rejects_a_normalizer_that_is_not_a_length(bad):
    X, y = separable()
    with pytest.raises(ValueError, match="normalizer"):
        RelevanceScorer().fit(X, y).pin_universe(bad)


def test_from_json_rejects_other_feature_names(tmp_path):
    X, y = separable()
    path = tmp_path / "w.json"
    fitted(X, y).to_json(path)
    blob = json.loads(path.read_text())
    blob["features"][0] = "something_else"
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="feature"):
        RelevanceScorer.from_json(path)


def test_from_json_rejects_a_truncated_vector(tmp_path):
    X, y = separable()
    path = tmp_path / "w.json"
    fitted(X, y).to_json(path)
    blob = json.loads(path.read_text())
    blob["w"] = blob["w"][:5]
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="length"):
        RelevanceScorer.from_json(path)


def test_validate_rejects_a_zero_scale(tmp_path):
    X, y = separable()
    sc = RelevanceScorer().fit(X, y)
    sc.std[2] = 0.0
    with pytest.raises(ValueError, match="scale"):
        sc.validate()


def test_validate_rejects_non_finite_weights():
    X, y = separable()
    sc = RelevanceScorer().fit(X, y)
    sc.w[1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        sc.validate()


def test_load_default_returns_none_when_the_file_is_absent(tmp_path, monkeypatch):
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "nope.json"))
    assert load_default() is None


def test_load_default_reads_the_file_named_by_the_environment(tmp_path, monkeypatch):
    X, y = separable()
    path = tmp_path / "w.json"
    sc = fitted(X, y)
    sc.to_json(path)
    monkeypatch.setenv(WEIGHTS_ENV, str(path))
    loaded = load_default()
    assert loaded is not None and loaded.score(X) == pytest.approx(sc.score(X))


def test_load_default_returns_none_on_a_defective_file(tmp_path, monkeypatch, capsys):
    path = tmp_path / "w.json"
    path.write_text("{ not json")
    monkeypatch.setenv(WEIGHTS_ENV, str(path))
    assert load_default() is None
    assert "rerank" in capsys.readouterr().err


def test_load_default_returns_none_when_the_universe_is_not_pinned(tmp_path, monkeypatch, capsys):
    X, y = separable()
    path = tmp_path / "w.json"
    fitted(X, y).to_json(path)
    blob = json.loads(path.read_text())
    del blob["bm25_avgdl"]
    path.write_text(json.dumps(blob))
    monkeypatch.setenv(WEIGHTS_ENV, str(path))
    assert load_default() is None
    assert "rerank" in capsys.readouterr().err


def test_load_default_returns_none_on_a_file_with_other_features(tmp_path, monkeypatch):
    X, y = separable()
    path = tmp_path / "w.json"
    fitted(X, y).to_json(path)
    blob = json.loads(path.read_text())
    blob["features"] = list(FEATURES)[:11]
    path.write_text(json.dumps(blob))
    monkeypatch.setenv(WEIGHTS_ENV, str(path))
    assert load_default() is None


def test_fit_is_invariant_to_the_scale_of_a_feature():
    """Standardizing inside the fit, not only at scoring time: rescaling one column
    affinely must leave every score where it was."""
    X, y = separable()
    rescaled = X.copy()
    rescaled[:, 3] = rescaled[:, 3] * 100.0 + 500.0
    plain = RelevanceScorer().fit(X, y)
    shifted = RelevanceScorer().fit(rescaled, y)
    assert shifted.score(rescaled) == pytest.approx(plain.score(X), abs=1e-6)


# -- the penalty in units that do not depend on the row count -------------------
def test_penalty_shrink_and_l2_for_shrink_invert_each_other():
    for rows in (200, 13800):
        for target in (0.9, 0.5, 0.05):
            l2 = l2_for_shrink(target, rows)
            assert penalty_shrink(l2, rows) == pytest.approx(target, rel=1e-9)


def test_the_same_l2_means_less_and_less_as_the_rows_grow():
    """The number in the spec is not a strength: at the training shape it is inert."""
    assert penalty_shrink(1.0, 200) < 0.5
    assert penalty_shrink(1.0, 13800) > 0.98


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, float("nan")])
def test_l2_for_shrink_rejects_a_factor_that_is_not_a_shrink(bad):
    with pytest.raises(ValueError, match="shrink"):
        l2_for_shrink(bad, 200)


def test_fit_refuses_a_penalty_that_does_nothing_at_this_row_count():
    """A penalty that is asked for has to happen: a fixed l2 carried to a large training
    set shrinks nothing, and the weights file would still report it as regularization."""
    X = np.zeros((20000, 12))
    y = np.zeros(20000)
    with pytest.raises(ValueError, match="shrink"):
        RelevanceScorer().fit(X, y, l2=1.0)


def test_fit_accepts_an_explicit_absence_of_a_penalty():
    X = np.zeros((20000, 12))
    y = np.zeros(20000)
    assert RelevanceScorer().fit(X, y, l2=0.0).l2 == 0.0


def test_shrink_states_the_penalty_independently_of_the_row_count():
    X, y = separable()
    doubled, labels = np.vstack([X, X]), np.concatenate([y, y])
    one = RelevanceScorer().fit(X, y, shrink=0.5)
    two = RelevanceScorer().fit(doubled, labels, shrink=0.5)
    assert two.w == pytest.approx(one.w, abs=2e-3)
    assert two.l2 == pytest.approx(2.0 * one.l2, rel=1e-6)
    assert one.l2_shrink == pytest.approx(0.5) and two.l2_shrink == pytest.approx(0.5)


def test_fit_records_the_effect_of_the_penalty_and_the_shape_it_was_measured_on():
    X, y = separable()
    sc = RelevanceScorer().fit(X, y, l2=2.0)
    assert sc.fit_rows == X.shape[0] and sc.fit_steps == 2000 and sc.fit_lr == pytest.approx(0.1)
    assert sc.l2_shrink == pytest.approx(penalty_shrink(2.0, X.shape[0]))


def test_weights_file_carries_the_effect_of_the_penalty(tmp_path):
    X, y = separable()
    path = tmp_path / "w.json"
    fitted(X, y, l2=2.0).to_json(path)
    blob = json.loads(path.read_text())
    assert set(blob) >= {"l2", "l2_shrink", "fit_rows", "fit_lr", "fit_steps"}
    assert blob["l2_shrink"] == pytest.approx(penalty_shrink(2.0, X.shape[0]))
    assert RelevanceScorer.from_json(path).l2_shrink == pytest.approx(blob["l2_shrink"])


@pytest.mark.parametrize("key", ["l2_shrink", "fit_rows", "fit_lr", "fit_steps"])
def test_from_json_rejects_a_file_that_does_not_say_what_the_penalty_did(tmp_path, key):
    X, y = separable()
    path = tmp_path / "w.json"
    fitted(X, y).to_json(path)
    blob = json.loads(path.read_text())
    del blob[key]
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="penalty"):
        RelevanceScorer.from_json(path)


def test_from_json_rejects_an_effect_that_does_not_follow_from_the_penalty(tmp_path):
    """The recorded effect is recomputed from l2, rows, steps and step size: a file cannot
    claim regularization it did not apply."""
    X, y = separable()
    path = tmp_path / "w.json"
    fitted(X, y).to_json(path)
    blob = json.loads(path.read_text())
    blob["l2_shrink"] = 0.5 * blob["l2_shrink"]
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="penalty"):
        RelevanceScorer.from_json(path)


# -- ask(scorer="learned"): candidates, context, order, fallback ------------------
LEARNED_DOC = ("The encoder maps text to a vector. "
               "The decoder maps text to a vector. "
               "Because the budget grew, the committee met early.")
LEARNED_TRIPLES = {
    "The encoder maps text to a vector.": [("encoder", "give", "text")],
    "The decoder maps text to a vector.": [("decoder", "give", "text")],
    "Because the budget grew, the committee met early.": [("budget", "do", "committee")],
}
LEARNED_Q = "What does the encoder do with text?"


def write_weights(path, weights, b=0.0, avgdl=10.0):
    """A shipped weights file over unit-scaled features: `weights` alone picks the order."""
    sc = RelevanceScorer(w=np.asarray(weights, dtype=float), b=float(b),
                         mean=np.zeros(len(FEATURES)), std=np.ones(len(FEATURES)),
                         l2=0.0, l2_shrink=1.0, fit_rows=64, fit_lr=0.1, fit_steps=2000,
                         bm25_avgdl=avgdl, bm25_universe="fixture")
    return sc.to_json(path)


def only(name, value=1.0):
    w = np.zeros(len(FEATURES))
    w[FEATURES.index(name)] = value
    return w


def col(X, name):
    return X[:, FEATURES.index(name)]


def unit_vectors(texts, lang="eng_Latn", dim=8):
    """Deterministic normalized bag-of-words vectors: a stand-in for a sentence encoder."""
    rows = []
    for t in texts:
        v = np.zeros(dim, dtype=np.float32)
        for w in re.findall(r"[a-z]+", t.lower()):
            v[sum(map(ord, w)) % dim] += 1.0
        n = float(np.linalg.norm(v))
        rows.append(v / n if n else v)
    return np.asarray(rows, dtype=np.float16)


@pytest.fixture
def learned_store(fake_extractor):
    def make(encode=None):
        ex = fake_extractor(LEARNED_TRIPLES)
        if encode is not None:
            ex.encode_batch = encode
        st = GraphStore(extractor=ex)
        st.ingest_text(LEARNED_DOC, "t")
        return st
    return make


@pytest.fixture
def weighted(tmp_path, monkeypatch):
    def use(weights, **over):
        monkeypatch.setenv(WEIGHTS_ENV, str(write_weights(tmp_path / "w.json", weights, **over)))
    return use


def sentence_lines(reply):
    lines = reply.split("\n")
    return lines[lines.index("--") + 1:] if "--" in lines else []


def test_learned_is_a_scorer_ask_accepts():
    assert "learned" in ASK_SCORERS


def test_ask_learned_names_itself_in_the_header(learned_store, weighted):
    weighted(only("lex_norm"))
    assert learned_store().ask(LEARNED_Q, scorer="learned").split("\n")[0].endswith(
        "· scorer=learned")


def test_ask_learned_reorders_the_sentence_block(learned_store, weighted):
    """The lexical order is the input, not the output: a weight on a different feature moves
    a sentence the lexical ranking put second."""
    st = learned_store()
    assert sentence_lines(st.ask(LEARNED_Q))[0].startswith("t s0:")
    weighted(only("rel_pos"))
    assert sentence_lines(st.ask(LEARNED_Q, scorer="learned"))[0].startswith("t s1:")


def test_ask_learned_ranks_only_the_lexical_candidates(learned_store, weighted):
    """A sentence with no overlap and no retrieved fact never reaches the model, so no weight
    can pull it into the reply."""
    weighted(only("rel_pos"))
    assert "committee" not in learned_store().ask(LEARNED_Q, scorer="learned")


def test_ask_learned_keeps_the_layout_and_the_budget(learned_store, weighted):
    weighted(only("lex_norm"))
    st = learned_store()
    lines = st.ask(LEARNED_Q, scorer="learned").split("\n")
    assert lines[0].startswith("entities: encoder, text ·") and "--" in lines
    for budget in (20, 40, 80, 200, 600):
        assert ntok(st.ask(LEARNED_Q, budget=budget, scorer="learned")) <= budget


def test_ask_learned_falls_back_with_a_note_when_the_weights_are_absent(
        learned_store, tmp_path, monkeypatch):
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "absent.json"))
    assert learned_store().ask(LEARNED_Q, scorer="learned").startswith(
        "entities: encoder, text · scorer=lexical (learned unavailable)")


def test_ask_learned_falls_back_when_the_weights_file_is_unusable(
        learned_store, tmp_path, monkeypatch):
    path = tmp_path / "w.json"
    path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv(WEIGHTS_ENV, str(path))
    assert "· scorer=lexical (learned unavailable)" in learned_store().ask(
        LEARNED_Q, scorer="learned")


def test_ask_learned_falls_back_when_nothing_is_worth_ranking(learned_store, weighted):
    weighted(only("lex_norm"))
    assert learned_store().ask("quantum chromodynamics", scorer="learned") == \
        "no material for 'quantum chromodynamics'"


def spy_features(monkeypatch):
    """Capture the matrix the store hands the model, still running the real featurizer."""
    from cogito_estella.mcp import store as store_mod
    seen: dict = {}
    real = store_mod.featurize

    def spy(question, cands, ctx):
        X = real(question, cands, ctx)
        seen["cands"], seen["X"], seen["ctx"] = list(cands), X, ctx
        return X
    monkeypatch.setattr(store_mod, "featurize", spy)
    return seen


def test_ask_learned_fills_the_feature_context_from_the_store(
        learned_store, weighted, monkeypatch):
    weighted(only("lex_norm"))
    seen = spy_features(monkeypatch)
    learned_store().ask(LEARNED_Q, scorer="learned")
    X = seen["X"]
    assert len(seen["cands"]) == 2 and X.shape == (2, len(FEATURES))
    assert seen["cands"][0].startswith("The encoder")         # candidates arrive lexically ordered
    assert col(X, "lex_norm")[0] == pytest.approx(1.0)
    assert col(X, "provenance").tolist() == [1.0, 1.0]        # both sentences back a retrieved fact
    assert col(X, "rel_pos") == pytest.approx([0.0, 0.5])     # third of three sentences
    assert col(X, "bm25_norm")[0] == pytest.approx(1.0)


def test_ask_learned_marks_the_dense_feature_missing_without_embeddings(
        learned_store, weighted, monkeypatch):
    weighted(only("lex_norm"))
    seen = spy_features(monkeypatch)
    learned_store().ask(LEARNED_Q, scorer="learned")
    assert col(seen["X"], "dense_missing").tolist() == [1.0, 1.0]
    assert col(seen["X"], "dense_cos").tolist() == [0.0, 0.0]


def test_ask_learned_uses_the_dense_feature_when_the_store_has_embeddings(
        learned_store, weighted, monkeypatch):
    weighted(only("lex_norm"))
    seen = spy_features(monkeypatch)
    learned_store(encode=unit_vectors).ask(LEARNED_Q, scorer="learned")
    cos = col(seen["X"], "dense_cos")
    assert col(seen["X"], "dense_missing").tolist() == [0.0, 0.0]
    assert np.all(np.abs(cos) <= 1.0) and float(np.abs(cos).max()) > 0.0


def test_ask_learned_marks_the_dense_feature_missing_when_the_question_will_not_encode(
        learned_store, weighted, monkeypatch):
    def boom(texts, lang="eng_Latn"):
        raise RuntimeError("no encoder")

    weighted(only("lex_norm"))
    st = learned_store(encode=unit_vectors)
    seen = spy_features(monkeypatch)
    st._ex.encode_batch = boom
    assert "· scorer=learned" in st.ask(LEARNED_Q, scorer="learned")
    assert col(seen["X"], "dense_missing").tolist() == [1.0, 1.0]


def test_ask_learned_scores_bm25_against_the_pinned_length_normalizer(
        learned_store, weighted, monkeypatch):
    """The ranking column is only comparable to the fitted one at the same divisor."""
    from cogito_estella.mcp import store as store_mod
    weighted(only("bm25_norm"), avgdl=7.5)
    seen: dict = {}
    real = store_mod.bm25_scores

    def spy(texts, idf, q_tokens, **kw):
        seen.update(kw)
        return real(texts, idf, q_tokens, **kw)
    monkeypatch.setattr(store_mod, "bm25_scores", spy)
    learned_store().ask(LEARNED_Q, scorer="learned")
    assert seen["avgdl"] == pytest.approx(7.5)


# -- one candidate rule for the fit path and the query path -----------------------
WIDE_Q = "What maps text to a vector?"
WIDE_DOC = " ".join(f"Item {i} maps text to a vector of size {i}." for i in range(150))


@pytest.fixture
def wide_store(fake_extractor):
    st = GraphStore(extractor=fake_extractor({}))
    st.ingest_text(WIDE_DOC, "w")
    return st


def test_ask_learned_marks_the_dense_feature_missing_when_the_widths_disagree(
        learned_store, weighted, monkeypatch):
    """A sidecar written by one encoder and a question encoded by another: the column is
    unavailable, and the ranking still happens."""
    weighted(only("lex_norm"))
    st = learned_store(encode=unit_vectors)
    st._ex.encode_batch = lambda texts, lang="eng_Latn": unit_vectors(texts, dim=4)
    seen = spy_features(monkeypatch)
    assert "· scorer=learned" in st.ask(LEARNED_Q, scorer="learned")
    assert col(seen["X"], "dense_missing").tolist() == [1.0, 1.0]
    assert col(seen["X"], "dense_cos").tolist() == [0.0, 0.0]


def test_the_shortlist_grows_with_the_budget(wide_store, weighted, monkeypatch):
    """A fixed shortlist would truncate the block at a budget the tool accepts, and compare
    two arms at two different depths."""
    weighted(only("lex_norm"))
    seen = spy_features(monkeypatch)
    wide_store.ask(WIDE_Q, scorer="learned")
    assert len(seen["cands"]) == ASK_CANDIDATES
    wide_store.ask(WIDE_Q, budget=4000, scorer="learned")
    assert len(seen["cands"]) == 150                    # every sentence the lexical arm would fill


def test_a_large_budget_fills_the_same_number_of_lines_as_the_lexical_arm(
        wide_store, weighted):
    weighted(only("lex_norm"))
    learned = sentence_lines(wide_store.ask(WIDE_Q, budget=4000, scorer="learned"))
    lexical = sentence_lines(wide_store.ask(WIDE_Q, budget=4000))
    assert len(learned) == len(lexical) > ASK_CANDIDATES


def test_learned_candidates_is_the_rule_the_query_path_ranks(
        learned_store, weighted, monkeypatch):
    """The fit path and the query path must select and describe the same rows: one helper
    answers for both, so a drifting shortlist or a drifting column cannot go unnoticed."""
    weighted(only("lex_norm"))
    st = learned_store(encode=unit_vectors)
    seen = spy_features(monkeypatch)
    st.ask(LEARNED_Q, scorer="learned")
    idx, texts, ctx = st.learned_candidates(LEARNED_Q, model=load_default())
    assert texts == seen["cands"]
    assert [st.sentence_index()[0][i] for i in idx] == texts
    assert np.allclose(featurize(LEARNED_Q, texts, ctx), seen["X"])


def test_learned_candidates_follows_the_budget_it_is_given(wide_store):
    assert len(wide_store.learned_candidates(WIDE_Q)[0]) == ASK_CANDIDATES
    assert len(wide_store.learned_candidates(WIDE_Q, budget=4000)[0]) == 150


def test_learned_candidates_normalizes_lengths_by_the_universe_before_a_fit_exists(
        learned_store):
    """No weights yet: the column a fit reads is scaled by the mean length of the universe,
    which is the number the fit then pins."""
    st = learned_store()
    _idx, cands, ctx = st.learned_candidates(LEARNED_Q)
    assert np.allclose(ctx.bm25, bm25_scores([tokenize(t) for t in cands],
                                             st._lexical_scorer().idf_of, tokenize(LEARNED_Q),
                                             avgdl=st.sentence_avgdl()))


def test_learned_candidates_prefers_the_normalizer_the_caller_pins(learned_store):
    st = learned_store()
    _idx, cands, ctx = st.learned_candidates(LEARNED_Q, avgdl=7.5)
    assert np.allclose(ctx.bm25, bm25_scores([tokenize(t) for t in cands],
                                             st._lexical_scorer().idf_of, tokenize(LEARNED_Q),
                                             avgdl=7.5))


def test_sentence_avgdl_is_the_mean_length_of_the_whole_universe(learned_store):
    st = learned_store()
    def mean():
        return corpus_avgdl([tokenize(t) for t in st.sentence_index()[0]])

    assert st.sentence_avgdl() == pytest.approx(mean())
    st.ingest_text("Short.", "u")                       # the universe grew: the divisor follows
    assert st.sentence_avgdl() == pytest.approx(mean())
