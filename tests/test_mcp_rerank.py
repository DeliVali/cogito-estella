"""Learned relevance scorer: feature extraction, BM25, logistic model, weights file."""
import json
import math

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
    featurize,
    is_quantity_question,
    load_default,
)

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
    weak = RelevanceScorer().fit(X, y, l2=0.1)
    strong = RelevanceScorer().fit(X, y, l2=1000.0)
    assert np.abs(strong.w).sum() < np.abs(weak.w).sum()


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
def test_json_round_trip_preserves_the_scores(tmp_path):
    X, y = separable()
    sc = RelevanceScorer().fit(X, y)
    sc.trained_on = "fixture"
    sc.cv = {"learned_recall_at_budget": 0.9, "lexical_recall_at_budget": 0.8, "folds": []}
    path = tmp_path / "w.json"
    sc.to_json(path)
    back = RelevanceScorer.from_json(path)
    assert back.score(X) == pytest.approx(sc.score(X))
    assert back.trained_on == "fixture" and back.cv["folds"] == []


def test_to_json_serializes_array_scalars_left_in_the_report(tmp_path):
    X, y = separable()
    sc = RelevanceScorer().fit(X, y)
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
    RelevanceScorer().fit(X, y).to_json(path)
    blob = json.loads(path.read_text())
    assert set(blob) >= {"features", "mean", "std", "w", "b", "trained_on", "cv"}
    assert blob["features"] == list(FEATURES)
    assert len(blob["mean"]) == len(blob["std"]) == len(blob["w"]) == 12


def test_from_json_rejects_other_feature_names(tmp_path):
    X, y = separable()
    path = tmp_path / "w.json"
    RelevanceScorer().fit(X, y).to_json(path)
    blob = json.loads(path.read_text())
    blob["features"][0] = "something_else"
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="feature"):
        RelevanceScorer.from_json(path)


def test_from_json_rejects_a_truncated_vector(tmp_path):
    X, y = separable()
    path = tmp_path / "w.json"
    RelevanceScorer().fit(X, y).to_json(path)
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
    sc = RelevanceScorer().fit(X, y)
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


def test_load_default_returns_none_on_a_file_with_other_features(tmp_path, monkeypatch):
    X, y = separable()
    path = tmp_path / "w.json"
    RelevanceScorer().fit(X, y).to_json(path)
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
