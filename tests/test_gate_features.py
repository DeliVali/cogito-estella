import numpy as np

from cogito_estella import gate_features as gf


def test_feature_vector_length_matches_names():
    v = gf.surface_vector("Hola mundo, esto es una prueba.")
    assert v.shape == (len(gf.FEATURE_NAMES),)
    assert v.dtype == np.float32


def test_digit_ratio():
    feats = gf.surface_features("abc123")
    assert feats["digit_ratio"] == 0.5


def test_brace_and_bracket_and_quote_counts_normalized():
    feats = gf.surface_features('{"a": [1]}')
    # normalizados por longitud → todos > 0 cuando existen
    assert feats["brace_ratio"] > 0
    assert feats["bracket_ratio"] > 0
    assert feats["quote_ratio"] > 0


def test_plain_prose_has_no_structural_chars():
    feats = gf.surface_features("Una oración simple sin símbolos estructurales")
    assert feats["brace_ratio"] == 0.0
    assert feats["bracket_ratio"] == 0.0


def test_empty_text_is_safe():
    feats = gf.surface_features("")
    assert feats["digit_ratio"] == 0.0
    assert feats["length"] == 0.0
