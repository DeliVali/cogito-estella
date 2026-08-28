import pytest

from cogito_estella import metrics as m


def test_normalize_ws():
    assert m.normalize_ws("  hola \n mundo\t") == "hola mundo"


def test_exact_match_ignores_whitespace():
    assert m.exact_match("hola  mundo", "hola mundo")
    assert not m.exact_match("hola mundo", "hola mundos")


def test_chrf_identical_is_100():
    assert m.chrf("the cat sat", "the cat sat") == pytest.approx(100.0)


def test_chrf_disjoint_is_low():
    assert m.chrf("aaaa bbbb", "zzzz yyyy") < 10.0


def test_number_fidelity_all_preserved():
    assert m.number_fidelity("pesa 3.5 kg y mide 40 cm", "pesa 3.5 kg, mide 40 cm") == 1.0


def test_number_fidelity_partial():
    assert m.number_fidelity("entre 10 y 20", "entre 10 y 200") == pytest.approx(0.5)


def test_number_fidelity_none_when_no_numbers():
    assert m.number_fidelity("sin cifras aquí", "nada") is None


def test_json_valid():
    assert m.json_valid('{"a": 1}')
    assert not m.json_valid('{"a": }')


def test_json_equiv_key_order_irrelevant():
    assert m.json_equiv('{"a": 1, "b": [2]}', '{"b": [2], "a": 1}')
    assert not m.json_equiv('{"a": 1}', '{"a": 2}')


def test_summarize():
    s = m.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["median"] == 3.0
    assert s["n"] == 5
    assert s["p10"] <= s["median"] <= s["p90"]
