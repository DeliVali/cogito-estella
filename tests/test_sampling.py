from cogito_estella import sampling as s


def test_synthetic_numbers_deterministic_and_sized():
    a = s.synthetic_numbers(10, seed=0)
    b = s.synthetic_numbers(10, seed=0)
    assert [x.text for x in a] == [x.text for x in b]
    assert len(a) == 10
    assert all(x.category == "numeros" for x in a)
    assert all(any(ch.isdigit() for ch in x.text) for x in a)


def test_synthetic_json_tools_are_valid_json():
    import json

    for sample in s.synthetic_json_tools(8, seed=1):
        json.loads(sample.text)
        assert sample.category == "json_tools"


def test_split_sentences_basic():
    text = "Hola mundo. Esto es una prueba! ¿Funciona bien? Sí."
    parts = s.split_sentences(text)
    assert len(parts) == 4
    assert parts[0] == "Hola mundo."


def test_good_unit_bounds():
    assert not s.good_unit("corto")
    assert s.good_unit("Esta es una oración de longitud perfectamente razonable para SONAR.")
    assert not s.good_unit("x" * 300)


def test_sample_local_code_from_repo(tmp_path):
    (tmp_path / "a.py").write_text("def f(x):\n    return x + 1\n\n\ndef g(y):\n    return y * 2\n")
    out = s.sample_local_code(str(tmp_path), n=2, seed=0)
    assert len(out) == 2
    assert all(o.category == "codigo" for o in out)
