from cogito_estella import preprocess as pp


def test_sanitize_code_removes_python_comments_and_docstrings():
    code = '''# comentario de linea
def foo(x):
    """Docstring extenso
    con varias lineas."""
    y = x + 1  # comentario inline
    return y


# codigo muerto comentado
'''
    out = pp.sanitize_code(code, lang="python")
    assert "comentario" not in out
    assert "Docstring" not in out
    assert "def foo(x):" in out and "return y" in out
    # sin lineas en blanco de sobra (max 1 consecutiva)
    assert "\n\n\n" not in out


def test_sanitize_code_multilang_c_style():
    code = "int main() { // comentario\n  /* bloque */\n  return 0;\n}"
    out = pp.sanitize_code(code, lang="c")
    assert "comentario" not in out and "bloque" not in out
    assert "return 0;" in out


def test_anonymize_secrets():
    t = "contacta a juan@example.com con la clave sk-abc123DEF456ghi789JKL012mno y AKIA1234567890ABCDEF"
    out = pp.anonymize_secrets(t)
    assert "juan@example.com" not in out and "<EMAIL>" in out
    assert "sk-abc123" not in out and "<SECRET>" in out
    assert "AKIA1234567890ABCDEF" not in out


def test_space_digits():
    assert pp.space_digits("limit 400 y id 42") == "limit 4 0 0 y id 4 2"
    assert pp.space_digits("sin numeros") == "sin numeros"


def test_clean_prose_rejects_corrupt():
    corrupt = "texto ������ corrupto"  # muchos replacement chars
    assert pp.clean_prose(corrupt) is None


def test_clean_prose_rejects_incomplete_short():
    assert pp.clean_prose("hola") is None  # demasiado corto


def test_clean_prose_keeps_good_and_strips_control():
    good = "Esta es una oración válida y completa en español.\x00\x07"
    out = pp.clean_prose(good)
    assert out is not None
    assert "\x00" not in out and "\x07" not in out
    assert "oración válida" in out


def test_preprocess_record_dispatch():
    from cogito_estella.multilingual_factory import DocRecord
    code_rec = DocRecord("# c\nx = 1\n", "eng_Latn", "code_py", "code")
    out = pp.preprocess_record(code_rec)
    assert out is not None and "# c" not in out.text and "x = 1" in out.text
    bad = DocRecord("ab", "eng_Latn", "en", "prose")
    assert pp.preprocess_record(bad) is None  # prosa muy corta -> descartada
