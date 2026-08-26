import pytest

pytestmark = pytest.mark.integration


def test_segment_prose_spanish():
    from cogito_estella.segmenter import Segmenter

    seg = Segmenter()
    text = ("La inteligencia artificial ha avanzado mucho en los últimos años. "
            "Los modelos de lenguaje son cada vez más capaces y eficientes. "
            "Sin embargo, todavía quedan muchos retos por resolver.")
    units = seg.segment(text)
    assert len(units) == 3
    assert all(len(u) >= 30 for u in units)


def test_segment_batch_returns_per_document_lists():
    from cogito_estella.segmenter import Segmenter

    seg = Segmenter()
    docs = [
        "Primera oración de prueba con longitud suficiente. Segunda oración igualmente larga aquí.",
        "Un documento distinto con su propia oración de prueba suficientemente larga.",
    ]
    out = seg.segment_batch(docs)
    assert len(out) == 2
    assert len(out[0]) == 2
    assert len(out[1]) == 1
