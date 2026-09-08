"""Document readers: plain text, markdown, LaTeXML/arXiv HTML, optional PDF, directories."""
import builtins

import pytest

from cogito_estella.mcp.readers import ReaderError, html_to_text, read_source
from cogito_estella.mcp.store import GraphStore

HTML = """<html><body><nav>Report issue</nav>
<article class="ltx_document ltx_authors_1line">
<h1 class="ltx_title">Large Concept Models</h1>
<div class="ltx_authors"><span class="ltx_personname">Jane Doe</span>
<span class="ltx_role_affiliation">Affiliation: FAIR</span></div>
<span class="ltx_pubnotes">preprint</span>
<div class="ltx_abstract"><p>LLMs process&nbsp;tokens. <math>x</math>We build concepts.</p></div>
<section><p>Concepts are decoded by SONAR.</p>
<figure><figcaption>Figure 1</figcaption></figure>
<table><tr><td>cell</td></tr></table>
<p>Second paragraph.</p></section>
<section class="ltx_bibliography"><p>Ref 1</p></section>
</article><footer>Bibliographic Explorer</footer></body></html>"""


def test_html_keeps_article_prose_and_drops_front_matter_and_noise():
    text = html_to_text(HTML)
    assert text.startswith("Large Concept Models")
    assert "Affiliation" not in text and "Jane Doe" not in text and "preprint" not in text
    for gone in ("Report issue", "Bibliographic Explorer", "Ref 1", "cell", "Figure 1"):
        assert gone not in text
    assert "<math>" not in text and "x</math>" not in text
    assert "LLMs process tokens." in text and "\xa0" not in text
    assert "Concepts are decoded by SONAR." in text and "Second paragraph." in text


def test_html_without_article_falls_back_to_body():
    assert html_to_text("<html><body><p>Only body.</p></body></html>") == "Only body."


def test_html_without_article_keeps_only_body():
    html = ("<html><head><title>Page Title Should Not Appear</title></head>"
            "<body><p>Only body.</p></body></html>")
    assert html_to_text(html) == "Only body."


def test_html_fragment_without_body_is_kept():
    assert html_to_text("<p>Only body.</p>") == "Only body."


def test_read_source_txt_md_html(tmp_path):
    (tmp_path / "a.txt").write_text("Plain text.")
    (tmp_path / "b.md").write_text("# Title\n\nBody.")
    (tmp_path / "c.html").write_text(HTML)
    out = dict(read_source(tmp_path))
    assert out == {"a.txt": "Plain text.", "b.md": "# Title\n\nBody.",
                   "c.html": html_to_text(HTML)}
    assert read_source(tmp_path / "a.txt") == [(str(tmp_path / "a.txt"), "Plain text.")]


def test_directory_walk_is_recursive_sorted_and_skips_other_suffixes(tmp_path):
    (tmp_path / "z.txt").write_text("z")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_text("a")
    (tmp_path / "skip.py").write_text("print(1)")
    assert [n for n, _ in read_source(tmp_path)] == ["sub/a.md", "z.txt"]


def test_pdf_without_pypdf_raises_reader_error(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pypdf":
            raise ImportError("no pypdf")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(ReaderError, match=r"cogito-estella\[pdf\]"):
        read_source(tmp_path / "x.pdf")


def test_unknown_suffix_raises(tmp_path):
    (tmp_path / "x.py").write_text("x")
    with pytest.raises(ReaderError, match="unsupported"):
        read_source(tmp_path / "x.py")


def test_ingest_path_reports_per_document_and_errors(tmp_path, fake_extractor):
    (tmp_path / "a.txt").write_text("The encoder maps text to a vector.")
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 broken")
    st = GraphStore(extractor=fake_extractor(
        {"The encoder maps text to a vector.": [("encoder", "give", "text")]}))
    out = st.ingest_path(tmp_path)
    lines = [o if isinstance(o, str) else o.line() for o in out]
    assert any(ln.startswith("source=a.txt status=ingested") for ln in lines)
    assert any(ln.startswith("source=bad.pdf error=") for ln in lines)


def test_ingest_path_single_bad_file_returns_error_line(tmp_path, fake_extractor):
    st = GraphStore(extractor=fake_extractor({}))
    bad = tmp_path / "x.py"
    bad.write_text("x")
    out = st.ingest_path(bad)
    assert len(out) == 1
    assert out[0].startswith(f"source={bad} error=unsupported")

    missing = tmp_path / "does-not-exist.txt"
    out = st.ingest_path(missing)
    assert len(out) == 1
    assert out[0].startswith(f"source={missing} error=no such file")


# -- finding 2: readers decode UTF-8 explicitly, not the locale encoding ---------------

def test_readers_decode_utf8(tmp_path):
    text = "café — naïve"
    (tmp_path / "a.txt").write_bytes(text.encode("utf-8"))
    assert read_source(tmp_path / "a.txt") == [(str(tmp_path / "a.txt"), text)]


# -- finding 4: <article>/<body> scope comes from the parse, not a raw substring -------

def test_article_like_text_inside_script_does_not_trigger_article_scope():
    html = "<script>var s='<article>';</script><p>Real body.</p>"
    assert html_to_text(html) == "Real body."


def test_article_like_text_inside_comment_does_not_trigger_article_scope():
    html = "<!-- <article> --><p>Body</p>"
    assert html_to_text(html) == "Body"


# -- finding 15: convert_charrefs already unescapes; no second unescape pass -----------

def test_html_to_text_does_not_double_unescape_entities():
    assert html_to_text("<p>&amp;lt;</p>") == "&lt;"
