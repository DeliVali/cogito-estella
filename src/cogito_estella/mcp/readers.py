"""Documents to plain text: txt/md verbatim, arXiv/LaTeXML-aware HTML, optional PDF."""
from __future__ import annotations

import html as htmllib
import re
from html.parser import HTMLParser
from pathlib import Path

SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf"}
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
         "param", "source", "track", "wbr"}
_BLOCK = {"p", "div", "section", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr",
          "table", "figure", "article", "blockquote"}
_DROP_TAGS = {"script", "style", "table", "figure", "math"}
# Exact class tokens; never substring-match: <article> itself carries "ltx_authors_1line".
_DROP_CLASSES = {"ltx_bibliography", "ltx_note_outer", "ltx_authors", "ltx_pubnotes"}


class ReaderError(Exception):
    pass


class _Extractor(HTMLParser):
    """Collects prose inside <article> when present, else inside <body>, else the
    whole document (a bare fragment with neither tag)."""

    def __init__(self, article_mode: bool, body_mode: bool):
        super().__init__(convert_charrefs=True)
        self.article_mode = article_mode
        self.body_mode = body_mode
        self.depth_article = 0
        self.depth_body = 0
        self.skip: list[str] = []
        self.chunks: list[str] = []

    def _trigger(self, tag, attrs) -> bool:
        tokens = (dict(attrs).get("class") or "").split()
        return tag in _DROP_TAGS or any(c in _DROP_CLASSES for c in tokens)

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.depth_article += 1
        if tag == "body":
            self.depth_body += 1
        trigger = self._trigger(tag, attrs)
        if tag in _BLOCK and not self.skip and not trigger:
            self.chunks.append("\n")
        if tag not in _VOID and (self.skip or trigger):
            self.skip.append(tag)

    def handle_endtag(self, tag):
        if tag == "article":
            self.depth_article -= 1
        if tag == "body":
            self.depth_body -= 1
        if self.skip and self.skip[-1] == tag:
            self.skip.pop()

    def handle_data(self, data):
        if self.article_mode:
            inside = self.depth_article > 0
        elif self.body_mode:
            inside = self.depth_body > 0
        else:
            inside = True
        if inside and not self.skip:
            self.chunks.append(data)


def html_to_text(raw: str) -> str:
    lower = raw.lower()
    parser = _Extractor(article_mode="<article" in lower, body_mode="<body" in lower)
    parser.feed(raw)
    text = htmllib.unescape("".join(parser.chunks)).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_text(path: Path) -> str:
    try:
        import pypdf
    except ImportError as exc:
        raise ReaderError("PDF support needs: pip install 'cogito-estella[pdf]'") from exc
    try:
        reader = pypdf.PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as exc:                      # pypdf raises many concrete types
        raise ReaderError(f"cannot read PDF {path.name}: {exc}") from exc


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(errors="ignore")
    if suffix in (".html", ".htm"):
        return html_to_text(path.read_text(errors="ignore"))
    if suffix == ".pdf":
        return pdf_to_text(path)
    raise ReaderError(f"unsupported file type {suffix!r} ({path.name}); "
                      f"supported: {', '.join(sorted(SUFFIXES))}")


def iter_sources(path: Path):
    """Yield (source_name, text) or (source_name, ReaderError) per document.
    A directory is walked recursively in sorted order, names relative to it;
    a single file (or a missing path) yields under its full path. Every
    per-document failure is isolated as a yielded ReaderError, never raised —
    including an unsupported suffix or a missing path on a single-file call."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES)
        for p in files:
            name = p.relative_to(path).as_posix()
            try:
                yield name, _read_file(p)
            except ReaderError as exc:
                yield name, exc
        return
    name = str(path)
    try:
        if not path.is_file():
            raise ReaderError(f"no such file or directory: {path}")
        yield name, _read_file(path)
    except ReaderError as exc:
        yield name, exc


def read_source(path: Path) -> list[tuple[str, str]]:
    """Strict variant of iter_sources: raises on the first unreadable document."""
    out = []
    for name, item in iter_sources(path):
        if isinstance(item, ReaderError):
            raise item
        out.append((name, item))
    return out
