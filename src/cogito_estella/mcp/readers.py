"""Documents to plain text: txt/md verbatim, arXiv/LaTeXML-aware HTML, optional PDF."""
from __future__ import annotations

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
    whole document (a bare fragment with neither tag). Scope is decided from the
    parse (real tag depth), never from a raw substring search — text that merely
    looks like a tag inside <script>/<style> CDATA or an HTML comment never
    reaches handle_starttag, so it cannot fake article/body scope."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth_article = 0
        self.depth_body = 0
        self.skip: list[str] = []
        # (text, in_article, in_body) at the moment each chunk was captured
        self.chunks: list[tuple[str, bool, bool]] = []

    def _trigger(self, tag, attrs) -> bool:
        tokens = (dict(attrs).get("class") or "").split()
        return tag in _DROP_TAGS or any(c in _DROP_CLASSES for c in tokens)

    def _append(self, text: str) -> None:
        self.chunks.append((text, self.depth_article > 0, self.depth_body > 0))

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.depth_article += 1
        if tag == "body":
            self.depth_body += 1
        trigger = self._trigger(tag, attrs)
        if tag in _BLOCK and not self.skip and not trigger:
            self._append("\n")
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
        if not self.skip:
            self._append(data)


def html_to_text(raw: str) -> str:
    parser = _Extractor()
    parser.feed(raw)
    article = [t for t, in_a, _ in parser.chunks if in_a]
    body = [t for t, _, in_b in parser.chunks if in_b]
    chosen = article or body or [t for t, _, _ in parser.chunks]
    text = "".join(chosen).replace("\xa0", " ")
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
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in (".html", ".htm"):
        return html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
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
