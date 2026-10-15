"""Documents to plain text: any text file verbatim (code, config, Dockerfile...), arXiv/LaTeXML-aware
HTML, optional PDF. Binary files are recognized by content, not by extension."""
from __future__ import annotations

import json
import re
import zipfile
from email import message_from_bytes, policy
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

OFFICE = {".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods", ".epub", ".rtf", ".eml", ".ipynb"}
SUFFIXES = {".txt", ".md", ".html", ".htm", ".xhtml", ".pdf", ".doc"} | OFFICE   # always read
MAX_BYTES = 2_000_000           # a directory walk skips larger files (lock files, dumps, data)
# never walked: dependencies, caches, build output; hidden entries (.git, .env) are skipped too,
# except the hidden folders that hold project configuration
SKIP_DIRS = {"node_modules", "__pycache__", "venv", "dist", "build", "target", "site-packages"}
KEEP_HIDDEN = {".github", ".gitlab", ".circleci", ".devcontainer"}
# prose is split into sentences; every other text file (code, config) into lines
PROSE = {".txt", ".md", ".html", ".htm", ".xhtml", ".pdf", ".rst", ".tex", ".org", ".adoc",
         ".docx", ".pptx", ".odt", ".odp", ".epub", ".rtf", ".eml"}
MAX_MEMBER = 50_000_000         # bytes read from one member of an office archive (zip bombs)


CODE_NAMES = ("dockerfile", "containerfile", "makefile", "jenkinsfile", "procfile", "gemfile",
              "vagrantfile", "justfile")


def is_prose(name: str) -> bool:
    """Raw text and extensionless notes read as prose; a code or config file does not."""
    p = Path(name)
    if p.name.lower().startswith(CODE_NAMES):
        return False
    return not p.suffix or p.suffix.lower() in PROSE
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


def is_text(path: Path) -> bool:
    """No NUL byte and valid UTF-8 in the first 8 KiB: the test git and grep use for binaries."""
    try:
        with open(path, "rb") as f:
            head = f.read(8192)
    except OSError:
        return False
    if b"\0" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:            # a multi-byte character cut at the boundary
        return exc.start >= len(head) - 3
    return True


def _readable(path: Path) -> bool:
    return path.suffix.lower() in SUFFIXES or is_text(path)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml(z: zipfile.ZipFile, name: str):
    info = z.getinfo(name)
    if info.file_size > MAX_MEMBER:
        raise ReaderError(f"{name} is larger than {MAX_MEMBER} bytes")
    return ElementTree.fromstring(z.read(name))


def _paragraphs(root, para: set[str], text: str, breaks=("tab", "br", "s")) -> str:
    """One line per paragraph element: Word/ODF/PowerPoint keep text in runs inside them."""
    out = []
    for el in root.iter():
        if _local(el.tag) in para:
            parts = []
            for x in el.iter():
                name = _local(x.tag)
                if name == text and x.text:
                    parts.append(x.text)
                elif name in breaks:
                    parts.append(" ")
                if x is not el and name in para:
                    break
            line = "".join(parts).strip()
            if line:
                out.append(line)
    return "\n".join(out)


def _odf_text(root) -> str:
    out = []
    for el in root.iter():
        if _local(el.tag) in ("p", "h"):
            line = " ".join("".join(el.itertext()).split())
            if line:
                out.append(line)
    return "\n".join(out)


def _slide_key(name: str) -> int:
    m = re.search(r"(\d+)\.xml$", name)
    return int(m.group(1)) if m else 0


def _xlsx_text(z: zipfile.ZipFile) -> str:
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        shared = ["".join(si.itertext()) for si in _xml(z, "xl/sharedStrings.xml")
                  if _local(si.tag) == "si"]
    lines = []
    for name in sorted((n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)),
                       key=_slide_key):
        for row in _xml(z, name).iter():
            if _local(row.tag) != "row":
                continue
            cells = []
            for c in row:
                if _local(c.tag) != "c":
                    continue
                v = next((x.text for x in c.iter() if _local(x.tag) in ("v", "t") and x.text), None)
                if v is None:
                    continue
                if c.get("t") == "s" and v.isdigit() and int(v) < len(shared):
                    v = shared[int(v)]
                cells.append(v)
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def _epub_text(z: zipfile.ZipFile) -> str:
    container = _xml(z, "META-INF/container.xml")
    opf = next(el.get("full-path") for el in container.iter() if _local(el.tag) == "rootfile")
    root = _xml(z, opf)
    base = opf.rsplit("/", 1)[0] + "/" if "/" in opf else ""
    items = {el.get("id"): el.get("href") for el in root.iter() if _local(el.tag) == "item"}
    order = [el.get("idref") for el in root.iter() if _local(el.tag) == "itemref"]
    parts = []
    for idref in order:
        href = items.get(idref)
        if href and base + href in z.namelist():
            parts.append(html_to_text(z.read(base + href).decode("utf-8", "ignore")))
    return "\n\n".join(p for p in parts if p)


def rtf_to_text(raw: str) -> str:
    """Plain text of an RTF file: groups that hold no body text are dropped, control words
    removed, escaped characters decoded."""
    raw = re.sub(r"\{\\\*[^{}]*\}", "", raw)
    raw = re.sub(r"\{\\(?:fonttbl|colortbl|stylesheet|info|pict)[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", raw)
    raw = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: bytes.fromhex(m.group(1)).decode("cp1252"), raw)
    raw = re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m.group(1)) % 65536), raw)
    raw = re.sub(r"\\(?:par|line)\b ?", "\n", raw)
    raw = re.sub(r"\\tab\b ?", "\t", raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", raw)
    raw = raw.replace("\\{", "{").replace("\\}", "}").replace("\\\\", "\\")
    raw = raw.replace("{", "").replace("}", "")
    return re.sub(r"\n{3,}", "\n\n", raw).strip()


def eml_to_text(data: bytes) -> str:
    msg = message_from_bytes(data, policy=policy.default)
    head = "\n".join(f"{k}: {msg[k]}" for k in ("Subject", "From", "To", "Date") if msg[k])
    body = msg.get_body(preferencelist=("plain", "html"))
    text = ""
    if body is not None:
        text = body.get_content()
        if body.get_content_type() == "text/html":
            text = html_to_text(text)
    return f"{head}\n\n{text}".strip()


def ipynb_to_text(raw: str) -> str:
    cells = json.loads(raw).get("cells", [])
    return "\n\n".join("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                        else c.get("source", "") for c in cells).strip()


def office_to_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".rtf":
            return rtf_to_text(path.read_text(encoding="latin-1"))
        if suffix == ".eml":
            return eml_to_text(path.read_bytes())
        if suffix == ".ipynb":
            return ipynb_to_text(path.read_text(encoding="utf-8", errors="ignore"))
        with zipfile.ZipFile(path) as z:
            if suffix == ".docx":
                return _paragraphs(_xml(z, "word/document.xml"), {"p"}, "t")
            if suffix == ".pptx":
                slides = sorted((n for n in z.namelist()
                                 if re.match(r"ppt/slides/slide\d+\.xml$", n)), key=_slide_key)
                return "\n\n".join(_paragraphs(_xml(z, n), {"p"}, "t") for n in slides)
            if suffix == ".xlsx":
                return _xlsx_text(z)
            if suffix == ".epub":
                return _epub_text(z)
            return _odf_text(_xml(z, "content.xml"))          # odt, odp, ods
    except ReaderError:
        raise
    except Exception as exc:                      # zip, XML, JSON and mail parsers raise many types
        raise ReaderError(f"cannot read {path.name}: {exc}") from exc


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in OFFICE:
        return office_to_text(path)
    if suffix == ".doc":
        raise ReaderError(f"legacy Word file {path.name}: save it as .docx")
    if suffix in (".html", ".htm", ".xhtml"):
        return html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        return pdf_to_text(path)
    if suffix in SUFFIXES or is_text(path):
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ReaderError(f"binary file, not indexed: {path.name}")


def _dir_files(path: Path) -> list[Path]:
    out = []
    for p in path.rglob("*"):
        parts = p.relative_to(path).parts
        if any((x.startswith(".") and x not in KEEP_HIDDEN) or x in SKIP_DIRS for x in parts) \
                or not p.is_file():
            continue
        if p.stat().st_size <= MAX_BYTES and _readable(p):
            out.append(p)
    return sorted(out)


def source_names(path: Path) -> list[str]:
    """The source names `iter_sources` would yield for `path`."""
    path = Path(path)
    if path.is_dir():
        return [p.relative_to(path).as_posix() for p in _dir_files(path)]
    return [str(path)] if path.is_file() and _readable(path) else []


def iter_sources(path: Path):
    """Yield (source_name, text) or (source_name, ReaderError) per document.
    A directory is walked recursively in sorted order, names relative to it;
    a single file (or a missing path) yields under its full path. Every
    per-document failure is isolated as a yielded ReaderError, never raised —
    including an unsupported suffix or a missing path on a single-file call."""
    path = Path(path)
    if path.is_dir():
        for p in _dir_files(path):
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
