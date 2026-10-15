"""Knowledge graph as an index: edges with span provenance, exact text on demand.
Edge ids are stable and never reused; documents are deduplicated by content hash."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cogito_estella.mcp.rank import LexicalScorer, SonarScorer, rank, singular_forms, tokenize
from cogito_estella.mcp.readers import is_prose
from cogito_estella.mcp.rerank import (
    FeatureContext,
    bigrams,
    bm25_scores,
    corpus_avgdl,
    featurize,
    load_default,
)
from cogito_estella.mcp.tokens import Ledger, ntok

BATCH = 64
DIVIDER = "~ class-only, verify with provenance:"
FACT_SHARE = 0.4                # of `ask`'s budget; the sentences take the rest
ASK_BUDGET = 600                # default token budget of one `ask`
ASK_ENTITIES = 3                # entities resolved from one question
ASK_MAX_SKIPS = 32              # consecutive oversize sentences before the fill stops
ASK_SCORERS = ("lexical", "dense", "learned")       # the scorer contract of `ask`
SCORER_ALIASES = {"sonar": "dense"}                 # older name of the dense ranking
ASK_CANDIDATES = 60             # smallest lexical shortlist the learned scorer re-orders
ASK_CAND_TOKENS = 10            # budget per extra shortlist slot: the block must not outrun it
ASK_MAX_IDS = 8                 # ids per fact line in `ask`: one hub group must not eat the cap
DENSE_COS_FLOOR = 0.2           # below this cosine a sentence is not about the question
DENSE_FLOOR = (DENSE_COS_FLOOR + 1.0) / 2.0        # the same floor on the scorer's [0, 1] scale
DENSE_OFFSET = 1.0              # embedded rows rank above lexical ones: no shared origin
NEIGHBOR_TOP = 2                # best sentences whose successor is pulled up with them
NEIGHBOR_WEIGHT = 0.9           # successor score as a share of its predecessor's
HOP_LINES = 2                   # best prose lines whose new names start a second search
HOP_NAMES = 4                   # names carried into it
_CAPS = re.compile(r"\b[A-Z][\w'\-]*(?:\s+(?:of|the|de|and|&)?\s*[A-Z][\w'\-]*)*")
CODE_NEIGHBORS = 2              # lines after a code hit pulled up with it
# a sentence ending here is cut mid-sentence: the splitter joins it to the next one
_ABBREV_END = re.compile(r"(?:\bet al|\be\.g|\bi\.e|\bFig|\bTab|\bEq|\bvs|\bcf|\bSec|\bNo)\.[\],)]*$")
VECTOR_ONLY_BELOW = 0.5        # share of question words in the corpus under which vectors rank alone
LOW_CONFIDENCE = 0.3            # learned p= under which the reply probably misses the answer
ASK_STYLES = ("full", "compact")
# compact replies speak plain language at the point of use: a small model follows a line
# placed next to the evidence, not a policy buried in a tool description
COMPACT_HEAD = "From the indexed documents, best match first:"
COMPACT_LOW = ("Low confidence: these lines may not answer the question. If none of them "
               "states the answer, reply that the documents do not say.")
COMPACT_WHOLE = "All of the indexed text fits here ({docs} document{s}):"
COMPACT_NONE = "Nothing in the indexed documents matches this question."
SECTION_DEPTH = 2               # heading levels a compact line names; 0 names none
SECTION_WIDTH = 40              # characters kept of one heading
SECTION_SEP = " › "
SONAR_COS_FLOOR, SONAR_FLOOR, SONAR_OFFSET = DENSE_COS_FLOOR, DENSE_FLOOR, DENSE_OFFSET   # older names
# question words too generic to be worth a graph hop
_GENERIC = ("model models paper use uses used result results work approach method "
            "methods data table figure")
ASK_STOPLIST = frozenset(_GENERIC.split())
_QWORD = re.compile(r"[a-z0-9][a-z0-9\-]*")


class _StdoutWire:
    """One reference-counted redirect of the process-global `sys.stdout`. Two plain
    `redirect_stdout` blocks overlapping in different threads restore it out of order:
    the wire returns while the other block still prints on it (corrupting the JSON-RPC
    framing) and is then lost for the rest of the process. Here the first block in swaps
    and the last one out restores, so blocks may overlap without ever nesting."""

    def __init__(self):
        self._lock = threading.Lock()      # held only across the swap, never across a load
        self._depth = 0
        self._wire = None

    @contextmanager
    def quiet(self):
        with self._lock:
            if not self._depth:
                self._wire, sys.stdout = sys.stdout, sys.stderr
            self._depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._depth -= 1
                if not self._depth:
                    sys.stdout, self._wire = self._wire, None


_quiet = _StdoutWire().quiet               # every stdout redirect in this process goes here


class TextOnlyExtractor:
    """`--lite`: the sentence index without the graph model. Sentences are split with
    spaCy and no encoder or decoder head is loaded, so a small session starts at once and
    downloads nothing; `ask` still ranks with the learned scorer, whose features include
    the absence of an encoder cosine. Documents indexed this way carry no facts, and are
    extracted again the next time a full server ingests them."""

    lite = True
    encoder_name = None

    def _scanner(self):
        # spaCy's rule-based sentencizer: no model to download, and on 1,140 SQuAD questions
        # the same answers reach `ask` as with en_core_web_sm (96.2 % both), indexing 15x
        # faster; the regex fallback lost 1.5 points
        if not hasattr(self, "_nlp"):
            try:
                import spacy
                self._nlp = spacy.blank("en")
                self._nlp.add_pipe("sentencizer")
            except ImportError:
                self._nlp = False
        return self._nlp

    def extract_batch_with_provenance(self, texts, doc_offsets=None):
        return [[] for _ in texts]


LONG_SENTENCE = 150             # tokens above which a lite "sentence" is cut into pieces
CHUNK = 80                      # target tokens of each piece


def _chunks(sentence: str) -> list[str]:
    """A sentence over LONG_SENTENCE tokens in pieces of about CHUNK tokens, cut at spaces
    where the script has them and between characters where it does not. Counted in tokens,
    not characters: a Thai or Chinese character costs several times an English one, and a
    piece that outgrows the reply budget is dropped whole."""
    if len(sentence) < 2 * LONG_SENTENCE or ntok(sentence) <= LONG_SENTENCE:
        return [sentence]
    width = max(1, len(sentence) * CHUNK // ntok(sentence))   # chars per piece, this text
    out, cur = [], ""
    for word in sentence.split(" "):
        while len(word) > width:                   # an unspaced run longer than a piece
            if cur:
                out.append(cur)
                cur = ""
            out.append(word[:width])
            word = word[width:]
        if cur and len(cur) + 1 + len(word) > width:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        out.append(cur)
    return out


def _full_line(src: str, si: int, text: str) -> str:
    return f'{src} s{si}: "{text}"'


def _compact_line(src: str, text: str, section: str = "") -> str:
    # one passage per line, every line tagged: a sentence spanning a markdown table or a
    # bullet list would otherwise continue on untagged lines
    where = f"{src} § {section}" if section else src
    return f"[{where}] {' '.join(_drop_shown_headings(text, section).split())}"


def _drop_shown_headings(text: str, section: str) -> str:
    """A sentence that opens with the heading its tag already names says it twice: spaCy
    keeps `## Setup` inside the first sentence under it."""
    shown = set(section.split(SECTION_SEP)) if section else set()
    while shown:
        first, _, rest = text.lstrip().partition("\n")
        if " ".join(first.lstrip("#").translate(_MARKUP).split()) not in shown or not rest:
            break
        text = rest
    return text


_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")
_MARKUP = str.maketrans("", "", "[]*`_")


def section_paths(text: str, offsets: list[int]) -> list[str]:
    """The markdown heading path each sentence sits under, "" outside any heading; the
    offsets are the sentences' starts, in document order.

    A sentence ranked on its own loses the section that gives it meaning: "networkx added
    to the core dependencies" says nothing about the version unless the reader sees it sits
    under `## [0.9.0]`. A lone level-1 heading is the document's title, not a section, and is
    left out and fenced code is skipped. The whole path is stored; `shown_section` trims it."""
    heads: list[tuple[int, int, str]] = []
    pos, fenced = 0, False
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if _FENCE.match(bare):
            fenced = not fenced
        elif not fenced and (m := _HEADING.match(bare)):
            title = " ".join(m.group(2).translate(_MARKUP).split())
            if title:
                heads.append((pos, len(m.group(1)), title[:SECTION_WIDTH]))
        pos += len(line)
    if sum(1 for _, level, _ in heads if level == 1) == 1:
        heads = [h for h in heads if h[1] != 1]
    out, stack, k = [], [], 0
    for off in offsets:
        while k < len(heads) and heads[k][0] <= off:
            _, level, title = heads[k]
            stack = [h for h in stack if h[0] < level] + [(level, title)]
            k += 1
        out.append(SECTION_SEP.join(t for _, t in stack))
    return out


def shown_section(path: str) -> str:
    """The deepest SECTION_DEPTH headings of a stored path: the depth is a rendering choice,
    so it changes without re-indexing."""
    if not path or SECTION_DEPTH <= 0:
        return ""
    return SECTION_SEP.join(path.split(SECTION_SEP)[-SECTION_DEPTH:])


@dataclass
class IngestResult:
    source: str
    status: str                 # ingested | unchanged | replaced
    sentences: int
    triples: int
    raw_tokens: int
    graph_tokens: int
    compression: float
    seconds: float

    def line(self) -> str:
        return (f"source={self.source} status={self.status} sentences={self.sentences} "
                f"triples={self.triples} raw_tokens={self.raw_tokens} "
                f"graph_tokens={self.graph_tokens} compression={self.compression} "
                f"seconds={self.seconds}")


@dataclass
class _AskSnapshot:
    """What one `ask` reads from the graph under the lock; scoring then runs outside it."""

    texts: list
    keys: list
    ents: list
    boost: set
    mat: object
    mask: object
    cover: str
    doc_sents: dict
    sections: dict = None       # source -> heading path per sentence (absent on old graphs)

    def section(self, key) -> str:
        src, si = key
        paths = (self.sections or {}).get(src) or ()
        return shown_section(paths[si]) if si < len(paths) else ""

    def compact(self, i: int) -> str:
        return _compact_line(self.keys[i][0], self.texts[i], self.section(self.keys[i]))

    def rel_pos(self, key) -> float:
        """Where a sentence sits in its own document, 0 at the head and 1 at the tail."""
        src, si = key
        return si / max(self.doc_sents.get(src, 1) - 1, 1)


def _line_source(line: str) -> str:
    """`[file § section] text` -> file."""
    return line[1:].split("]", 1)[0].split(" § ", 1)[0]


def _norm_words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _code_graph(text: str, source: str, offsets: list[int], n: int) -> list[list[dict]]:
    """Syntax edges of a code file grouped by line, the shape the extraction loop yields."""
    from cogito_estella.mcp import codegraph
    per: list[list[dict]] = [[] for _ in range(n)]
    for e in codegraph.extract(text, source, offsets):
        per[e.pop("sent_idx")].append(e)
    return per


def _rebuild(sents: list) -> str:
    """Stored lines back at their offsets: each line keeps its column, which Python's
    indentation-based syntax needs to parse again."""
    out, end = [], 0
    for s, o in sents:
        if o > end:
            out.append("\n" + " " * (o - end - 1) if end else " " * o)
        out.append(s)
        end = o + len(s)
    return "".join(out)


def _sections(source: str, text: str, offsets: list[int]) -> list[str]:
    """Heading paths for prose; a code file has none (`# comment` is not a heading)."""
    return section_paths(text, offsets) if is_prose(source) else [""] * len(offsets)


def _code_terms(line: str, path: str) -> str:
    """What a code line is searched by: its text, its identifiers split at camelCase and
    snake_case boundaries (`timeoutSeconds` -> timeout seconds) and its file path words."""
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", line)
    return f"{line} {re.sub(r'[^A-Za-z0-9]+', ' ', parts)} {re.sub(r'[^A-Za-z0-9]+', ' ', path)}"


def _pull_up_successors(scores, keys):
    """The sentence after each of the best ones scores just below it: an answer often sits in
    the next sentence, which opens with a pronoun and shares no word with the question."""
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:NEIGHBOR_TOP]
    where = {k: i for i, k in enumerate(keys)}
    out = list(scores)
    for i in top:
        if scores[i] <= 0:
            break
        # a code line is often a comment or a name whose value sits on the next lines
        reach = 1 if is_prose(keys[i][0]) else CODE_NEIGHBORS
        for d in range(1, reach + 1):
            j = where.get((keys[i][0], keys[i][1] + d))
            if j is not None:
                out[j] = max(out[j], NEIGHBOR_WEIGHT ** d * scores[i])
    return out


class GraphStore:
    def __init__(self, extractor=None, extractor_factory=None, path: Path | None = None,
                 full_factory=None):
        self._ex = extractor
        self._factory = extractor_factory
        self._full_factory = full_factory          # loaded once by use_full(), then kept
        self.path = Path(path) if path else None
        self.edges: dict[int, dict] = {}
        self.next_id = 0
        self.adj: dict[str, list[int]] = defaultdict(list)
        self.docs: dict[str, dict] = {}
        self.emb: dict[str, np.ndarray] = {}       # source -> [n_sents, D] float16, normalized
        self._sindex: tuple | None = None          # flat sentence universe, rebuilt on change
        self._lex: LexicalScorer | None = None     # IDF over that universe, rebuilt with it
        self._avgdl: float | None = None           # mean sentence length of that universe
        self._emb_warned = False
        self._emb_keep_warned = False
        self.emb_encoder: str | None = None       # encoder that produced the sidecar rows
        self._emb_space_warned = False
        self.ledger = Ledger()
        self.persisted_at: str | None = None
        self._lock = threading.RLock()     # sync tools run in worker threads: serialize mutation
        self._ex_lock = threading.Lock()   # one extractor load per store, not one per caller

    @property
    def extractor(self):
        if self._ex is None:
            self._warm_extractor()
        return self._ex

    def _warm_extractor(self) -> None:
        """Materialize the extractor once: two concurrent first calls must not each load
        the encoder. Off the store lock, so a load never serializes the other tools."""
        with self._ex_lock:
            if self._ex is None:
                if self._factory is None:
                    raise RuntimeError("GraphStore needs an extractor or an extractor_factory")
                with _quiet():                     # stdout is the MCP wire
                    self._ex = self._factory()

    @property
    def is_lite(self) -> bool:
        """True while no graph model serves this store (nothing is loaded to find out)."""
        if self._ex is not None:
            return bool(getattr(self._ex, "lite", False))
        return bool(getattr(self._factory, "lite", False))

    def use_full(self) -> str | None:
        """Load the graph model once and extract facts and embeddings for every lite
        document; the model then stays resident, so alternating lite and full questions
        never pays a second load. Returns why it could not, or None."""
        if not self.is_lite:
            return None
        if self._full_factory is None:
            return "the graph model is not available here: pip install 'cogito-estella[full]'"
        with self._ex_lock:
            if getattr(self._ex, "lite", False) or self._ex is None:
                try:
                    with _quiet():                 # stdout is the MCP wire
                        ex = self._full_factory()
                except (Exception, SystemExit) as exc:   # noqa: BLE001 - a failed load must not stop the server
                    return f"the graph model could not be loaded ({exc})"
                self._ex = ex
        with self._lock:
            lite_docs = [s for s, d in self.docs.items() if d.get("lite")]
            for s in lite_docs:
                # a file is read again so the full model splits it with its own scanner,
                # exactly as a full index would; raw text keeps its stored sentences
                if not (Path(s).is_file() and self.ingest_path(Path(s), save=False)
                        and not self.docs[s].get("lite")):
                    self._reextract(s)
            if lite_docs:
                self.save()
        return None

    def _reextract(self, source: str) -> None:
        """Facts and embeddings for a lite document, from the sentences it stored."""
        prev = self.docs[source]
        sents = [(s, o) for s, o in prev["sents"]]
        texts = [s for s, _ in sents]
        offsets = [o for _, o in sents]
        ex = self._ex
        prose = is_prose(source)
        recs_per: list[list[dict]] = []
        if not prose:
            recs_per = _code_graph(_rebuild(sents), source, offsets, len(texts))
        with _quiet():
            for i in range(0, len(texts) if prose else 0, BATCH):
                recs_per.extend(ex.extract_batch_with_provenance(
                    texts[i:i + BATCH], doc_offsets=offsets[i:i + BATCH]))
        for k, recs in enumerate(recs_per):
            for rec in recs:
                rec.pop("sentence", None)
                rec.update(id=self.next_id, source=source, sent_idx=k)
                self._add_edge(rec)
        prev.pop("lite", None)
        if prose:
            self._embed_source(ex, source, texts)
        self._invalidate_index()

    # -- ingestion -----------------------------------------------------------
    def split(self, text: str, by_line: bool = False) -> list[tuple[str, int]]:
        """(sentence, absolute char offset) pairs; regex fallback without spaCy. Code and
        config are split by line: a sentence splitter reads a whole file as one sentence."""
        if by_line:
            out, cursor = [], 0
            for line in text.splitlines(keepends=True):
                if line.strip():
                    out.append((line.strip(), cursor + len(line) - len(line.lstrip())))
                cursor += len(line)
            return out
        with _quiet():                             # the first scan may load spaCy
            nlp = self.extractor._scanner()
        sents = ([s.text.strip() for s in nlp(text).sents] if nlp
                 else [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)])
        if getattr(self.extractor, "lite", False):     # Thai and similar end sentences
            sents = [c for s in sents for c in _chunks(s)]   # with a space, not a stop
        out, cursor = [], 0
        for s in sents:
            if not s:
                continue
            i = text.find(s, cursor)
            i = cursor if i < 0 else i
            if out and _ABBREV_END.search(out[-1][0]):
                out[-1] = (text[out[-1][1]:i + len(s)].strip(), out[-1][1])
            else:
                out.append((s, i))
            cursor = i + len(s)
        return out

    def ingest_text(self, text: str, source: str, save: bool = True) -> IngestResult:
        with self._lock:
            t0 = time.time()
            digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
            prev = self.docs.get(source)
            upgrade = bool(prev and prev.get("lite")) and not getattr(self.extractor, "lite",
                                                                       False)
            if prev and prev["sha256"] == digest and not upgrade:
                if "sections" not in prev:         # indexed before sections existed: the
                    prev["sections"] = _sections(source,  # text and offsets are enough, no
                        text, [o for _, o in prev["sents"]])   # extraction runs again
                    if save:
                        self.save()
                return IngestResult(source, "unchanged", prev["sentences"], 0, prev["tokens"],
                                    0, 0.0, round(time.time() - t0, 2))
            status = "replaced" if prev else "ingested"
            if prev:
                self.drop_source(source)
            ex = self.extractor
            # the graph heads and the encoder were trained on prose: on code lines they add
            # wrong facts and a cosine that misleads the ranking (22/30 against 29/30 lite)
            prose = is_prose(source)
            sents = self.split(text, by_line=not prose)
            texts = [s for s, _ in sents]
            offsets = [o for _, o in sents]
            recs_per: list[list[dict]] = []
            if not prose and not getattr(ex, "lite", False):
                recs_per = _code_graph(text, source, offsets, len(texts))
            with _quiet():
                for i in range(0, len(texts) if prose else 0, BATCH):
                    recs_per.extend(ex.extract_batch_with_provenance(
                        texts[i:i + BATCH], doc_offsets=offsets[i:i + BATCH]))
            n_new = 0
            for k, recs in enumerate(recs_per):
                for rec in recs:
                    rec.pop("sentence", None)
                    rec.update(id=self.next_id, source=source, sent_idx=k)
                    self._add_edge(rec)
                    n_new += 1
            raw = ntok(text)
            self.docs[source] = {"sha256": digest, "tokens": raw, "sentences": len(sents),
                                 "sents": sents, "sections": _sections(source, text, offsets)}
            if getattr(ex, "lite", False):
                self.docs[source]["lite"] = True
            if prose:
                self._embed_source(ex, source, texts)
            self._invalidate_index()
            self._sync_raw_tokens()
            graph_tok = ntok(self.render_edges(
                [e for e in self.edges.values() if e["source"] == source]))
            if save:
                self.save()
            return IngestResult(source, status, len(sents), n_new, raw, graph_tok,
                                round(raw / max(graph_tok, 1), 1), round(time.time() - t0, 2))

    def ingest_path(self, path: str | Path, save: bool = True) -> list[IngestResult | str]:
        """Ingest a file or directory; one IngestResult per document, or an
        error line `source=<name> error=<message>` (unreadable or empty).
        Saves once at the end, not once per document."""
        from cogito_estella.mcp.readers import ReaderError, iter_sources
        out: list[IngestResult | str] = []
        with self._lock:
            for name, item in iter_sources(Path(path)):
                if isinstance(item, ReaderError):
                    out.append(f"source={name} error={item}")
                elif not item.strip():
                    out.append(f"source={name} error=empty document")
                else:
                    out.append(self.ingest_text(item, name, save=False))
            if not out:
                out.append(f"source={path} error=no supported documents "
                           "(text files, html, pdf)")
            if save:
                self.save()
        return out

    def _add_edge(self, rec: dict) -> None:
        self.edges[rec["id"]] = rec
        self.adj[rec["s"]].append(rec["id"])
        self.adj[rec["o"]].append(rec["id"])
        self.next_id = max(self.next_id, rec["id"] + 1)

    def drop_source(self, source: str) -> int:
        """Retire every edge of `source`; ids are not reused. Returns edges dropped."""
        with self._lock:
            gone = [i for i, e in self.edges.items() if e["source"] == source]
            for i in gone:
                del self.edges[i]
            self.docs.pop(source, None)
            self.emb.pop(source, None)
            self._invalidate_index()
            self._rebuild_adj()
            self._sync_raw_tokens()
            return len(gone)

    @staticmethod
    def _adjacency(edges: dict[int, dict]) -> dict[str, list[int]]:
        """Adjacency for a parsed edges mapping; raises KeyError on a malformed edge."""
        adj: dict[str, list[int]] = defaultdict(list)
        for i in sorted(edges):
            e = edges[i]
            adj[e["s"]].append(i)
            adj[e["o"]].append(i)
        return adj

    def _rebuild_adj(self) -> None:
        self.adj = self._adjacency(self.edges)

    def _sync_raw_tokens(self) -> None:
        """raw_tokens is derived from the currently stored documents, never accumulated:
        a replaced document must not double-count its old and new token totals."""
        self.ledger.raw_tokens = sum(d["tokens"] for d in self.docs.values())

    def _embed_source(self, ex, source: str, texts: list[str]) -> None:
        """Sentence embeddings when the extractor can encode; absence means lexical ranking."""
        encode = getattr(ex, "encode_batch", None)
        if encode is None or not texts:
            return
        self._drop_foreign_embeddings(ex)
        try:
            with _quiet():                         # stdout is the MCP wire
                emb = np.asarray(encode(texts), dtype=np.float16)
        except Exception as exc:                   # noqa: BLE001 - encoder failure is a fallback
            if not self._emb_warned:               # one line per process, not per document
                self._emb_warned = True
                print(f"cogito-mcp: sentence embeddings unavailable ({exc}); "
                      "ask ranks lexically", file=sys.stderr)
            return
        defect = self._emb_defect(emb, len(texts), getattr(ex, "dim", None))
        if defect:
            if not self._emb_warned:               # one line per process, not per document
                self._emb_warned = True
                print(f"cogito-mcp: sentence embeddings rejected ({defect}); "
                      "ask ranks lexically", file=sys.stderr)
            return
        self.emb[source] = emb
        self.emb_encoder = getattr(ex, "encoder_name", None) or self.emb_encoder

    @staticmethod
    def _emb_defect(emb: np.ndarray, n_texts: int, dim: int | None) -> str:
        """Why `emb` cannot rank sentences, or "" when it can. Cosine ranking assumes
        unit rows in the extractor's own space: an off-space or unnormalized block would
        score silently wrong, so it is refused rather than stored."""
        if emb.ndim != 2 or emb.shape[0] != n_texts:
            return f"shape {tuple(emb.shape)} for {n_texts} sentences"
        if dim is not None and emb.shape[1] != dim:
            return f"width {emb.shape[1]}, the extractor encodes {dim}"
        off = np.abs(np.linalg.norm(emb.astype(np.float32), axis=1) - 1.0)
        if emb.shape[0] and float(off.max()) >= 0.01:
            return f"a row is {float(off.max()):.3f} off the unit sphere"
        return ""

    def _invalidate_index(self) -> None:
        """The sentence universe changed: index, IDF scorer and length normalizer are stale."""
        self._sindex = None
        self._lex = None
        self._avgdl = None

    def _lexical_scorer(self) -> LexicalScorer:
        """IDF over the whole universe: built once per corpus, not once per question."""
        with self._lock:
            if self._lex is None:
                texts, keys, _ = self.sentence_index()
                # a code line rarely names its file ("which port does the Dockerfile expose?"):
                # its path words count as its own for matching, never for display
                self._lex = LexicalScorer([t if is_prose(k[0]) else _code_terms(t, k[0])
                                           for t, k in zip(texts, keys)])
            return self._lex

    def sentence_index(self) -> tuple[list, list, dict]:
        """Flat sentence universe: texts, their (source, sent_idx) keys, and the reverse map."""
        with self._lock:
            if self._sindex is None:
                texts: list[str] = []
                keys: list[tuple[str, int]] = []
                pos: dict[tuple[str, int], int] = {}
                for src, d in self.docs.items():
                    for i, (s, _) in enumerate(d["sents"]):
                        pos[(src, i)] = len(texts)
                        keys.append((src, i))
                        texts.append(s)
                self._sindex = (texts, keys, pos)
            return self._sindex

    def _embedding_matrix(self, keys: list):
        """(matrix aligned with the sentence universe, mask of embedded rows) or (None, None)."""
        if not self.emb:
            return None, None
        dim = next(iter(self.emb.values())).shape[1]
        mat = np.zeros((len(keys), dim), dtype=np.float16)
        mask = [False] * len(keys)
        for i, (src, si) in enumerate(keys):
            block = self.emb.get(src)
            if block is not None and si < block.shape[0] and block.shape[1] == dim:
                mat[i] = block[si]
                mask[i] = True
        return (mat, mask) if any(mask) else (None, None)

    def sentence_of(self, edge: dict) -> str:
        return self.docs[edge["source"]]["sents"][edge["sent_idx"]][0]

    # -- rendering helpers --------------------------------------------------------
    @staticmethod
    def render_edges(edges: list[dict]) -> str:
        groups: dict[tuple, list[int]] = {}
        for e in edges:
            groups.setdefault((e["s"], e["r"], e["o"]), []).append(e["id"])
        return "\n".join(f"{s} {r} {o} #{','.join(map(str, ids))}"
                         for (s, r, o), ids in groups.items())

    # -- retrieval ----------------------------------------------------------------
    def resolve(self, name: str) -> str | None:
        if name.strip() in self.adj:             # code keeps case: `Pay` the class, `pay` the method
            return name.strip()
        q = name.strip().lower()
        if not q:
            return None
        if q in self.adj:
            return q
        same = [e for e in self.adj if e.lower() == q]
        if same:
            return max(same, key=lambda e: len(self.adj[e]))
        if len(q) < 2:               # too short for prefix/substring/plural guessing
            return None
        cands = [e for e in self.adj
                 if e.lower().startswith(q) or (len(e) >= 3 and q.startswith(e.lower()))
                 or (len(q) >= 4 and q in e.lower())]
        if not cands and q.endswith("s"):
            return self.resolve(q[:-1])
        return max(cands, key=lambda e: len(self.adj[e])) if cands else None

    def question_entities(self, question: str, k: int = ASK_ENTITIES) -> list[str]:
        """Exact entity hits in a question, rarest first; no prefix or substring guessing."""
        out: list[str] = []
        for raw in _QWORD.findall(question.lower()):
            if raw in ASK_STOPLIST:
                continue
            # -es/-ies plurals too: `ask` must not be weaker than `query.resolve` here
            cand = next((c for c in singular_forms(raw) if c in self.adj), None)
            if cand is None or cand in ASK_STOPLIST or cand in out:
                continue
            out.append(cand)
        out.sort(key=lambda e: len(self.adj[e]))     # stable: ties keep question order
        return out[:k]

    def neighborhood(self, entity: str, hops: int = 1) -> list[dict]:
        seen_e, seen_edges, out = {entity}, set(), []
        frontier = deque([(entity, 0)])
        while frontier:
            node, d = frontier.popleft()
            if d >= hops:
                continue
            for eid in self.adj.get(node, ()):
                if eid in seen_edges:
                    continue
                seen_edges.add(eid)
                e = self.edges[eid]
                out.append(e)
                other = e["o"] if e["s"] == node else e["s"]
                if other not in seen_e:
                    seen_e.add(other)
                    frontier.append((other, d + 1))
        return out

    def top_entities(self, limit: int = 30, prefix: str = "") -> list[str]:
        ents = [e for e in self.adj if e.startswith(prefix.lower())]
        ents.sort(key=lambda e: -len(self.adj[e]))
        return ents[:limit]

    # -- tool-facing renderers ------------------------------------------------------
    def query(self, entity: str, hops: int = 1, limit: int = 25) -> str:
        node = self.resolve(entity)
        if node is None:
            known = ", ".join(self.top_entities(10)) or "(empty graph)"
            return f"no entity matches '{entity}'. known (by degree): {known}"
        edges = self.neighborhood(node, hops)
        lex = [e for e in edges if e["r_lex"]]
        fb = [e for e in edges if not e["r_lex"]]
        lex_lines = self.render_edges(lex).split("\n") if lex else []
        fb_lines = self.render_edges(fb).split("\n") if fb else []
        total = len(lex_lines) + len(fb_lines)
        shown_lex = lex_lines[:limit]
        shown_fb = fb_lines[:max(limit - len(shown_lex), 0)]
        body = shown_lex + ([DIVIDER] + shown_fb if shown_fb else [])
        shown = len(shown_lex) + len(shown_fb)
        more = "" if shown >= total else f"\n+{total - shown} more (raise limit)"
        return f"{node} ({total} facts, hops={hops})\n" + "\n".join(body) + more

    def provenance(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            e = self.edges.get(i)
            if e is None:
                out.append(f"#{i}: retired edge (document replaced)" if 0 <= i < self.next_id
                           else f"#{i}: unknown edge")
                continue
            sp = lambda x: f"[{x[0]}:{x[1]}]" if x else "[-]"
            out.append(f"#{i} {e['s']} {e['r']} {e['o']}  {e['source']} s{e['sent_idx']} "
                       f"{sp(e['s_span'])}/{sp(e['o_span'])}\n  \"{self.sentence_of(e)}\"")
        return "\n".join(out)

    def search(self, term: str, limit: int = 8) -> str:
        """Every word of `term` must occur in the sentence; a multi-word query is an AND
        of its words, not one contiguous phrase. A caller composing a natural multi-word
        search rarely repeats the source text verbatim, so requiring the whole term as one
        substring dropped sentences a single-word retry of the same search then found."""
        words = [w for w in term.strip().lower().split() if w]
        hits = [f"{src} s{i}: \"{s}\"" for src, d in self.docs.items()
                for i, (s, _) in enumerate(d["sents"])
                if words and all(w in s.lower() for w in words)]
        more = "" if len(hits) <= limit else f"\n+{len(hits) - limit} more (raise limit)"
        return ("\n".join(hits[:limit]) + more) if hits else f"no sentence mentions '{term}'"

    def entities(self, prefix: str = "", limit: int = 30) -> str:
        ents = self.top_entities(limit, prefix)
        return ", ".join(f"{e}({len(self.adj[e])})" for e in ents) or "(empty graph)"

    def ask(self, question: str, budget: int = ASK_BUDGET, scorer: str = "lexical",
            style: str = "full") -> str:
        """One call from a question to the facts and sentences that answer it, within budget.
        IDF is the primary engine; `dense` (older name `sonar`) and `learned` re-rank the
        sentence block only when asked for. `compact` drops the graph facts and the scorer
        jargon: sentences tagged by source under one plain-language line."""
        return self._ask(question, budget, scorer, style)[0]

    def ask_compact(self, question: str, budget: int = ASK_BUDGET, scorer: str = "lexical",
                    widen_to: int | None = None) -> str:
        """`ask` in the compact style, widened once inside the same call when the learned
        scorer doubts its best sentence. The measured escalation ("ask again at 1200 when
        p < 0.3") is a policy a small model does not follow; here it costs no extra turn."""
        text, conf = self._ask(question, budget, scorer, "compact")
        if widen_to and widen_to > budget and conf is not None and conf < LOW_CONFIDENCE:
            text, _ = self._ask(question, widen_to, scorer, "compact")
        return self._second_hop(question, text, scorer)

    def _second_hop(self, question: str, text: str, scorer: str) -> str:
        """Bridge questions ("the producer of the film scored by X") name one entity and
        need a passage about another: the names the best prose lines bring in are asked
        again, and both replies interleave within the first one's size. HotpotQA, 200
        questions not used to choose it: the answer reaches the reply 151 times instead
        of 136; papers unchanged, code worse (18 to 16 of 30), so code replies keep one hop."""
        lines = text.split("\n")
        if any(x.startswith("  ") for x in lines[1:]):     # grouped code: one search only
            return text
        head, body = lines[0], [x for x in lines[1:] if x.startswith("[")]
        top = body[:HOP_LINES]
        if not top or head.startswith(COMPACT_WHOLE.split("{")[0]) or not all(is_prose(_line_source(x)) for x in top):
            return text
        asked = _norm_words(question)
        names: list[str] = []
        for line in top:
            for m in _CAPS.findall(line.split("] ", 1)[-1]):
                if len(m) > 2 and _norm_words(m) and not _norm_words(m) <= asked \
                        and m not in names:
                    names.append(m)
        if not names:
            return text
        second, _ = self._ask(" ".join(names[:HOP_NAMES]) + " " + question,
                              ASK_BUDGET, scorer, "compact")
        more = [x for x in second.split("\n")[1:] if x.startswith("[")]
        merged, seen = [], set()
        for pair in zip(body + [None] * len(more), more + [None] * len(body)):
            for x in pair:
                if x and x not in seen:
                    seen.add(x)
                    merged.append(x)
        cap, out = ntok(text), [head]
        for x in merged:
            if ntok("\n".join([*out, x])) > cap:
                break
            out.append(x)
        return "\n".join(out)

    def _ask(self, question, budget, scorer, style) -> tuple[str, float | None]:
        requested = str(scorer).strip().lower()
        requested = SCORER_ALIASES.get(requested, requested)
        if requested not in ASK_SCORERS:           # a stray value must not pick a scorer by luck
            raise ValueError(f"unknown scorer {str(scorer).strip()[:24]!r}; "
                             f"use one of: {', '.join(ASK_SCORERS)}")
        if style not in ASK_STYLES:
            raise ValueError(f"unknown style {str(style)[:24]!r}; use one of: "
                             f"{', '.join(ASK_STYLES)}")
        if requested != "lexical" and self.emb:
            self._drop_foreign_embeddings(self.extractor)
        lex, snap, edges = self._ask_snapshot(question, dense=requested != "lexical")
        scores, name, conf = self._score_sentences(question, lex, snap, requested, budget)
        scores = _pull_up_successors(scores, snap.keys)
        if not snap.ents and not any(s > 0 for s in scores):
            return (COMPACT_NONE if style == "compact" else f"no material for '{question}'"), None
        if style == "compact":
            whole = self._whole_corpus(snap, budget)
            if whole is not None:
                return whole, None
            low = conf is not None and conf < LOW_CONFIDENCE
            body = [COMPACT_LOW if low else COMPACT_HEAD]
            sents = self._compact_lines(snap, scores, body, budget)
            return ("\n".join(body + sents) if sents else COMPACT_NONE), conf
        mark = f" · p={conf:.2f}" if conf is not None else ""
        head = (f"entities: {', '.join(snap.ents) or '(none)'}{mark} · scorer={name}")
        body = [head, *self._fact_lines(edges, head, int(budget * FACT_SHARE))]
        sents = self._sentence_lines(snap.texts, snap.keys, scores, body, budget)
        return "\n".join(body + (["--"] + sents if sents else [])), conf

    def _whole_corpus(self, snap, budget) -> str | None:
        """Every sentence in document order when the whole corpus fits the budget, else None.
        Below that size ranking can only drop text the caller could have had for the same
        tokens: on a 1.9k-token document one `ask` saved 1.3x and could miss the answer."""
        if not snap.texts or self.ledger.raw_tokens > budget:
            return None
        docs = len(snap.doc_sents)
        lines, prev = [COMPACT_WHOLE.format(docs=docs, s="" if docs == 1 else "s")], None
        for (src, si), t in zip(snap.keys, snap.texts):
            where = (src, snap.section((src, si)))   # in document order a tag that repeats
            lines.append(_compact_line(where[0], t, where[1]) if where != prev    # says nothing
                         else f"[{src}] {' '.join(t.split())}")
            prev = where
        text = "\n".join(lines)
        return text if ntok(text) <= budget else None

    def _ask_snapshot(self, question, dense: bool):
        """(lexical scorer, snapshot, retrieved edges). Read under the lock, scored outside it:
        one model load must not serialize every other reader and writer behind this call."""
        with self._lock:
            texts, keys, pos = self.sentence_index()
            lex = self._lexical_scorer()
            ents = self.question_entities(question)
            edges, seen = [], set()
            for ent in ents:
                for e in self.neighborhood(ent, 1):
                    if e["id"] not in seen:
                        seen.add(e["id"])
                        edges.append(e)
            # every retrieved fact boosts its own sentence, whether or not the line survives;
            # code facts do not: a name like `pay` sits on dozens of lines (15/30 against 18/30)
            boost = {pos[(e["source"], e["sent_idx"])] for e in edges
                     if (e["source"], e["sent_idx"]) in pos and e.get("pattern") != "syntax"}
            mat, mask = self._embedding_matrix(keys) if dense else (None, None)
            snap = _AskSnapshot(texts=texts, keys=keys, ents=ents, boost=boost, mat=mat,
                                mask=mask, cover=self._embedded_note(),
                                doc_sents={src: d["sentences"] for src, d in self.docs.items()},
                                sections={src: d.get("sections") for src, d in self.docs.items()})
        return lex, snap, edges

    def _embedded_note(self) -> str:
        """`(n/m docs)` when only part of the corpus is embedded: a mixed ranking must say so."""
        embedded = sum(1 for src in self.docs if src in self.emb)
        return "" if embedded >= len(self.docs) else f" ({embedded}/{len(self.docs)} docs)"

    def _score_sentences(self, question, lex, snap, requested, budget=ASK_BUDGET):
        """(scores, scorer name, confidence). SONAR ranks the embedded sentences, the lexical
        ones rank strictly after them: (cos + 1) / 2 floors near 0.5 while an overlap-free
        sentence scores 0, so the two scales must never be compared row by row. Confidence is
        the learned model's own probability for its best sentence; the other scorers report
        none, their numbers being an order and not a likelihood."""
        lexical = lex.score(question, snap.boost)
        if requested == "lexical" or not lex.n:
            return lexical, "lexical", None
        if requested == "learned" and snap.mat is not None \
                and getattr(self.extractor, "vectors", False):
            return self._interleaved(question, lex, lexical, snap, budget)
        if requested == "learned":
            learned, conf = self._learned_scores(question, lex, lexical, snap, budget)
            return (learned, "learned", conf) if learned is not None \
                else (lexical, "lexical (learned unavailable)", None)
        note = "lexical (dense unavailable)"
        if snap.mat is None:
            return lexical, note, None
        q = self._encode_question(question, width=snap.mat.shape[1])
        if q is None:
            return lexical, note, None
        sonar = SonarScorer(snap.mat, lambda _texts: q).score(question, snap.boost)
        out = []
        for i in range(lex.n):
            if not snap.mask[i]:
                out.append(lexical[i])
            else:                                  # the floor keeps `no material` reachable
                out.append(DENSE_OFFSET + sonar[i] if sonar[i] >= DENSE_FLOOR else 0.0)
        return out, f"dense{snap.cover}", None

    def _interleaved(self, question, lex, lexical, snap, budget):
        """`--multilingual`: the learned lexical order and the vector order taken in turns,
        or the vector order alone when few question words occur in the corpus (another
        language). XQuAD, against either ranking alone: Thai 63.4 % (vectors 52.4, lexical
        62.9), Arabic 82.2 (81.7, 71.9); Spanish over English text 95.7 (lexical 43)."""
        dense, _, _ = self._score_sentences(question, lex, snap, "dense", budget)
        q = set(tokenize(question))
        if sum(t in lex.df for t in q) < VECTOR_ONLY_BELOW * max(1, len(q)):
            return dense, "vectors", None
        plain = replace(snap, mat=None, mask=None)    # the scorer was fitted
        learned, _ = self._learned_scores(question, lex, lexical, plain, budget)  # without e5
        if learned is None:
            learned = lexical
        n = len(dense)
        by_learned = {i: r for r, i in enumerate(rank(learned, n))}
        by_dense = {i: r for r, i in enumerate(rank(dense, n))}
        out = []
        for i in range(n):
            a = by_learned[i] if learned[i] > 0 else n
            out.append(1.0 / (1 + min(a, by_dense[i])) + 1e-6 / (1 + max(a, by_dense[i])))
        return out, "interleaved", None

    def _encode_question(self, question, width=None):
        """The question as a flat row, or None when no encoder answers for it or its width
        does not match the stored rows: a mismatched product is a crash, not a ranking."""
        if not hasattr(self.extractor, "encode_batch"):    # --lite: nothing to encode with
            return None
        try:
            with _quiet():                         # stdout is the MCP wire
                encode = getattr(self.extractor, "encode_query", self.extractor.encode_batch)
                q = np.asarray(encode([question]), dtype=np.float32)
        except Exception as exc:                   # noqa: BLE001 - a missing encoder is a fallback
            print(f"cogito-mcp: question encoding failed ({exc}); ask ranks lexically",
                  file=sys.stderr)
            return None
        q = q.reshape(-1)
        if width is not None and q.size != int(width):
            print(f"cogito-mcp: question width {q.size} does not match the stored {int(width)}; "
                  "ask ranks lexically", file=sys.stderr)
            return None
        return q

    def _question_cosines(self, question, snap):
        """(cosine per sentence, availability per sentence) for the dense feature; both None
        when the corpus carries no embedding or the question will not encode."""
        if snap.mat is None:
            return None, None
        q = self._encode_question(question, width=snap.mat.shape[1])
        if q is None:
            return None, None
        norm = float(np.linalg.norm(q))
        if norm:
            q = q / norm                           # a stray scale must not skew cosine
        cos = np.clip(np.asarray(snap.mat, dtype=np.float32) @ q, -1.0, 1.0)
        return cos, snap.mask

    def sentence_avgdl(self) -> float:
        """Mean token length of the whole sentence universe: the length normalizer a fit pins
        and the query path hands back to the ranking column."""
        with self._lock:
            if self._avgdl is None:
                self._avgdl = corpus_avgdl([tokenize(t) for t in self.sentence_index()[0]])
            return self._avgdl

    def learned_candidates(self, question: str, budget: int = ASK_BUDGET, model=None,
                           avgdl: float | None = None):
        """(indices into the sentence universe, their texts, feature context) for the learned
        ranking. The fit path and the query path go through this one rule, so the shortlist and
        the feature columns cannot drift apart between them."""
        lex, snap, _edges = self._ask_snapshot(question, dense=True)
        return self._candidates(question, lex, lex.score(question, snap.boost), snap,
                                budget, model, avgdl)

    def _candidates(self, question, lex, lexical, snap, budget, model, avgdl=None):
        # the shortlist grows with the budget: a fixed cap would truncate the block that the
        # lexical arm keeps filling, and silently compare two different depths
        width = max(ASK_CANDIDATES, int(budget) // ASK_CAND_TOKENS)
        cand = [i for i in rank(lexical, width) if lexical[i] > 0]
        ctx = self._feature_context(question, lex, lexical, snap, cand, model, avgdl)
        return cand, [snap.texts[i] for i in cand], ctx

    def _learned_scores(self, question, lex, lexical, snap, budget=ASK_BUDGET):
        """Model scores over the lexical shortlist, mapped onto the sentence universe, or None
        when no usable weights ship: the caller then keeps the lexical order.

        The returned values carry the ranking and nothing else - a logit has no floor, and the
        fill stops at the first non-positive score. The probability of the best candidate is
        returned beside them: the order discards it, and a caller deciding whether to trust
        the reply needs the likelihood, not the rank."""
        model = load_default()
        if model is None:
            return None, None
        cand, texts, ctx = self._candidates(question, lex, lexical, snap, budget, model)
        if not cand:
            return None, None
        p = model.predict_proba(featurize(question, texts, ctx))   # monotonic in the score
        out = [0.0] * len(lexical)
        order = rank(p)
        for place, j in enumerate(order, start=1):
            out[cand[j]] = 1.0 / place
        return out, float(p[order[0]])

    def _feature_context(self, question, lex, lexical, snap, cand, model, avgdl=None):
        q_tokens = tokenize(question)
        cos, avail = self._question_cosines(question, snap) if cand else (None, None)
        return FeatureContext(
            lex_scores=[lexical[i] for i in cand],
            bm25=bm25_scores([tokenize(snap.texts[i]) for i in cand], lex.idf_of, q_tokens,
                             avgdl=self._bm25_avgdl(model, avgdl)),
            q_tokens=q_tokens, q_bigrams=bigrams(q_tokens), q_ents=snap.ents,
            prov_flags=[i in snap.boost for i in cand],
            dense_cos=[float(cos[i]) for i in cand] if cos is not None else 0.0,
            dense_available=[bool(avail[i]) for i in cand] if avail is not None else False,
            rel_pos=[snap.rel_pos(snap.keys[i]) for i in cand])

    def _bm25_avgdl(self, model, avgdl=None) -> float:
        """The length normalizer of the ranking column: what the caller pins, else what the
        weights were fitted with, else the mean length of this universe."""
        pinned = getattr(model, "bm25_avgdl", None) if model is not None else None
        for value in (avgdl, pinned):
            if value is not None:
                return float(value)
        return self.sentence_avgdl()

    @staticmethod
    def _fit(block: list, lines: list, cap: int) -> list:
        """Lines that keep the joined block within `cap` tokens; truncation is line-granular.
        An oversize line is skipped, not a stop: one grouped fact line must not delete the
        shorter ones behind it (the same rule `_sentence_lines` follows)."""
        out: list[str] = []
        skips = 0
        for line in lines:
            if ntok("\n".join(block + out + [line])) > cap:
                skips += 1
                if skips >= ASK_MAX_SKIPS:
                    break
                continue
            skips = 0
            out.append(line)
        return out

    @staticmethod
    def _cap_ids(line: str) -> str:
        """Trim a grouped fact line's `#id` list; the ellipsis holds no digit that a
        `provenance` call could read back as an edge id."""
        head, sep, ids = line.rpartition(" #")
        parts = ids.split(",")
        if not sep or len(parts) <= ASK_MAX_IDS:
            return line
        return f"{head} #{','.join(parts[:ASK_MAX_IDS])} …"

    def _ask_edge_lines(self, edges: list) -> list:
        return [self._cap_ids(x) for x in self.render_edges(edges).split("\n")] if edges else []

    def _fact_lines(self, edges: list, head: str, cap: int) -> list:
        lex = [e for e in edges if e["r_lex"]]
        cls = [e for e in edges if not e["r_lex"]]
        out = self._fit([head], self._ask_edge_lines(lex), cap)
        if cls:
            tail = self._fit([head, *out, DIVIDER], self._ask_edge_lines(cls), cap)
            if tail:                               # a divider with nothing under it is noise
                out += [DIVIDER, *tail]
        return out

    def _sentence_lines(self, texts, keys, scores, body, budget, sep=("--",),
                        fmt=None) -> list:
        out: list[str] = []
        seen: set[str] = set()
        skips = 0
        fmt = fmt or _full_line
        for i in rank(scores):
            if scores[i] <= 0:
                break
            if texts[i] in seen:
                continue
            src, si = keys[i]
            line = fmt(src, si, texts[i])
            if ntok("\n".join(body + [*sep, *out, line])) > budget:
                skips += 1                         # oversize line: shorter ones may still fit
                if skips >= ASK_MAX_SKIPS:
                    break
                continue
            skips = 0
            seen.add(texts[i])
            out.append(line)
        return out

    def _compact_lines(self, snap, scores, body, budget) -> list[str]:
        """Compact reply lines. Prose keeps one tagged line per sentence; code lines are
        grouped under their file, once, in file order: on code the repeated tag was 40 %
        of a reply."""
        picked: list[int] = []
        seen: set[str] = set()
        skips = 0
        for i in rank(scores):
            if scores[i] <= 0:
                break
            if snap.texts[i] in seen:
                continue
            trial = self._render_compact(snap, [*picked, i])
            if ntok("\n".join(body + trial)) > budget:
                skips += 1
                if skips >= ASK_MAX_SKIPS:
                    break
                continue
            skips = 0
            seen.add(snap.texts[i])
            picked.append(i)
        return self._render_compact(snap, picked)

    @staticmethod
    def _render_compact(snap, picked: list[int]) -> list[str]:
        groups: dict[str, list[int]] = {}
        out: list[object] = []                     # a prose line, or a code file's group key
        for i in picked:
            src, si = snap.keys[i]
            if is_prose(src):
                out.append(_compact_line(src, snap.texts[i], snap.section((src, si))))
            else:
                if src not in groups:
                    groups[src] = []
                    out.append(("code", src))
                groups[src].append(i)
        lines: list[str] = []
        for item in out:
            if isinstance(item, str):
                lines.append(item)
                continue
            src = item[1]
            lines.append(f"[{src}]")
            lines.extend(f"  {' '.join(snap.texts[i].split())}"
                         for i in sorted(groups[src], key=lambda i: snap.keys[i][1]))
        return lines

    def stats(self) -> str:
        L = self.ledger
        full = ntok(self.render_edges(list(self.edges.values()))) if self.edges else 0
        calls = sum(L.calls.values())
        by = ", ".join(f"{k}={v}" for k, v in L.served_by.items()) or "-"
        lex = sum(1 for e in self.edges.values() if e["r_lex"])
        gf = (f"graph_file={self.path} size={self.path.stat().st_size} "
              f"persisted_at={self.persisted_at}") if self.path and self.path.exists() \
            else "graph_file=none"
        return "\n".join([
            (f"docs={len(self.docs)} sentences={sum(d['sentences'] for d in self.docs.values())} "
             f"edges={len(self.edges)} lexical={lex} entities={len(self.adj)} "
             f"embedded_docs={len(self.emb)}/{len(self.docs)}"),
            f"raw_tokens_ingested={L.raw_tokens}  (what Read-ing every doc costs)",
            f"full_graph_tokens={full}  (compression {L.raw_tokens / max(full, 1):.1f}x)",
            f"tokens_served_to_agent={L.served} over {calls} calls: {by}",
            f"savings_vs_read={L.raw_tokens / max(L.served, 1):.1f}x",
            *([f"tool_definitions_tokens={L.schema_tokens}  (re-sent with every model turn)"]
              if L.schema_tokens else []),
            gf,
        ])

    # -- persistence ----------------------------------------------------------------
    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            data = {"version": 1, "next_id": self.next_id,
                    "raw_tokens": self.ledger.raw_tokens, "docs": self.docs,
                    "edges": {str(i): e for i, e in self.edges.items()}}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # unique per writer: two concurrent ingests must not share (and race on)
            # the same temp file
            tmp = self.path.with_name(
                f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
            try:
                self._save_embeddings()
            except (OSError, ValueError) as exc:   # the graph itself is already committed
                print(f"cogito-mcp: embeddings sidecar not written ({exc}); "
                      "ask ranks lexically after a reload", file=sys.stderr)
            self.persisted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        with self._lock:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("graph file is not a JSON object")  # noqa: TRY004
                v = data.get("version")
                if v != 1:
                    raise ValueError(
                        f"unsupported graph file version {v!r}; this build reads version 1")
                edges_raw, docs_raw = data.get("edges"), data.get("docs")
                if not isinstance(edges_raw, dict) or not isinstance(docs_raw, dict):
                    raise ValueError("graph file has malformed sections")  # noqa: TRY004
                edges = {int(i): e for i, e in edges_raw.items()}
                docs = {src: {**d, "sents": [tuple(x) for x in d["sents"]]}
                        for src, d in docs_raw.items()}
                next_id = int(data["next_id"])
                adj = self._adjacency(edges)    # raises KeyError on a malformed edge
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: UP017
                bad = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
                os.replace(self.path, bad)
                print(f"cogito-mcp: graph file corrupt ({exc}); moved to {bad}; starting empty",
                      file=sys.stderr)
                return
            # only commit once every section parsed cleanly: never half-load
            self.edges, self.docs, self.adj = edges, docs, adj
            self.next_id = max(next_id, max(edges, default=-1) + 1)
            self._invalidate_index()
            self._load_embeddings()
            self._sync_raw_tokens()
            self.persisted_at = datetime.fromtimestamp(
                self.path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")  # noqa: UP017

    # -- embedding sidecar (<graph>.emb.npz) -----------------------------------------
    def _current_encoder(self) -> str | None:
        """Encoder behind a materialized extractor; None until one exists."""
        return getattr(self._ex, "encoder_name", None) if self._ex is not None else None

    def _drop_foreign_embeddings(self, ex) -> None:
        """Rows encoded in another encoder's space are dropped before use or extension:
        a cosine across spaces is noise, and a sidecar must not mix spaces."""
        cur = getattr(ex, "encoder_name", None)
        if not (self.emb and cur and self.emb_encoder and cur != self.emb_encoder):
            return
        with self._lock:
            self.emb = {}
        if not self._emb_space_warned:
            self._emb_space_warned = True
            print(f"cogito-mcp: stored sentence embeddings come from {self.emb_encoder}, the "
                  f"server runs {cur}; ask ranks lexically until the documents are re-ingested",
                  file=sys.stderr)

    def _emb_path(self) -> Path | None:
        return self.path.with_suffix(".emb.npz") if self.path else None

    def _save_embeddings(self) -> None:
        p = self._emb_path()
        if p is None:
            return
        if not self.emb:
            if not self.docs:
                p.unlink(missing_ok=True)          # never leave a sidecar without a graph
            elif p.exists() and not self._emb_keep_warned:
                self._emb_keep_warned = True       # a rejected load must not destroy the file
                print(f"cogito-mcp: no embeddings in memory; {p} left untouched",
                      file=sys.stderr)
            return
        sources = list(self.emb)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with tmp.open("wb") as fh:             # a handle keeps savez from appending .npz
                np.savez(fh, sources=np.array(sources),
                         counts=np.array([self.emb[s].shape[0] for s in sources]),
                         # content hash per block: a same-length replacement must not pass
                         shas=np.array([self.docs.get(s, {}).get("sha256", "") for s in sources]),
                         encoder=np.array(self._current_encoder() or self.emb_encoder or ""),
                         emb=np.concatenate([self.emb[s] for s in sources], axis=0))
            os.replace(tmp, p)
        finally:
            tmp.unlink(missing_ok=True)            # a failed write leaves no tmp behind

    def _load_embeddings(self) -> None:
        p = self._emb_path()
        self.emb = {}
        if p is None or not p.exists():
            return
        try:
            with np.load(p, allow_pickle=False) as z:
                sources = [str(s) for s in z["sources"]]
                counts = [int(c) for c in z["counts"]]
                shas = [str(s) for s in z["shas"]]      # absent in pre-0.15.0 sidecars: KeyError
                emb = z["emb"]
                tag = str(z["encoder"]) if "encoder" in z.files else "sonar"   # pre-0.16.0: SONAR
            if emb.ndim != 2 or emb.dtype.kind != "f" or len(sources) != len(counts) \
                    or len(sources) != len(shas) or sum(counts) != emb.shape[0]:
                raise ValueError("sidecar matrix does not match its index")
            out, off = {}, 0
            for src, n, sha in zip(sources, counts, shas, strict=True):
                block, off = emb[off:off + n], off + n
                doc = self.docs.get(src, {})
                # the hash pins the block to the exact text it was encoded from;
                # the count is a cheap redundancy. Any disagreement: lexical is safer
                if doc.get("sha256") == sha and doc.get("sentences") == n:
                    out[src] = block
            self.emb = out
            self.emb_encoder = tag if out else None
        except (OSError, ValueError, KeyError, EOFError, IndexError) as exc:
            print(f"cogito-mcp: embeddings sidecar unusable ({exc}); ask ranks lexically",
                  file=sys.stderr)
