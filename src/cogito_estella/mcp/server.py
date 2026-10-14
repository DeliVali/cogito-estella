"""Cogito MCP server (stdio): ingest once, route every question through `ask`,
go deeper with query/provenance/search. Every tool reply is charged to the token ledger."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections import Counter
from pathlib import Path

from cogito_estella.encoders import ENCODERS, EncoderMismatch, encoder_revision
from cogito_estella.encoders.pooling import PoolWeightsError
from cogito_estella.mcp.readers import source_names
from cogito_estella.mcp.store import ASK_SCORERS, SCORER_ALIASES, GraphStore, TextOnlyExtractor
from cogito_estella.mcp.tokens import ntok
from cogito_estella.mcp.weights import WeightsError, ensure_spacy_model, resolve

INSTRUCTIONS = ("Knowledge-graph memory over documents. `ingest` a file, directory or text "
                "once (zero LLM tokens). Start every question with `ask`: it returns the graph "
                "facts and the sentences that answer it within a token budget. Use `query`, "
                "`provenance` and `search` only to go deeper when `ask` is not enough; facts "
                "under the '~ class-only' divider need `provenance` before you rely on them.")

# compact: the surface for models with little context. One tool (two when nothing was
# indexed at startup), one argument, short descriptions that name what is indexed, and
# replies in plain language; the escalation policy runs in the server, not in the model
COMPACT_INSTRUCTIONS = ("Answers questions from local documents without reading them whole. "
                        "Call `ask` with the question before opening any indexed file; one "
                        "call is usually enough.")
# push: the client fetches the context and sends it with the question, so a model that
# never decides to call a tool (measured: a 1.5B model calls none, under any prompt) still
# answers from the documents
CONTEXT_TEMPLATE = "Context from the indexed documents for this question:\n{context}\n\n" \
                   "Question: {question}"
PROFILES = ("full", "compact")
CORPUS_NAMES = 6                  # sources named in the compact `ask` description
INDEX_WAIT = 60.0                 # seconds a tool waits for the startup indexing
COMPACT_ERRORS = 5                # error lines a compact ingest reply keeps

ASK_BUDGET_MIN, ASK_BUDGET_MAX = 100, 4000
DEFAULT_SCORER = "learned"        # what `ask` ranks with when the operator sets nothing
SCORER_ENV = "COGITO_ASK_SCORER"


def resolve_scorer(value: str | None = None) -> str:
    """The configured scorer. An unreadable setting is refused rather than ignored: a typo
    would otherwise be served, and measured, as the scorer it was meant to replace."""
    raw = os.environ.get(SCORER_ENV, "") if value is None else value
    name = (raw or "").strip().lower()
    if not name:
        return DEFAULT_SCORER
    name = SCORER_ALIASES.get(name, name)
    if name not in ASK_SCORERS:
        raise ValueError(f"{SCORER_ENV}={(raw or '').strip()[:24]!r} is not a scorer; "
                         f"use one of: {', '.join(ASK_SCORERS)}")
    return name


def _existing_path(value: str) -> Path | None:
    """Path when `value` names an existing file or directory; None for raw text."""
    try:
        p = Path(value)
        return p if p.exists() else None
    except (OSError, ValueError):     # e.g. "File name too long" for a pasted paragraph
        return None


def _clamp_budget(budget) -> int:
    return min(max(int(budget), ASK_BUDGET_MIN), ASK_BUDGET_MAX)


def _ingest_summary(results, errors: int = COMPACT_ERRORS) -> str:
    """One line for a whole ingest, plus its first error lines: a directory of forty
    documents must not cost a small model forty lines of counters."""
    done = [r for r in results if not isinstance(r, str)]
    failed = [r for r in results if isinstance(r, str)]
    counts = Counter(r.status for r in done)
    parts = ", ".join(f"{counts[s]} {s}" for s in ("ingested", "replaced", "unchanged")
                      if counts[s])
    tokens = sum(r.raw_tokens for r in done)
    head = (f"indexed {len(done)} document{'' if len(done) == 1 else 's'} ({parts}), "
            f"{tokens} tokens of text; `ask` answers from them now") if done \
        else "nothing indexed"
    more = [f"+{len(failed) - errors} more errors"] if len(failed) > errors else []
    return "\n".join([head, *failed[:errors], *more])


def corpus_label(names) -> str:
    """The indexed sources as a model reads them in a tool description: a model picks a
    tool by matching its need to the description, and "documents" matches nothing."""
    names = list(names)
    if not names:
        return ""
    more = f" and {len(names) - CORPUS_NAMES} more" if len(names) > CORPUS_NAMES else ""
    return ", ".join(names[:CORPUS_NAMES]) + more


class Tools:
    """Tool bodies bound to one store; plain callables so they can be unit-tested."""

    def __init__(self, store: GraphStore, profile: str = "full", budget: int = 600):
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; use one of: {', '.join(PROFILES)}")
        self.store = store
        self.profile = profile
        self.budget = _clamp_budget(budget)
        # read once: one run must rank every question the same way
        self.scorer = resolve_scorer()
        self._ready = threading.Event()
        self._ready.set()
        self._index_error: str | None = None
        self._deferred: list[Path] = []

    def _charge(self, tool: str, out: str) -> str:
        self.store.ledger.charge(tool, out)
        return out

    # -- startup indexing ----------------------------------------------------------
    def index_in_background(self, paths) -> threading.Thread:
        """Ingest `paths` off the stdio loop, so the handshake does not wait on a model
        load. Reading tools wait for it: a reply from a half-built index carries the same
        confidence as a complete one, and is wrong."""
        self._ready.clear()
        th = threading.Thread(target=self._index, args=([Path(p) for p in paths],),
                              name="cogito-index", daemon=True)
        th.start()
        return th

    def _index(self, paths: list[Path]) -> None:
        try:
            results = [r for p in paths for r in self.store.ingest_path(p)]
            print(f"cogito-mcp: {_ingest_summary(results, errors=len(results))}",
                  file=sys.stderr)
        except BaseException as exc:               # noqa: BLE001 - a dead thread must say why
            self._index_error = f"{type(exc).__name__}: {exc}"
            print(f"cogito-mcp: startup indexing failed ({self._index_error})", file=sys.stderr)
        finally:
            self._ready.set()

    def defer_indexing(self, paths) -> None:
        """Index `paths` when the first tool call arrives: by then the handshake is done and
        the transport owns stdout. Reading tools wait for it from the start."""
        self._deferred = [Path(p) for p in paths]
        if self._deferred:
            self._ready.clear()

    def _not_ready(self) -> str | None:
        """Why a reading tool cannot answer yet, or None once the index is complete."""
        if self._deferred:
            paths, self._deferred = self._deferred, []
            self.index_in_background(paths)
        if not self._ready.wait(INDEX_WAIT):
            return "still indexing the documents; call again in a minute"
        if self._index_error:
            return f"the documents could not be indexed ({self._index_error})"
        return None

    # -- tools -------------------------------------------------------------------------
    def ask(self, question: str, budget: int | None = None, use_dense: bool = False,
            use_sonar: bool = False) -> str:
        budget = self.budget if budget is None else _clamp_budget(budget)
        wait = self._not_ready()
        if wait:
            return self._charge("ask", wait)
        scorer = "dense" if (use_dense or use_sonar) else self.scorer   # use_sonar: older name
        if self.profile == "compact":
            return self._charge("ask", self.store.ask_compact(
                question, budget, scorer, widen_to=min(2 * budget, ASK_BUDGET_MAX)))
        return self._charge("ask", self.store.ask(question, budget, scorer))

    def context(self, question: str) -> str:
        """The question with the compact `ask` reply in front of it, for the client to send
        as the user turn: the push form of `ask`, whatever the profile."""
        wait = self._not_ready()
        if wait:
            return self._charge("context", wait)
        reply = self.store.ask_compact(question, self.budget, self.scorer,
                                       widen_to=min(2 * self.budget, ASK_BUDGET_MAX))
        return self._charge("context", CONTEXT_TEMPLATE.format(context=reply,
                                                               question=question))

    def ingest(self, path_or_text: str, source: str = "") -> str:
        if not path_or_text.strip():
            src = source or f"text{len(self.store.docs) + 1}"
            return self._charge("ingest", f"source={src} error=empty input")
        p = _existing_path(path_or_text)
        if p:
            results = self.store.ingest_path(p)
        else:
            src = source or f"text{len(self.store.docs) + 1}"
            results = [self.store.ingest_text(path_or_text, src)]
        if self.profile == "compact":
            return self._charge("ingest", _ingest_summary(results))
        return self._charge("ingest", "\n".join(r if isinstance(r, str) else r.line()
                                                 for r in results))

    def query(self, entity: str, hops: int = 1, limit: int = 25) -> str:
        limit = max(1, int(limit))
        return self._charge("query", self._not_ready() or self.store.query(entity, hops, limit))

    def provenance(self, edge_ids: str) -> str:
        wait = self._not_ready()
        if wait:
            return self._charge("provenance", wait)
        ids = [int(x) for x in re.findall(r"\d+", edge_ids)]
        if not ids:
            return self._charge("provenance", f"no edge ids in {edge_ids!r}")
        return self._charge("provenance", self.store.provenance(ids))

    def search(self, term: str, limit: int = 8) -> str:
        limit = max(1, int(limit))
        return self._charge("search", self._not_ready() or self.store.search(term, limit))

    def entities(self, prefix: str = "", limit: int = 30) -> str:
        limit = max(1, int(limit))
        return self._charge("entities",
                            self._not_ready() or self.store.entities(prefix, limit))

    def stats(self) -> str:
        # introspection is not charged: it would count itself
        return self.store.stats()


def build_server(store: GraphStore, tools: Tools | None = None, corpus=None,
                 indexed_at_startup: bool = False):
    """The MCP server over `store`. `corpus` names the sources the compact `ask`
    describes (default: the documents already in the store); with `indexed_at_startup`
    the compact profile leaves `ingest` out, the operator having indexed the corpus."""
    from mcp.server.mcpserver import MCPServer
    t = tools or Tools(store)
    if t.profile == "compact":
        mcp = MCPServer("cogito", instructions=COMPACT_INSTRUCTIONS)
        _register_compact(mcp, t, sorted(store.docs) if corpus is None else list(corpus),
                          indexed_at_startup)
    else:
        mcp = MCPServer("cogito", instructions=INSTRUCTIONS)
        _register_full(mcp, t)
    _register_context(mcp, t)
    store.ledger.schema_tokens = schema_tokens(mcp)
    return mcp


def serve_stdio(server, on_claimed=None) -> None:
    """Serve over stdio and call `on_claimed` once the transport owns stdout.

    The SDK picks the descriptor it answers on from `sys.stdout` when it claims it, and the
    store silences stdout around model loads by pointing `sys.stdout` at stderr. Indexing
    started before the claim could land in that window: every reply then went to stderr and
    the client never saw the handshake. Started after it, the SDK has already diverted fd 1
    to stderr, so nothing printed by a load can reach the wire."""
    import anyio
    from mcp.server.stdio import stdio_server
    lowlevel = getattr(server, "_lowlevel_server", None)

    async def run() -> None:
        async with stdio_server() as (read, write):
            if on_claimed is not None:
                on_claimed()
            await lowlevel.run(read, write, lowlevel.create_initialization_options())

    anyio.run(run)


def _register_context(mcp, t: Tools) -> None:
    """A prompt, not a tool: the client or the user invokes it, never the model, and it is
    not part of the definitions re-sent on every model turn."""

    @mcp.prompt(description="The question preceded by the passages of the indexed documents "
                            "that answer it; send it as the user message.")
    def context(question: str) -> str:
        return t.context(question)


def schema_tokens(mcp) -> int:
    """Tokens of the tool definitions as a client forwards them: re-sent on every model
    turn whether or not a tool is called, so they are part of what the server costs. The
    listing is SDK-internal: a version that moves it reports 0 rather than stop the server."""
    manager = getattr(mcp, "_tool_manager", None)
    if manager is None:
        return 0
    return sum(ntok(json.dumps({"name": tool.name, "description": tool.description,
                                "input_schema": tool.parameters}))
               for tool in manager.list_tools())


def _register_compact(mcp, t: Tools, corpus: list, indexed_at_startup: bool) -> None:
    label = corpus_label(corpus)
    where = f"the indexed documents ({label})" if label else "the documents indexed with `ingest`"

    # measured on a 3B model: it split its calls between `ask` and a text search, and every
    # search fed the whole question as a pattern and found nothing; the description names
    # what `ask` does that a pattern cannot
    @mcp.tool(description=f"Answer a question from {where}. Pass the question as asked: it "
                          "finds the lines that answer it even when the wording differs, "
                          "which an exact text search does not. Use it before searching or "
                          "reading those files; one call is enough.")
    def ask(question: str) -> str:
        return t.ask(question)

    if indexed_at_startup:
        return

    @mcp.tool(description="Index a file or directory (txt, md, html, pdf) so `ask` can "
                          "answer from it. Uses no LLM tokens.")
    def ingest(path: str) -> str:
        return t.ingest(path)


def _register_full(mcp, t: Tools) -> None:
    default_budget = t.budget

    @mcp.tool()
    def ask(question: str, budget: int = default_budget, use_dense: bool = False,
            use_sonar: bool = False) -> str:
        """Start here for any question: graph facts, then a '--' line, then the source
        sentences that answer it, ranked. `budget` caps the reply in tokens (100-4000,
        40 % facts / 60 % sentences). Sentences are shortlisted by IDF and ordered by the
        learned relevance scorer; use_dense=True ranks the block by encoder cosine instead
        when the graph has embeddings (falls back to lexical with a note). The header's
        `p=` is that scorer's probability for its best sentence: under ~0.3 the material is
        probably not in this reply. Do not answer from a low-`p=` reply. First ask again
        with a larger budget; if `p=` is still low, use `query`, `provenance` or `search`
        before answering. Once any of them names the specific fact asked, even briefly or
        in passing, state it — do not keep answering NOT FOUND once you have found it. If
        they only surface something related without naming that fact, answer NOT FOUND;
        do not substitute a nearby or similar-sounding fact for the one actually asked."""
        return t.ask(question, budget, use_dense, use_sonar)

    @mcp.tool()
    def ingest(path_or_text: str, source: str = "") -> str:
        """Ingest a file, a directory (txt/md/html/pdf), or raw text into the graph.
        Runs the local Cogito model: costs 0 LLM tokens. Returns one line per document."""
        return t.ingest(path_or_text, source)

    @mcp.tool()
    def query(entity: str, hops: int = 1, limit: int = 25) -> str:
        """Facts about an entity, one per line `subject relation object #edge_ids`.
        Facts after '~ class-only, verify with provenance:' carry a coarse class label."""
        return t.query(entity, hops, limit)

    @mcp.tool()
    def provenance(edge_ids: str) -> str:
        """Exact source sentence + character spans for edge ids (comma-separated)."""
        return t.provenance(edge_ids)

    @mcp.tool()
    def search(term: str, limit: int = 8) -> str:
        """Sentences containing every word of `term`, in any order: the text fallback
        when the graph has no fact. Pass a few distinctive words, not a full question or
        a phrase copied verbatim from the question — the source text rarely repeats a
        multi-word phrase exactly."""
        return t.search(term, limit)

    @mcp.tool()
    def entities(prefix: str = "", limit: int = 30) -> str:
        """Known entities with their degree, optionally filtered by prefix."""
        return t.entities(prefix, limit)

    @mcp.tool()
    def stats() -> str:
        """Token ledger: raw tokens ingested vs tokens served, persistence status."""
        return t.stats()


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="cogito-mcp", description=INSTRUCTIONS)
    ap.add_argument("--dir", type=Path, default=Path(".cogito"),
                    help="project directory holding graph.json (default: ./.cogito)")
    ap.add_argument("--checkpoint", action="append", help="decoder checkpoint (repeatable)")
    ap.add_argument("--vocab", help="vocabulary json matching the checkpoints")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default=None,
                    help="text encoder (default: whatever the checkpoints were trained on)")
    ap.add_argument("--pool", type=Path, default=None,
                    help="learned pooling weights for m2m100-pool (default: the published file)")
    ap.add_argument("--no-download", dest="download", action="store_false",
                    help="never download weights or the spaCy model")
    ap.add_argument("--profile", choices=PROFILES, default="full",
                    help="tool surface: full (seven tools) or compact (`ask` alone, one "
                         "argument, plain replies; for models with little context)")
    ap.add_argument("--ingest", action="append", type=Path, default=None, metavar="PATH",
                    help="file or directory indexed at startup, in the background "
                         "(repeatable); the model then never has to call `ingest`")
    ap.add_argument("--lite", action="store_true",
                    help="index sentences only: no graph model is loaded or downloaded, so "
                         "the server starts at once; `ask` keeps its learned ranking, the "
                         "graph tools have no facts for documents indexed this way")
    ap.add_argument("--budget", type=int, default=600,
                    help=f"token budget of one `ask` reply ({ASK_BUDGET_MIN}-{ASK_BUDGET_MAX}; "
                         "default 600)")
    return ap.parse_args(argv)


def build_extractor(weights, ns: argparse.Namespace):
    """Extractor plus the startup canary. An encoder that disagrees with the
    checkpoints stops the server instead of serving triples from the wrong space.
    `ns.encoder` (not the resolved name) is forwarded: an unset flag lets legacy
    checkpoints keep their own encoder, and their space is announced here, once the
    checkpoints have decided it."""
    from cogito_estella.integrations import llamaindex_connector as connector
    point = weights.operating_point or (None, None)
    try:
        ex = connector.CogitoGraphExtractor(
            [str(c) for c in weights.checkpoints], str(weights.vocab),
            device=ns.device, encoder=ns.encoder, download=ns.download,
            pool_path=weights.pool, threshold=point[0], adj_threshold=point[1])
        ex.check_canary()
    except (EncoderMismatch, PoolWeightsError) as exc:
        sys.exit(f"cogito-mcp: {exc}")
    if weights.encoder is None:
        print(f"cogito-mcp: {_encoder_line(ex.encoder_name)}", file=sys.stderr)
    return ex


def _encoder_line(encoder: str) -> str:
    return f"encoder={encoder} revision={encoder_revision(encoder)}"


def _exit_missing_dependency(exc: ImportError) -> None:
    sys.exit(f"cogito-mcp: missing optional dependency {exc.name}; "
             "install with: pip install 'cogito-estella[mcp]'")


def main(argv=None) -> None:
    ns = parse_args(argv)
    try:                                       # a knob that names no scorer stops the run here,
        resolve_scorer()                       # before a model load makes the failure expensive
    except ValueError as exc:
        sys.exit(f"cogito-mcp: {exc}")
    try:
        weights = None if ns.lite else resolve(ns.checkpoint, ns.vocab, download=ns.download,
                                               encoder=ns.encoder, pool=ns.pool)
        ensure_spacy_model(download=ns.download)
    except WeightsError as exc:
        sys.exit(f"cogito-mcp: {exc}")
    except ImportError as exc:
        _exit_missing_dependency(exc)

    factory = TextOnlyExtractor if ns.lite else (lambda: build_extractor(weights, ns))
    store = GraphStore(extractor_factory=factory, path=ns.dir / "graph.json")
    store.load()
    # Explicit legacy checkpoints resolve their own encoder inside the extractor, which
    # is built lazily: name it there rather than guessing the default here.
    if ns.lite:
        space = "encoder=none (lite)"
    else:
        space = _encoder_line(weights.encoder) if weights.encoder else "encoder=from-checkpoints"
    print(f"cogito-mcp: graph {store.path} edges={len(store.edges)} docs={len(store.docs)} "
          + space, file=sys.stderr)
    tools = Tools(store, ns.profile, ns.budget)
    paths = ns.ingest or []
    corpus = sorted(set(store.docs) | {n for p in paths for n in source_names(p)})
    try:
        server = build_server(store, tools, corpus, indexed_at_startup=bool(paths))
        if getattr(server, "_lowlevel_server", None) is None:   # an SDK without the hook:
            tools.defer_indexing(paths)                         # index at the first call
            server.run("stdio")
        else:
            serve_stdio(server, (lambda: tools.index_in_background(paths)) if paths else None)
    except ImportError as exc:
        _exit_missing_dependency(exc)


if __name__ == "__main__":
    main()
