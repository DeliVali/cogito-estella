"""Cogito MCP server (stdio): ingest once, route every question through `ask`,
go deeper with query/provenance/search. Every tool reply is charged to the token ledger."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from cogito_estella.encoders import ENCODERS, EncoderMismatch, encoder_revision
from cogito_estella.encoders.pooling import PoolWeightsError
from cogito_estella.mcp.store import GraphStore
from cogito_estella.mcp.weights import WeightsError, ensure_spacy_model, resolve

INSTRUCTIONS = ("Knowledge-graph memory over documents. `ingest` a file, directory or text "
                "once (zero LLM tokens). Start every question with `ask`: it returns the graph "
                "facts and the sentences that answer it within a token budget. Use `query`, "
                "`provenance` and `search` only to go deeper when `ask` is not enough; facts "
                "under the '~ class-only' divider need `provenance` before you rely on them.")

ASK_BUDGET_MIN, ASK_BUDGET_MAX = 100, 4000


def _existing_path(value: str) -> Path | None:
    """Path when `value` names an existing file or directory; None for raw text."""
    try:
        p = Path(value)
        return p if p.exists() else None
    except (OSError, ValueError):     # e.g. "File name too long" for a pasted paragraph
        return None


class Tools:
    """Tool bodies bound to one store; plain callables so they can be unit-tested."""

    def __init__(self, store: GraphStore):
        self.store = store

    def _charge(self, tool: str, out: str) -> str:
        self.store.ledger.charge(tool, out)
        return out

    def ask(self, question: str, budget: int = 600, use_sonar: bool = False) -> str:
        budget = min(max(int(budget), ASK_BUDGET_MIN), ASK_BUDGET_MAX)
        scorer = "sonar" if use_sonar else "lexical"
        return self._charge("ask", self.store.ask(question, budget, scorer))

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
        return self._charge("ingest", "\n".join(r if isinstance(r, str) else r.line()
                                                 for r in results))

    def query(self, entity: str, hops: int = 1, limit: int = 25) -> str:
        limit = max(1, int(limit))
        return self._charge("query", self.store.query(entity, hops, limit))

    def provenance(self, edge_ids: str) -> str:
        ids = [int(x) for x in re.findall(r"\d+", edge_ids)]
        if not ids:
            return self._charge("provenance", f"no edge ids in {edge_ids!r}")
        return self._charge("provenance", self.store.provenance(ids))

    def search(self, term: str, limit: int = 8) -> str:
        limit = max(1, int(limit))
        return self._charge("search", self.store.search(term, limit))

    def entities(self, prefix: str = "", limit: int = 30) -> str:
        limit = max(1, int(limit))
        return self._charge("entities", self.store.entities(prefix, limit))

    def stats(self) -> str:
        # introspection is not charged: it would count itself
        return self.store.stats()


def build_server(store: GraphStore):
    from mcp.server.mcpserver import MCPServer
    mcp = MCPServer("cogito", instructions=INSTRUCTIONS)
    t = Tools(store)

    @mcp.tool()
    def ask(question: str, budget: int = 600, use_sonar: bool = False) -> str:
        """Start here for any question: graph facts, then a '--' line, then the source
        sentences that answer it, ranked. `budget` caps the reply in tokens (100-4000,
        40 % facts / 60 % sentences). IDF ranking by default; use_sonar=True ranks the
        sentence block with SONAR embeddings when the graph has them (falls back to
        lexical with a note). Go deeper with `query`, `provenance` and `search` only
        when this reply is not enough."""
        return t.ask(question, budget, use_sonar)

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
        """Sentences mentioning a term: the text fallback when the graph has no fact."""
        return t.search(term, limit)

    @mcp.tool()
    def entities(prefix: str = "", limit: int = 30) -> str:
        """Known entities with their degree, optionally filtered by prefix."""
        return t.entities(prefix, limit)

    @mcp.tool()
    def stats() -> str:
        """Token ledger: raw tokens ingested vs tokens served, persistence status."""
        return t.stats()

    return mcp


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
                    help="learned pooling weights (m2m100-pool; env COGITO_POOL)")
    ap.add_argument("--no-download", dest="download", action="store_false",
                    help="never download weights or the spaCy model")
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
    try:
        weights = resolve(ns.checkpoint, ns.vocab, download=ns.download,
                          encoder=ns.encoder, pool=ns.pool)
        ensure_spacy_model(download=ns.download)
    except WeightsError as exc:
        sys.exit(f"cogito-mcp: {exc}")
    except ImportError as exc:
        _exit_missing_dependency(exc)

    store = GraphStore(extractor_factory=lambda: build_extractor(weights, ns),
                       path=ns.dir / "graph.json")
    store.load()
    # Explicit legacy checkpoints resolve their own encoder inside the extractor, which
    # is built lazily: name it there rather than guessing the default here.
    print(f"cogito-mcp: graph {store.path} edges={len(store.edges)} docs={len(store.docs)} "
          + (_encoder_line(weights.encoder) if weights.encoder else "encoder=from-checkpoints"),
          file=sys.stderr)
    try:
        build_server(store).run("stdio")
    except ImportError as exc:
        _exit_missing_dependency(exc)


if __name__ == "__main__":
    main()
