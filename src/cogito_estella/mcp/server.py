"""Cogito MCP server (stdio): ingest once, answer entity questions from the graph,
verify with provenance. Every tool reply is charged to the token ledger."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from cogito_estella.mcp.store import GraphStore
from cogito_estella.mcp.weights import WeightsError, ensure_spacy_model, resolve

INSTRUCTIONS = ("Knowledge-graph memory over documents. `ingest` a file, directory or text "
                "once (zero LLM tokens), then answer entity questions with `query`; facts under "
                "the '~ class-only' divider need `provenance` before you rely on them. "
                "`search` is the text fallback when the graph has no fact.")


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
        return self._charge("query", self.store.query(entity, hops, limit))

    def provenance(self, edge_ids: str) -> str:
        ids = [int(x) for x in re.findall(r"\d+", edge_ids)]
        return self._charge("provenance", self.store.provenance(ids))

    def search(self, term: str, limit: int = 8) -> str:
        return self._charge("search", self.store.search(term, limit))

    def entities(self, prefix: str = "", limit: int = 30) -> str:
        return self._charge("entities", self.store.entities(prefix, limit))

    def stats(self) -> str:
        return self.store.stats()


def build_server(store: GraphStore):
    from mcp.server.mcpserver import MCPServer
    mcp = MCPServer("cogito", instructions=INSTRUCTIONS)
    t = Tools(store)

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
    ap.add_argument("--no-download", dest="download", action="store_false",
                    help="never download weights or the spaCy model")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    ns = parse_args(argv)
    try:
        ckpts, vocab = resolve(ns.checkpoint, ns.vocab, download=ns.download)
        ensure_spacy_model(download=ns.download)
    except WeightsError as exc:
        sys.exit(f"cogito-mcp: {exc}")

    def factory():
        from cogito_estella.integrations.llamaindex_connector import CogitoGraphExtractor
        return CogitoGraphExtractor([str(c) for c in ckpts], str(vocab), device=ns.device)

    store = GraphStore(extractor_factory=factory, path=ns.dir / "graph.json")
    store.load()
    print(f"cogito-mcp: graph {store.path} edges={len(store.edges)} docs={len(store.docs)}",
          file=sys.stderr)
    build_server(store).run("stdio")


if __name__ == "__main__":
    main()
