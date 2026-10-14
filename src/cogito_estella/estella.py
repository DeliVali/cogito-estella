"""The shortest way in: `Estella("docs/").context(question)`.

A harness that wants the passages of its documents that answer a question, without MCP,
without a model download and in a few hundred tokens instead of the whole file."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from cogito_estella.mcp.server import ASK_BUDGET_MAX, CONTEXT_TEMPLATE, DEFAULT_SCORER
from cogito_estella.mcp.store import GraphStore, TextOnlyExtractor


class Estella:
    """Documents indexed for questions.

    `Estella("docs/", "notes.md")` indexes files or directories (txt, md, html, pdf) with
    the lite index: spaCy only, ready in seconds. `full=True` loads the graph model instead
    (needs `pip install 'cogito-estella[full]'`). `save_to=".cogito"` keeps the index on
    disk so the next run does not index unchanged files again."""

    def __init__(self, *paths, full: bool = False, budget: int = 600,
                 save_to: str | Path | None = None):
        if full:
            from cogito_estella.mcp.server import build_extractor
            from cogito_estella.mcp.weights import ensure_spacy_model, resolve
            weights = resolve(None, None)
            ensure_spacy_model()
            factory = lambda: build_extractor(
                weights, Namespace(device=None, encoder=None, download=True))
        else:
            factory = TextOnlyExtractor
        self.budget = budget
        self.store = GraphStore(extractor_factory=factory,
                                path=Path(save_to) / "graph.json" if save_to else None)
        self.store.load()
        for p in paths:
            self.add(p)

    def add(self, path_or_text: str | Path, name: str = "") -> None:
        """Index a file, a directory, or raw text (named `name`)."""
        from cogito_estella.mcp.server import _existing_path   # raw text is never a path
        p = path_or_text if isinstance(path_or_text, Path) else _existing_path(path_or_text)
        if p is not None:
            self.store.ingest_path(p)
        else:
            self.store.ingest_text(str(path_or_text), name or f"text{len(self.store.docs) + 1}")

    def ask(self, question: str) -> str:
        """The passages that answer `question`, one `[file § section] passage` per line."""
        return self.store.ask_compact(question, self.budget, DEFAULT_SCORER,
                                      widen_to=min(2 * self.budget, ASK_BUDGET_MAX))

    def context(self, question: str) -> str:
        """`question` preceded by those passages: send it to any model as the user message."""
        return CONTEXT_TEMPLATE.format(context=self.ask(question), question=question)
