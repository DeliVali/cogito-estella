"""Knowledge graph as an index: edges with span provenance, exact text on demand.
Edge ids are stable and never reused; documents are deduplicated by content hash."""
from __future__ import annotations

import hashlib
import re
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from cogito_estella.mcp.tokens import Ledger, ntok

BATCH = 64


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


class GraphStore:
    def __init__(self, extractor=None, extractor_factory=None, path: Path | None = None):
        self._ex = extractor
        self._factory = extractor_factory
        self.path = Path(path) if path else None
        self.edges: dict[int, dict] = {}
        self.next_id = 0
        self.adj: dict[str, list[int]] = defaultdict(list)
        self.docs: dict[str, dict] = {}
        self.ledger = Ledger()
        self.persisted_at: str | None = None

    @property
    def extractor(self):
        if self._ex is None:
            if self._factory is None:
                raise RuntimeError("GraphStore needs an extractor or an extractor_factory")
            with redirect_stdout(sys.stderr):      # stdout is the MCP wire
                self._ex = self._factory()
        return self._ex

    # -- ingestion -----------------------------------------------------------
    def split(self, text: str) -> list[tuple[str, int]]:
        """(sentence, absolute char offset) pairs; regex fallback without spaCy."""
        nlp = self.extractor._scanner()
        sents = ([s.text.strip() for s in nlp(text).sents] if nlp
                 else [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)])
        out, cursor = [], 0
        for s in sents:
            if not s:
                continue
            i = text.find(s, cursor)
            i = cursor if i < 0 else i
            out.append((s, i))
            cursor = i + len(s)
        return out

    def ingest_text(self, text: str, source: str) -> IngestResult:
        t0 = time.time()
        digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
        prev = self.docs.get(source)
        if prev and prev["sha256"] == digest:
            return IngestResult(source, "unchanged", prev["sentences"], 0, prev["tokens"],
                                0, 0.0, round(time.time() - t0, 2))
        status = "replaced" if prev else "ingested"
        if prev:
            self.drop_source(source)
        ex = self.extractor
        sents = self.split(text)
        texts = [s for s, _ in sents]
        offsets = [o for _, o in sents]
        recs_per: list[list[dict]] = []
        with redirect_stdout(sys.stderr):
            for i in range(0, len(texts), BATCH):
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
        self.ledger.raw_tokens += raw
        self.docs[source] = {"sha256": digest, "tokens": raw, "sentences": len(sents),
                             "sents": sents}
        graph_tok = ntok(self.render_edges(
            [e for e in self.edges.values() if e["source"] == source]))
        self.save()
        return IngestResult(source, status, len(sents), n_new, raw, graph_tok,
                            round(raw / max(graph_tok, 1), 1), round(time.time() - t0, 2))

    def _add_edge(self, rec: dict) -> None:
        self.edges[rec["id"]] = rec
        self.adj[rec["s"]].append(rec["id"])
        self.adj[rec["o"]].append(rec["id"])
        self.next_id = max(self.next_id, rec["id"] + 1)

    def drop_source(self, source: str) -> int:
        """Retire every edge of `source`; ids are not reused. Returns edges dropped."""
        gone = [i for i, e in self.edges.items() if e["source"] == source]
        for i in gone:
            del self.edges[i]
        self.docs.pop(source, None)
        self._rebuild_adj()
        return len(gone)

    def _rebuild_adj(self) -> None:
        self.adj = defaultdict(list)
        for i in sorted(self.edges):
            e = self.edges[i]
            self.adj[e["s"]].append(i)
            self.adj[e["o"]].append(i)

    def sentence_of(self, edge: dict) -> str:
        return self.docs[edge["source"]]["sents"][edge["sent_idx"]][0]

    # -- rendering helpers (tool renderers arrive in Task 4) --------------------
    @staticmethod
    def render_edges(edges: list[dict]) -> str:
        groups: dict[tuple, list[int]] = {}
        for e in edges:
            groups.setdefault((e["s"], e["r"], e["o"]), []).append(e["id"])
        return "\n".join(f"{s} {r} {o} #{','.join(map(str, ids))}"
                         for (s, r, o), ids in groups.items())

    # -- persistence (Task 5) ---------------------------------------------------
    def save(self) -> None:
        return None

    def load(self) -> None:
        return None
