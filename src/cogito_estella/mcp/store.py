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
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cogito_estella.mcp.rank import LexicalScorer, SonarScorer, rank, singular_forms
from cogito_estella.mcp.tokens import Ledger, ntok

BATCH = 64
DIVIDER = "~ class-only, verify with provenance:"
FACT_SHARE = 0.4                # of `ask`'s budget; the sentences take the rest
ASK_ENTITIES = 3                # entities resolved from one question
ASK_MAX_SKIPS = 32              # consecutive oversize sentences before the fill stops
SONAR_COS_FLOOR = 0.2           # below this cosine a sentence is not about the question
SONAR_FLOOR = (SONAR_COS_FLOOR + 1.0) / 2.0        # the same floor on the scorer's [0, 1] scale
SONAR_OFFSET = 1.0              # embedded rows rank above lexical ones: no shared origin
# question words too generic to be worth a graph hop
_GENERIC = ("model models paper use uses used result results work approach method "
            "methods data table figure")
ASK_STOPLIST = frozenset(_GENERIC.split())
_QWORD = re.compile(r"[a-z0-9][a-z0-9\-]*")


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
        self.emb: dict[str, np.ndarray] = {}       # source -> [n_sents, D] float16, normalized
        self._sindex: tuple | None = None          # flat sentence universe, rebuilt on change
        self._emb_warned = False
        self.ledger = Ledger()
        self.persisted_at: str | None = None
        self._lock = threading.RLock()     # sync tools run in worker threads: serialize mutation

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

    def ingest_text(self, text: str, source: str, save: bool = True) -> IngestResult:
        with self._lock:
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
            self.docs[source] = {"sha256": digest, "tokens": raw, "sentences": len(sents),
                                 "sents": sents}
            self._embed_source(ex, source, texts)
            self._sindex = None
            self._sync_raw_tokens()
            graph_tok = ntok(self.render_edges(
                [e for e in self.edges.values() if e["source"] == source]))
            if save:
                self.save()
            return IngestResult(source, status, len(sents), n_new, raw, graph_tok,
                                round(raw / max(graph_tok, 1), 1), round(time.time() - t0, 2))

    def ingest_path(self, path: str | Path) -> list[IngestResult | str]:
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
                           "(txt, md, html, htm, pdf)")
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
            self._sindex = None
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
        try:
            with redirect_stdout(sys.stderr):      # stdout is the MCP wire
                emb = np.asarray(encode(texts), dtype=np.float16)
        except Exception as exc:                   # noqa: BLE001 - encoder failure is a fallback
            if not self._emb_warned:               # one line per process, not per document
                self._emb_warned = True
                print(f"cogito-mcp: sentence embeddings unavailable ({exc}); "
                      "ask ranks lexically", file=sys.stderr)
            return
        if emb.ndim == 2 and emb.shape[0] == len(texts):
            self.emb[source] = emb

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
        q = name.strip().lower()
        if not q:
            return None
        if q in self.adj:
            return q
        if len(q) < 2:               # too short for prefix/substring/plural guessing
            return None
        cands = [e for e in self.adj
                 if e.startswith(q) or (len(e) >= 3 and q.startswith(e))
                 or (len(q) >= 4 and q in e)]
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
        t = term.strip().lower()
        hits = [f"{src} s{i}: \"{s}\"" for src, d in self.docs.items()
                for i, (s, _) in enumerate(d["sents"]) if t in s.lower()]
        more = "" if len(hits) <= limit else f"\n+{len(hits) - limit} more (raise limit)"
        return ("\n".join(hits[:limit]) + more) if hits else f"no sentence mentions '{term}'"

    def entities(self, prefix: str = "", limit: int = 30) -> str:
        ents = self.top_entities(limit, prefix)
        return ", ".join(f"{e}({len(self.adj[e])})" for e in ents) or "(empty graph)"

    def ask(self, question: str, budget: int = 600, scorer: str = "auto") -> str:
        """One call from a question to the facts and sentences that answer it, within budget."""
        with self._lock:
            texts, keys, pos = self.sentence_index()
            ents = self.question_entities(question)
            edges, seen = [], set()
            for ent in ents:
                for e in self.neighborhood(ent, 1):
                    if e["id"] not in seen:
                        seen.add(e["id"])
                        edges.append(e)
            # every retrieved fact boosts its own sentence, whether or not the line survives
            boost = {pos[(e["source"], e["sent_idx"])] for e in edges
                     if (e["source"], e["sent_idx"]) in pos}
            scores, name = self._score_sentences(question, texts, keys, scorer, boost)
            if not ents and not any(s > 0 for s in scores):
                return f"no material for '{question}'"
            head = f"entities: {', '.join(ents) or '(none)'} · scorer={name}"
            body = [head, *self._fact_lines(edges, head, int(budget * FACT_SHARE))]
            sents = self._sentence_lines(texts, keys, scores, body, budget)
            return "\n".join(body + (["--"] + sents if sents else []))

    def _embedded_note(self) -> str:
        """`(n/m docs)` when only part of the corpus is embedded: a mixed ranking must say so."""
        embedded = sum(1 for src in self.docs if src in self.emb)
        return "" if embedded >= len(self.docs) else f" ({embedded}/{len(self.docs)} docs)"

    def _score_sentences(self, question, texts, keys, requested, boost):
        """(scores, scorer name). SONAR ranks the embedded sentences, the lexical ones rank
        strictly after them: (cos + 1) / 2 floors near 0.5 while an overlap-free sentence
        scores 0, so the two scales must never be compared row by row."""
        lexical = LexicalScorer(texts).score(question, boost)
        note = "lexical (sonar unavailable)" if requested == "sonar" else "lexical"
        if requested == "lexical" or not texts:
            return lexical, "lexical"
        mat, mask = self._embedding_matrix(keys)
        if mat is None:
            return lexical, note
        try:
            with redirect_stdout(sys.stderr):
                q = np.asarray(self.extractor.encode_batch([question]), dtype=np.float32)
                sonar = SonarScorer(mat, lambda _texts: q).score(question, boost)
        except Exception as exc:                   # noqa: BLE001 - a missing encoder is a fallback
            print(f"cogito-mcp: question encoding failed ({exc}); ask ranks lexically",
                  file=sys.stderr)
            return lexical, note
        out = []
        for i in range(len(texts)):
            if not mask[i]:
                out.append(lexical[i])
            else:                                  # the floor keeps `no material` reachable
                out.append(SONAR_OFFSET + sonar[i] if sonar[i] >= SONAR_FLOOR else 0.0)
        return out, f"sonar{self._embedded_note()}"

    @staticmethod
    def _fit(block: list, lines: list, cap: int) -> list:
        """Lines that keep the joined block within `cap` tokens; truncation is line-granular."""
        out: list[str] = []
        for line in lines:
            if ntok("\n".join(block + out + [line])) > cap:
                break
            out.append(line)
        return out

    def _fact_lines(self, edges: list, head: str, cap: int) -> list:
        lex = [e for e in edges if e["r_lex"]]
        cls = [e for e in edges if not e["r_lex"]]
        out = self._fit([head], self.render_edges(lex).split("\n") if lex else [], cap)
        if cls:
            tail = self._fit([head, *out, DIVIDER], self.render_edges(cls).split("\n"), cap)
            if tail:                               # a divider with nothing under it is noise
                out += [DIVIDER, *tail]
        return out

    def _sentence_lines(self, texts, keys, scores, body, budget) -> list:
        out: list[str] = []
        seen: set[str] = set()
        skips = 0
        for i in rank(scores):
            if scores[i] <= 0:
                break
            if texts[i] in seen:
                continue
            src, si = keys[i]
            line = f'{src} s{si}: "{texts[i]}"'
            if ntok("\n".join(body + ["--", *out, line])) > budget:
                skips += 1                         # oversize line: shorter ones may still fit
                if skips >= ASK_MAX_SKIPS:
                    break
                continue
            skips = 0
            seen.add(texts[i])
            out.append(line)
        return out

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
            self._save_embeddings()
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
            self._sindex = None
            self._load_embeddings()
            self._sync_raw_tokens()
            self.persisted_at = datetime.fromtimestamp(
                self.path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")  # noqa: UP017

    # -- embedding sidecar (<graph>.emb.npz) -----------------------------------------
    def _emb_path(self) -> Path | None:
        return self.path.with_suffix(".emb.npz") if self.path else None

    def _save_embeddings(self) -> None:
        p = self._emb_path()
        if p is None:
            return
        if not self.emb:
            p.unlink(missing_ok=True)              # never leave a sidecar without a graph
            return
        sources = list(self.emb)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with tmp.open("wb") as fh:                 # a handle keeps savez from appending .npz
            np.savez(fh, sources=np.array(sources),
                     counts=np.array([self.emb[s].shape[0] for s in sources]),
                     # content hash per block: a same-length replacement must not pass
                     shas=np.array([self.docs.get(s, {}).get("sha256", "") for s in sources]),
                     emb=np.concatenate([self.emb[s] for s in sources], axis=0))
        os.replace(tmp, p)

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
            if len(sources) != len(counts) or len(sources) != len(shas) \
                    or sum(counts) != emb.shape[0]:
                raise ValueError("sidecar counts do not match the matrix")
            out, off = {}, 0
            for src, n, sha in zip(sources, counts, shas, strict=True):
                block, off = emb[off:off + n], off + n
                doc = self.docs.get(src, {})
                # the hash pins the block to the exact text it was encoded from;
                # the count is a cheap redundancy. Any disagreement: lexical is safer
                if doc.get("sha256") == sha and doc.get("sentences") == n:
                    out[src] = block
            self.emb = out
        except (OSError, ValueError, KeyError, EOFError) as exc:
            print(f"cogito-mcp: embeddings sidecar unusable ({exc}); ask ranks lexically",
                  file=sys.stderr)
