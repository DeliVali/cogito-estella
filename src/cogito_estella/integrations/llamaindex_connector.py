"""Plug-and-play graph extractor for LlamaIndex / LangChain pipelines.

Three lines to production:

    extractor = CogitoGraphExtractor("cogito-prose-candidates-ft.pt", "vocab-prose.json")
    triples = extractor.extract("The committee approved the new budget.")
    extractor.to_neo4j(driver, triples, source="doc-42")

No hard dependency on either framework: `extract` takes plain text (what both hand
you), returns `(subject, relation, object)` string triples. `to_neo4j` maps them to
standard Cypher MERGE statements with per-edge provenance; the neo4j driver ships in
the `[graph]` extra. SONAR (the `[sonar]` extra) is loaded lazily on first call.
"""
import json
import math
import re
from pathlib import Path

import torch

from cogito_estella.model.candidate_decoder import (
    CandidateDecoderConfig, CandidateGraphDecoder, decode_triples_coo)

_CYPHER = (
    "MERGE (a:Entity {name: $s}) "
    "MERGE (b:Entity {name: $o}) "
    "MERGE (a)-[r:REL {type: $r, source: $src}]->(b)"
)
_CYPHER_LITERAL = (
    "MERGE (e:Entity {name: $ent}) "
    "MERGE (l:Literal {value: $value, kind: $kind}) "
    "MERGE (e)-[r:HAS_LITERAL {source: $src}]->(l)"
)

# Exact literals never enter the semantic space: they are detected deterministically in
# the source text and stored VERBATIM. Semantic decoding carries meaning; this channel
# carries the strings that must never be approximated.
_LITERAL_PATTERNS = {
    "url": re.compile(r"https?://[^\s]+"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "uuid": re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                       r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    "token": re.compile(r"[A-Za-z0-9_\-+/=.]{16,}"),
    "hash": re.compile(r"\b[0-9a-f]{12,64}\b"),
    "phone": re.compile(r"\+?\d[\d\s().-]{6,16}\d"),
    "number": re.compile(r"\b\d+\.\d+\b|\b\d{5,}\b"),
    "id": re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])"
                     r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+){1,4}\b"),
}


def _entropy(s: str) -> float:
    from collections import Counter
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_literals(text: str, extra_patterns: dict = None,
                     redact_sensitive: bool = True) -> dict:
    """Deterministic verbatim literal detection. Returns {kind: [exact substrings]}.
    Precedence removes overlaps (a phone inside a URL is just the URL).

    `token` is the DYNAMIC detector: any opaque high-entropy string (nanoid, JWT,
    base64 ciphertext, API keys) is caught by its Shannon-entropy signature — no
    per-format pattern needed. Because such tokens are often secrets, they are
    stored as sha256 fingerprints by default (queryable, irrecoverable); pass
    redact_sensitive=False for verbatim. `extra_patterns` ({name: compiled_regex})
    lets callers add domain formats with top precedence."""
    kinds = list(extra_patterns or {}) + ["url", "email", "uuid", "token", "hash",
                                          "phone", "number", "id"]
    patterns = {**(extra_patterns or {}), **_LITERAL_PATTERNS}
    out = {k: [] for k in kinds}
    taken = []

    def overlaps(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    for kind in kinds:
        for m in patterns[kind].finditer(text):
            span, val = m.span(), m.group(0)
            if any(overlaps(span, t) for t in taken):
                continue
            if kind == "phone":
                digits = sum(c.isdigit() for c in val)
                # bare digit runs are numbers/ids; phones carry separators or '+'
                has_sep = val.startswith("+") or any(c in val for c in " ()-.")
                if not (8 <= digits <= 15 and has_sep):
                    continue
            if kind == "id" and sum(c.isdigit() for c in val) < 2:
                continue
            if kind == "token":
                has_digit = any(c.isdigit() for c in val)
                mixed_case = val.lower() != val and val.upper() != val
                if not ((has_digit or mixed_case) and _entropy(val) >= 3.9):
                    continue
                if redact_sensitive:
                    import hashlib
                    val = "sha256:" + hashlib.sha256(val.encode()).hexdigest()[:16]
            taken.append(span)
            out[kind].append(val)
    return out


def score_report(exist_logits, adj_logits, cand: list, rels: list,
                 threshold: float, adj_threshold: float, force_top1: bool) -> dict:
    """Explain mode: per-candidate existence probabilities and per-edge confidences.
    Same decision rule as decode_triples_coo, with every probability exposed and the
    force-top1 floor edge flagged (its low confidence stays visible)."""
    import torch as _t
    ep = _t.sigmoid(exist_logits).tolist()
    ap = _t.sigmoid(adj_logits)
    report = {"candidates": [{"name": c, "exist_prob": round(float(p), 4)}
                             for c, p in zip(cand, ep)],
              "edges": [], "triples": []}
    present = [p > threshold for p in ep]
    n = len(cand)
    for r in range(len(rels)):
        for i in range(n):
            for j in range(n):
                if i != j and present[i] and present[j]                         and float(ap[r, i, j]) > adj_threshold:
                    report["edges"].append(
                        {"s": cand[i], "r": rels[r], "o": cand[j],
                         "adj_prob": round(float(ap[r, i, j]), 4),
                         "exist_s": round(ep[i], 4), "exist_o": round(ep[j], 4),
                         "forced": False})
    if force_top1 and not report["edges"] and n >= 2:
        best, arg = -1.0, None
        for r in range(len(rels)):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    sc = float(ap[r, i, j]) * ep[i] * ep[j]
                    if sc > best:
                        best, arg = sc, (r, i, j)
        r, i, j = arg
        report["edges"].append({"s": cand[i], "r": rels[r], "o": cand[j],
                                "adj_prob": round(float(ap[r, i, j]), 4),
                                "exist_s": round(ep[i], 4), "exist_o": round(ep[j], 4),
                                "forced": True})
    report["triples"] = [(e["s"], e["r"], e["o"]) for e in report["edges"]]
    return report


class CogitoGraphExtractor:
    """text -> knowledge-graph triples in one non-autoregressive pass."""

    def __init__(self, checkpoint, vocab_path: str, device: str = None,
                 threshold: float = None, adj_threshold: float = None,
                 force_top1: bool = True):
        """`checkpoint`: a single path, or a list of paths for prob-averaged ensemble
        decoding (the validated 0.827 recipe ships as 5 checkpoints). Defaults follow
        the validated operating points: single model (0.15, 0.15); ensemble (0.1, 0.8)
        — precision-heavy edges with force-top1 as the recall floor."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        vocab = json.loads(Path(vocab_path).read_text())
        self.ent2id = vocab["ent2id"]
        self.rels = sorted(vocab["rel2id"], key=vocab["rel2id"].get)
        is_ens = not isinstance(checkpoint, (str, Path)) and len(list(checkpoint)) > 1
        self.threshold = threshold if threshold is not None else (0.1 if is_ens else 0.15)
        self.adj_threshold = (adj_threshold if adj_threshold is not None
                              else (0.8 if is_ens else 0.15))
        self.force_top1 = force_top1
        paths = [checkpoint] if isinstance(checkpoint, (str, Path)) else list(checkpoint)
        self.decs = []
        for path in paths:
            dec = CandidateGraphDecoder(
                CandidateDecoderConfig(n_relations=len(self.rels))).to(self.device)
            ck = torch.load(path, map_location=self.device, weights_only=False)
            dec.load_state_dict(ck["dec"])
            dec.eval()
            self.decs.append(dec)
        self._pipe = None

    def _encode(self, texts, lang):
        if self._pipe is None:
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
            self._pipe = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder",
                device=torch.device(self.device))
        return self._pipe.predict(texts, source_lang=lang).to(self.device)

    def _scanner(self):
        # spaCy scan (NOUN/PROPN lemmas) when available — matches the recall-1.0
        # production protocol; degrades to token matching with naive singularization
        if not hasattr(self, "_nlp"):
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm", disable=["ner"])
            except Exception:
                self._nlp = False
        return self._nlp

    def _candidates(self, text, extra):
        nlp = self._scanner()
        cand = []
        if nlp:
            for tok in nlp(text):
                if tok.pos_ in ("NOUN", "PROPN"):
                    lem = tok.lemma_.lower()
                    if lem in self.ent2id and lem not in cand:
                        cand.append(lem)
        else:
            seen = set()
            for raw in text.split():
                t = raw.strip(".,;:!?()[]\"'").lower()
                for form in (t, t[:-1] if t.endswith("s") else None,
                             t[:-2] if t.endswith("es") else None):
                    if form and form in self.ent2id and form not in seen:
                        seen.add(form); cand.append(form)
                        break
        for e in extra or []:
            if e in self.ent2id and e not in cand:
                cand.append(e)
        return cand

    def _decode(self, emb: torch.Tensor, cand: list) -> list:
        ids = torch.tensor([[self.ent2id[c] for c in cand]], device=self.device)
        mask = torch.ones(1, len(cand), dtype=torch.bool, device=self.device)
        eps, aps = [], []
        with torch.no_grad():
            for dec in self.decs:
                out = dec(emb[None].float().to(self.device), ids, mask)
                eps.append(torch.sigmoid(out["exist_logits"]))
                aps.append(torch.sigmoid(out["adj_logits"]))
        ep = torch.stack(eps).mean(0).clamp(1e-6, 1 - 1e-6)
        ap = torch.stack(aps).mean(0).clamp(1e-6, 1 - 1e-6)
        coo = decode_triples_coo(torch.logit(ep).cpu(), torch.logit(ap).cpu(),
                                 mask.cpu(), threshold=self.threshold,
                                 adj_threshold=self.adj_threshold,
                                 force_top1=self.force_top1)[0]
        return [(cand[i], self.rels[r], cand[j]) for i, r, j in sorted(coo)]

    def extract_with_literals(self, text: str, candidates: list = None,
                              lang: str = "eng_Latn"):
        """(triples, literals): semantic graph + verbatim exact strings, one call."""
        return self.extract(text, candidates=candidates, lang=lang), extract_literals(text)

    @staticmethod
    def ensure_schema(driver, database: str = None):
        """Create idempotent uniqueness constraints. REQUIRED for concurrent ingestion:
        without them, simultaneous MERGEs can race and duplicate nodes (measured).
        Migrating a pre-existing database: deduplicate first — constraint creation
        fails if duplicates already exist."""
        stmts = [
            "CREATE CONSTRAINT cogito_entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT cogito_literal_vk IF NOT EXISTS "
            "FOR (l:Literal) REQUIRE (l.value, l.kind) IS UNIQUE",
        ]
        with driver.session(database=database) as session:
            for q in stmts:
                session.run(q)

    @staticmethod
    def literals_to_neo4j(driver, triples: list, literals: dict,
                          source: str = "unknown", database: str = None):
        """Attach verbatim literals to the sentence's entities (co-occurrence linking;
        provenance on every HAS_LITERAL edge disambiguates)."""
        ents = sorted({e for s, r, o in triples for e in (s, o)})
        if not ents:
            return
        with driver.session(database=database) as session:
            for kind, values in literals.items():
                for value in values:
                    for ent in ents:
                        session.run(_CYPHER_LITERAL, ent=ent, value=value,
                                    kind=kind, src=source)

    def extract(self, text: str, candidates: list = None, lang: str = "eng_Latn",
                embedding: torch.Tensor = None, return_scores: bool = False):
        """Returns [(subject, relation, object), ...]; with return_scores=True, a full
        explain report: per-candidate existence probabilities and per-edge confidences
        (forced floor edges flagged with their low confidence visible)."""
        cand = self._candidates(text, candidates)
        if len(cand) < 2:
            return {"candidates": [], "edges": [], "triples": []} if return_scores else []
        emb = embedding if embedding is not None else self._encode([text], lang)[0]
        if not return_scores:
            return self._decode(emb, cand)
        ids = torch.tensor([[self.ent2id[c] for c in cand]], device=self.device)
        mask = torch.ones(1, len(cand), dtype=torch.bool, device=self.device)
        eps, aps = [], []
        with torch.no_grad():
            for dec in self.decs:
                out = dec(emb[None].float().to(self.device), ids, mask)
                eps.append(torch.sigmoid(out["exist_logits"]))
                aps.append(torch.sigmoid(out["adj_logits"]))
        ep = torch.stack(eps).mean(0).clamp(1e-6, 1 - 1e-6)[0]
        ap = torch.stack(aps).mean(0).clamp(1e-6, 1 - 1e-6)[0]
        return score_report(torch.logit(ep).cpu(), torch.logit(ap).cpu(), cand,
                            self.rels, self.threshold, self.adj_threshold,
                            self.force_top1)

    def extract_batch(self, texts: list, candidates: list = None,
                      lang: str = "eng_Latn") -> list:
        """Batched ingestion: ONE encoder call for N texts. Returns a list of triple
        lists aligned with `texts`. `candidates` is an optional per-text list."""
        cands = [self._candidates(t, (candidates or [None] * len(texts))[i])
                 for i, t in enumerate(texts)]
        emb = self._encode(texts, lang)
        results = []
        for i, cand in enumerate(cands):
            if len(cand) < 2:
                results.append([])
                continue
            results.append(self._decode(emb[i], cand))
        return results

    def to_neo4j(self, driver, triples: list, source: str = "unknown",
                 database: str = None):
        """MERGE triples into Neo4j with per-edge provenance. `driver` is a
        neo4j.Driver (pip install cogito-estella[graph])."""
        with driver.session(database=database) as session:
            for s, r, o in triples:
                session.run(_CYPHER, s=s, r=r, o=o, src=source)

    def to_cypher(self, triples: list, source: str = "unknown") -> list:
        """Driver-free variant: returns (query, params) pairs for any executor."""
        return [(_CYPHER, {"s": s, "r": r, "o": o, "src": source})
                for s, r, o in triples]
