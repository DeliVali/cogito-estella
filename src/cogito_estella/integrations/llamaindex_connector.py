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
from pathlib import Path

import torch

from cogito_estella.model.candidate_decoder import (
    CandidateDecoderConfig, CandidateGraphDecoder, decode_triples_coo)

_CYPHER = (
    "MERGE (a:Entity {name: $s}) "
    "MERGE (b:Entity {name: $o}) "
    "MERGE (a)-[r:REL {type: $r, source: $src}]->(b)"
)


class CogitoGraphExtractor:
    """text -> knowledge-graph triples in one non-autoregressive pass."""

    def __init__(self, checkpoint: str, vocab_path: str, device: str = None,
                 threshold: float = 0.15, force_top1: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        vocab = json.loads(Path(vocab_path).read_text())
        self.ent2id = vocab["ent2id"]
        self.rels = sorted(vocab["rel2id"], key=vocab["rel2id"].get)
        self.threshold, self.force_top1 = threshold, force_top1
        self.dec = CandidateGraphDecoder(
            CandidateDecoderConfig(n_relations=len(self.rels))).to(self.device)
        ck = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.dec.load_state_dict(ck["dec"])
        self.dec.eval()
        self._pipe = None

    def _encode(self, texts, lang):
        if self._pipe is None:
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
            self._pipe = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder",
                device=torch.device(self.device))
        return self._pipe.predict(texts, source_lang=lang).to(self.device)

    def _candidates(self, text, extra):
        # default scan: in-vocab tokens of the text; callers with a graph memory pass
        # their known nodes via `extra` (the hybrid text+memory candidate set)
        toks = {t.strip(".,;:!?()[]\"'").lower() for t in text.split()}
        cand = [t for t in toks if t in self.ent2id]
        for e in extra or []:
            if e in self.ent2id and e not in cand:
                cand.append(e)
        return cand

    def extract(self, text: str, candidates: list = None, lang: str = "eng_Latn",
                embedding: torch.Tensor = None) -> list:
        """Returns [(subject, relation, object), ...] as plain strings."""
        cand = self._candidates(text, candidates)
        if len(cand) < 2:
            return []
        emb = embedding if embedding is not None else self._encode([text], lang)[0]
        ids = torch.tensor([[self.ent2id[c] for c in cand]], device=self.device)
        mask = torch.ones(1, len(cand), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            out = self.dec(emb[None].float().to(self.device), ids, mask)
        coo = decode_triples_coo(out["exist_logits"].cpu(), out["adj_logits"].cpu(),
                                 mask.cpu(), threshold=self.threshold,
                                 force_top1=self.force_top1)[0]
        return [(cand[i], self.rels[r], cand[j]) for i, r, j in sorted(coo)]

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
