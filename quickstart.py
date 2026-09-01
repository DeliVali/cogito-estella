"""Quickstart: raw sentence -> knowledge-graph triples in one forward pass.

Setup:
    uv sync
    Download from the v0.7.0 release (or Hugging Face: DeliVali/cogito-estella):
        cogito-prose-candidates-ft.pt  vocab-prose.json
    Place both next to this script. SONAR weights download on first use.
"""
import json

import torch

from cogito_estella.model.candidate_decoder import (
    CandidateDecoderConfig, CandidateGraphDecoder, decode_triples_coo)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

SENTENCES = [
    "The committee approved the new budget for the hospital.",
    "Researchers published a study that links exercise and memory.",
]
# Candidate entities per sentence: in production these come from a noun scan of the
# text plus the agent's existing graph nodes. Lemmatized lowercase forms.
CANDIDATES = [
    ["committee", "budget", "hospital", "career", "molecule"],
    ["researcher", "study", "exercise", "memory", "kitten", "engine"],
]


def main():
    vocab = json.load(open("vocab-prose.json"))
    ent2id = vocab["ent2id"]
    rels = sorted(vocab["rel2id"], key=vocab["rel2id"].get)

    dec = CandidateGraphDecoder(CandidateDecoderConfig(n_relations=len(rels))).to(DEV)
    dec.load_state_dict(torch.load("cogito-prose-candidates-ft.pt",
                                   map_location=DEV, weights_only=False)["dec"])
    dec.eval()

    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    pipe = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                        tokenizer="text_sonar_basic_encoder",
                                        device=torch.device(DEV))
    emb = pipe.predict(SENTENCES, source_lang="eng_Latn").to(DEV)

    for text, cands, e in zip(SENTENCES, CANDIDATES, emb):
        known = [c for c in cands if c in ent2id]
        ids = torch.tensor([[ent2id[c] for c in known]], device=DEV)
        mask = torch.ones(1, len(known), dtype=torch.bool, device=DEV)
        with torch.no_grad():
            out = dec(e[None].float(), ids, mask)
        triples = decode_triples_coo(out["exist_logits"].cpu(), out["adj_logits"].cpu(),
                                     mask.cpu(), threshold=0.15, force_top1=True)[0]
        print(f"\n{text}")
        for i, r, j in sorted(triples):
            print(f"  ({known[i]}) -[{rels[r]}]-> ({known[j]})")


if __name__ == "__main__":
    main()
