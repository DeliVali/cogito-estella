"""Tool-call observability demo — the F1 = 1.000 head, live.

Ingests a raw asynchronous tool-call stream (the exact text an agent framework logs),
encodes each event once with SONAR, and decodes the full adjacency graph in a single
non-autoregressive pass. Prints verbatim JSON triples per event plus split timings
(encoder vs decoder) to substantiate real-time deterministic stability.

Usage:
    uv run python tools/demo_observability.py [--checkpoint cogito-toolcalls-graphdecoder.pt]

The checkpoint ships with release v0.7.0 (GitHub Releases / Hugging Face).
"""
import argparse
import json
import time

import torch

from cogito_estella import graph_target as gt
from cogito_estella.model.graph_decoder import (GraphDecoder, GraphDecoderConfig,
                                                decode_triples)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, R, V, D = 24, 48, 8192, 448     # production preset (exp020)

STREAM = [
    'ASSISTANT: <functioncall> {"name": "get_weather", "arguments": {"query": "clima CDMX", "limit": 3, "verbose": false}}',
    'ASSISTANT: <functioncall> {"name": "search_web", "arguments": {"query": "flights to Madrid", "limit": 10, "verbose": true}}',
    'ASSISTANT: <functioncall> {"name": "query_db", "arguments": {"query": "SELECT * FROM users", "limit": 25, "verbose": false}}',
    'ASSISTANT: <functioncall> {"name": "create_event", "arguments": {"query": "reunión lunes", "limit": 1, "verbose": true}}',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="cogito-toolcalls-graphdecoder.pt")
    args = ap.parse_args()

    vocab = gt.build_toolcall_vocab()          # fully deterministic, no data required
    id2label = {i: l for l, i in vocab.label2id.items()}
    id2rel = {i: r for r, i in vocab.rel2id.items()}

    cfg = GraphDecoderConfig(concept_dim=1024, max_nodes=K, node_dim=D,
                             node_vocab=V, n_relations=R)
    dec = GraphDecoder(cfg).to(DEV)
    dec.load_state_dict(torch.load(args.checkpoint, map_location=DEV,
                                   weights_only=False)["model"])
    dec.eval()

    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    pipe = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                        tokenizer="text_sonar_basic_encoder",
                                        device=torch.device(DEV))
    t0 = time.perf_counter()
    emb = pipe.predict(STREAM, source_lang="eng_Latn").to(DEV)
    t_enc = (time.perf_counter() - t0) * 1000 / len(STREAM)

    with torch.no_grad():
        for _ in range(3):                     # warmup for honest decoder timing
            dec(emb.float())
        if DEV == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = dec(emb.float())
        if DEV == "cuda":
            torch.cuda.synchronize()
        t_dec = (time.perf_counter() - t1) * 1000 / len(STREAM)
    triples = decode_triples(out["exist_logits"].cpu(), out["label_logits"].cpu(),
                             out["adj_logits"].cpu())

    print(f"device={DEV} · encoder {t_enc:.2f} ms/event · decoder {t_dec:.4f} ms/event\n")
    for text, tr in zip(STREAM, triples):
        graph = sorted([id2label[i], id2rel[r], id2label[j]] for i, r, j in tr)
        print(text)
        print(json.dumps({"triples": graph}, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
