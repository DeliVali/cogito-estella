# Cogito Estella: Latent Graph Engine (v0.7.0)

Non-autoregressive inference backend that decodes SONAR (Meta) semantic embeddings
directly into knowledge graphs, bypassing token-based text decoding entirely.
Mapped, trained, and validated end-to-end on a single RTX 5070 (12 GB).

## Why

GraphRAG over millions of documents with autoregressive LLMs is budget-prohibitive:
per-token decoding, per-token API pricing, and malformed-output retry loops. Cogito
Estella emits the full adjacency structure in one dense forward pass — structurally
valid by construction, at consumer-hardware cost.

---

## Inference Architecture

The engine attaches compact non-autoregressive decoder heads (5.8M–45M parameters)
to Meta's frozen multilingual latent space. Input: a concept embedding
$e \in \mathbb{R}^{1024}$ (sentence segmented with SaT, projected by SONAR). No time
dimension — a single pass emits the complete graph.

```
e ∈ ℝ^1024
  │  deep trunk (optional): 1–3 × [Linear → GELU → LayerNorm]
  ▼
N ∈ ℝ^{K × d}          # K node slots (fixed grid) or C candidates (elastic)
  ├─ existence   s ∈ ℝ^K            = σ(W_s · N)
  ├─ labels      ℓ ∈ ℝ^{K × V}      = W_ℓ · N          (open-vocab head)
  └─ adjacency   A ∈ ℝ^{R × K × K}  = N_i^T W_r N_j    (low-rank: W_r = U_r V_r^T)
```

Two production head families:

* **`GraphDecoder`** (open-vocab): fixed K-slot grid, V-way label softmax per slot,
  bilinear adjacency. Optional deep GELU/LayerNorm trunk (`trunk_layers`).
* **`CandidateGraphDecoder`** (entity-conditioned): caller supplies candidate
  entities (text scan and/or graph-memory nodes); candidates act as cross-attention
  queries over learned concept views via `nn.Embedding` lookup. Output space is
  restricted by construction — zero vocabulary leakage. Elastic COO adjacency sized
  to the candidate list; low-rank bilinear relations (rank 64); calibrated decode
  with a force-top1 recall floor.

Training objective — structural loss, no sampling, no teacher forcing:
$\mathcal{L} = \mathrm{BCE}(\hat{s}, s) + \mathrm{CE}_{\text{masked}}(\hat{\ell}, \ell) + \mathrm{BCE}(\hat{A}, A)$.

---

## Metrics & Experimental Benchmark Summary

Held-out evaluation, split **by combination** (an entity/relation combination never
seen in training, eliminating duplicate leakage). Prose numbers are validated on
virgin slices of a 119,911-sample held-out pool that took no part in any model,
threshold, or ensemble selection; structured modalities use their own
combination-held splits (tool-calls: 12,147 held-out concepts spanning 240 unseen
combinations).

| Modality | Triple F1 | Recipe |
| :--- | :--- | :--- |
| Tool-calling / API control | **1.000** | 24.3M `GraphDecoder`, deterministic decode, zero-leakage combination split |
| Source code structure (Python AST oracle) | **0.781** | LoRA-adapted SONAR (r=32) + decoder, fixed-threshold decode |
| Entity-conditioned prose (restricted candidates) | **0.827** | 5-seed `CandidateGraphDecoder` ensemble, cross-attention entity conditioning, calibrated decode + force-top1 |
| Open-vocab prose (fallback, no candidates) | **0.6514** | 3-seed `GraphDecoder` ensemble + trained cascade as empty-decode fallback |

Oracles: exact-by-construction JSON (tool-calls), native Python `ast` parser (code),
deterministic dependency-parse SVO (prose). Adjacency thresholding ships three
strategies (`fixed`, `otsu`, `noise_floor`); the sparsity-prior `noise_floor`
dominates variance-based Otsu on sparse graphs (8/8 vs 3/8 stress scenarios).

## Core Latency & Compute Footprint

Measured on consumer silicon (RTX 5070, 12 GB, CUDA 12.8), batch 1024:

* Non-autoregressive forward pass: **0.013 ms per sentence concept**
  (single `CandidateGraphDecoder`; 0.052 ms for the 24.3M open-vocab production
  config; ~0.06 ms for the 5-model prose ensemble). 269–1074× faster wall-clock
  than autoregressive text decoding of the same content.
* Context state: **336 KB per active prefill thread** vs. 2.1 GB KV-cache on a
  standard 7B autoregressive model at 4k context — a **6,400× reduction**.
* FLOP reduction: **36×** against a matched char-level token baseline (dense
  structural blocks) up to **2,881×** for exact string literals via the parallel
  char-grid mapping (digits/characters as nodes, no copy loop).
* Encode throughput (SaT + sanitization + SONAR): ~273 concepts/s; storage density
  ~2.19 KB/concept (fp16).

## Production Integration Patterns

* **GraphRAG ingestion pipelines** — vector-to-knowledge-graph indexing without
  per-token commercial API calls; graphs land as COO triples ready for `MERGE`
  into any property-graph store, with per-edge provenance.
* **Non-autoregressive tool dispatchers** — raw text streams decoded into typed
  call structures in one pass; output is structurally valid by construction
  (no JSON repair, no syntax-breaking retries).
* **Incremental AST repository tracking** — hook file-save events to re-encode
  changed units and merge live call/import dependency sub-graphs into memory.
* **Multilingual agent memory shards** — chat history compressed into
  language-agnostic 1024-d concepts (SONAR, 200 languages) and decoded to a
  queryable graph; entity-conditioning accepts the agent's existing graph nodes
  as candidates.

---

## The Map of Literals (Open-Vocab Handling)

Continuous embeddings lose exact values (measured: MAE 36, 1.5% exact on held-out
integers via linear probe). The ceiling breaks at the data level, not the model:

* **Integers:** digits-as-nodes + atomic spacing (`"400"` → `"4 0 0"`): **98.3%**
  exact on unseen values.
* **Short strings (≤4 chars):** characters-as-nodes: **89.9%** exact, open vocab.
* **Long arbitrary strings:** hybrid copy channel (the only autoregressive path).

## Core Structure (`src/cogito_estella/`)

```
src/cogito_estella/
  model/graph_decoder.py       # non-AR GraphDecoder + deep trunk + decode strategies
  model/candidate_decoder.py   # entity-conditioned head: cross-attention, COO, force-top1
  model/transformer.py         # ConceptTransformer (RoPE, RMSNorm, SwiGLU)
  model/token_baseline.py      # matched-FLOP token baseline
  model/sonar_loss.py          # CE propagated through the frozen SONAR decoder
  model/train_graph.py         # resumable training loop
  sonar_codec.py               # SONAR encode/decode, OOM-resilient
  segmenter.py                 # SaT multilingual segmentation
  multilingual_factory.py      # canonicalized ingestion from open sources
  preprocess.py                # sanitization + atomic spacing (script-gated)
  graph_target.py              # target-graph construction
  graph_metrics.py             # Triple F1, GED proxy, tool-call F1
  compute.py                   # FLOP/bandwidth accounting, roofline
```

Training data, experiment scaffolding, and logs are untracked; the unit-test suite
ships with the repository — **113 tests**, fully reproducible offline:

```bash
uv sync
uv run pytest tests/
```

Requires Python 3.12; SONAR/SaT weights download on first use.

## License

Distributed under the **Apache License 2.0**. See the `LICENSE` file for details on
relational-patent protection and commercial use.
