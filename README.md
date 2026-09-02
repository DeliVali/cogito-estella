# Cogito Estella: Latent Graph Engine (v0.11.0)

Non-autoregressive inference backend that decodes SONAR (Meta) semantic embeddings
directly into knowledge graphs, bypassing token-based text decoding entirely.
Mapped, trained, and validated end-to-end on a single RTX 5070 (12 GB).

![Cogito Estella demo: text to knowledge graph in one forward pass, with cross-lingual extraction](assets/demo.gif)

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

<details>
<summary><b>Demo: tool-call observability (F1 = 1.000)</b> — raw agent stream → exact per-event call graphs</summary>

![Tool-call observability demo: raw functioncall stream decoded into exact star graphs per event](assets/demo_tools.gif)
</details>

<details>
<summary><b>Demo: code → call/import graph (F1 = 0.781)</b> — one line can yield multiple edges</summary>

![Code demo: imports and calls decoded into a dependency graph, including two edges from one nested call](assets/demo_code.gif)
</details>

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

## Use Cases

* **Agent long-term memory ("second brain")** — every message, note, or document an
  agent touches becomes graph triples in a property store (Neo4j, SQLite); sessions
  end, knowledge persists and stays queryable.
* **GraphRAG grounding** — answer-time retrieval over structured facts instead of
  (or alongside) vector similarity; the LLM cites edges that exist, reducing
  hallucinated recall.
* **Tool-call observability** — agent traces decoded at F1 1.000 make "which tools
  touched service X" an exact graph query, at near-zero compute.
* **Codebase intelligence** — call/import dependency graphs over whole repositories
  at a cost where re-indexing on every save is affordable.
* **Multilingual knowledge consolidation** — SONAR's language-agnostic space lets
  documents in 200 languages land in one shared graph.
* **Log & trace mining** — high-volume streams (support tickets, incident reports,
  transcripts) structured continuously on one consumer GPU, where per-token LLM
  extraction is cost-prohibitive.
* **Compliance & provenance KBs** — every edge carries source metadata; audits
  answer "where does this fact come from" by construction.

## Global Summarization (`GraphSummarizer`)

GraphRAG-style sensemaking where the expensive stage (per-chunk extraction) costs zero
LLM tokens; the LLM only writes per-community summaries (~10 calls per book instead of
~600), with automatic entity-grounding rejection of unverifiable output:

```python
from cogito_estella.graph_summary import GraphSummarizer

gs = GraphSummarizer(llm_fn=my_llm)          # any callable: prompt -> str
report = gs.summarize(triples, top_k=8)      # triples from the extractor
for c in report:
    print(c["accepted"], c["grounding"], c["top_entities"][:5], c["summary"])
```

Validated on a full novel: coherent thematic clusters at ~155x fewer LLM tokens than
per-chunk extraction pipelines; low-cohesion clusters are gated out and summaries that
mention entities absent from their cluster are rejected automatically.

## Pretrained Weights & Quickstart

Champion checkpoints ship via [GitHub Releases](../../releases) and
[Hugging Face (`DeliVali/cogito-estella`)](https://huggingface.co/DeliVali/cogito-estella):

| Asset | Modality | F1 |
| :--- | :--- | :--- |
| `cogito-toolcalls-graphdecoder.pt` | tool-calls | 1.000 |
| `cogito-code-lora-adapters.pt` | code (LoRA + decoder) | 0.781 |
| `cogito-prose-candidates-{ft,cal,base,s2,s3}.pt` | entity-conditioned prose (5-seed ensemble) | 0.827 |
| `cogito-prose-openvocab{,-s4,-s5}.pt` + `cogito-prose-cascade-fallback.pt` | open-vocab prose stack | 0.6514 |
| `vocab-prose.json` | entity/relation vocabulary (20k/60) | — |

```bash
pip install "cogito-estella[sonar]"     # or: uv sync (from a clone)
# download cogito-prose-candidates-ft.pt + vocab-prose.json next to quickstart.py
python quickstart.py
```

`quickstart.py` encodes two raw sentences and prints their decoded triples in one
non-autoregressive pass.

**Exact literals** (phones, hashes, IDs, precise amounts) never round-trip through the
embedding: `extract_with_literals` detects them deterministically in the source text and
`literals_to_neo4j` stores them verbatim with provenance — character-exact recovery by
query, guaranteed by copying rather than decoding.

**Weight licensing:** all from-scratch decoder heads (GraphDecoder,
CandidateGraphDecoder, trunks, ensembles) are Apache-2.0. The code-modality LoRA
adapters modify Meta's SONAR encoder, whose weights are distributed under
CC-BY-NC 4.0 — the adapted encoder inherits those non-commercial terms.

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
