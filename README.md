# Cogito Estella: Latent Graph Engine (v0.4.4)

An ultra-efficient inference backend that decodes SONAR (Meta) semantic embeddings directly into non-autoregressive knowledge graphs, sidestepping the bottleneck and cost of traditional token-based LLMs.

Mapped and validated end-to-end on a single RTX 5070 (12 GB).

## Why?
Running GraphRAG with traditional autoregressive LLMs over millions of documents destroys any infrastructure budget. Translating latent concepts back into structured text (JSON) is massively inefficient and prone to formatting errors.

**Cogito Estella changes the paradigm:** instead of decoding text, it projects logical adjacency matrices in a single inference pass.

* **1074× faster** wall-clock than decoding prose token by token.
* **Drastic FLOP reduction:** $1.2 \times 10^7$ vs. $3.0 \times 10^{10}$ for classical decoding.
* **Perfect Structural Fidelity (Triple F1 = 1.000)** on held-out, independent tool-calls.

---

## Inference Architecture

The engine is a **5.8M-parameter** non-autoregressive `GraphDecoder` bolted onto Meta's multilingual latent space.

Input: a concept embedding $e \in \mathbb{R}^{1024}$ (a sentence segmented with SaT and projected into SONAR space). There is no time dimension — a single pass emits the full graph over $K$ fixed node slots.

```
e ∈ ℝ^1024
  │  W_in ∈ ℝ^{1024 × K·d}
  ▼
N ∈ ℝ^{K × d}          # K node slots
  ├─ existence   s ∈ ℝ^K            = σ(W_s · N)
  ├─ labels      ℓ ∈ ℝ^{K × V}      = W_ℓ · N
  └─ adjacency   A ∈ ℝ^{R × K × K}
```

`medium` config (5.8M): $K = 12$ nodes, $d = 256$, vocab $V = 2048$, $R = 32$ relations.

Adjacency is a per-relation bilinear score. For node $i$, node $j$, relation $r$:

$$A_{r,i,j} = N_i^{\top} W_r N_j, \qquad W_r \in \mathbb{R}^{d \times d}$$

The output graph is the set of triples $(\ell_i, r, \ell_j)$ where $\sigma(s_i), \sigma(s_j) > \tau$ and $\sigma(A_{r,i,j}) > \tau$, with $\tau = 0.5$.

**Training objective** — structural loss, no sampling, no teacher forcing:

$$\mathcal{L} = \mathrm{BCE}(\hat{s}, s) \;+\; \underbrace{\frac{1}{\sum s}\sum_k s_k \cdot \mathrm{CE}(\hat{\ell}_k, \ell_k)}_{\text{label CE, masked by existence}} \;+\; \mathrm{BCE}(\hat{A}, A)$$

---

## Performance and Cost per Concept (RTX 5070)

| Metric | This Work (GraphDecoder 5.8M) | Baseline (TokenTransformer) | SONAR Text Decoder (605M) |
| :--- | :--- | :--- | :--- |
| **Latency (wall-clock)** | **0.013 ms** | 4.1 ms (315× slower) | 13.96 ms (1074× slower) |
| **Compute (FLOPs)** | **$1.2 \times 10^7$** | $4.1 \times 10^8$ | $3.0 \times 10^{10}$ |
| **Structural Validity** | **100% (F1 = 1.0)** | 100% (F1 = 1.0) | 0.3% (JSON collapse) |

Wall-clock (1074×) beats the FLOP ratio (144×): the text decoder is autoregressive (token-by-token loop + beam, poor GPU utilization); the `GraphDecoder` is a single dense matmul.

---

## The Map of Literals (Open-Vocab Handling)

Continuous embeddings inherently struggle to extract exact values (v0.4.2 documented a **MAE of 36** on held-out integers). A linear Ridge probe on the embedding, interpolation split over $[1, 500]$:

```
MAE = 36        (chance ≈ 125)
exact match = 1.5%
```

SONAR encodes the number's approximate magnitude (MAE 36 < chance) but not the exact digit. The system breaks this technical ceiling by redesigning the physical data instead of enlarging the model — the atomic-spacing trick:

* **Logical structure:** native graph mapping (F1 ~1.0).
* **Integers:** digits-as-nodes + spaced text (`"400"` → `"4 0 0"`). **98.3%** exact accuracy on unseen values.
* **Short strings ($\leq$ 4 chars):** characters-as-nodes + spacing. **89.9%** exact accuracy (open vocabulary).
* **Long arbitrary strings:** automatic fallback to a hybrid copy channel via pointers (the only autoregressive case).

A single unseen integer becomes a new combination of known digit-nodes — compositional generalization. The residual is data coverage (digit-by-position seen in training), not a SONAR limit.

---

## Core Structure (`src/cogito_estella/`)

The public repository exposes the 18 clean architecture modules, ready for integration:

* `GraphDecoder` & `ConceptTransformer` — the core of the latent pipeline.
* `sonar_loss` & `compute.py` — the math harness for the propagated cross-entropy (CE) loss.
* `preprocess` — rigid sanitization factory and atomic spacing.

```
src/cogito_estella/
  model/graph_decoder.py     # non-AR GraphDecoder + graph_loss + decode_triples
  model/transformer.py       # ConceptTransformer (RoPE, RMSNorm, SwiGLU)
  model/token_baseline.py    # token baseline (matched-FLOP comparison)
  model/sonar_loss.py        # cross-entropy propagated through the frozen SONAR decoder
  model/train_graph.py       # training loop with resumable checkpoints
  sonar_codec.py             # SONAR encode/decode, OOM-resilient
  segmenter.py               # SaT (multilingual segmentation)
  multilingual_factory.py    # canonicalized ingestion (DocRecord) from 14 sources
  preprocess.py              # sanitization: code (pygments), secrets (regex), prose
  graph_target.py            # target-graph construction (tool-calls)
  graph_metrics.py           # Triple F1, GED proxy, tool-call F1
  compute.py                 # FLOP accounting (concept vs. tokens)
```

*(Note: training data, experiment scaffolding, and logs are kept out of tracking; the unit-test suite is included.)*

---

## Hardware

| | |
| :--- | :--- |
| GPU | NVIDIA RTX 5070 · 12 GB · Blackwell · CUDA 12.8 |
| Latent encoder / decoder | SONAR `text_sonar_basic` (Meta), 200 languages, frozen |
| Segmenter | SaT `sat-3l-sm` (wtpsplit), half-precision on GPU |
| Encode throughput (full pipeline) | ~273 concepts/s (SaT + SONAR + sanitization) |
| Density | ~2.19 KB / concept (fp16, 1024-d) |
| Training (24.3M production config) | batch 1024 → 6.72 GB peak VRAM (measured) |
| Inference | 0.013 ms / concept |

## Massive Deployment Status

Currently running the resumable production pipeline for the massive encode of the target corpus: **3 million multilingual documents**. Pipeline state is recorded deterministically through local, power-failure-proof checkpoints.

## Installation

```bash
uv sync
```

Requires Python 3.12; SONAR/SaT weights download on first use.

Run the test suite with `pytest tests/`.

## License

Distributed under the **Apache License 2.0**. See the `LICENSE` file for details on relational-patent protection and commercial use.
