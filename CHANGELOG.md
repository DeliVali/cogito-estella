# Changelog

Format: [Keep a Changelog 1.1](https://keepachangelog.com/) · Versioning: [SemVer 2.0.0](https://semver.org/).

## [0.8.0] - 2026-09-01

### Added
- `CogitoGraphExtractor` (LlamaIndex/LangChain-ready graph connector): single-checkpoint and 5-checkpoint prob-averaged ensemble modes with the validated operating points as defaults (single 0.15/0.15; ensemble 0.1/0.8 + force-top1); spaCy candidate scan with naive fallback; one-encoder-call `extract_batch`; Cypher MERGE mapping with per-edge provenance (`[graph]` extra). Cross-lingual extraction verified (Spanish input, English entity vocabulary).
- Standalone tool-call observability demo (`tools/demo_observability.py`): raw async stream -> verbatim JSON triples, split encoder/decoder timings.

### Changed
- Lean packaging: core runtime is torch+numpy; SONAR, corpus tooling, and the neo4j driver move to `[sonar]`/`[data]`/`[graph]` extras.

## [0.7.0] - 2026-09-01

Consolidates 0.5.0 → 0.7.0: modality champions across tool-calls, code, and prose; the entity-conditioned decoder; validated ensemble recipes.

### Added
- **`CandidateGraphDecoder`** (entity-conditioned prose head): caller-supplied candidates as cross-attention queries over learned concept views; output space restricted by construction, elastic COO adjacency, low-rank bilinear relations (rank 64), calibrated decode with force-top1 recall floor. End-to-end prose triple F1 **0.827** as a 5-seed ensemble, validated on a never-touched held-out slice (119,911-sample pool, selection and validation slices fully separated).
- **Deep trunk** for `GraphDecoder` (`trunk_layers`/`trunk_dim`): GELU+LayerNorm MLP before slot projection — the safe capacity axis under a frozen encoder. Open-vocab prose 0.192 → 0.592 across the width/depth/data ladder, overfit-free.
- **Open-vocab prose stack**: 3-seed ensemble with a trained self-proposal cascade as empty-decode fallback → F1 **0.6514** (virgin-slice validated; up from 0.5922 single-model).
- **LoRA-adapted SONAR for code** (r=32, α=64, manual injection into fairseq2 sharded layers): code→graph 0.652 → **0.781** with the fixed-threshold decode sweep; low-rank adaptation shown to act as protective regularization (full-rank unfreeze degrades to 0.633).
- `noise_floor_threshold` adjacency decoding: sparsity-prior quantile threshold, dominates variance-based Otsu on sparse graphs (8/8 vs 3/8 stress scenarios); `decode_triples(adj_threshold=...)` strategy selector.
- Script-safety gate for digit spacing (`SPACING_UNSAFE_LANGS`) and roofline/bandwidth accounting in `compute` (arithmetic intensity, crossover batch, encoder-toll and copy-ceiling analyses).

### Changed
- README rewritten: per-modality benchmark table with split protocol, measured latency/compute footprint per configuration, production integration patterns.
- Unit-test suite grown to **113 tests** (trunk, candidate decoder, decode strategies, audit additions).

### Findings (negative results, documented)
- Specialized losses (Focal, AST reward) and hybrid encoder unfreezing degrade code F1; BCE + LoRA + early stop is the production recipe.
- Character-level entity generation from sentence embeddings fails (0.006): exact surface recovery needs decoder-scale capacity; selection over candidates replaces generation.
- LLM-oracle distillation with per-sentence labels fails (0.263): dense but inconsistent labels are unlearnable — label consistency dominates label density.

## [0.4.4] - 2026-08-26

Consolidates 0.3.0 → 0.4.4: the structured-knowledge (concept → graph) paradigm, its fidelity benchmarks, and the literal-recovery data-level fixes.

### Added
- **Structured-knowledge paradigm (concept → graph):** non-autoregressive `GraphDecoder` (concept → nodes + labeled adjacency, presets 1.4M/5.8M/29M), `graph_metrics` (triple F1, GED-proxy, tool-call F1), paradigm FLOP accounting in `compute`, and **exp011**: RTX 5070 benchmark → GraphDecoder **1074× faster** wall-clock (0.013 vs 13.96 ms/concept) and 144× in FLOPs than the SONAR text decoder. Product decision: a structured-knowledge model, not a prose model.
- **exp012** (graph fidelity): trained GraphDecoder recovers tool-call graphs from SONAR embeddings at **Triple F1 = 1.000** held-out, including never-seen combinations — the same content where the 605M text decoder produced 0.3% valid JSON.
- **exp013/exp014** (numeric ceiling + token baseline): single-label targets fail on unseen exact integers (0.000); the char-level token baseline reads them verbatim (0.983). Verdict framed per field: graphs win logical structure at 36× fewer FLOPs / 315× less wall-clock; tokens win the exact literal.
- **exp016** (digits-as-nodes + digit spacing): decomposing integers into digit nodes recovers unseen exact integers at 0.956; spacing the digits in the source text ("400" → "4 0 0") raises it to **0.983**. Includes the honest correction: the exp013 "SONAR does not encode the value" conclusion was an artifact of the linear probe and single-label target.
- **exp017** (chars-as-nodes): the same trick mapped to open-vocab strings — 0.899 exact on unseen short strings (L=4, spaced); cost advantage holds at **2881×** even at L=32. Long arbitrary strings (L≥12) remain copy-channel territory.
- Production preprocessing (`preprocess`): language-aware code sanitization (pygments), secret anonymization (curated regexes), prose corruption filtering (unicodedata), and gated digit spacing.
- Multilingual concept factory (`multilingual_factory`): 14 open sources normalized to a canonical `DocRecord`, frozen 75/15/10 prose/code/tool-call mix, modality-aware segmentation, resumable at-scale encoding.
- `train_loop_ce`: training with the propagated-CE objective in the production `train_loop`, memory-safe minibatching and **resumable checkpoints** (enables long CE runs / spot GPUs). Test with a simulated celoss on CPU.
- **exp002b** (composite-label gate): closes the exp002 limitation. With "bad = chrF<60 OR structural JSON break", the JSON failure rate goes from 0.8% to 99.8%, and the surface gate improves to AUC 0.969 / 91% precision @ 90% recall. Reinforces adaptive-resolution viability for the agentic case.
- **exp007** (pilot scaling): tiny (0.66M) and 39M give the same held-out CE (~7.88) → at small data scale the bottleneck is data, not capacity. The next experiment must scale data, not the model.
- **exp008** (data scaling, 4 points + 3 seeds): scaling 160→800→2500→5000 docs drops model CE 7.89→7.58→7.29→6.72 and the gap vs the marginal prior changes sign and GROWS (+0.133→+0.052→−0.265→−0.934). The advantage not only appears but grows with data. The 2500-doc crossover is robust across 3 seeds (gaps −0.265/−0.294/−0.108).

### Fixed
- **Metric bug in the generalization evaluation** (`eval_ce`): CE was weighted by number of concepts instead of tokens, making it batch-dependent. Fixed with `SonarCELoss.loss_sum` (batch-invariant bits/token, test `test_loss_sum_matches_mean`). **This corrects the exp006 verdict**: the model beats persistence but NOT the mean at pilot scale (it was previously, and wrongly, claimed to beat both).

## [0.2.3] - 2026-08-26

Token-baseline infrastructure + matched-compute methodology (for review before execution).

### Added
- `cogito_estella.model.token_baseline.TokenTransformer`: decoder-only token baseline reusing the ConceptTransformer blocks (same recipe: RoPE, RMSNorm, SwiGLU) → the comparison is about the "currency" (concept vs token), not the architecture. Tests: forward, causality, overfit.
- Rigorous protocol for the matched-compute comparison (the experiment that defines the efficiency thesis), with confounds, compute definition and victory criteria, subject to review before execution.

### Notes
- Confirmed: with the NLLB vocab (256206), the token baseline is dominated by embedding+head (262M of 300M at dim 512) → the comparison must be matched-FLOPs, not matched-params.

## [0.2.2] - 2026-08-26

Generalization pilot + memory robustness.

### Added
- **exp006** (generalization pilot): the CE-trained model generalizes to unseen documents — held-out CE 7.90 vs persistence 9.76 vs mean 9.18. Small-scale pilot (0.66M, 160 docs); the matched-compute token baseline is the next milestone.
- `SonarCELoss.max_tokens` (default 96): truncates long units to bound logits memory (256k vocab) — one pathological unit inflated the padding of the whole minibatch.

## [0.2.1] - 2026-08-26

The REAL training objective: cross-entropy propagated through the frozen SONAR decoder (SONAR-LLM mechanism).

### Added
- `cogito_estella.model.sonar_loss.SonarCELoss`: computes CE with teacher forcing through the frozen SONAR decoder; the gradient flows to the predicted embedding, not the decoder. Loads in bf16 for memory.
- `cogito_estella.model.train.next_concept_ce`: next-concept CE objective (flattens [B,T] concepts, groups by language for the correct lang tag).
- **exp005** (overfit with real CE): tiny reduces CE 16.67→4.62 (−72%) on the RTX 5070 → the scientific objective works end-to-end. Sanity check: true embedding CE~0.3, random CE~20.

### Notes
- The production `train_loop` remains on MSE; CE integration (with a text-carrying dataset) and EOS handling are v0.2.2.

## [0.2.0] - 2026-08-26

Concept-model backbone + training loop, validated end-to-end.

### Added
- `cogito_estella.model.transformer.ConceptTransformer`: Llama-3-style decoder-only (RoPE, RMSNorm, SwiGLU, causal attention) operating in the 1024-dim SONAR space; tiny/39M/100M/300M presets.
- `cogito_estella.model.train`: training loop (MSE next-concept), AdamW + cosine LR + grad clip + bf16, resumable checkpoints; `build_sequences` groups concepts by doc without crossing boundaries.
- **exp003** (overfit smoke test): tiny (6221×) and 39M (6162×) overfit real SONAR embeddings on the RTX 5070 → full pipeline validated.

### Notes
- v0.2.0 validates engineering (forward/backward/pipeline), not science. The real objective (CE propagated through the frozen SONAR decoder, SONAR-LLM style) is v0.2.1.

## [0.1.0] - 2026-08-26

Concept factory (SaT segmentation + store) and adaptive-resolution validation.

### Added
- `cogito_estella.segmenter.Segmenter`: SaT wrapper (sat-3l-sm, half precision on GPU) robust to style/corruption.
- `cogito_estella.concept_store`: memory-mapped shards (fp16 embedding + raw text + metadata) with `ShardWriter`/`ConceptDataset`.
- `cogito_estella.gate_features`: 9 cheap surface features for the gate.
- **exp001** (segmentation ablation): SaT rescues code (chrF 39.9→89.0, collapse 12.4%→0%); length/boundary confound documented.
- **exp002** (gate feasibility): round-trip failure is predictable at AUC 0.93 from surface features (they beat the SONAR embedding) → empirical basis for adaptive resolution.

### Fixed
- `SonarCodec` resilient to CUDA OOM (adaptive batch splitting) when decoding long code units on a 12 GB GPU.

### Changed
- `SonarCodec` exposes `encode`/`decode` separately (roundtrip composes them).

## [0.0.1] - 2026-08-25

### Added
- Repository scaffold: structure, versioning standards (SemVer, Keep a Changelog, Conventional Commits).
- `cogito_estella.metrics`: chrF, exact match, numeric fidelity, JSON validity/equivalence.
- `cogito_estella.sampling`: category-based sampling (deterministic synthetics, HuggingFace streaming, local code fallback).
- `cogito_estella.sonar_codec.SonarCodec`: round-trip wrapper over SONAR (fairseq2).
- **Experiment zero**: SONAR round-trip fidelity measured across 5 categories, N=300.

### Fixed
- Round-trip sample grouping must be by (category, language), not category alone — otherwise SONAR translates instead of reconstructing.

### Findings
- English and Spanish prose: SONAR reconstructs with high fidelity (median chrF 93.0 / 80.4) — frozen-encoder path validated for v0.1–v0.3, with a watch on the es-vs-en gap.
- JSON/tool-calls: high textual fidelity (chrF 86.3) but near-zero structural validity (json_equiv 0.3%) — confirms concept↔token adaptive resolution is necessary, not optional, for structured output.
- Code: severe degradation (median chrF 49.5) with a catastrophic collapse mode on out-of-distribution content (shell/security commands, Windows APIs).
