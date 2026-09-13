# Changelog

Format: [Keep a Changelog 1.1](https://keepachangelog.com/) · Versioning: [SemVer 2.0.0](https://semver.org/).

## [Unreleased]

### Added
- **Confidence in the `ask` header** (`p=`): the learned scorer's own probability for the best sentence it found, reported before the scorer name so that field stays last. The ranking uses reciprocal rank, whose top is 1.0 for every question, so the likelihood was being discarded at the point a caller needs it. Measured over the 50 benchmark questions: median 0.64 where the answer was right, 0.46 where the agent abstained; a floor at 0.30 flags 5 of the 11 failures and 1 of the 39 correct replies. `p=` is absent for the lexical and dense scorers, whose numbers are an order and not a likelihood.

## [0.16.0] - 2026-09-11

### Added
- **`m2m100-pool` is the default encoder** (`DEFAULT_ENCODER`): Meta's frozen M2M-100 418M encoder (MIT, revision `55c2e61b`) under the learned attention pooling trained here. The default `cogito-mcp` path carries no non-commercial term; SONAR moves to the `[sonar]` extra (non-commercial, CC-BY-NC 4.0 at runtime). Ontology ensemble ×3 retrained in the pooled space: held-out Triple F1 0.852 on its own virgin slice (the SONAR ensemble: 0.796 on its own), and 0.819 vs 0.730 for the SONAR ensemble on 5,000 rows held out under both protocols (1.12×).
- **`cogito_estella.encoders`**: one `TextEncoder` contract (`encode(texts, lang, batch_size, normalize)`) behind a registry — `sonar`, `m2m100-pool` (frozen M2M-100 418M + `AttnPool`, 128-token window) and an experimental `bge-m3` adapter without published weights. Checkpoints carry `encoder`, `encoder_revision`, `dim`, `normalize`, `vocab_sha256` and `pool_sha256`; a mixed ensemble or a checkpoint decoded in another encoder's space is refused, and a cosine canary (`encoders/canary.json`) checks the loaded encoder at startup. `--encoder` and `COGITO_ENCODER` select the encoder; a value that disagrees with the checkpoints stops the load.
- **Encoder-aware weight resolution** (`cogito_estella.mcp.weights`): `ENCODER_ASSETS` maps each encoder to its published checkpoints, vocabulary, pooling weights and repository subfolder, overridable by the manifest `encoders.json` in the Hugging Face repository. Of the manifest, the files, the subfolder and the operating point govern resolution; `revision`, `dim`, `normalize` and `licence` are descriptive, the binding contract travelling in the checkpoint sidecars. `resolve(...) -> ResolvedWeights(checkpoints, vocab, pool, encoder, operating_point)`; explicit paths still win and must all exist. Explicit paths given without `--encoder` leave `encoder` unset — the checkpoints decide the space, and the server names it once the extractor has resolved it rather than announcing the process default. The published operating point reaches the extractor as its decode thresholds; an absent manifest falls back to the built-in table, a corrupt one (or one whose operating point is not a pair of numbers) is a defect.
- **safetensors release format**: checkpoints and pooling ship as `.safetensors` with their contract (encoder, encoder revision, dim, normalize, vocab and pool sha256, F1, step) in a JSON sidecar and in the file header; both loaders still read `.pt`. Round-trip verified bit-exact (max weight deviation 0, max logit deviation 0 on three fixture embeddings) and end to end (identical triples through the real encoder).
- **Per-encoder operating point** (`ENSEMBLE_OPERATING_POINT`): the extractor's ensemble thresholds follow the encoder that was swept — (0.1, 0.8) for both published ensembles, each from its own sweep — instead of one hard-coded pair. A single model keeps (0.15, 0.15).
- `--pool` and the resolved pooling weights are threaded from the server to the encoder; the startup line names the graph and, as soon as it is settled, the encoder and its revision.
- **Learned relevance scorer for `ask`** (`cogito_estella.mcp.rerank`): twelve scale-free features per question–sentence pair (lexical score and rank, BM25, query token and bigram coverage, entity hits, provenance, encoder cosine and its absence, length, position, numeric match) under an L2 logistic model, trained without LLM tokens from the benchmark questions' evidence sentences and shipped as `rerank_weights.json` (`COGITO_RERANK_WEIGHTS` points at another file). It re-orders the lexical shortlist (`ASK_CANDIDATES`); leave-one-paper-out recall at the 600-token budget 0.869 vs 0.770 for plain IDF, fitted and measured on the released encoder's graph, so `learned` is the default (`DEFAULT_SCORER`); `COGITO_ASK_SCORER` selects `lexical`, `dense` or `learned` for a server and is read once at startup, an unreadable value stops the server. Missing or unusable weights fall back to lexical with a header note.

### Changed
- `ask(question, budget, use_dense=False, use_sonar=False)`: the encoder-cosine ranking is named `dense` (`ASK_SCORERS = ("lexical", "dense", "learned")`); `sonar` stays accepted as an alias of `dense` and `use_sonar` as an alias of `use_dense`. Header notes read `scorer=dense`, `scorer=lexical (dense unavailable)`.
- `[mcp]` extra: `transformers>=4.57,<5`, `sentencepiece` and `safetensors` replace `sonar-space`/`wtpsplit`, which stay in `[sonar]`.
- Checkpoints without encoder metadata are resolved as SONAR at the extractor boundary, whatever the process default is; an explicit `--encoder` or `COGITO_ENCODER` that disagrees still stops the load.

### Fixed
- The shipped canary reference for `m2m100-pool` was baselined on the exp058 probe pooling, not the released exp059 pooling (cosine deviation 0.0704 > tolerance 0.02): the startup canary would have refused the released weights. Re-baselined on the pooling file that ships, `pool.safetensors` (`pool:abb6bbade64b`), so the reference names the published artefact rather than the `.pt` it was converted from — the same restamping applies to the checkpoint sidecars' `pool_sha256` and to the manifest revision, with the training-time digests kept under `source_*` keys.

## [0.15.0] - 2026-09-09

### Added
- **`ask(question, budget=600, use_sonar=False)`**: one call from a question to the material that answers it — the graph facts of the entities recognized in the question (exact hits, rarest first, 1 hop), then a `--` line, then the ranked source sentences, all within `budget` tokens (clamped to 100-4000; 40 % facts / 60 % sentences, line-granular truncation). Charged to the ledger like every other tool. Lexical IDF is the primary ranking engine; `use_sonar=True` is an opt-in toggle.
- **Sentence scorers** (`cogito_estella.mcp.rank`): `LexicalScorer` (idf overlap normalized to [0, 1]) and `SonarScorer` (cosine mapped to [0, 1]), both with a provenance bonus for the sentences that back the retrieved facts. The two scales are never merged row by row: embedded sentences rank above lexical ones and are dropped below a cosine floor, so `ask` can still answer `no material`; a partially embedded corpus is reported in the header as `scorer=sonar (3/5 docs)`. SONAR is a toggle, not a blend: it ranks only when `use_sonar=True` and falls back to lexical with a note when the graph has no embeddings.
- **Sentence embeddings at ingest**: `CogitoGraphExtractor.encode_batch(texts)` (float16, L2-normalized) feeds a per-document embedding block stored in the sidecar `<graph>.emb.npz` (atomic write, tolerated when missing or unreadable); `stats` reports `embedded_docs=<n>/<docs>`. Documents ingested without an encoder are ranked lexically.

### Changed
- Server instructions make `ask` the routing gate: `query`, `provenance` and `search` are for going deeper.

## [0.14.0] - 2026-09-08

### Added
- **MCP server** (`cogito_estella.mcp`, console script `cogito-mcp`, extra `[mcp]`): six tools (`ingest`, `query`, `provenance`, `search`, `entities`, `stats`) over a per-project graph persisted in `<dir>/graph.json`; documents deduplicated by content hash and replaced when they change; `query` separates syntax-derived facts from class-only facts with a divider so agents know what to verify.
- Readers for `.txt`, `.md`, arXiv/LaTeXML `.html` and directories; `.pdf` via extra `[pdf]`.
- Default weights resolved from Hugging Face (`cogito-prose-ontology*.pt`, `vocab-onto.json`) with `--checkpoint`/`--vocab` overrides and `--no-download`.
- Persistence is thread-safe (`GraphStore` serializes ingestion and save/load behind a lock) and reads/writes `graph.json` as UTF-8; a malformed graph file (non-dict sections, a broken edge) is recovered atomically instead of half-loading; `raw_tokens` is derived from the currently stored documents so a replaced document is never double-counted.

### Changed
- `CogitoGraphExtractor.extract_batch_with_provenance(texts, doc_offsets)`: batched provenance records (one encoder call per batch); `extract_with_provenance` shares the same lexicalization helper; `doc_offsets`, when given, must align with `texts`.

## [0.13.0] - 2026-09-07

### Added
- **Relation lexicalizer** (`cogito_estella.relation_lexicalizer`): edge labels read from the dependency path between the two entity spans (passive agent inversion, verb+preposition, copula `is_a`, nominal prepositions, compounds, `not_` negation, verb particles). `extract_with_provenance` now returns `r_lex`/`r_class`/`pattern`/`swapped`; the 76-class label is kept in `r_class`. Measured on 4 ingested papers: lexical coverage 0.334 of head-proposed edges (503/1504); blind fidelity rate class 0.11 / lexicalized (A) 0.96 / phrase-span control (B) 0.61 (n=100); token cost +1.45% per fact.

### Changed
- `extract_with_provenance` now sets `r` to `r_lex or r_class` (the lexical label when syntax yields one, else the 76-class label); when syntax reverses the head's pair order, `s`/`o` and `s_span`/`o_span` are exchanged and `swapped` is `True`. Consumers that need the canonical 76-class label read `r_class`.

## [0.12.0] - 2026-09-02

### Added
- **`ensure_schema(driver)`**: idempotent uniqueness constraints (Entity.name; Literal value+kind). Concurrent `MERGE` ingestion measurably races and duplicates nodes without them; deterministic with them. Migration note: deduplicate pre-existing databases before constraining.
- **Explain mode**: `extract(..., return_scores=True)` returns per-candidate existence probabilities and per-edge confidences; force-top1 floor edges are flagged with their low confidence visible.

## [0.11.0] - 2026-09-02

### Added
- **Dynamic literal detection**: `uuid` pattern plus an entropy-based `token` detector — any opaque high-entropy string (nanoid, JWT, base64 ciphertext, API keys) is caught by its Shannon-entropy signature without per-format patterns. High-entropy tokens are stored as sha256 fingerprints by default (queryable, irrecoverable — they are often secrets); `redact_sensitive=False` opts into verbatim. `extra_patterns` lets callers register domain formats (invoices, SKUs) with top precedence.

## [0.10.0] - 2026-09-01

### Added
- **Exact-literal channel** in the connector: deterministic verbatim detection of phones, emails, URLs, hashes, IDs, and precise numbers in the source text (`extract_literals`, `extract_with_literals`, `literals_to_neo4j`). Literals never enter the semantic space — they are stored verbatim as `(:Literal)` nodes linked to the sentence's entities with per-edge provenance, and recover character-exact by query. Closes the "continuous latent spaces lose exact strings" concern at the modeling level: meaning travels through embeddings; exact strings travel through copy.

## [0.9.0] - 2026-09-01

### Added
- **`GraphSummarizer`** (`graph_summary`): GraphRAG-style global sensemaking over extracted triples at ~155x fewer LLM tokens (validated novel-scale). Deterministic pipeline: PPMI edge weighting (chance-level co-occurrence self-suppresses, no stoplists), Louvain communities, scale-free cohesion gating, salience-ranked evidence (frequency x PMI — keeps protagonists, drops generic hubs and rare trivia), and entity-level grounding verification with automatic rejection. The summarizer LLM is caller-injected (`llm_fn`), model-agnostic.
- `networkx` added to core dependencies (pure Python).

## [0.8.1] - 2026-09-01

### Added
- CI workflow (test + wheel build) and PyPI trusted-publishing release workflow. First version distributed on PyPI.

### Changed
- sonar-stack tests skip cleanly on core-only installs.

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
