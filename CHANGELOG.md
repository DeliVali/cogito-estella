# Changelog

Formato: [Keep a Changelog 1.1](https://keepachangelog.com/) · Versionamiento: [SemVer 2.0.0](https://semver.org/).

## [Unreleased]

### Added
- **exp002b** (gate con etiqueta compuesta): cierra la limitación de exp002. Con "malo = chrF<60 O ruptura estructural JSON", la tasa de fallo de JSON pasa de 0.8% a 99.8%, y el gate de superficie mejora a AUC 0.969 / 91% precisión @ 90% recall. Refuerza la viabilidad de la resolución adaptativa para el caso agéntico.
- **exp007** (escalado del piloto): tiny (0.66M) y 39M dan la misma CE held-out (~7.88) → a escala de datos pequeña el cuello de botella son los datos, no la capacidad. El siguiente experimento debe escalar datos, no modelo.
- **exp008** (escalar datos): 5× más datos (160→800 docs) baja la CE del modelo de 7.89 a 7.58 y estrecha la brecha con el baseline de media de +0.133 a +0.052 (aún no la cruza, pero la tendencia apunta a que lo hará con más datos). Confirma el diagnóstico de exp007.

### Fixed
- **Bug de métrica en la evaluación de generalización** (`eval_ce`): ponderaba la CE por número de conceptos en vez de por tokens, haciéndola dependiente del batch. Corregido con `SonarCELoss.loss_sum` (bits/token batch-invariante, test `test_loss_sum_matches_mean`). **Esto corrige el veredicto de exp006**: el modelo bate la persistencia pero NO la media a escala piloto (antes se afirmaba, erróneamente, que batía ambos).

## [0.2.3] - 2026-08-26

Infraestructura del baseline de tokens + metodología matched-compute (para revisión antes de ejecutar).

### Added
- `cogito_estella.model.token_baseline.TokenTransformer`: baseline decoder-only de tokens que reutiliza los bloques del ConceptTransformer (misma receta: RoPE, RMSNorm, SwiGLU) → la comparación es de "moneda" (concepto vs token), no de arquitectura. Tests: forward, causalidad, overfit.
- `docs/internal/specs/2026-08-26-matched-compute-metodologia.md`: protocolo riguroso para la comparación matched-compute (el experimento que define la tesis de eficiencia), con confounds, definición de compute y criterios de victoria. **Pendiente de revisión de Jeffrey antes de ejecutar** — la afirmación central no se produce sin validar la metodología.

### Notes
- Confirmado: con vocab NLLB (256206), el baseline de tokens está dominado por embedding+head (262M de 300M a dim 512) → la comparación debe ser matched-FLOPs, no matched-params.

## [0.2.2] - 2026-08-26

Piloto de generalización + robustez de memoria.

### Added
- **exp006** (piloto de generalización): el modelo entrenado con CE generaliza a documentos no vistos — CE held-out 7.90 vs persistencia 9.76 vs media 9.18. Piloto a escala pequeña (0.66M, 160 docs); el baseline de tokens matched-compute es el siguiente hito.
- `SonarCELoss.max_tokens` (default 96): trunca unidades largas para acotar la memoria de logits (vocab 256k) — una unidad patológica inflaba el padding de todo el minibatch.

## [0.2.1] - 2026-08-26

El objetivo de entrenamiento REAL: cross-entropy propagada por el decoder SONAR congelado (mecanismo de SONAR-LLM).

### Added
- `cogito_estella.model.sonar_loss.SonarCELoss`: computa CE con teacher forcing a través del decoder SONAR congelado; el gradiente fluye al embedding predicho, no al decoder. Carga en bf16 para memoria.
- `cogito_estella.model.train.next_concept_ce`: objetivo CE next-concept (aplana [B,T] conceptos, agrupa por idioma para el lang tag correcto).
- **exp005** (overfit con CE real): tiny reduce CE 16.67→4.62 (−72%) en la RTX 5070 → el objetivo científico funciona end-to-end. Test de cordura: embedding verdadero CE~0.3, aleatorio CE~20.

### Notes
- `train_loop` de producción sigue en MSE; la integración de la CE (con dataset que acarree textos) y el manejo de EOS son v0.2.2.

## [0.2.0] - 2026-08-26

Backbone del modelo de conceptos + loop de entrenamiento, validados end-to-end.

### Added
- `cogito_estella.model.transformer.ConceptTransformer`: decoder-only estilo Llama 3 (RoPE, RMSNorm, SwiGLU, atención causal) operando en el espacio SONAR de 1024 dims; presets tiny/39M/100M/300M.
- `cogito_estella.model.train`: loop de entrenamiento (MSE next-concept), AdamW + cosine LR + grad clip + bf16, checkpoints reanudables; `build_sequences` agrupa conceptos por doc sin cruzar fronteras.
- **exp003** (overfit smoke test): tiny (6221×) y 39M (6162×) sobreajustan embeddings SONAR reales en la RTX 5070 → pipeline completo validado.

### Notes
- v0.2.0 valida ingeniería (forward/backward/pipeline), no ciencia. El objetivo real (CE propagada por el decoder SONAR congelado, estilo SONAR-LLM) es v0.2.1.

## [0.1.0] - 2026-08-26

Fábrica de conceptos (segmentación SaT + almacén) y validación de la resolución adaptativa.

### Added
- `cogito_estella.segmenter.Segmenter`: wrapper SaT (sat-3l-sm, half en GPU) robusto a estilo/corrupción.
- `cogito_estella.concept_store`: shards memory-mapped (embedding fp16 + texto crudo + metadatos) con `ShardWriter`/`ConceptDataset`.
- `cogito_estella.gate_features`: 9 features de superficie baratas para el gate.
- **exp001** (ablación de segmentación): SaT rescata el código (chrF 39.9→89.0, colapso 12.4%→0%); confound longitud/frontera documentado.
- **exp002** (viabilidad del gate): el fallo de round-trip es predecible con AUC 0.93 desde features de superficie (superan al embedding SONAR) → base empírica de la resolución adaptativa.

### Fixed
- `SonarCodec` resiliente a CUDA OOM (split adaptativo del batch) al decodificar unidades largas de código en GPU de 12 GB.

### Changed
- `SonarCodec` expone `encode`/`decode` por separado (roundtrip los compone).

## [0.0.1] - 2026-08-25

### Added
- Scaffold del repositorio: estructura, estándares de versionamiento (SemVer, Keep a Changelog, Conventional Commits).
- `cogito_estella.metrics`: chrF, exact match, fidelidad numérica, validez/equivalencia JSON.
- `cogito_estella.sampling`: muestreo por categoría (sintéticos deterministas, streaming de HuggingFace, fallback de código local).
- `cogito_estella.sonar_codec.SonarCodec`: wrapper round-trip sobre SONAR (fairseq2).
- **Experimento cero** (`experiments/exp000_roundtrip/`): fidelidad round-trip de SONAR medida en 5 categorías, N=300. Ver `REPORT.md` para resultados y decisión.

### Fixed
- El agrupamiento de muestras para el round-trip debe ser por (categoría, idioma), no solo por categoría — de lo contrario SONAR traduce en vez de reconstruir.

### Findings
- Prosa en inglés y español: SONAR reconstruye con alta fidelidad (chrF mediana 93.0 / 80.4) — camino de encoder congelado validado para v0.1-v0.3, con vigilancia sobre la brecha es vs. en.
- JSON/tool-calls: fidelidad textual alta (chrF 86.3) pero validez estructural casi nula (json_equiv 0.3%) — confirma que la resolución adaptativa concepto↔token es necesaria, no opcional, para salida estructurada.
- Código: degradación severa (chrF mediana 49.5) con un modo de colapso catastrófico ante contenido fuera de distribución (comandos de shell/seguridad, APIs de Windows).
