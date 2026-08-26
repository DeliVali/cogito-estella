# COGITO_ESTELLA

Modelo de conceptos (Large Concept Model) entrenado desde cero, con **resolución adaptativa concepto↔token** como contribución arquitectónica central y la **eficiencia medible** (calidad por FLOP) como criterio de éxito.

- **Diseño y veredictos de investigación:** ver `docs/internal/specs/`.
- **Qué se publica:** el código de este repositorio y los pesos del modelo. El dataset parseado (embeddings de conceptos) **no** se publica ni se versiona aquí (`data/` está en `.gitignore`).

## Arquitectura (v0, resumen)

Segmentación de oraciones (SaT) → espacio de embeddings SONAR congelado → transformer decoder que predice el siguiente concepto, entrenado con **cross-entropy propagada** a través del decoder SONAR congelado (estilo SONAR-LLM) → controlador adaptativo que cae a predicción token a token cuando el riesgo predicho supera el umbral ε.

## Escalera de versiones

| Versión | Hito | Estado |
|---|---|---|
| v0.0.1 | Experimento cero: fidelidad round-trip de SONAR por tipo de dato (exp000) | ✅ hecho |
| v0.1.0 | Fábrica de conceptos: SaT + concept_store + gate_features; ablación de segmentación (exp001) y viabilidad del gate (exp002/002b) | ✅ hecho |
| v0.2.0 | Backbone: ConceptTransformer + train_loop; overfit end-to-end (exp003) | ✅ hecho |
| v0.2.1 | Objetivo real: CE propagada por decoder SONAR congelado (exp005) | ✅ hecho |
| v0.2.2 | Generalización: el modelo cruza el prior marginal al escalar datos (exp006/007/008) | ✅ hecho |
| v0.2.3 | Baseline de tokens + metodología matched-compute (spec) | ✅ hecho (spec pend. revisión) |
| v0.2.4 | `train_loop_ce`: objetivo CE en producción con checkpoints reanudables | ✅ hecho |
| v0.3.x | Baseline de tokens matched-compute a escala (declara la tesis de eficiencia) | ⏳ pendiente de revisión de metodología |
| v0.4.x | Escalado (renta, solo si valida) + integración agéntica | pendiente |

Hallazgos clave (ver `experiments/*/REPORT.md` y la bitácora): la resolución adaptativa concepto↔token tiene base empírica (fallo de round-trip predecible, AUC 0.93–0.97); el objetivo CE propagada entrena en la RTX 5070; el modelo generaliza y cruza el prior marginal con datos suficientes (robusto a 3 semillas). Nada es aún comparable a un LM de tokens — ese es el hito v0.3.

## Estándares del repositorio

- **Versionamiento:** [SemVer 2.0.0](https://semver.org/). Pre-1.0: la API puede cambiar entre minors. Tags de git `vX.Y.Z`; la versión vive en `pyproject.toml` y `cogito_estella.__version__` (fuente única).
- **Changelog:** [Keep a Changelog 1.1](https://keepachangelog.com/) en `CHANGELOG.md`.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `exp:` para experimentos, `docs:`, `chore:`).
- **Experimentos:** cada experimento vive en `experiments/expNNN_nombre/` con su `REPORT.md`; los resultados se publican además como Artifacts (bitácora entre sesiones).
- **Especificaciones y planes:** `docs/internal/specs/` y `docs/internal/plans/`.

## Desarrollo

```bash
uv sync            # crea .venv con Python 3.12 y dependencias (incl. SONAR/fairseq2)
```

El paquete `cogito_estella` expone la arquitectura completa: el `GraphDecoder` (concepto → grafo, no-autorregresivo), el `ConceptTransformer`, la fábrica de conceptos multilingüe y el pipeline de sanitización. Ver `experiments/*/REPORT.md` para la evidencia y `docs/REPORTE-FINAL-paradigma-grafos.md` para el reporte consolidado.

Hardware de referencia: RTX 5070 12 GB · CUDA (Blackwell) · fallback CPU para inferencia pequeña.
