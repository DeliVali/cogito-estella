# COGITO_ESTELLA

Modelo de conceptos (Large Concept Model) entrenado desde cero, con **resolución adaptativa concepto↔token** como contribución arquitectónica central y la **eficiencia medible** (calidad por FLOP) como criterio de éxito.

- **Diseño y veredictos de investigación:** ver `docs/internal/specs/`.
- **Qué se publica:** el código de este repositorio y los pesos del modelo. El dataset parseado (embeddings de conceptos) **no** se publica ni se versiona aquí (`data/` está en `.gitignore`).

## Arquitectura (v0, resumen)

Segmentación de oraciones (SaT) → espacio de embeddings SONAR congelado → transformer decoder que predice el siguiente concepto, entrenado con **cross-entropy propagada** a través del decoder SONAR congelado (estilo SONAR-LLM) → controlador adaptativo que cae a predicción token a token cuando el riesgo predicho supera el umbral ε.

## Escalera de versiones

| Versión | Hito | Estado |
|---|---|---|
| v0.0.x | Experimento cero: fidelidad round-trip de SONAR por tipo de dato | en curso |
| v0.1.x | Fábrica de conceptos: pipeline HF → SaT → SONAR → shards | pendiente |
| v0.2.x | Modelo semilla 100-300M + baseline matched-compute | pendiente |
| v0.3.x | Escalado ~1B (renta, solo si v0.2 valida) | pendiente |
| v0.4.x | Integración agéntica | pendiente |

## Estándares del repositorio

- **Versionamiento:** [SemVer 2.0.0](https://semver.org/). Pre-1.0: la API puede cambiar entre minors. Tags de git `vX.Y.Z`; la versión vive en `pyproject.toml` y `cogito_estella.__version__` (fuente única).
- **Changelog:** [Keep a Changelog 1.1](https://keepachangelog.com/) en `CHANGELOG.md`.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `exp:` para experimentos, `docs:`, `test:`, `chore:`).
- **Experimentos:** cada experimento vive en `experiments/expNNN_nombre/` con su `REPORT.md`; los resultados se publican además como Artifacts (bitácora entre sesiones).
- **Especificaciones y planes:** `docs/internal/specs/` y `docs/internal/plans/`.

## Desarrollo

```bash
uv sync            # crea .venv con Python 3.12 y dependencias
uv run pytest      # tests
```

Hardware de referencia: RTX 5070 12 GB · CUDA (Blackwell) · fallback CPU para experimentos pequeños.
