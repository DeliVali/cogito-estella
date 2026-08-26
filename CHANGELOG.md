# Changelog

Formato: [Keep a Changelog 1.1](https://keepachangelog.com/) · Versionamiento: [SemVer 2.0.0](https://semver.org/).

## [Unreleased]

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
