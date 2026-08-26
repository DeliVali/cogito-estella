"""Accounting de FLOPs para la comparación matched-compute (concepto vs. tokens).

Convención de factores (estándar Kaplan/Chinchilla):
- Un parámetro en un matmul cuesta ~2 FLOPs por posición en el forward (multiply-add).
- Entrenamiento con pesos ENTRENABLES: forward 2ND + backward 4ND = **6ND**
  (2ND para grad de entrada, 2ND para grad de pesos).
- Módulo CONGELADO en el grafo (p. ej. el decoder SONAR en la CE propagada): forward 2ND
  + backward SOLO grad de entrada 2ND = **4ND** (no se computa grad de pesos).
- Inferencia (con KV-cache, por posición generada): **2ND**.

N = parámetros que participan en matmuls (excluye tablas de embedding, que son lookups).
D = número de posiciones procesadas (tokens para el modelo de tokens; conceptos para el
    ConceptTransformer; tokens para el decoder/encoder SONAR).

El punto de justicia CRÍTICO: el modelo de conceptos, para computar su pérdida, invoca el
decoder SONAR congelado (~N_dec grande). Ignorar ese costo favorece injustamente al modelo
de conceptos. Este módulo lo cuenta explícitamente.
"""
from dataclasses import dataclass


@dataclass
class FlopBreakdown:
    concept_transformer: float = 0.0
    sonar_decoder: float = 0.0
    sonar_encoder_amortized: float = 0.0
    token_transformer: float = 0.0

    @property
    def total(self) -> float:
        return (self.concept_transformer + self.sonar_decoder
                + self.sonar_encoder_amortized + self.token_transformer)

    def as_dict(self) -> dict:
        return {"concept_transformer": self.concept_transformer,
                "sonar_decoder": self.sonar_decoder,
                "sonar_encoder_amortized": self.sonar_encoder_amortized,
                "token_transformer": self.token_transformer,
                "total": self.total}


def concept_training_flops(n_ct: int, n_dec: int, n_enc: int, n_concepts: int,
                           tokens_per_concept: float, n_epochs: float) -> FlopBreakdown:
    """FLOPs de entrenar el modelo de conceptos sobre `n_concepts` conceptos, `n_epochs` veces.

    - ConceptTransformer (entrenable): 6 · N_ct · (conceptos procesados).
    - Decoder SONAR (congelado, teacher forcing sobre los tokens de cada concepto):
      4 · N_dec · (tokens decodificados).
    - Encoder SONAR (congelado, se paga UNA vez para construir el dataset, se amortiza):
      2 · N_enc · (tokens del corpus) / n_epochs  → aquí se devuelve el costo total del
      encode dividido entre épocas (el costo por época).
    """
    concepts_processed = n_concepts * n_epochs
    tokens_decoded = concepts_processed * tokens_per_concept
    corpus_tokens = n_concepts * tokens_per_concept
    return FlopBreakdown(
        concept_transformer=6 * n_ct * concepts_processed,
        sonar_decoder=4 * n_dec * tokens_decoded,
        sonar_encoder_amortized=2 * n_enc * corpus_tokens,  # total del encode (una vez)
    )


def token_training_flops(n_tt: int, n_concepts: int, tokens_per_concept: float,
                         n_epochs: float) -> FlopBreakdown:
    """FLOPs de entrenar el baseline de tokens sobre el MISMO texto (mismos tokens).
    6 · N_tt · (tokens procesados). N_tt incluye bloques + cabeza LM (matmul d×vocab),
    excluye la tabla de embedding (lookup).
    """
    tokens_processed = n_concepts * tokens_per_concept * n_epochs
    return FlopBreakdown(token_transformer=6 * n_tt * tokens_processed)


def concept_inference_flops_per_token(n_ct: int, n_dec: int, tokens_per_concept: float) -> float:
    """FLOPs por token GENERADO en inferencia (generación corta).
    Por concepto: 2·N_ct (predecir el embedding) + 2·N_dec·L_tok (decodificarlo).
    Por token: 2·N_ct/L_tok + 2·N_dec.
    """
    return 2 * n_ct / tokens_per_concept + 2 * n_dec


def token_inference_flops_per_token(n_tt: int) -> float:
    """FLOPs por token generado por el baseline de tokens (con KV-cache): 2·N_tt."""
    return 2 * n_tt


def context_processing_flops(n_model: int, n_positions: int, dim: int) -> float:
    """FLOPs de PROCESAR un contexto de `n_positions` posiciones (prefill), incluyendo
    el término cuadrático de atención. 2·N·P (matmuls) + 2·(P²·dim) (atención QK^T y AV).
    Aquí está la ventaja del modelo de conceptos: comprime el contexto ~L_tok×, así que
    P_conceptos = P_tokens / L_tok, reduciendo tanto el término lineal como el cuadrático.
    """
    return 2 * n_model * n_positions + 2 * (n_positions ** 2) * dim


def flops_summary(n_ct, n_dec, n_enc, n_tt, n_concepts, tokens_per_concept, n_epochs) -> dict:
    """Reporte comparativo completo para un presupuesto dado."""
    ct = concept_training_flops(n_ct, n_dec, n_enc, n_concepts, tokens_per_concept, n_epochs)
    tt = token_training_flops(n_tt, n_concepts, tokens_per_concept, n_epochs)
    return {
        "training_flops": {
            "concept_model": ct.as_dict(),
            "token_model": tt.as_dict(),
            "ratio_concept_over_token": ct.total / tt.total if tt.total else None,
        },
        "inference_flops_per_generated_token": {
            "concept_model": concept_inference_flops_per_token(n_ct, n_dec, tokens_per_concept),
            "token_model": token_inference_flops_per_token(n_tt),
        },
        "params": {"n_ct": n_ct, "n_dec": n_dec, "n_enc": n_enc, "n_tt": n_tt},
        "assumptions": {"n_concepts": n_concepts, "tokens_per_concept": tokens_per_concept,
                        "n_epochs": n_epochs},
    }
