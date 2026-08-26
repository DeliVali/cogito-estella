"""Configuración del ConceptTransformer y presets nombrados."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    mlp_ratio: float = 8 / 3  # SwiGLU: hidden ~ 8/3 * dim (convención Llama)
    concept_dim: int = 1024   # dimensión del espacio SONAR
    max_seq_len: int = 128    # en conceptos (oraciones)
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        assert self.dim % self.n_heads == 0, "dim debe ser divisible por n_heads"
        return self.dim // self.n_heads


# Presets: dim, n_layers, n_heads elegidos para caer cerca del conteo objetivo
# (contando solo el backbone; SONAR va congelado y aparte).
_PRESETS = {
    "tiny": ModelConfig(dim=128, n_layers=2, n_heads=4, max_seq_len=64),
    "39M": ModelConfig(dim=512, n_layers=12, n_heads=8),
    "100M": ModelConfig(dim=768, n_layers=14, n_heads=12),
    "300M": ModelConfig(dim=1024, n_layers=24, n_heads=16),
}


def named_config(name: str) -> ModelConfig:
    if name not in _PRESETS:
        raise KeyError(f"preset desconocido: {name}; opciones: {list(_PRESETS)}")
    return _PRESETS[name]


def param_count(cfg: ModelConfig) -> int:
    d = cfg.dim
    hidden = int(cfg.mlp_ratio * d)
    # por bloque: atención (q,k,v,o = 4 * d*d) + MLP SwiGLU (gate,up,down = 3 * d*hidden)
    per_block = 4 * d * d + 3 * d * hidden
    blocks = cfg.n_layers * per_block
    # proyecciones de entrada/salida entre concept_dim y dim
    io = cfg.concept_dim * d + d * cfg.concept_dim
    # normas (RMSNorm): 2 por bloque + 1 final, cada una d params (despreciable pero se cuenta)
    norms = (2 * cfg.n_layers + 1) * d
    return blocks + io + norms
