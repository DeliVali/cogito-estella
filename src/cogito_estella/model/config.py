"""ConceptTransformer config and named presets."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    mlp_ratio: float = 8 / 3  # SwiGLU hidden ~ 8/3 * dim (Llama convention)
    concept_dim: int = 1024   # SONAR space
    max_seq_len: int = 128    # in concepts
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        return self.dim // self.n_heads


# Backbone-only counts; SONAR is frozen and separate.
_PRESETS = {
    "tiny": ModelConfig(dim=128, n_layers=2, n_heads=4, max_seq_len=64),
    "39M": ModelConfig(dim=512, n_layers=12, n_heads=8),
    "100M": ModelConfig(dim=768, n_layers=14, n_heads=12),
    "300M": ModelConfig(dim=1024, n_layers=24, n_heads=16),
}


def named_config(name: str) -> ModelConfig:
    if name not in _PRESETS:
        raise KeyError(f"unknown preset: {name}; options: {list(_PRESETS)}")
    return _PRESETS[name]


def param_count(cfg: ModelConfig) -> int:
    d = cfg.dim
    hidden = int(cfg.mlp_ratio * d)
    per_block = 4 * d * d + 3 * d * hidden  # attn (4 d^2) + SwiGLU MLP (3 d*hidden)
    blocks = cfg.n_layers * per_block
    io = cfg.concept_dim * d + d * cfg.concept_dim  # in/out projections
    norms = (2 * cfg.n_layers + 1) * d
    return blocks + io + norms
