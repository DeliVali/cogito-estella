"""Baseline de tokens: transformer decoder-only estándar (next-token).

Reutiliza los bloques del ConceptTransformer (RoPE, RMSNorm, SwiGLU) pero opera
sobre tokens: embedding de vocabulario -> bloques -> cabeza LM. Es el baseline
contra el que se mide la eficiencia del modelo de conceptos (comparación
matched-compute, Q4). Comparte el patrón de capa para que la comparación sea de
'moneda' (concepto vs token), no de arquitectura.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from cogito_estella.model.config import ModelConfig
from cogito_estella.model.transformer import Block, RMSNorm, _rope_cache


@dataclass
class TokenConfig:
    vocab_size: int
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    mlp_ratio: float = 8 / 3
    max_seq_len: int = 512
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        assert self.dim % self.n_heads == 0
        return self.dim // self.n_heads

    def _as_model_config(self) -> ModelConfig:
        # los Block reutilizan ModelConfig (dim, n_heads, mlp_ratio, rope_theta, head_dim)
        return ModelConfig(dim=self.dim, n_layers=self.n_layers, n_heads=self.n_heads,
                           mlp_ratio=self.mlp_ratio, max_seq_len=self.max_seq_len,
                           rope_theta=self.rope_theta)


class TokenTransformer(nn.Module):
    def __init__(self, cfg: TokenConfig):
        super().__init__()
        self.cfg = cfg
        mc = cfg._as_model_config()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(mc) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T] -> logits [B, T, vocab]
        B, T = tokens.shape
        h = self.embed(tokens)
        cos, sin = _rope_cache(T, self.cfg.head_dim, self.cfg.rope_theta, tokens.device, h.dtype)
        for block in self.blocks:
            h = block(h, cos, sin)
        h = self.final_norm(h)
        return self.lm_head(h)


@torch.no_grad()
def greedy_generate(model: "TokenTransformer", prefix: torch.Tensor, max_new: int,
                    eos_id: int = None) -> torch.Tensor:
    """Generación greedy autorregresiva. prefix: [B, T0]. Devuelve [B, T0+generados]."""
    model.eval()
    seq = prefix
    for _ in range(max_new):
        logits = model(seq[:, -model.cfg.max_seq_len:])
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
        if eos_id is not None and bool((nxt == eos_id).all()):
            break
    return seq


def token_param_count(cfg: TokenConfig) -> int:
    d = cfg.dim
    hidden = int(cfg.mlp_ratio * d)
    per_block = 4 * d * d + 3 * d * hidden
    blocks = cfg.n_layers * per_block
    embed = cfg.vocab_size * d       # tabla de embeddings
    head = d * cfg.vocab_size        # cabeza LM (sin atar)
    norms = (2 * cfg.n_layers + 1) * d
    return blocks + embed + head + norms
