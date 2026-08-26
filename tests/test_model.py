import torch

from cogito_estella.model.config import ModelConfig, named_config, param_count
from cogito_estella.model.transformer import ConceptTransformer


def test_forward_shape_tiny():
    cfg = named_config("tiny")
    model = ConceptTransformer(cfg)
    x = torch.randn(2, 5, cfg.concept_dim)
    out = model(x)
    assert out.shape == (2, 5, cfg.concept_dim)


def test_causality():
    # Cambiar un concepto en la posición t no debe alterar las salidas < t.
    cfg = named_config("tiny")
    model = ConceptTransformer(cfg).eval()
    x = torch.randn(1, 6, cfg.concept_dim)
    with torch.no_grad():
        out_a = model(x)
        x2 = x.clone()
        x2[0, 4] += 5.0  # perturbar posición 4
        out_b = model(x2)
    # posiciones 0..3 no deben cambiar
    assert torch.allclose(out_a[0, :4], out_b[0, :4], atol=1e-4)
    # posición 4 sí debe cambiar
    assert not torch.allclose(out_a[0, 4], out_b[0, 4], atol=1e-4)


def test_param_count_presets_within_tolerance():
    for name, target in [("39M", 39e6), ("100M", 100e6), ("300M", 300e6)]:
        cfg = named_config(name)
        n = param_count(cfg)
        assert 0.85 * target <= n <= 1.15 * target, f"{name}: {n/1e6:.1f}M fuera de rango"


def test_config_roundtrip():
    cfg = ModelConfig(dim=256, n_layers=4, n_heads=8)
    assert cfg.concept_dim == 1024
    assert cfg.head_dim == 32
