import torch

from cogito_estella.model import train
from cogito_estella.model.config import named_config
from cogito_estella.model.transformer import ConceptTransformer


class FakeCELoss:
    """celoss simulado: CE = MSE entre el embedding predicho y un objetivo fijo por texto.
    Diferenciable respecto a pred_emb, sin GPU ni SONAR. Valida el plumbing de
    aplanado/reagrupado por idioma de next_concept_ce."""
    def __init__(self):
        self.calls = []

    def loss(self, pred_emb, texts, lang):
        self.calls.append((tuple(texts), lang))
        # objetivo determinista por texto (hash -> vector)
        tgt = torch.stack([
            torch.full((pred_emb.shape[1],), float(len(t) % 7), device=pred_emb.device)
            for t in texts
        ])
        return ((pred_emb - tgt) ** 2).mean()


def test_next_concept_ce_shapes_and_grouping():
    cfg = named_config("tiny")
    model = ConceptTransformer(cfg)
    B, T = 2, 4
    batch = torch.randn(B, T, cfg.concept_dim)
    texts = [[f"s{b}_{t}" for t in range(T)] for b in range(B)]
    langs = ["eng_Latn", "spa_Latn"]
    fake = FakeCELoss()
    loss = train.next_concept_ce(model, batch, texts, langs, fake)
    assert loss.ndim == 0
    loss.backward()
    # gradiente llega a los parámetros del modelo
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    # se agrupó por idioma: dos llamadas (una por lang), cada una con B*(T-1)/... textos
    langs_called = {c[1] for c in fake.calls}
    assert langs_called == {"eng_Latn", "spa_Latn"}


def test_next_concept_ce_targets_next_concept_text():
    # el objetivo del texto en posición t debe ser el concepto t+1
    cfg = named_config("tiny")
    model = ConceptTransformer(cfg)
    batch = torch.randn(1, 3, cfg.concept_dim)
    texts = [["a", "b", "c"]]
    fake = FakeCELoss()
    train.next_concept_ce(model, batch, texts, ["eng_Latn"], fake)
    # posiciones 0 y 1 predicen textos "b" y "c" (no "a")
    called_texts = [t for c in fake.calls for t in c[0]]
    assert set(called_texts) == {"b", "c"}
