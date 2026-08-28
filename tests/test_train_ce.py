import torch

from cogito_estella.model import train
from cogito_estella.model.config import named_config
from cogito_estella.model.transformer import ConceptTransformer


class FakeCELoss:
    """simulated celoss: CE = MSE between the predicted embedding and a target fijo por texto.
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
    # gradient reaches the model parameters
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    # grouped by language: two calls (one per lang), each with B*(T-1)/... texts
    langs_called = {c[1] for c in fake.calls}
    assert langs_called == {"eng_Latn", "spa_Latn"}


def test_train_loop_ce_runs_and_checkpoints(tmp_path):
    import numpy as np

    cfg = named_config("tiny")
    model = ConceptTransformer(cfg)
    N, T = 6, 3
    seq_emb = np.random.default_rng(0).standard_normal((N, T, cfg.concept_dim)).astype(np.float32)
    seq_text = [[f"doc{n}_s{t}" for t in range(T)] for n in range(N)]
    seq_lang = ["eng_Latn"] * N
    fake = FakeCELoss()
    metrics = train.train_loop_ce(model, seq_emb, seq_text, seq_lang, fake,
                                  steps=20, lr=1e-3, batch_size=3, device="cpu",
                                  out_dir=str(tmp_path), log_every=1)
    assert len(metrics) >= 1
    assert (tmp_path / "last.pt").exists()  # checkpoint escrito
    # resumable: loading the checkpoint returns the step
    from cogito_estella.model.train import load_checkpoint
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert load_checkpoint(tmp_path / "last.pt", ConceptTransformer(cfg), opt) == 20


def test_next_concept_ce_targets_next_concept_text():
    # the text target at position t must be concept t+1
    cfg = named_config("tiny")
    model = ConceptTransformer(cfg)
    batch = torch.randn(1, 3, cfg.concept_dim)
    texts = [["a", "b", "c"]]
    fake = FakeCELoss()
    train.next_concept_ce(model, batch, texts, ["eng_Latn"], fake)
    # posiciones 0 y 1 predicen textos "b" y "c" (no "a")
    called_texts = [t for c in fake.calls for t in c[0]]
    assert set(called_texts) == {"b", "c"}
