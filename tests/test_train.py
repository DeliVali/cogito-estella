import numpy as np
import torch

from cogito_estella.model.config import named_config
from cogito_estella.model.transformer import ConceptTransformer
from cogito_estella.model import train


class FakeDataset:
    """Imita ConceptDataset: __len__ y __getitem__ -> (emb[1024], meta)."""
    def __init__(self, items):
        self._items = items  # list of (np.ndarray, dict)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


def _fake_ds(n_docs=3, per_doc=10, dim=1024, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for d in range(n_docs):
        for _ in range(per_doc):
            items.append((rng.standard_normal(dim).astype(np.float16), {"doc_id": f"doc{d}"}))
    return FakeDataset(items)


def test_build_sequences_groups_by_doc_and_chunks():
    ds = _fake_ds(n_docs=2, per_doc=10)
    seqs = train.build_sequences(ds, seq_len=4)
    # cada doc: 10 conceptos -> chunks de 4 -> 2 chunks completos (descarta resto de 2)
    assert seqs.shape == (4, 4, 1024)
    assert seqs.dtype == np.float32


def test_build_sequences_does_not_cross_doc_boundaries():
    # doc con 3 conceptos y seq_len 4 -> 0 secuencias (no completa un chunk)
    ds = _fake_ds(n_docs=1, per_doc=3)
    seqs = train.build_sequences(ds, seq_len=4)
    assert seqs.shape[0] == 0


def test_mse_loss_decreases_overfit_cpu():
    torch.manual_seed(0)
    cfg = named_config("tiny")
    model = ConceptTransformer(cfg)
    # un batch fijo diminuto de datos aleatorios pero estructurados
    ds = _fake_ds(n_docs=2, per_doc=8, dim=cfg.concept_dim)
    seqs = train.build_sequences(ds, seq_len=8)
    metrics = train.train_loop(model, seqs, steps=150, lr=1e-3, batch_size=len(seqs),
                               device="cpu", loss="mse", log_every=1000)
    first = metrics[0]["loss"]
    last = metrics[-1]["loss"]
    assert last < first / 5.0, f"overfit falló: {first:.4f} -> {last:.4f}"


def test_checkpoint_save_load_roundtrip(tmp_path):
    cfg = named_config("tiny")
    model = ConceptTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train.save_checkpoint(tmp_path / "ck.pt", model, opt, step=42)
    model2 = ConceptTransformer(cfg)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    step = train.load_checkpoint(tmp_path / "ck.pt", model2, opt2)
    assert step == 42
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
