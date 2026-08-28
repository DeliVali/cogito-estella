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
    # each doc: 10 concepts -> chunks of 4 -> 2 full chunks (discarda resto de 2)
    assert seqs.shape == (4, 4, 1024)
    assert seqs.dtype == np.float32


def test_build_sequences_does_not_cross_doc_boundaries():
    # doc with 3 concepts and seq_len 4 -> 0 sequences (does not fill a chunk)
    ds = _fake_ds(n_docs=1, per_doc=3)
    seqs = train.build_sequences(ds, seq_len=4)
    assert seqs.shape[0] == 0


def _fake_ds_with_text(n_docs=2, per_doc=6, dim=1024, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for d in range(n_docs):
        for i in range(per_doc):
            meta = {"doc_id": f"doc{d}", "text": f"doc{d}_sent{i}", "lang": "eng_Latn"}
            items.append((rng.standard_normal(dim).astype(np.float16), meta))
    return FakeDataset(items)


def test_build_sequences_with_text():
    ds = _fake_ds_with_text(n_docs=2, per_doc=6)
    emb, texts, langs = train.build_sequences_with_text(ds, seq_len=3)
    # 2 docs x 6 conceptos -> 2 chunks de 3 por doc -> 4 secuencias
    assert emb.shape == (4, 3, 1024)
    assert emb.dtype == np.float32
    assert len(texts) == 4 and len(texts[0]) == 3
    assert texts[0] == ["doc0_sent0", "doc0_sent1", "doc0_sent2"]
    assert langs == ["eng_Latn"] * 4


def test_build_sequences_with_text_no_cross_doc():
    ds = _fake_ds_with_text(n_docs=1, per_doc=2)  # < seq_len 3
    emb, texts, langs = train.build_sequences_with_text(ds, seq_len=3)
    assert emb.shape[0] == 0 and texts == [] and langs == []


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
