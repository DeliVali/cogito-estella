import numpy as np
import torch

from cogito_estella.model.graph_decoder import GraphDecoder, GraphDecoderConfig
from cogito_estella.model import train_graph


def test_train_loop_graph_checkpoints_and_curve(tmp_path):
    cfg = GraphDecoderConfig(max_nodes=6, node_dim=64, node_vocab=32, n_relations=4)
    dec = GraphDecoder(cfg)
    n, K, R = 40, 6, 4
    emb = np.random.default_rng(0).standard_normal((n, cfg.concept_dim)).astype(np.float32)
    exist = np.zeros((n, K), np.float32); exist[:, :3] = 1.0
    labels = np.random.default_rng(1).integers(0, 32, (n, K)).astype(np.int64)
    adj = np.zeros((n, R, K, K), np.float32); adj[:, 0, 0, 1] = 1.0

    metrics = train_graph.train_loop_graph(
        dec, emb, exist, labels, adj, steps=40, lr=2e-3, batch_size=16,
        device="cpu", out_dir=str(tmp_path), ckpt_every=10, log_every=10)

    # checkpoints escritos (cada 10 de 40 pasos)
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "metrics.jsonl").exists()
    assert len(metrics) >= 3
    # la pérdida baja (aprende el batch)
    assert metrics[-1]["loss"] < metrics[0]["loss"]


def test_train_loop_graph_resumes(tmp_path):
    cfg = GraphDecoderConfig(max_nodes=6, node_dim=64, node_vocab=32, n_relations=4)
    n, K, R = 20, 6, 4
    emb = np.random.default_rng(0).standard_normal((n, cfg.concept_dim)).astype(np.float32)
    exist = np.ones((n, K), np.float32)
    labels = np.zeros((n, K), np.int64)
    adj = np.zeros((n, R, K, K), np.float32)

    train_graph.train_loop_graph(GraphDecoder(cfg), emb, exist, labels, adj, steps=20,
                                 lr=1e-3, batch_size=8, device="cpu", out_dir=str(tmp_path),
                                 ckpt_every=10)
    # reanudar: continúa desde el checkpoint (paso 20) hasta 30
    from cogito_estella.model.train import load_checkpoint
    dec2 = GraphDecoder(cfg)
    opt2 = torch.optim.AdamW(dec2.parameters(), lr=1e-3)
    assert load_checkpoint(tmp_path / "last.pt", dec2, opt2) == 20
