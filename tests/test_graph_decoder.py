import torch

from cogito_estella.model.graph_decoder import GraphDecoder, GraphDecoderConfig, decode_triples


def test_forward_shapes():
    cfg = GraphDecoderConfig(concept_dim=1024, max_nodes=8, node_dim=128,
                             node_vocab=512, n_relations=16)
    dec = GraphDecoder(cfg)
    concept = torch.randn(3, 1024)
    out = dec(concept)
    assert out["exist_logits"].shape == (3, 8)
    assert out["label_logits"].shape == (3, 8, 512)
    assert out["adj_logits"].shape == (3, 16, 8, 8)


def test_non_autoregressive_single_forward():
    # una sola pasada produce todo el grafo (no hay dimensión de tiempo/pasos)
    cfg = GraphDecoderConfig()
    dec = GraphDecoder(cfg)
    concept = torch.randn(2, cfg.concept_dim)
    out = dec(concept)
    # el número de "posiciones" de salida es fijo (K nodos), independiente del contenido
    assert out["adj_logits"].shape[-1] == cfg.max_nodes


def test_param_count_is_light():
    cfg = GraphDecoderConfig(max_nodes=8, node_dim=128, node_vocab=512, n_relations=16)
    dec = GraphDecoder(cfg)
    n = sum(p.numel() for p in dec.parameters())
    assert n < 5e6  # << 605M del decoder SONAR


def test_graph_loss_decreases_on_overfit():
    import torch
    from cogito_estella.model.graph_decoder import graph_loss
    torch.manual_seed(0)
    cfg = GraphDecoderConfig(max_nodes=6, node_dim=64, node_vocab=32, n_relations=4)
    dec = GraphDecoder(cfg)
    opt = torch.optim.AdamW(dec.parameters(), lr=3e-3)
    concept = torch.randn(4, cfg.concept_dim)
    tgt_exist = torch.zeros(4, 6); tgt_exist[:, :3] = 1.0
    tgt_labels = torch.randint(0, 32, (4, 6))
    tgt_adj = torch.zeros(4, 4, 6, 6); tgt_adj[:, 0, 0, 1] = 1.0
    first = last = None
    for step in range(150):
        out = dec(concept)
        loss = graph_loss(out, tgt_exist, tgt_labels, tgt_adj)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first / 3


def test_decode_triples_recovers_planted_graph():
    # construimos logits que codifican un triple claro: nodo0(label 3) --rel 1--> nodo1(label 7)
    B, K, V, R = 1, 4, 16, 4
    exist = torch.full((B, K), -10.0); exist[0, 0] = 10; exist[0, 1] = 10
    label = torch.full((B, K, V), -10.0); label[0, 0, 3] = 10; label[0, 1, 7] = 10
    adj = torch.full((B, R, K, K), -10.0); adj[0, 1, 0, 1] = 10
    triples = decode_triples(exist, label, adj, threshold=0.5)[0]
    assert (3, 1, 7) in triples
    assert len(triples) == 1
