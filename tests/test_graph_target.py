import json

from cogito_estella import graph_target as gt


def test_vocab_covers_generator_values():
    v = gt.build_toolcall_vocab()
    # ROOT + tools + queries + limits + bools
    assert "ROOT" in v.label2id
    assert "tool:search_web" in v.label2id
    assert v.n_relations >= 4


def test_toolcall_to_triples_roundtrip():
    call = {"name": "search_web", "arguments": {"query": "clima CDMX", "limit": 5, "verbose": True}}
    v = gt.build_toolcall_vocab()
    triples = gt.toolcall_to_triples(call, v)
    root = v.label2id["ROOT"]
    assert (root, v.rel2id["has_name"], v.label2id["tool:search_web"]) in triples
    assert (root, v.rel2id["has_limit"], v.label2id["limit:5"]) in triples
    assert (root, v.rel2id["has_verbose"], v.label2id["bool:True"]) in triples


def test_target_tensors_shapes_and_consistency():
    import numpy as np
    call = {"name": "get_weather", "arguments": {"query": "reunión lunes", "limit": 12, "verbose": False}}
    v = gt.build_toolcall_vocab()
    exist, labels, adj = gt.toolcall_to_target(call, v, max_nodes=8)
    assert exist.shape == (8,)
    assert labels.shape == (8,)
    assert adj.shape == (v.n_relations, 8, 8)
    # 5 nodos existen (ROOT + 4 valores)
    assert int(exist.sum()) == 5
    # el nodo ROOT (slot 0) tiene 4 aristas salientes
    assert int(adj[:, 0, :].sum()) == 4


def test_triples_match_between_target_and_decode():
    # el target construido debe decodificarse a los mismos triples que toolcall_to_triples
    import numpy as np
    import torch
    from cogito_estella.model.graph_decoder import decode_triples
    call = {"name": "query_db", "arguments": {"query": "SELECT * FROM users", "limit": 3, "verbose": True}}
    v = gt.build_toolcall_vocab()
    exist, labels, adj = gt.toolcall_to_target(call, v, max_nodes=8)
    # convertir target (0/1) a "logits" fuertes
    ex = torch.tensor(np.where(exist > 0.5, 10.0, -10.0))[None]
    K, V = 8, len(v.label2id)
    lab = torch.full((1, K, V), -10.0)
    for k in range(K):
        lab[0, k, int(labels[k])] = 10.0
    aj = torch.tensor(np.where(adj > 0.5, 10.0, -10.0))[None].float()
    got = decode_triples(ex, lab, aj, threshold=0.5)[0]
    gold = gt.toolcall_to_triples(call, v)
    assert got == gold
