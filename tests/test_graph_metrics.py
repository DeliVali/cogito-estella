from cogito_estella import graph_metrics as gm


def test_triple_f1_perfect():
    gold = {(1, 0, 2), (2, 1, 3)}
    pred = {(1, 0, 2), (2, 1, 3)}
    p, r, f1 = gm.triple_prf1(pred, gold)
    assert p == r == f1 == 1.0


def test_triple_f1_partial():
    gold = {(1, 0, 2), (2, 1, 3)}
    pred = {(1, 0, 2), (9, 9, 9)}      # 1 correcto, 1 espurio, 1 faltante
    p, r, f1 = gm.triple_prf1(pred, gold)
    assert p == 0.5 and r == 0.5 and f1 == 0.5


def test_triple_f1_empty_gold_and_pred():
    p, r, f1 = gm.triple_prf1(set(), set())
    assert p == r == f1 == 1.0        # both empty = perfect agreement


def test_graph_edit_distance_proxy():
    # normalized GED-proxy: |symmetric diff of nodes| + |symmetric diff of edges|
    gold_nodes, gold_edges = {1, 2, 3}, {(1, 2), (2, 3)}
    pred_nodes, pred_edges = {1, 2}, {(1, 2)}          # falta nodo 3 y arista (2,3)
    ged = gm.graph_edit_distance_proxy(pred_nodes, pred_edges, gold_nodes, gold_edges)
    assert ged == 2                                    # 1 nodo + 1 arista de diferencia


def test_graph_edit_distance_normalized():
    gold_nodes, gold_edges = {1, 2, 3}, {(1, 2), (2, 3)}
    pred_nodes, pred_edges = set(), set()
    n = gm.graph_edit_distance_normalized(pred_nodes, pred_edges, gold_nodes, gold_edges)
    assert n == 1.0                                    # everything different -> max distance


def test_tool_call_exact_and_arg_f1():
    gold = '{"name": "search", "arguments": {"q": "clima", "n": 5}}'
    pred_ok = '{"name": "search", "arguments": {"n": 5, "q": "clima"}}'   # orden distinto, igual
    pred_bad = '{"name": "search", "arguments": {"q": "clima", "n": 6}}'  # valor distinto
    assert gm.tool_call_exact(pred_ok, gold) is True
    assert gm.tool_call_exact(pred_bad, gold) is False
    # F1 de argumentos: pred_bad acierta name+q, falla n
    f1 = gm.tool_call_arg_f1(pred_bad, gold)
    assert 0.0 < f1 < 1.0
