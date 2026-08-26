"""Structural fidelity metrics (graph paradigm).

Once raw text is dropped, "bits/token" no longer applies. These measure structure
recovery (graph/triples/tool-call), applied identically to the concept model and the
token baseline. Pure, no GPU.
"""
import json


def triple_prf1(pred: set, gold: set) -> tuple[float, float, float]:
    """Precision, recall, F1 over (subject, relation, object) triple sets."""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def graph_edit_distance_proxy(pred_nodes: set, pred_edges: set,
                              gold_nodes: set, gold_edges: set) -> int:
    """Symmetric-difference proxy for GED (optimal alignment is NP-hard). Cheap upper
    bound; fine for small per-sentence graphs."""
    return len(pred_nodes ^ gold_nodes) + len(pred_edges ^ gold_edges)


def graph_edit_distance_normalized(pred_nodes, pred_edges, gold_nodes, gold_edges) -> float:
    """GED proxy normalized to [0, 1] by total size."""
    ged = graph_edit_distance_proxy(pred_nodes, pred_edges, gold_nodes, gold_edges)
    denom = (len(pred_nodes) + len(gold_nodes) + len(pred_edges) + len(gold_edges))
    return ged / denom if denom else 0.0


def _parse_tool_call(text: str):
    try:
        obj = json.loads(text)
        return obj.get("name"), obj.get("arguments", {})
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None, None


def tool_call_exact(pred: str, gold: str) -> bool:
    """Exact match on name + arguments (key order irrelevant)."""
    pn, pa = _parse_tool_call(pred)
    gn, ga = _parse_tool_call(gold)
    if pn is None or gn is None:
        return False
    return pn == gn and pa == ga


def tool_call_arg_f1(pred: str, gold: str) -> float:
    """F1 over the (key, value) argument set plus the name (rewards partial matches)."""
    pn, pa = _parse_tool_call(pred)
    gn, ga = _parse_tool_call(gold)
    if pa is None or ga is None:
        return 0.0
    pred_items = {("name", pn)} | {(k, json.dumps(v, sort_keys=True)) for k, v in pa.items()}
    gold_items = {("name", gn)} | {(k, json.dumps(v, sort_keys=True)) for k, v in ga.items()}
    _, _, f1 = triple_prf1(pred_items, gold_items)
    return f1
