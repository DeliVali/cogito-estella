"""Métricas de fidelidad ESTRUCTURAL (paradigma de grafos).

Al abandonar el texto crudo, "bits/token" ya no aplica. Estas métricas miden qué tan
bien se recupera la estructura (grafo/triples/tool-call), manteniendo la comparación
justa: se aplican IGUAL al output del modelo de conceptos (GraphDecoder) y al del
baseline de tokens (que serializa el grafo). Puras, sin GPU.
"""
import json


def triple_prf1(pred: set, gold: set) -> tuple[float, float, float]:
    """Precision, recall y F1 sobre conjuntos de triples (sujeto, relación, objeto)."""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def graph_edit_distance_proxy(pred_nodes: set, pred_edges: set,
                              gold_nodes: set, gold_edges: set) -> int:
    """GED-proxy: nodos y aristas con etiquetas fijas (sin buscar alineamiento óptimo,
    que es NP-hard). Cuenta la diferencia simétrica de nodos + la de aristas. Es un
    límite superior barato de la GED real; adecuado para grafos pequeños por-oración.
    """
    return len(pred_nodes ^ gold_nodes) + len(pred_edges ^ gold_edges)


def graph_edit_distance_normalized(pred_nodes, pred_edges, gold_nodes, gold_edges) -> float:
    """GED-proxy normalizada a [0,1] por el tamaño total (peor caso: todo distinto)."""
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
    """True si nombre y argumentos coinciden exactamente (orden de claves irrelevante)."""
    pn, pa = _parse_tool_call(pred)
    gn, ga = _parse_tool_call(gold)
    if pn is None or gn is None:
        return False
    return pn == gn and pa == ga


def tool_call_arg_f1(pred: str, gold: str) -> float:
    """F1 sobre el conjunto de pares (clave, valor) de argumentos, más el nombre.
    Recompensa aciertos parciales (un tool-call casi correcto no vale 0)."""
    pn, pa = _parse_tool_call(pred)
    gn, ga = _parse_tool_call(gold)
    if pa is None or ga is None:
        return 0.0
    pred_items = {("name", pn)} | {(k, json.dumps(v, sort_keys=True)) for k, v in pa.items()}
    gold_items = {("name", gn)} | {(k, json.dumps(v, sort_keys=True)) for k, v in ga.items()}
    _, _, f1 = triple_prf1(pred_items, gold_items)
    return f1
