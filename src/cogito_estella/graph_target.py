"""Construcción del GRAFO OBJETIVO para el paradigma de conocimiento estructurado.

Para tool-calls la estructura es exacta por construcción (no se necesita oráculo): un
tool-call es un grafo estrella ROOT --relación--> valor. Vocab controlado derivado del
generador sintético (sampling.synthetic_json_tools). Para PROSA, el oráculo es graphify
(se parsea su JSON GraphRAG a triples) — no cubierto aquí.

Slots (canónicos, evitan el problema de asignación en el primer experimento):
  slot 0 = ROOT ; slot r (1..R) = objeto de la relación r.
"""
import json
from dataclasses import dataclass, field

import numpy as np

# Conjuntos de valores del generador sintético (sampling.synthetic_json_tools)
_TOOL_NAMES = ["search_web", "get_weather", "send_email", "create_event", "query_db", "run_code"]
_QUERIES = ["clima CDMX", "flights to Madrid", "SELECT * FROM users", "reunión lunes"]
_LIMITS = list(range(1, 51))
_BOOLS = [False, True]
# relaciones en orden de slot (slot r = objeto de la relación con índice r-1... ver abajo)
_RELATIONS = ["has_name", "has_query", "has_limit", "has_verbose"]


@dataclass
class ToolcallVocab:
    label2id: dict = field(default_factory=dict)
    rel2id: dict = field(default_factory=dict)

    @property
    def n_labels(self) -> int:
        return len(self.label2id)

    @property
    def n_relations(self) -> int:
        return len(self.rel2id)


def build_toolcall_vocab() -> ToolcallVocab:
    labels = ["ROOT"]
    labels += [f"tool:{t}" for t in _TOOL_NAMES]
    labels += [f"query:{q}" for q in _QUERIES]
    labels += [f"limit:{n}" for n in _LIMITS]
    labels += [f"bool:{b}" for b in _BOOLS]
    label2id = {lab: i for i, lab in enumerate(labels)}
    rel2id = {r: i for i, r in enumerate(_RELATIONS)}
    return ToolcallVocab(label2id=label2id, rel2id=rel2id)


def _value_label(rel: str, args: dict) -> str:
    if rel == "has_name":
        return f"tool:{args['_name']}"
    if rel == "has_query":
        return f"query:{args['query']}"
    if rel == "has_limit":
        return f"limit:{args['limit']}"
    if rel == "has_verbose":
        return f"bool:{args['verbose']}"
    raise KeyError(rel)


def toolcall_to_triples(call: dict, v: ToolcallVocab) -> set:
    args = dict(call["arguments"]); args["_name"] = call["name"]
    root = v.label2id["ROOT"]
    triples = set()
    for rel in _RELATIONS:
        triples.add((root, v.rel2id[rel], v.label2id[_value_label(rel, args)]))
    return triples


def toolcall_to_target(call: dict, v: ToolcallVocab, max_nodes: int = 8, n_relations: int = None):
    """Devuelve (exist[K], labels[K], adj[R,K,K]) como np.float32 con slots canónicos.
    n_relations permite padear a las R del decoder (las relaciones reales van en 0..3)."""
    K = max_nodes
    R = n_relations if n_relations is not None else v.n_relations
    exist = np.zeros(K, dtype=np.float32)
    labels = np.zeros(K, dtype=np.int64)
    adj = np.zeros((R, K, K), dtype=np.float32)
    args = dict(call["arguments"]); args["_name"] = call["name"]
    # slot 0 = ROOT
    exist[0] = 1.0
    labels[0] = v.label2id["ROOT"]
    # slot r+1 = objeto de la relación r
    for ri, rel in enumerate(_RELATIONS):
        slot = ri + 1
        exist[slot] = 1.0
        labels[slot] = v.label2id[_value_label(rel, args)]
        adj[v.rel2id[rel], 0, slot] = 1.0   # ROOT --rel--> valor
    return exist, labels, adj


def parse_toolcall(text: str) -> dict | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
