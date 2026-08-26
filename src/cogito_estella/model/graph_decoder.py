"""GraphDecoder — decodifica un embedding de concepto a un GRAFO, no a texto.

Motivación (idea de Jeffrey): el decoder SONAR de texto (605M, autorregresivo) domina
el costo del modelo de conceptos en generación corta y entrenamiento. Un grafo es la
representación natural del contenido estructurado (tool-calls, hechos, relaciones) —
justo donde SONAR falla (JSON round-trip 0.3%, exp002b). Un decoder de grafo:
  - es NO-AUTORREGRESIVO: una sola pasada produce todo el grafo (K nodos fijos), sin
    dimensión de tiempo → O(1) en longitud, ~100-1000× menos params que SONAR.
  - mapea concepto [1024] -> (existencia de nodos, etiquetas de nodos, matriz de
    adyacencia por relación).

Arquitectura mínima: proyección lineal a K slots de nodo + cabezas lineales + scoring
bilineal de aristas. Sin GNN por ahora (se puede añadir una capa de message-passing
ligera si la fidelidad lo pide).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GraphDecoderConfig:
    concept_dim: int = 1024
    max_nodes: int = 8       # K: número fijo de slots de nodo
    node_dim: int = 128      # d: dimensión de cada nodo
    node_vocab: int = 512    # V: vocabulario de etiquetas de nodo (tipos/entidades)
    n_relations: int = 16    # R: tipos de relación (aristas)


class GraphDecoder(nn.Module):
    def __init__(self, cfg: GraphDecoderConfig = None):
        super().__init__()
        cfg = cfg or GraphDecoderConfig()
        self.cfg = cfg
        self.to_nodes = nn.Linear(cfg.concept_dim, cfg.max_nodes * cfg.node_dim)
        self.node_exist = nn.Linear(cfg.node_dim, 1)
        self.node_label = nn.Linear(cfg.node_dim, cfg.node_vocab)
        # adyacencia bilineal: adj[b,r,i,j] = n_i^T W_r n_j
        self.rel = nn.Parameter(torch.empty(cfg.n_relations, cfg.node_dim, cfg.node_dim))
        nn.init.xavier_uniform_(self.rel)

    def forward(self, concept: torch.Tensor) -> dict:
        # concept: [B, concept_dim]
        B = concept.shape[0]
        nodes = self.to_nodes(concept).view(B, self.cfg.max_nodes, self.cfg.node_dim)
        exist_logits = self.node_exist(nodes).squeeze(-1)          # [B, K]
        label_logits = self.node_label(nodes)                      # [B, K, V]
        # adj: [B,R,K,K] = einsum(nodes[b,i,d], rel[r,d,e], nodes[b,j,e])
        adj_logits = torch.einsum("bid,rde,bje->brij", nodes, self.rel, nodes)
        return {"exist_logits": exist_logits, "label_logits": label_logits,
                "adj_logits": adj_logits, "nodes": nodes}


def decode_triples(exist_logits, label_logits, adj_logits, threshold: float = 0.5):
    """Convierte logits en un conjunto de triples (label_i, relación, label_j) por muestra.
    Un nodo existe si sigmoid(exist) > threshold; una arista si sigmoid(adj) > threshold.
    """
    B, K = exist_logits.shape
    exist = torch.sigmoid(exist_logits) > threshold          # [B, K]
    labels = label_logits.argmax(dim=-1)                     # [B, K]
    adj = torch.sigmoid(adj_logits) > threshold              # [B, R, K, K]
    out = []
    for b in range(B):
        triples = set()
        present = [k for k in range(K) if bool(exist[b, k])]
        R = adj.shape[1]
        for r in range(R):
            for i in present:
                for j in present:
                    if i != j and bool(adj[b, r, i, j]):
                        triples.add((int(labels[b, i]), r, int(labels[b, j])))
        out.append(triples)
    return out
