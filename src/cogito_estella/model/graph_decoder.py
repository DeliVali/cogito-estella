"""Non-autoregressive decoder: concept embedding -> knowledge graph.

Single forward pass over K fixed node slots (no time dimension, O(1) in length).
Replaces the 605M autoregressive SONAR text decoder for structured content.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GraphDecoderConfig:
    concept_dim: int = 1024
    max_nodes: int = 8       # K
    node_dim: int = 128      # d
    node_vocab: int = 512    # V
    n_relations: int = 16    # R


class GraphDecoder(nn.Module):
    def __init__(self, cfg: GraphDecoderConfig = None):
        super().__init__()
        cfg = cfg or GraphDecoderConfig()
        self.cfg = cfg
        self.to_nodes = nn.Linear(cfg.concept_dim, cfg.max_nodes * cfg.node_dim)
        self.node_exist = nn.Linear(cfg.node_dim, 1)
        self.node_label = nn.Linear(cfg.node_dim, cfg.node_vocab)
        self.rel = nn.Parameter(torch.empty(cfg.n_relations, cfg.node_dim, cfg.node_dim))
        nn.init.xavier_uniform_(self.rel)

    def forward(self, concept: torch.Tensor) -> dict:
        B = concept.shape[0]
        nodes = self.to_nodes(concept).view(B, self.cfg.max_nodes, self.cfg.node_dim)
        exist_logits = self.node_exist(nodes).squeeze(-1)          # [B, K]
        label_logits = self.node_label(nodes)                      # [B, K, V]
        # bilinear adjacency: A[b,r,i,j] = n_i^T W_r n_j
        adj_logits = torch.einsum("bid,rde,bje->brij", nodes, self.rel, nodes)
        return {"exist_logits": exist_logits, "label_logits": label_logits,
                "adj_logits": adj_logits, "nodes": nodes}


def graph_loss(out: dict, tgt_exist, tgt_labels, tgt_adj):
    """BCE(existence) + existence-masked CE(labels) + BCE(adjacency)."""
    import torch.nn.functional as F

    exist_loss = F.binary_cross_entropy_with_logits(out["exist_logits"], tgt_exist)
    B, K, V = out["label_logits"].shape
    label_ce = F.cross_entropy(out["label_logits"].reshape(B * K, V),
                               tgt_labels.reshape(B * K), reduction="none").reshape(B, K)
    label_loss = (label_ce * tgt_exist).sum() / tgt_exist.sum().clamp(min=1)  # ignore empty slots
    adj_loss = F.binary_cross_entropy_with_logits(out["adj_logits"], tgt_adj)
    return exist_loss + label_loss + adj_loss


def otsu_threshold(probs: torch.Tensor, nbins: int = 64) -> float:
    """Unsupervised per-matrix threshold (Otsu): maximizes inter-class variance of the
    sigmoid distribution. Adapts to dense/oversmoothed adjacency where a fixed 0.5 either
    floods false positives or collapses to empty."""
    p = probs.flatten()
    if p.numel() == 0:
        return 0.5
    hist = torch.histc(p, bins=nbins, min=0.0, max=1.0)
    w = hist / hist.sum().clamp(min=1)
    centers = (torch.arange(nbins, device=p.device) + 0.5) / nbins
    omega = torch.cumsum(w, 0)
    mu = torch.cumsum(w * centers, 0)
    mu_t = mu[-1]
    denom = (omega * (1 - omega)).clamp(min=1e-12)
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    # flat maximum across an empty inter-class gap: take the plateau midpoint,
    # and threshold at the bin's UPPER edge so class-0 bin contents stay below it
    flat = torch.nonzero(sigma_b2 >= sigma_b2.max() - 1e-12).flatten()
    k = int(flat[len(flat) // 2])
    return float((k + 1) / nbins)


def decode_triples(exist_logits, label_logits, adj_logits, threshold: float = 0.5,
                   adj_threshold: float | str | None = None):
    """Logits -> set of (label_i, relation, label_j) triples per sample.

    adj_threshold controls the adjacency decision: None reuses `threshold` (default,
    backward-compatible); a float overrides it; "otsu" derives a per-sample threshold
    from that sample's adjacency probabilities (dynamic thresholding for dense graphs).
    """
    B, K = exist_logits.shape
    exist = torch.sigmoid(exist_logits) > threshold
    labels = label_logits.argmax(dim=-1)
    adj_p = torch.sigmoid(adj_logits)
    R = adj_p.shape[1]
    out = []
    for b in range(B):
        if adj_threshold == "otsu":
            at = otsu_threshold(adj_p[b])
        elif adj_threshold is None:
            at = threshold
        else:
            at = adj_threshold
        adj_b = adj_p[b] > at
        triples = set()
        present = [k for k in range(K) if bool(exist[b, k])]
        for r in range(R):
            for i in present:
                for j in present:
                    if i != j and bool(adj_b[r, i, j]):
                        triples.add((int(labels[b, i]), r, int(labels[b, j])))
        out.append(triples)
    return out
