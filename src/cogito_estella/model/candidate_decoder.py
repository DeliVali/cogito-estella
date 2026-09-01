"""Candidate-restricted graph decoder (production prose head, v0.7).

Validated E2E 0.827 as a 5-checkpoint ensemble (selection slice A, virgin-slice C).

Frozen SONAR concept -> champion deep trunk (warm-started from exp042 best.pt, frozen
during the baseline phase) -> 8 concept views. Candidates enter as learned table rows
(nn.Embedding 20k x 512, warm-started from the champion's per-entity output vectors) and
act as queries in 2 cross-attention+FFN blocks over the views. Heads: per-candidate
existence + low-rank bilinear adjacency (rank 64). Elastic C, COO decode.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4


@dataclass
class CandidateDecoderConfig:
    concept_dim: int = 1024
    trunk_dim: int = 2048
    node_dim: int = 512
    n_views: int = 8
    n_relations: int = 60
    n_entities: int = 20000
    n_heads: int = 8
    rel_rank: int = 64
    ffn_mult: int = 2
    n_blocks: int = 2


class CrossBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn_mult: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.GELU(),
                                nn.Linear(d * ffn_mult, d))
        self.n2 = nn.LayerNorm(d)

    def forward(self, x, kv):
        a, _ = self.attn(x, kv, kv)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x))


class CandidateGraphDecoder(nn.Module):
    def __init__(self, cfg: CandidateDecoderConfig = None):
        super().__init__()
        cfg = cfg or CandidateDecoderConfig()
        self.cfg = cfg
        # exact exp042 trunk layout (indices 0..8) so its state dict loads verbatim
        self.trunk = nn.Sequential(
            nn.Linear(cfg.concept_dim, cfg.trunk_dim), nn.GELU(), nn.LayerNorm(cfg.trunk_dim),
            nn.Linear(cfg.trunk_dim, cfg.trunk_dim), nn.GELU(), nn.LayerNorm(cfg.trunk_dim),
            nn.Linear(cfg.trunk_dim, cfg.trunk_dim), nn.GELU(), nn.LayerNorm(cfg.trunk_dim))
        self.table = nn.Embedding(cfg.n_entities, cfg.node_dim)
        self.to_views = nn.Linear(cfg.trunk_dim, cfg.n_views * cfg.node_dim)
        self.blocks = nn.ModuleList(
            [CrossBlock(cfg.node_dim, cfg.n_heads, cfg.ffn_mult) for _ in range(cfg.n_blocks)])
        self.exist = nn.Linear(cfg.node_dim, 1)
        self.rel_u = nn.Parameter(torch.empty(cfg.n_relations, cfg.node_dim, cfg.rel_rank))
        self.rel_v = nn.Parameter(torch.empty(cfg.n_relations, cfg.node_dim, cfg.rel_rank))
        nn.init.xavier_uniform_(self.rel_u)
        nn.init.xavier_uniform_(self.rel_v)

    def load_champion(self, path: str):
        """Warm-start: trunk verbatim; table rows from the champion's node_label weight
        (its per-entity output vectors, same ent2id ordering). Returns the ckpt ent2id."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["dec"]
        trunk_sd = {k[len("trunk."):]: v for k, v in sd.items() if k.startswith("trunk.")}
        self.trunk.load_state_dict(trunk_sd)
        with torch.no_grad():
            self.table.weight.copy_(sd["head.node_label.weight"][: self.cfg.n_entities])
        return ck.get("ent2id")

    def freeze_trunk(self):
        for p in self.trunk.parameters():
            p.requires_grad_(False)

    def forward(self, concept: torch.Tensor, cand_ids: torch.Tensor,
                cand_mask: torch.Tensor) -> dict:
        B, C = cand_ids.shape
        h = self.trunk(concept)
        views = self.to_views(h).view(B, self.cfg.n_views, self.cfg.node_dim)
        x = self.table(cand_ids)
        for blk in self.blocks:
            x = blk(x, views)
        exist_logits = self.exist(x).squeeze(-1).masked_fill(~cand_mask, NEG)
        left = torch.einsum("bid,rdk->brik", x, self.rel_u)
        right = torch.einsum("bjd,rdk->brjk", x, self.rel_v)
        adj_logits = torch.einsum("brik,brjk->brij", left, right)
        pair = cand_mask[:, None, :, None] & cand_mask[:, None, None, :]
        adj_logits = adj_logits.masked_fill(~pair, NEG)
        return {"exist_logits": exist_logits, "adj_logits": adj_logits, "nodes": x}


def candidate_graph_loss(out: dict, tgt_exist, tgt_adj, cand_mask):
    m = cand_mask.float()
    ex = F.binary_cross_entropy_with_logits(out["exist_logits"], tgt_exist, reduction="none")
    exist_loss = (ex * m).sum() / m.sum().clamp(min=1)
    pair = (cand_mask[:, None, :, None] & cand_mask[:, None, None, :]).float()
    ad = F.binary_cross_entropy_with_logits(out["adj_logits"], tgt_adj, reduction="none")
    adj_loss = (ad * pair).sum() / pair.sum().clamp(min=1)
    return exist_loss + adj_loss


def decode_triples_coo(exist_logits, adj_logits, cand_mask, threshold: float = 0.5,
                       adj_threshold: float = None, force_top1: bool = False):
    """Per-sample set of (i, r, j) coordinates into the candidate list (COO). Masked
    candidates and self-loops excluded structurally. force_top1 emits the argmax edge
    (joint score adj * exist_i * exist_j) when a sample would otherwise decode empty —
    the recall floor for populations known to carry at least one relation."""
    at = threshold if adj_threshold is None else adj_threshold
    ep = torch.sigmoid(exist_logits)
    ap = torch.sigmoid(adj_logits)
    present = (ep > threshold) & cand_mask
    keep = (ap > at) & present[:, None, :, None] & present[:, None, None, :]
    C = exist_logits.shape[1]
    eye = torch.eye(C, dtype=torch.bool, device=exist_logits.device)
    keep = keep & ~eye[None, None]
    pair_ok = cand_mask[:, None, :, None] & cand_mask[:, None, None, :] & ~eye[None, None]
    out = []
    for b in range(exist_logits.shape[0]):
        coords = keep[b].nonzero(as_tuple=False)
        triples = {(int(i), int(r), int(j)) for r, i, j in coords}
        if force_top1 and not triples and bool(pair_ok[b].any()):
            score = (ap[b] * ep[b][None, :, None] * ep[b][None, None, :]).masked_fill(~pair_ok[b], -1.0)
            r, i, j = np.unravel_index(int(score.argmax()), tuple(score.shape))
            triples = {(int(i), int(r), int(j))}
        out.append(triples)
    return out


def build_candidates(text_noun_ids, gold, mem_pool, rng, cap=32, n_mem=8):
    """Production candidate set: in-vocab text nouns (hard negatives; gold present only
    if the scanner caught it — no injection) + simulated memory distractors. Returns
    (cand_ids list, exist target, pos map entity_id -> slot)."""
    import numpy as np
    cand = list(dict.fromkeys(text_noun_ids))[: cap - n_mem]
    need = min(n_mem, cap - len(cand))
    if need > 0:
        avail = np.setdiff1d(np.asarray(mem_pool), np.asarray(cand, dtype=np.int64))
        take = min(need, len(avail))
        if take > 0:
            for e in rng.choice(avail, size=take, replace=False):
                cand.append(int(e))
    gents = {x for a, r, b in gold for x in (a, b)}
    exist = [1.0 if e in gents else 0.0 for e in cand]
    pos = {e: i for i, e in enumerate(cand)}
    return cand, exist, pos


class EMA:
    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: nn.Module):
        for k, v in module.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def swap_in(self, module: nn.Module):
        live = {k: v.detach().clone() for k, v in module.state_dict().items()}
        module.load_state_dict(self.shadow)
        return live
