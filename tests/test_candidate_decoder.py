"""Candidate decoder invariants: constructor, elastic COO, leak-free decode, warm-start,
EMA, frozen-trunk gradients, force-top1 recall floor."""
from pathlib import Path

import numpy as np
import pytest
import torch

from cogito_estella.model.candidate_decoder import (
    CandidateDecoderConfig, CandidateGraphDecoder, EMA, build_candidates,
    candidate_graph_loss, decode_triples_coo)

CKPT = Path(__file__).resolve().parents[1] / "experiments/exp042_prose_polish/best.pt"


def _tiny():
    return CandidateDecoderConfig(concept_dim=32, trunk_dim=48, node_dim=16,
                                  n_views=4, n_relations=3, n_entities=50,
                                  n_heads=4, rel_rank=8)


def test_constructor_no_injection_no_duplicates_and_targets():
    rng = np.random.default_rng(0)
    gold = {(5, 0, 9), (9, 1, 7)}
    nouns = [5, 9, 3, 3, 12]              # scanner caught 5 and 9, NOT 7; dup 3
    mem = np.arange(30, 50)
    cand, exist, pos = build_candidates(nouns, gold, mem, rng, cap=32, n_mem=8)
    assert len(cand) == len(set(cand)), "duplicate candidates"
    assert cand[:4] == [5, 9, 3, 12], "text nouns must lead, dedup preserved order"
    assert 7 not in cand, "missed gold must NOT be injected (end-to-end honesty)"
    assert exist[cand.index(5)] == 1.0 and exist[cand.index(9)] == 1.0
    assert exist[cand.index(3)] == 0.0 and exist[cand.index(12)] == 0.0
    mems = cand[4:]
    assert len(mems) == 8 and all(30 <= e < 50 for e in mems), "memory distractors appended"
    assert all(pos[e] == i for i, e in enumerate(cand))


def test_constructor_cap_binds_and_small_pool_terminates():
    rng = np.random.default_rng(1)
    cand, exist, _ = build_candidates(list(range(40)), {(0, 0, 1)}, np.arange(45, 50),
                                      rng, cap=32, n_mem=8)
    assert cand[:24] == list(range(24)), "text nouns capped at cap - n_mem"
    # pool has only 5 unique entities available: take them all, never loop forever
    assert len(cand) == 24 + 5
    assert len(cand) == len(set(cand))


def test_elastic_shapes_scale_with_candidate_count():
    torch.manual_seed(0)
    dec = CandidateGraphDecoder(_tiny())
    for C in (5, 20):
        ids = torch.randint(0, 50, (2, C))
        out = dec(torch.randn(2, 32), ids, torch.ones(2, C, dtype=torch.bool))
        assert out["exist_logits"].shape == (2, C)
        assert out["adj_logits"].shape == (2, 3, C, C)


def test_decode_never_leaks_masked_candidates():
    torch.manual_seed(1)
    dec = CandidateGraphDecoder(_tiny())
    ids = torch.randint(0, 50, (3, 6))
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[:, 4] = False
    out = dec(torch.randn(3, 32), ids, mask)
    assert torch.sigmoid(out["exist_logits"][:, 4]).max() < 1e-4
    for tr in decode_triples_coo(out["exist_logits"], out["adj_logits"], mask, threshold=0.0):
        for i, r, j in tr:
            assert i != 4 and j != 4


@pytest.mark.skipif(not CKPT.exists(), reason="champion checkpoint missing")
def test_warm_start_trunk_matches_champion_and_freezes():
    dec = CandidateGraphDecoder(CandidateDecoderConfig())
    ent2id = dec.load_champion(str(CKPT))
    assert ent2id is not None and len(ent2id) == 20000
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)["dec"]
    x = torch.randn(2, 1024)
    with torch.no_grad():
        ref = x
        import torch.nn as nn
        trunk = nn.Sequential(*dec.trunk)          # same object, sanity forward
        got = trunk(ref)
    assert torch.allclose(dec.trunk(ref), got)
    assert torch.allclose(dec.trunk[0].weight, ck["trunk.0.weight"])
    assert torch.allclose(dec.table.weight[7], ck["head.node_label.weight"][7])
    dec.freeze_trunk()
    assert all(not p.requires_grad for p in dec.trunk.parameters())
    assert dec.table.weight.requires_grad


def test_ema_swap_restores_live_weights():
    torch.manual_seed(2)
    dec = CandidateGraphDecoder(_tiny())
    ema = EMA(dec, decay=0.5)
    before = dec.exist.weight.detach().clone()
    with torch.no_grad():
        dec.exist.weight.add_(1.0)
    ema.update(dec)
    live = ema.swap_in(dec)                        # eval on shadow
    assert not torch.allclose(dec.exist.weight, before + 1.0)
    dec.load_state_dict(live)                      # restore
    assert torch.allclose(dec.exist.weight, before + 1.0)


def test_loss_is_finite_and_backward_reaches_head_not_frozen_trunk():
    torch.manual_seed(3)
    dec = CandidateGraphDecoder(_tiny())
    dec.freeze_trunk()
    ids = torch.randint(0, 50, (4, 7))
    mask = torch.ones(4, 7, dtype=torch.bool)
    out = dec(torch.randn(4, 32), ids, mask)
    ex = torch.zeros(4, 7); ex[:, 0] = 1.0
    adj = torch.zeros(4, 3, 7, 7); adj[:, 0, 0, 1] = 1.0
    loss = candidate_graph_loss(out, ex, adj, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert dec.exist.weight.grad is not None
    assert all(p.grad is None for p in dec.trunk.parameters())


def test_force_top1_floors_empty_decodes_only():
    exist = torch.tensor([[-10.0, -10.0, -10.0], [10.0, 10.0, -10.0]])
    adj = torch.full((2, 2, 3, 3), -10.0)
    adj[0, 1, 0, 2] = -5.0            # sample 0: nothing above threshold; argmax edge
    adj[1, 0, 0, 1] = 10.0            # sample 1: normal decode, force must not touch it
    mask = torch.ones(2, 3, dtype=torch.bool)
    plain = decode_triples_coo(exist, adj, mask)
    assert plain[0] == set() and plain[1] == {(0, 0, 1)}
    forced = decode_triples_coo(exist, adj, mask, force_top1=True)
    assert forced[0] == {(0, 1, 2)}, "empty sample must emit its argmax edge"
    assert forced[1] == {(0, 0, 1)}, "non-empty sample must be untouched"
