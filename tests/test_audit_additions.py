"""Tests for audit additions: roofline accounting + dynamic thresholding."""
import math

import torch

from cogito_estella import compute as C
from cogito_estella.model.graph_decoder import (
    decode_triples, noise_floor_threshold, otsu_threshold)


def _stats_medium():
    return C.graph_decoder_flops_bytes(K=12, d=256, V=2048, R=32)


def test_graph_decoder_params_match_module():
    from cogito_estella.model.graph_decoder import GraphDecoder, GraphDecoderConfig
    cfg = GraphDecoderConfig(max_nodes=12, node_dim=256, node_vocab=2048, n_relations=32)
    n_params = sum(p.numel() for p in GraphDecoder(cfg).parameters())
    assert _stats_medium()["params"] == n_params


def test_arithmetic_intensity_monotone_in_batch():
    s = _stats_medium()
    ais = [C.arithmetic_intensity(s, b) for b in (1, 8, 64, 1024)]
    assert all(a < b for a, b in zip(ais, ais[1:]))
    ceiling = s["flops_per_sample"] / s["act_bytes_per_sample"]
    assert ais[-1] < ceiling <= ais[-1] * 1.05


def test_weight_traffic_vanishes_with_batch():
    s = _stats_medium()
    assert C.weight_traffic_fraction(s, 1) > 0.9
    assert C.weight_traffic_fraction(s, 1024) < 0.03


def test_roofline_crossover():
    s = _stats_medium()
    b_fp32 = C.roofline_crossover_batch(s, C.RTX5070_PEAKS_FLOPS["fp32_cuda"])
    assert 1 < b_fp32 < 64  # compute-bound reachable at small batch
    assert C.roofline_crossover_batch(s, C.RTX5070_PEAKS_FLOPS["bf16_tensor_high"]) == math.inf


def test_otsu_separates_bimodal():
    p = torch.tensor([0.01, 0.03, 0.02, 0.95, 0.97, 0.99])
    t = otsu_threshold(p)
    assert 0.05 < t < 0.95


def test_decode_triples_backward_compatible():
    el = torch.tensor([[5.0, 5.0, -5.0]])
    ll = torch.zeros(1, 3, 4); ll[0, 0, 1] = 9; ll[0, 1, 2] = 9
    al = torch.full((1, 2, 3, 3), -5.0); al[0, 0, 0, 1] = 5.0
    assert decode_triples(el, ll, al) == [{(1, 0, 2)}]
    assert decode_triples(el, ll, al, adj_threshold=None) == [{(1, 0, 2)}]


def test_encoder_prefill_vs_kv_state():
    L = 4096
    assert C.sonar_encoder_prefill_flops(L) < 2 * 7e9 * L  # cheaper than 7B prefill
    cs = C.concept_context_state_bytes(L)
    kv = C.token_kv_cache_bytes(L, n_layers=32, dim=4096)
    assert kv / cs > 5000  # orders of magnitude less per-user state


def test_copy_fraction_ceiling_binds_for_ar_only():
    ar = C.copy_fraction_ceiling(100, t_copy_ms=0.4)
    assert 0.30 < ar < 0.36                      # AR fallback: hard ceiling ~33%
    assert C.copy_fraction_ceiling(1000, t_copy_ms=0.4) < 0.01
    assert C.copy_fraction_ceiling(2881, t_copy_ms=C.CHAR_GRID_MS_K32) == 1.0  # grid: unbound


def test_hybrid_mean_latency_monotone():
    a = C.hybrid_mean_latency_ms(0.0, t_copy_ms=0.4)
    b = C.hybrid_mean_latency_ms(0.5, t_copy_ms=0.4)
    assert a == C.GRAPH_DECODE_MS and b > a


def test_space_digits_lang_gate():
    from cogito_estella.preprocess import space_digits
    assert space_digits("id 48213 ok", min_run=5) == "id 4 8 2 1 3 ok"
    assert space_digits("id 48213 ok", min_run=5, lang="rus_Cyrl") == "id 48213 ok"
    assert space_digits("id 48213 ok", min_run=5, lang="jpn_Jpan") == "id 48213 ok"
    assert space_digits("id 48213 ok", min_run=5, lang="deu_Latn") == "id 4 8 2 1 3 ok"


def _shaped(R, K, chunks, seed=0):
    g = torch.Generator().manual_seed(seed)
    vals = torch.cat([lo + (hi - lo) * torch.rand(n, generator=g) for n, lo, hi in chunks])
    assert vals.numel() == R * K * K
    return vals[torch.randperm(vals.numel(), generator=g)].view(R, K, K)


def test_noise_floor_spike_plus_subtle():
    # saturated spike must not raise the bar above subtle true edges
    p = _shaped(48, 24, [(27632, 0.01, 0.025), (12, 0.28, 0.32), (4, 0.99, 0.995)])
    t = noise_floor_threshold(p)
    assert 0.025 < t < 0.28


def test_noise_floor_empty_graph_stays_empty():
    p = _shaped(48, 24, [(27648, 0.005, 0.03)])
    assert noise_floor_threshold(p) > 0.03


def test_noise_floor_ultra_sparse_signal():
    # 3 edges among 27648 cells: variance-invisible to Otsu, caught by sparsity prior
    p = _shaped(48, 24, [(27645, 0.01, 0.025), (3, 0.28, 0.32)])
    t = noise_floor_threshold(p)
    assert 0.025 < t < 0.28


def test_noise_floor_shape_aware_quantile():
    # small R: fixed q=0.995 would land inside the signal; shape-derived q must not
    p = _shaped(2, 16, [(500, 0.01, 0.025), (6, 0.28, 0.32), (6, 0.99, 0.995)])
    t = noise_floor_threshold(p)
    assert 0.025 < t < 0.28


def test_decode_triples_noise_floor_mode():
    el = torch.tensor([[5.0, 5.0, 5.0]])
    ll = torch.zeros(1, 3, 8)
    for k in range(3):
        ll[0, k, k] = 9.0
    al = torch.full((1, 4, 3, 3), -4.0)          # noise ~0.018
    al[0, 0, 0, 1] = -0.85                        # subtle edge ~0.30
    al[0, 1, 1, 2] = 5.0                          # saturated edge ~0.99
    got = decode_triples(el, ll, al, adj_threshold="noise_floor")[0]
    assert got == {(0, 0, 1), (1, 1, 2)}


def test_decode_triples_otsu_recovers_low_logit_edges():
    # miscalibrated head: true edges sit at sigmoid ~0.3, noise at ~0.05
    el = torch.tensor([[5.0, 5.0, 5.0]])
    ll = torch.zeros(1, 3, 8)
    for k in range(3):
        ll[0, k, k] = 9.0
    al = torch.full((1, 1, 3, 3), -3.0)          # noise ~0.047
    al[0, 0, 0, 1] = -0.85                        # edge ~0.30 (below fixed 0.5)
    al[0, 0, 1, 2] = -0.85
    assert decode_triples(el, ll, al)[0] == set()               # fixed 0.5 collapses
    assert decode_triples(el, ll, al, adj_threshold="otsu")[0] == {(0, 0, 1), (1, 0, 2)}
