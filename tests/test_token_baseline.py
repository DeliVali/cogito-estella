import torch

from cogito_estella.model.token_baseline import TokenTransformer, TokenConfig


def test_forward_shape():
    cfg = TokenConfig(vocab_size=100, dim=64, n_layers=2, n_heads=4, max_seq_len=32)
    model = TokenTransformer(cfg)
    x = torch.randint(0, 100, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, 100)


def test_causality():
    cfg = TokenConfig(vocab_size=50, dim=64, n_layers=2, n_heads=4, max_seq_len=32)
    model = TokenTransformer(cfg).eval()
    x = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        out_a = model(x)
        x2 = x.clone()
        x2[0, 5] = (x2[0, 5] + 1) % 50
        out_b = model(x2)
    assert torch.allclose(out_a[0, :5], out_b[0, :5], atol=1e-4)
    assert not torch.allclose(out_a[0, 5], out_b[0, 5], atol=1e-4)


def test_overfit_cpu():
    torch.manual_seed(0)
    cfg = TokenConfig(vocab_size=40, dim=64, n_layers=2, n_heads=4, max_seq_len=16)
    model = TokenTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 40, (4, 12))
    import torch.nn.functional as F
    first = last = None
    for step in range(200):
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 40), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first / 5
