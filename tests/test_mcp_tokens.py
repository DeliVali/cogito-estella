"""Token proxy and ledger for the MCP server."""
from cogito_estella.mcp.tokens import Ledger, ntok


def test_ntok_counts_roughly_by_words():
    n = ntok("The encoder maps text to a vector.")
    assert 5 <= n <= 12


def test_ntok_fallback_when_tiktoken_missing(monkeypatch):
    import cogito_estella.mcp.tokens as t
    monkeypatch.setattr(t, "_ENC", None)
    monkeypatch.setattr(t, "_TRIED", True)
    assert t.ntok("abcdefgh") == 2 and t.ntok("") == 1


def test_ledger_charges_per_tool():
    L = Ledger()
    n = L.charge("query", "sonar decode concept #0")
    L.charge("query", "x")
    L.charge("search", "y")
    assert n >= 1 and L.served == L.served_by["query"] + L.served_by["search"]
    assert L.calls == {"query": 2, "search": 1}
