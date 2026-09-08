"""Token proxy and ledger for the MCP server."""
import sys
import types

import pytest

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


def test_ntok_falls_back_when_encoding_load_raises_value_error(monkeypatch):
    """Verify ntok falls back when tiktoken.get_encoding raises ValueError (hash mismatch)."""
    import cogito_estella.mcp.tokens as t
    t._TRIED = False
    t._ENC = None

    fake_tiktoken = types.ModuleType("tiktoken")
    def raise_value_error(*args, **kwargs):
        raise ValueError("hash mismatch")
    fake_tiktoken.get_encoding = raise_value_error

    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)

    assert t.ntok("abcdefgh") == 2
    assert t._ENC is None


def test_ntok_counts_text_with_special_tokens():
    """Verify ntok counts text containing disallowed special tokens without raising."""
    pytest.importorskip("tiktoken")
    import cogito_estella.mcp.tokens as t
    t._TRIED = False
    t._ENC = None

    n = t.ntok("hello <|endoftext|> world")
    assert n >= 3
