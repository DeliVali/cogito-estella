"""Code to graph by syntax: names, relations and the line each fact sits on."""
import pytest

pytest.importorskip("tree_sitter_language_pack")

from cogito_estella.mcp import codegraph
from cogito_estella.mcp.store import GraphStore

PY = ("import os\nfrom net import post\n\nclass Wallet(Base):\n    def pay(self, amount):\n"
      "        return post(self.url, amount)\n\ndef main():\n    Wallet().pay(3)\n")


class _Full:
    lite = False
    encoder_name = None

    def _scanner(self):
        return False


def _triples(text, name):
    st = GraphStore(extractor=_Full())
    st.ingest_text(text, name)
    return {(e["s"], e["r"], e["o"], e["sent_idx"]) for e in st.edges.values()}, st


def test_python_definitions_calls_imports_and_bases():
    got, _ = _triples(PY, "w.py")
    assert {("w.py", "imports", "os", 0), ("w.py", "defines", "Wallet", 2),
            ("Wallet", "inherits", "Base", 2), ("Wallet", "contains", "pay", 3),
            ("pay", "calls", "post", 4), ("main", "calls", "pay", 6)} <= got


@pytest.mark.parametrize("name,src,want", [
    ("Pay.kt", "import com.acme.Net\nclass Pay : Base() {\n  fun send() { client.post(url) }\n}\n",
     {("Pay", "inherits", "Base"), ("Pay", "contains", "send"), ("send", "calls", "post")}),
    ("W.swift", "import UIKit\nclass Wallet: Base {\n  func pay() { charge(amount) }\n}\n",
     {("W.swift", "imports", "UIKit"), ("Wallet", "inherits", "Base"), ("pay", "calls", "charge")}),
    ("cart.ts", "class Cart extends Base {\n  total() { return sum(this.items) }\n}\n",
     {("Cart", "inherits", "Base"), ("Cart", "contains", "total"), ("total", "calls", "sum")}),
    ("Pay.java", "class Pay extends Base {\n  void send() { client.post(url); }\n}\n",
     {("Pay", "inherits", "Base"), ("send", "calls", "post")}),
    ("pay.go", 'package main\nimport "fmt"\nfunc Pay() { fmt.Println(1) }\n',
     {("pay.go", "imports", "fmt"), ("Pay", "calls", "Println")}),
])
def test_other_languages(name, src, want):
    got, _ = _triples(src, name)
    assert want <= {(s, r, o) for s, r, o, _ in got}


def test_query_keeps_case_apart_and_finds_names_case_insensitively():
    _, st = _triples(PY + "\ndef Pay():\n    pass\ndef pay():\n    pass\n", "w.py")
    assert st.query("wallet").startswith("Wallet (")
    assert st.resolve("Pay") == "Pay" and st.resolve("pay") == "pay"
    assert "~ class-only" not in st.query("Wallet")     # syntax facts are exact


def test_unknown_language_and_lite_mode_give_no_graph():
    assert codegraph.extract("x = 1\n", "notes.unknownext", [0]) == []
    from cogito_estella.mcp.store import TextOnlyExtractor
    st = GraphStore(extractor=TextOnlyExtractor())
    st.ingest_text(PY, "w.py")
    assert not st.edges and st.docs["w.py"]["lite"]
