"""Exact-literal channel: deterministic regex detection + verbatim preservation.
Semantic decoding handles meaning; literals (phones, hashes, IDs, precise numbers)
travel through the channel that cannot fail — copying source text."""
import pytest

from cogito_estella.integrations.llamaindex_connector import extract_literals


def test_detects_phone_email_url_hash_and_precise_numbers():
    text = ("Call +52 55 1234 5678 or write to clinic@salud.mx — order a3f9c2d84b1e77aa "
            "cost 1249.99 at https://pay.example.com/x ref 4415523901")
    lits = extract_literals(text)
    assert "+52 55 1234 5678" in lits["phone"]
    assert "clinic@salud.mx" in lits["email"]
    assert "https://pay.example.com/x" in lits["url"]
    assert "a3f9c2d84b1e77aa" in lits["hash"]
    assert "1249.99" in lits["number"]
    assert "4415523901" in lits["number"]


def test_verbatim_no_normalization():
    text = "ID: AB-9921-x7 y monto 3.14159"
    lits = extract_literals(text)
    flat = [v for vs in lits.values() for v in vs]
    for v in flat:
        assert v in text, "literals must be exact substrings of the source"


def test_ignores_ordinary_text_years_and_small_numbers():
    lits = extract_literals("In 1999 the committee approved 3 budgets for 12 hospitals.")
    assert all(not v for v in lits.values()), f"false positives: {lits}"


def test_neo4j_literal_linking_with_mock_driver():
    from cogito_estella.integrations.llamaindex_connector import CogitoGraphExtractor

    class FakeSession:
        def __init__(self):
            self.calls = []

        def run(self, q, **p):
            self.calls.append((q, p))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class FakeDriver:
        def __init__(self):
            self.s = FakeSession()

        def session(self, database=None):
            return self.s

    drv = FakeDriver()
    triples = [("clinic", "have", "phone")]
    lits = {"phone": ["+52 55 1234 5678"], "email": [], "url": [], "hash": [],
            "number": []}
    CogitoGraphExtractor.literals_to_neo4j(drv, triples, lits, source="doc-9")
    joined = " ".join(q for q, _ in drv.s.calls)
    assert "Literal" in joined and "HAS_LITERAL" in joined
    params = [p for _, p in drv.s.calls]
    assert any(p.get("value") == "+52 55 1234 5678" and p.get("kind") == "phone"
               and p.get("src") == "doc-9" for p in params)
    # linked to the sentence's entities (co-occurrence semantics)
    assert any(p.get("ent") in ("clinic", "phone") for p in params)
