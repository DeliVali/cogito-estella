"""Shared fixtures. spaCy is optional: tests that need it skip cleanly when absent."""
import pytest


@pytest.fixture(scope="session")
def nlp():
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        pytest.skip("en_core_web_sm not installed")
