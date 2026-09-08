"""Shared fixtures. spaCy is optional: tests that need it skip cleanly when absent."""
import pytest


@pytest.fixture(scope="session")
def nlp():
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        pytest.skip("en_core_web_sm not installed")


@pytest.fixture
def fake_extractor(nlp):
    """Real CogitoGraphExtractor methods over a fake head: `extract`/`extract_batch`
    return fixed triples per sentence; spans and lexicalization run for real."""
    from cogito_estella.integrations.llamaindex_connector import CogitoGraphExtractor

    def make(triples_by_sentence, ent2id=None):
        ex = object.__new__(CogitoGraphExtractor)
        ex.ent2id = ent2id or {"concept": 1, "sonar": 2, "encoder": 3, "text": 4,
                               "vector": 5, "budget": 6, "committee": 7, "decoder": 8}
        ex._nlp = nlp
        ex.device, ex.rels = "cpu", []
        ex.extract = lambda text, candidates=None, lang="eng_Latn", return_scores=False: \
            list(triples_by_sentence.get(text, []))
        ex.extract_batch = lambda texts, candidates=None, lang="eng_Latn": \
            [list(triples_by_sentence.get(t, [])) for t in texts]
        return ex
    return make
