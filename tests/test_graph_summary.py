"""Graph-summary invariants: PPMI weighting, salience ranking, scale-free cohesion
gate, entity-level grounding. All deterministic and offline; the summarizer LLM is an
injected callable (mocked here)."""
import math

import pytest

from cogito_estella.graph_summary import (
    GraphSummarizer,
    build_pmi_graph,
    cohesion,
    entity_grounding,
    salience_ranked_edges,
)

TRIPLES = (
    # ubiquitous pair: "mr"–"way" co-occur often but each is everywhere
    [(("mr", "have", "way"), f"mr way sentence {i}") for i in range(6)]
    + [(("mr", "have", x), f"mr {x}") for x in ("a", "b", "c", "d", "e", "f")]
    + [(("way", "have", x), f"way {x}") for x in ("a", "b", "c", "d", "e", "f")]
    # informative pair: lydia–wickham only ever co-occur with each other
    + [(("lydia", "meet", "wickham"), f"lydia wickham sentence {i}") for i in range(3)]
)


def test_ppmi_downweights_ubiquitous_pairs():
    G, pair_count, _ = build_pmi_graph(TRIPLES)
    assert G["lydia"]["wickham"]["weight"] > G["mr"]["way"]["weight"], \
        "surprising co-occurrence must outweigh chance-level co-occurrence"
    assert min(d["weight"] for _, _, d in G.edges(data=True)) >= 0.0  # PPMI


def test_salience_prefers_frequent_and_informative():
    G, pair_count, _ = build_pmi_graph(TRIPLES)
    ranked = salience_ranked_edges(G, G.nodes, pair_count)
    top = tuple(sorted(ranked[0][:2]))
    assert top == ("lydia", "wickham"), \
        "salience = freq x PMI must rank the protagonist edge first"


def test_cohesion_gate_is_scale_free():
    # a big community diluted with weak edges must not be auto-penalized vs global mean
    big = [((f"n{i}", "r", f"n{i+1}"), "t") for i in range(20)] * 2
    tight = [(("a", "r", "b"), "t")] * 5
    G, pc, _ = build_pmi_graph(big + tight)
    global_mean = sum(d["weight"] for _, _, d in G.edges(data=True)) / G.number_of_edges()
    assert cohesion(G, {"a", "b"}) >= global_mean


def test_entity_grounding_allows_paraphrase_punishes_invented_entities():
    ents = {"elizabeth": 10, "carta": 5, "collins": 4}
    comm = {"elizabeth", "carta"}
    evidence = "elizabeth recibe una carta"
    # paraphrased relation, grounded entities -> high score
    good = entity_grounding("la carta agita el animo de elizabeth", comm, evidence, ents)
    # invented entity (collins is a known entity but NOT in this cluster/evidence)
    bad = entity_grounding("collins agita el animo de elizabeth", comm, evidence, ents)
    assert good == 1.0
    assert bad < good


def test_summarizer_end_to_end_with_mock_llm_and_rejection():
    calls = []

    def mock_llm(prompt):
        calls.append(prompt)
        # first community gets a grounded reply; others an invented-entity reply
        return ("lydia meets wickham" if "lydia" in prompt
                else "napoleon invades the cluster")

    gs = GraphSummarizer(llm_fn=mock_llm, min_size=2, min_grounding=0.75)
    out = gs.summarize(TRIPLES, top_k=5)
    assert len(calls) >= 1, "gated communities must reach the llm"
    accepted = [s for s in out if s["accepted"]]
    rejected = [s for s in out if not s["accepted"]]
    for s in accepted:
        assert s["grounding"] >= 0.75
    for s in rejected:
        assert s["grounding"] < 0.75
    assert all("summary" in s and "top_entities" in s for s in out)
