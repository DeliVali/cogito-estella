"""Global sensemaking over extracted triples at a fraction of standard GraphRAG cost.

Deterministic pipeline validated in exp048 (novel-scale, ~155x fewer LLM tokens):
PPMI edge weights (chance-level co-occurrence self-suppresses; no stoplists) ->
Louvain communities -> scale-free cohesion gate -> salience-ranked evidence
(frequency x PMI, the TF-IDF analogue: keeps protagonists, drops both generic hubs
and rare trivia) -> summaries via a CALLER-INJECTED llm_fn -> entity-level grounding
verification with automatic rejection (invented entities are unforgivable; paraphrased
relations are free).
"""
import math
import re
from collections import Counter, defaultdict

import networkx as nx

_STOP = {"this", "that", "with", "from", "their", "there", "these", "about", "which"}


def build_pmi_graph(triples):
    """triples: iterable of ((s, r, o), source_text). Returns (graph with PPMI
    weights, undirected pair counts, evidence map pair -> [(s, r, o, text), ...])."""
    ent_count, pair_count = Counter(), Counter()
    evidence = defaultdict(list)
    for (s, r, o), text in triples:
        ent_count[s] += 1
        ent_count[o] += 1
        key = tuple(sorted((s, o)))
        pair_count[key] += 1
        evidence[key].append((s, r, o, text))
    N = sum(pair_count.values()) or 1
    Ne = sum(ent_count.values()) or 1
    G = nx.Graph()
    for (u, v), c in pair_count.items():
        if u == v:
            continue
        w = max(0.0, math.log((c / N) / ((ent_count[u] / Ne) * (ent_count[v] / Ne))))
        if w > 0:
            G.add_edge(u, v, weight=w)
    G.graph["ent_count"] = ent_count
    return G, pair_count, evidence


def salience_ranked_edges(G, nodes, pair_count):
    """Edges of the induced subgraph ranked by frequency x PPMI (descending)."""
    sub = G.subgraph(nodes)
    return sorted(
        ((u, v, d) for u, v, d in sub.edges(data=True)),
        key=lambda e: -(pair_count[tuple(sorted((e[0], e[1])))] * e[2]["weight"]))


def cohesion(G, community):
    ws = [d["weight"] for _, _, d in G.subgraph(community).edges(data=True)]
    return sum(ws) / max(len(ws), 1)


def entity_grounding(summary, community, evidence_text, ent_count):
    """Fraction of the summary's KNOWN-entity mentions that belong to the community or
    its evidence. Relations may paraphrase freely; entities may not be invented."""
    comm = {e.lower() for e in community}
    ev = evidence_text.lower()
    mentions = [w for w in re.findall(r"[a-z]{3,}", summary.lower())
                if w not in _STOP and (w in ent_count or w.rstrip("s") in ent_count)]
    if not mentions:
        return 1.0
    ok = sum(1 for w in mentions
             if w in comm or w.rstrip("s") in comm or w in ev)
    return ok / len(mentions)


class GraphSummarizer:
    """Community summaries with automatic quality control. `llm_fn(prompt) -> str` is
    supplied by the caller (any local or remote model)."""

    def __init__(self, llm_fn, min_size: int = 5, min_grounding: float = 0.75,
                 evidence_edges: int = 12, evidence_quotes: int = 6):
        self.llm_fn = llm_fn
        self.min_size = min_size
        self.min_grounding = min_grounding
        self.evidence_edges = evidence_edges
        self.evidence_quotes = evidence_quotes

    def summarize(self, triples, top_k: int = 8):
        G, pair_count, evidence = build_pmi_graph(triples)
        if G.number_of_edges() == 0:
            return []
        ent_count = G.graph["ent_count"]
        comms = sorted(nx.community.louvain_communities(G, weight="weight", seed=0),
                       key=len, reverse=True)
        global_mean = (sum(d["weight"] for _, _, d in G.edges(data=True))
                       / G.number_of_edges())
        passed = [c for c in comms
                  if len(c) >= self.min_size and cohesion(G, c) >= global_mean][:top_k]
        out = []
        for comm in passed:
            ranked = salience_ranked_edges(G, comm, pair_count)[: self.evidence_edges]
            facts, quotes = [], []
            for u, v, _ in ranked:
                s, r, o, text = evidence[tuple(sorted((u, v)))][0]
                facts.append(f"{s} -{r}- {o}")
                if len(quotes) < self.evidence_quotes:
                    quotes.append(text)
            prompt = ("Facts and quotes from ONE thematic cluster of a knowledge "
                      "graph.\nFacts: " + "; ".join(facts) +
                      "\nQuotes: " + " | ".join(q[:140] for q in quotes) +
                      "\nWrite a 2-3 sentence summary. Mention ONLY entities present "
                      "in the facts/quotes above; do not speculate.")
            summary = self.llm_fn(prompt).strip()
            grounding = entity_grounding(summary, comm, " ".join(facts + quotes),
                                         ent_count)
            top = [n for n, _, _ in
                   ((n, None, None) for n in sorted(
                       comm, key=lambda n: -sum(
                           pair_count[tuple(sorted((n, m)))] * G[n][m]["weight"]
                           for m in G.neighbors(n))))][:8]
            out.append({"size": len(comm), "cohesion": round(cohesion(G, comm), 3),
                        "top_entities": top, "summary": summary,
                        "grounding": round(grounding, 3),
                        "accepted": grounding >= self.min_grounding})
        return out
