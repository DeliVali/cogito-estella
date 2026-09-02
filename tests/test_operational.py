"""Operational-robustness batch: schema constraints for concurrent ingestion and the
explain-mode score report. Pure/mocked — no GPU, no DB."""
import torch

from cogito_estella.integrations.llamaindex_connector import (
    CogitoGraphExtractor, score_report)


class FakeSession:
    def __init__(self):
        self.calls = []

    def run(self, q, **p):
        self.calls.append(q)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class FakeDriver:
    def __init__(self):
        self.s = FakeSession()

    def session(self, database=None):
        return self.s


def test_ensure_schema_creates_idempotent_uniqueness_constraints():
    drv = FakeDriver()
    CogitoGraphExtractor.ensure_schema(drv)
    joined = " ".join(drv.s.calls)
    assert "IF NOT EXISTS" in joined, "must be idempotent"
    assert "Entity" in joined and "name" in joined and "UNIQUE" in joined
    assert "Literal" in joined, "Literal nodes need uniqueness too"


def test_score_report_exposes_probabilities_and_forced_flag():
    cand = ["clinic", "budget", "noise"]
    rels = ["have", "support"]
    exist = torch.tensor([3.0, 2.0, -6.0])              # clinic/budget in, noise out
    adj = torch.full((2, 3, 3), -8.0)
    adj[1, 0, 1] = 2.5                                   # clinic -support-> budget
    rep = score_report(exist, adj, cand, rels, threshold=0.15, adj_threshold=0.15,
                       force_top1=True)
    probs = {c["name"]: c["exist_prob"] for c in rep["candidates"]}
    assert probs["clinic"] > 0.9 and probs["noise"] < 0.01
    edges = rep["edges"]
    assert edges and edges[0]["s"] == "clinic" and edges[0]["r"] == "support" \
        and edges[0]["o"] == "budget"
    assert edges[0]["adj_prob"] > 0.9 and edges[0]["forced"] is False
    assert rep["triples"] == [("clinic", "support", "budget")]


def test_score_report_marks_forced_floor_edge():
    cand = ["a", "b"]
    rels = ["r0"]
    exist = torch.tensor([-2.0, -2.0])
    adj = torch.full((1, 2, 2), -3.0)
    adj[0, 0, 1] = -1.0                                  # best of a weak lot
    rep = score_report(exist, adj, cand, rels, threshold=0.15, adj_threshold=0.15,
                       force_top1=True)
    assert len(rep["edges"]) == 1 and rep["edges"][0]["forced"] is True
    assert rep["edges"][0]["adj_prob"] < 0.5, "forced edge must expose its low confidence"
