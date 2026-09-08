"""Tool bodies over a store (offline) and the stdio protocol smoke (integration)."""
import sys
from pathlib import Path

import pytest

from cogito_estella.mcp.server import Tools, build_server, parse_args
from cogito_estella.mcp.store import GraphStore

DOC = "The generated concepts are decoded by SONAR. The encoder maps text to a vector."
TRIPLES = {
    "The generated concepts are decoded by SONAR.": [("concept", "improve", "sonar")],
    "The encoder maps text to a vector.": [("encoder", "give", "text")],
}


@pytest.fixture
def tools(fake_extractor, tmp_path):
    return Tools(GraphStore(extractor=fake_extractor(TRIPLES), path=tmp_path / "graph.json"))


def test_ingest_raw_text_and_query(tools):
    out = tools.ingest(DOC)
    assert out.startswith("source=text1 status=ingested")
    assert "sonar decode concept #0" in tools.query("sonar")
    assert tools.store.ledger.calls["query"] == 1 and tools.store.ledger.calls["ingest"] == 1


def test_ingest_path_and_directory(tools, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text(DOC)
    out = tools.ingest(str(tmp_path / "docs"))
    assert out.startswith("source=a.txt status=ingested")
    assert tools.ingest(str(tmp_path / "docs")).startswith("source=a.txt status=unchanged")


def test_provenance_parses_ids_and_stats_mentions_file(tools):
    tools.ingest(DOC)
    assert tools.provenance("0, 7").count("#") >= 2
    assert "graph_file=" in tools.stats() and "docs=1" in tools.stats()


def test_build_server_registers_six_tools(tools):
    server = build_server(tools.store)
    names = {t.name for t in server._tool_manager.list_tools()} if hasattr(server, "_tool_manager") \
        else set(getattr(server, "_tools", {}))
    assert {"ingest", "query", "provenance", "search", "entities", "stats"} <= names


def test_ingest_long_raw_text_is_not_treated_as_a_path(tools):
    text = ("The generated concepts are decoded by SONAR into a sequence of subwords. " * 4).strip()
    out = tools.ingest(text)
    assert out.startswith("source=text1 status=ingested")


def test_ingest_empty_text_returns_error_line(tools, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trap.txt").write_text("Do not ingest me.")
    out = tools.ingest("   ")
    assert out == "source=text1 error=empty input" and tools.store.docs == {}


def test_parse_args_defaults():
    ns = parse_args([])
    assert ns.dir == Path(".cogito") and ns.checkpoint is None and ns.vocab is None
    assert ns.download is True and ns.device is None
    ns = parse_args(["--dir", "x", "--checkpoint", "a.pt", "--checkpoint", "b.pt",
                     "--vocab", "v.json", "--no-download", "--device", "cpu"])
    assert ns.checkpoint == ["a.pt", "b.pt"] and ns.download is False and ns.device == "cpu"


@pytest.mark.integration
def test_stdio_smoke_persists_across_restarts(tmp_path):
    """Needs GPU-or-CPU weights on disk (COGITO_MCP_CHECKPOINTS / COGITO_MCP_VOCAB env)."""
    import asyncio
    import os

    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    cks = os.environ.get("COGITO_MCP_CHECKPOINTS", "").split(":")
    vocab = os.environ.get("COGITO_MCP_VOCAB", "")
    if not cks[0] or not vocab:
        pytest.skip("set COGITO_MCP_CHECKPOINTS=a.pt:b.pt:c.pt and COGITO_MCP_VOCAB")
    args = ["-m", "cogito_estella.mcp.server", "--dir", str(tmp_path), "--vocab", vocab]
    for c in cks:
        args += ["--checkpoint", c]
    text = "The generated concepts are decoded by SONAR into a sequence of subwords."

    async def run(calls):
        params = StdioServerParameters(command=sys.executable, args=args)
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            out = []
            for name, a in calls:
                res = await s.call_tool(name, a, read_timeout_seconds=600)
                out.append("\n".join(c.text for c in res.content if hasattr(c, "text")))
            return out
    first = asyncio.run(run([("ingest", {"path_or_text": text}), ("query", {"entity": "sonar"})]))
    assert "sonar decode concept" in first[1]
    second = asyncio.run(run([("query", {"entity": "sonar"})]))
    assert "sonar decode concept" in second[0]
