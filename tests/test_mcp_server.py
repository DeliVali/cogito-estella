"""Tool bodies over a store (offline) and the stdio protocol smoke (integration)."""
import sys
from pathlib import Path

import pytest

from cogito_estella.mcp.server import (
    INSTRUCTIONS,
    Tools,
    build_extractor,
    build_server,
    parse_args,
)
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


def test_build_server_registers_the_tool_set(tools):
    server = build_server(tools.store)
    names = {t.name for t in server._tool_manager.list_tools()} if hasattr(server, "_tool_manager") \
        else set(getattr(server, "_tools", {}))
    assert {"ask", "ingest", "query", "provenance", "search", "entities", "stats"} <= names


def test_ingest_long_raw_text_is_not_treated_as_a_path(tools):
    text = ("The generated concepts are decoded by SONAR into a sequence of subwords. " * 4).strip()
    out = tools.ingest(text)
    assert out.startswith("source=text1 status=ingested")


def test_ingest_empty_text_returns_error_line(tools, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trap.txt").write_text("Do not ingest me.")
    out = tools.ingest("   ")
    assert out == "source=text1 error=empty input" and tools.store.docs == {}


# -- ask: the default routing gate ----------------------------------------------------

def test_ask_returns_facts_and_sentences_and_is_charged(tools):
    tools.ingest(DOC)
    out = tools.ask("What is decoded by SONAR?")
    assert out.startswith("entities: sonar")
    assert "sonar decode concept #0" in out
    assert '\n--\n' in out and 'text1 s0: "' in out
    assert tools.store.ledger.calls["ask"] == 1
    assert tools.store.ledger.served_by["ask"] > 0


def test_ask_clamps_the_budget(tools, monkeypatch):
    seen = []
    monkeypatch.setattr(tools.store, "ask",
                        lambda q, budget, scorer: seen.append(budget) or "ok")
    tools.ask("q", budget=1)
    tools.ask("q", budget=99_999)
    tools.ask("q")
    assert seen == [100, 4000, 600]


def test_ask_use_sonar_without_embeddings_falls_back_with_a_note(tools):
    tools.ingest(DOC)
    assert "scorer=lexical (sonar unavailable)" in tools.ask("sonar decoder", use_sonar=True)


def test_ask_ranks_lexically_by_default(tools):
    tools.ingest(DOC)
    out = tools.ask("sonar decoder")
    assert "scorer=lexical" in out and "sonar unavailable" not in out


def test_ask_maps_the_toggle_to_the_store_scorer(tools, monkeypatch):
    seen = []
    monkeypatch.setattr(tools.store, "ask",
                        lambda q, budget, scorer: seen.append(scorer) or "ok")
    tools.ask("q")
    tools.ask("q", use_sonar=True)
    assert seen == ["lexical", "sonar"]


def test_instructions_route_every_question_to_ask():
    assert "Start every question with `ask`" in INSTRUCTIONS
    assert "go deeper" in INSTRUCTIONS and "class-only" in INSTRUCTIONS


def test_parse_args_defaults():
    ns = parse_args([])
    assert ns.dir == Path(".cogito") and ns.checkpoint is None and ns.vocab is None
    assert ns.download is True and ns.device is None and ns.encoder is None
    ns = parse_args(["--dir", "x", "--checkpoint", "a.pt", "--checkpoint", "b.pt",
                     "--vocab", "v.json", "--no-download", "--device", "cpu"])
    assert ns.checkpoint == ["a.pt", "b.pt"] and ns.download is False and ns.device == "cpu"


# -- finding 5: an empty (or all-unsupported) directory reports an error line ----------

def test_ingest_empty_directory_reports_no_supported_documents(tools, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out = tools.ingest(str(empty_dir))
    assert out.startswith("source=")
    assert "no supported documents" in out


# -- finding 10: limit is clamped to at least 1 for query/search/entities -------------

def test_query_limit_zero_or_negative_is_clamped_to_one(tools):
    tools.ingest(DOC)
    out0 = tools.query("sonar", limit=0)
    out1 = tools.query("sonar", limit=1)
    outm1 = tools.query("sonar", limit=-1)
    assert out0 == out1 == outm1


def test_search_limit_zero_or_negative_is_clamped_to_one(tools):
    tools.ingest(DOC)
    assert tools.search("The", limit=0) == tools.search("The", limit=1)
    assert tools.search("The", limit=-1) == tools.search("The", limit=1)


def test_entities_limit_zero_or_negative_is_clamped_to_one(tools):
    tools.ingest(DOC)
    assert tools.entities(limit=0) == tools.entities(limit=1)
    assert tools.entities(limit=-1) == tools.entities(limit=1)


# -- finding 11: an empty provenance query is a charged, explicit reply ---------------

def test_provenance_empty_string_reports_no_ids(tools):
    before = tools.store.ledger.calls["provenance"]
    out = tools.provenance("")
    assert out == "no edge ids in ''"
    assert tools.store.ledger.calls["provenance"] == before + 1


# -- finding 13: a missing optional dependency exits with an install hint ------------

def test_main_missing_dependency_at_weight_resolution_exits_with_install_hint(monkeypatch, tmp_path):
    import cogito_estella.mcp.server as srv

    def boom(*a, **k):
        raise ImportError("No module named 'spacy'", name="spacy")
    monkeypatch.setattr(srv, "resolve", lambda *a, **k: ([], Path("v.json")))
    monkeypatch.setattr(srv, "ensure_spacy_model", boom)
    with pytest.raises(SystemExit) as exc:
        srv.main(["--dir", str(tmp_path)])
    assert "spacy" in str(exc.value) and "cogito-estella[mcp]" in str(exc.value)


def test_main_missing_mcp_sdk_exits_with_install_hint(monkeypatch, tmp_path):
    import cogito_estella.mcp.server as srv

    def boom(store):
        raise ImportError("No module named 'mcp'", name="mcp")
    monkeypatch.setattr(srv, "resolve", lambda *a, **k: ([], Path("v.json")))
    monkeypatch.setattr(srv, "ensure_spacy_model", lambda **k: None)
    monkeypatch.setattr(srv, "build_server", boom)
    with pytest.raises(SystemExit) as exc:
        srv.main(["--dir", str(tmp_path)])
    assert "mcp" in str(exc.value) and "cogito-estella[mcp]" in str(exc.value)


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
    second = asyncio.run(run([("query", {"entity": "sonar"}),
                              ("ask", {"question": "What does SONAR decode?"})]))
    assert "sonar decode concept" in second[0]
    assert second[1].startswith("entities: sonar") and "\n--\n" in second[1]


# -- --encoder and the startup canary --------------------------------------------------

def test_parse_args_accepts_an_encoder_choice():
    assert parse_args([]).encoder is None
    assert parse_args(["--encoder", "bge-m3"]).encoder == "bge-m3"
    with pytest.raises(SystemExit):
        parse_args(["--encoder", "sonar-v2"])


class _StubExtractor:
    def __init__(self, ckpts, vocab, device=None, encoder=None, download=True):
        self.args = {"ckpts": ckpts, "vocab": vocab, "device": device,
                     "encoder": encoder, "download": download}
        self.canaries = 0

    def check_canary(self):
        self.canaries += 1


def test_build_extractor_forwards_the_flags_and_runs_the_canary_once(monkeypatch):
    import cogito_estella.integrations.llamaindex_connector as lc
    monkeypatch.setattr(lc, "CogitoGraphExtractor", _StubExtractor)
    ns = parse_args(["--encoder", "bge-m3", "--no-download", "--device", "cpu"])
    ex = build_extractor([Path("a.pt")], Path("v.json"), ns)
    assert ex.canaries == 1
    assert ex.args == {"ckpts": ["a.pt"], "vocab": "v.json", "device": "cpu",
                       "encoder": "bge-m3", "download": False}


@pytest.mark.parametrize("where", ["init", "canary"])
def test_build_extractor_stops_the_server_on_an_encoder_mismatch(monkeypatch, where):
    import cogito_estella.integrations.llamaindex_connector as lc
    from cogito_estella.encoders import EncoderMismatch

    def boom(*_a, **_k):
        raise EncoderMismatch("checkpoint a.pt was trained on sonar, active encoder is bge-m3")

    class Failing(_StubExtractor):
        check_canary = boom

    monkeypatch.setattr(lc, "CogitoGraphExtractor", boom if where == "init" else Failing)
    with pytest.raises(SystemExit) as exc:
        build_extractor([Path("a.pt")], Path("v.json"), parse_args(["--encoder", "bge-m3"]))
    assert "trained on sonar" in str(exc.value) and str(exc.value).startswith("cogito-mcp:")
