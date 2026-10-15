"""The 1.0 public surface (README, "Stable API"). A change that breaks one of these
assertions breaks callers: it needs a major version, not an edit of this file."""
import inspect

from cogito_estella import Estella
from cogito_estella.mcp.server import Tools, build_server, parse_args
from cogito_estella.mcp.store import GraphStore


def _params(fn):
    return {n: p.default for n, p in inspect.signature(fn).parameters.items() if n != "self"}


def _tools(server):
    manager = getattr(server, "_tool_manager", None)
    tools = manager.list_tools() if manager else getattr(server, "_tools", {}).values()
    return {t.name: set(t.parameters.get("properties", {})) for t in tools}


def test_the_python_api():
    init = _params(Estella.__init__)
    assert {"paths", "full", "budget", "save_to", "vectors"} <= set(init)
    assert init["full"] is False and init["save_to"] is None
    assert {"path_or_text", "name"} <= set(_params(Estella.add))
    assert {"question", "deep"} <= set(_params(Estella.ask))
    assert "question" in _params(Estella.context)
    assert callable(Estella.deep)


def test_the_mcp_tools(fake_extractor, tmp_path):
    store = GraphStore(extractor=fake_extractor({}), path=tmp_path / "graph.json")
    full = _tools(build_server(store, Tools(store, "full")))
    assert {"question"} <= full["ask"] and {"path_or_text"} <= full["ingest"]
    assert {"query", "provenance", "search", "entities", "stats"} <= set(full)
    compact = _tools(build_server(store, Tools(store, "compact")))
    assert compact["ask"] == {"question"} and compact["ingest"] == {"path"}


def test_the_server_flags():
    ns = parse_args(["--profile", "compact", "--budget", "300", "--lite", "--full",
                     "--ingest", "docs", "--dir", ".cogito", "--http", "8000",
                     "--no-download", "--device", "cpu", "--no-vectors"])
    assert ns.profile == "compact" and ns.budget == 300 and ns.vectors is False
