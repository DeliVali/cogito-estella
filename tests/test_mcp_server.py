"""Tool bodies over a store (offline) and the stdio protocol smoke (integration)."""
import sys
from pathlib import Path

import pytest

from cogito_estella.mcp.server import (
    DEFAULT_SCORER,
    INSTRUCTIONS,
    SCORER_ENV,
    Tools,
    build_extractor,
    build_server,
    parse_args,
    resolve_scorer,
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
    assert "scorer=lexical (dense unavailable)" in tools.ask("sonar decoder", use_sonar=True)


def test_ask_ranks_with_the_learned_scorer_by_default(tools):
    tools.ingest(DOC)
    out = tools.ask("sonar decoder")
    assert "scorer=learned" in out and "sonar unavailable" not in out


def test_ask_maps_the_toggle_to_the_store_scorer(tools, monkeypatch):
    seen = []
    monkeypatch.setattr(tools.store, "ask",
                        lambda q, budget, scorer: seen.append(scorer) or "ok")
    tools.ask("q")
    tools.ask("q", use_sonar=True)
    assert seen == ["learned", "dense"]


# -- the configured scorer: module default plus an operator knob -----------------------

def test_the_module_default_scorer_is_learned():
    assert DEFAULT_SCORER == "learned" and resolve_scorer("") == "learned"


def test_ask_sends_the_configured_scorer_to_the_store(fake_extractor, monkeypatch):
    seen = []
    monkeypatch.setenv(SCORER_ENV, "learned")
    t = Tools(GraphStore(extractor=fake_extractor(TRIPLES)))
    monkeypatch.setattr(t.store, "ask", lambda q, budget, scorer: seen.append(scorer) or "ok")
    t.ask("q")
    t.ask("q", use_sonar=True)                 # the explicit toggle still wins over the knob
    assert seen == ["learned", "dense"]


def test_the_knob_is_read_once_so_one_run_cannot_change_scorer_midway(fake_extractor, monkeypatch):
    seen = []
    t = Tools(GraphStore(extractor=fake_extractor(TRIPLES)))
    monkeypatch.setattr(t.store, "ask", lambda q, budget, scorer: seen.append(scorer) or "ok")
    monkeypatch.setenv(SCORER_ENV, "lexical")
    t.ask("q")
    assert seen == ["learned"]


def test_resolve_scorer_normalizes_case_and_padding():
    assert resolve_scorer("  LEARNED \n") == "learned"


def test_an_unknown_knob_value_is_refused_instead_of_silently_ignored(fake_extractor, monkeypatch):
    monkeypatch.setenv(SCORER_ENV, "lexcial")
    with pytest.raises(ValueError, match=SCORER_ENV):
        Tools(GraphStore(extractor=fake_extractor(TRIPLES)))


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

def _resolved(checkpoints=(), pool=None, encoder="m2m100-pool", point=None):
    from cogito_estella.mcp.weights import ResolvedWeights

    return ResolvedWeights([Path(c) for c in checkpoints], Path("v.json"), pool, encoder,
                           point)


def test_main_missing_dependency_at_weight_resolution_exits_with_install_hint(monkeypatch, tmp_path):
    import cogito_estella.mcp.server as srv

    def boom(*a, **k):
        raise ImportError("No module named 'spacy'", name="spacy")
    monkeypatch.setattr(srv, "resolve", lambda *a, **k: _resolved())
    monkeypatch.setattr(srv, "ensure_spacy_model", boom)
    with pytest.raises(SystemExit) as exc:
        srv.main(["--dir", str(tmp_path)])
    assert "spacy" in str(exc.value) and "pip install -U cogito-estella" in str(exc.value)


def test_main_missing_mcp_sdk_exits_with_install_hint(monkeypatch, tmp_path):
    import cogito_estella.mcp.server as srv

    def boom(store, *_a, **_k):
        raise ImportError("No module named 'mcp'", name="mcp")
    monkeypatch.setattr(srv, "resolve", lambda *a, **k: _resolved())
    monkeypatch.setattr(srv, "ensure_spacy_model", lambda **k: None)
    monkeypatch.setattr(srv, "build_server", boom)
    with pytest.raises(SystemExit) as exc:
        srv.main(["--dir", str(tmp_path)])
    assert "mcp" in str(exc.value) and "pip install -U cogito-estella" in str(exc.value)


def test_the_startup_line_names_the_encoder_and_its_revision(monkeypatch, tmp_path, capsys):
    """Operators must see which space the served graph is in before any tool call."""
    import cogito_estella.mcp.server as srv
    from cogito_estella.encoders import encoder_revision

    def boom(store, *_a, **_k):
        raise ImportError("No module named 'mcp'", name="mcp")
    monkeypatch.setattr(srv, "resolve", lambda *a, **k: _resolved())
    monkeypatch.setattr(srv, "ensure_spacy_model", lambda **k: None)
    monkeypatch.setattr(srv, "build_server", boom)
    with pytest.raises(SystemExit):
        srv.main(["--dir", str(tmp_path)])
    err = capsys.readouterr().err
    rev = encoder_revision("m2m100-pool")
    assert "encoder=m2m100-pool" in err and f"revision={rev}" in err and rev != "unknown"


def test_the_startup_line_does_not_guess_the_encoder_of_explicit_checkpoints(
        monkeypatch, tmp_path, capsys):
    """Legacy `.pt` paths resolve to sonar inside the extractor: announcing the process
    default here would name the wrong semantic space."""
    import cogito_estella.mcp.server as srv

    def boom(store, *_a, **_k):
        raise ImportError("No module named 'mcp'", name="mcp")
    monkeypatch.setattr(srv, "resolve", lambda *a, **k: _resolved(["a.pt"], encoder=None))
    monkeypatch.setattr(srv, "ensure_spacy_model", lambda **k: None)
    monkeypatch.setattr(srv, "build_server", boom)
    with pytest.raises(SystemExit):
        srv.main(["--dir", str(tmp_path), "--checkpoint", "a.pt", "--vocab", "v.json"])
    err = capsys.readouterr().err
    assert "encoder=m2m100-pool" not in err and "encoder=from-checkpoints" in err


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
    assert parse_args(["--encoder", "m2m100-pool"]).encoder == "m2m100-pool"
    with pytest.raises(SystemExit):
        parse_args(["--encoder", "sonar-v2"])


def test_parse_args_accepts_pooling_weights():
    assert parse_args([]).pool is None
    assert parse_args(["--pool", "w/pool.pt"]).pool == Path("w/pool.pt")


def test_build_extractor_stops_the_server_when_the_pooling_weights_are_missing(monkeypatch):
    import cogito_estella.integrations.llamaindex_connector as lc
    from cogito_estella.encoders.pooling import PoolWeightsError

    def boom(*_a, **_k):
        raise PoolWeightsError("pooling weights not found: pool.pt")

    class Failing(_StubExtractor):
        check_canary = boom

    monkeypatch.setattr(lc, "CogitoGraphExtractor", Failing)
    with pytest.raises(SystemExit) as exc:
        build_extractor(_resolved(["a.pt"], Path("pool.pt")),
                        parse_args(["--encoder", "m2m100-pool", "--pool", "pool.pt"]))
    assert "pooling weights not found" in str(exc.value)


class _StubExtractor:
    encoder_name = "sonar"

    def __init__(self, ckpts, vocab, device=None, encoder=None, download=True,
                 pool_path=None, threshold=None, adj_threshold=None):
        self.args = {"ckpts": ckpts, "vocab": vocab, "device": device,
                     "encoder": encoder, "download": download, "pool_path": pool_path,
                     "threshold": threshold, "adj_threshold": adj_threshold}
        self.canaries = 0

    def check_canary(self):
        self.canaries += 1


def test_build_extractor_forwards_the_flags_and_runs_the_canary_once(monkeypatch):
    import cogito_estella.integrations.llamaindex_connector as lc
    monkeypatch.setattr(lc, "CogitoGraphExtractor", _StubExtractor)
    ns = parse_args(["--encoder", "bge-m3", "--no-download", "--device", "cpu",
                     "--pool", "pool.pt"])
    ex = build_extractor(_resolved(["a.pt"], Path("pool.pt"), "bge-m3"), ns)
    assert ex.canaries == 1
    assert ex.args == {"ckpts": ["a.pt"], "vocab": "v.json", "device": "cpu",
                       "encoder": "bge-m3", "download": False,
                       "pool_path": Path("pool.pt"), "threshold": None,
                       "adj_threshold": None}


def test_build_extractor_forwards_the_resolved_pool_not_the_flag(monkeypatch):
    """`--pool` may be unset: the pooling weights the resolver downloaded are the ones
    the encoder must load."""
    import cogito_estella.integrations.llamaindex_connector as lc
    monkeypatch.setattr(lc, "CogitoGraphExtractor", _StubExtractor)
    ex = build_extractor(_resolved(["a.pt"], Path("hub/pool.safetensors")), parse_args([]))
    assert ex.args["pool_path"] == Path("hub/pool.safetensors")
    assert ex.args["encoder"] is None               # legacy checkpoints keep deciding


def test_build_extractor_applies_the_published_operating_point(monkeypatch):
    """Thresholds the publisher validated the weights under, not the module table."""
    import cogito_estella.integrations.llamaindex_connector as lc
    monkeypatch.setattr(lc, "CogitoGraphExtractor", _StubExtractor)
    ex = build_extractor(_resolved(["a.pt"], point=(0.2, 0.6)), parse_args([]))
    assert (ex.args["threshold"], ex.args["adj_threshold"]) == (0.2, 0.6)


def test_build_extractor_names_the_encoder_the_checkpoints_resolved(monkeypatch, capsys):
    """With explicit legacy paths the startup line stays silent about the space; the
    extractor announces the one it actually built."""
    import cogito_estella.integrations.llamaindex_connector as lc
    monkeypatch.setattr(lc, "CogitoGraphExtractor", _StubExtractor)
    build_extractor(_resolved(["a.pt"], encoder=None), parse_args([]))
    err = capsys.readouterr().err
    assert "encoder=sonar" in err and "encoder=m2m100-pool" not in err


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
        build_extractor(_resolved(["a.pt"]), parse_args(["--encoder", "bge-m3"]))
    assert "trained on sonar" in str(exc.value) and str(exc.value).startswith("cogito-mcp:")


def test_ask_use_dense_and_the_older_toggle_select_the_same_scorer(tools, monkeypatch):
    seen = []
    monkeypatch.setattr(tools.store, "ask", lambda q, budget=600, scorer="lexical": seen.append(scorer) or "")
    tools.ask("q", use_dense=True)
    tools.ask("q", use_sonar=True)
    assert seen == ["dense", "dense"]
def test_main_refuses_an_unreadable_scorer_knob_before_it_loads_anything(monkeypatch, tmp_path):
    import cogito_estella.mcp.server as srv

    def never(*a, **k):
        raise AssertionError("the knob must be read before any loading")
    monkeypatch.setenv(SCORER_ENV, "lexcial")
    monkeypatch.setattr(srv, "resolve", never)
    with pytest.raises(SystemExit) as exc:
        srv.main(["--dir", str(tmp_path)])
    assert SCORER_ENV in str(exc.value) and str(exc.value).startswith("cogito-mcp: ")


# -- compact profile: the surface for models with little context ------------------------

FILLER = " ".join(f"Gardens need water on day {i} of the season." for i in range(120))
BIG = FILLER + " The encoder maps text to a vector."


@pytest.fixture
def compact(fake_extractor, tmp_path):
    return Tools(GraphStore(extractor=fake_extractor(TRIPLES), path=tmp_path / "graph.json"),
                 profile="compact")


def test_an_unknown_profile_is_refused(fake_extractor):
    with pytest.raises(ValueError, match="profile"):
        Tools(GraphStore(extractor=fake_extractor(TRIPLES)), profile="tiny")


def test_compact_ask_returns_the_whole_corpus_when_it_fits_the_budget(compact):
    compact.ingest(DOC)
    out = compact.ask("What is decoded by SONAR?")
    assert out.splitlines() == ["All of the indexed text fits here (1 document):",
                                "[text1] The generated concepts are decoded by SONAR.",
                                "[text1] The encoder maps text to a vector."]


def test_compact_ask_still_says_nothing_matches_an_unrelated_question(compact):
    compact.ingest(DOC)
    assert compact.ask("quantum chromodynamics") == \
        "Nothing in the indexed documents matches this question."


def test_compact_ask_ranks_sentences_without_graph_jargon(compact):
    compact.ingest(BIG)
    out = compact.ask("What does the encoder map text to?")
    lines = out.splitlines()
    assert lines[0].startswith(("From the indexed documents", "Low confidence"))
    assert lines[1] == "[text1] The encoder maps text to a vector."
    assert all(line.startswith("[text1] ") for line in lines[1:])
    assert "entities:" not in out and "\n--\n" not in out and "#" not in out
    from cogito_estella.mcp.tokens import ntok
    assert ntok(out) <= 1200 and compact.store.ledger.calls["ask"] == 1


@pytest.mark.parametrize("conf, budgets", [(0.1, [600, 1200]), (0.9, [600]), (None, [600])])
def test_compact_ask_widens_once_in_the_same_call_when_the_scorer_doubts(compact, monkeypatch,
                                                                         conf, budgets):
    seen = []
    monkeypatch.setattr(compact.store, "_ask",
                        lambda q, budget, scorer, style: seen.append(budget) or ("ok", conf))
    assert compact.ask("q") == "ok"
    assert seen == budgets


def test_compact_ingest_is_one_summary_line_per_call(compact, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text(DOC)
    (tmp_path / "docs" / "b.md").write_text("The decoder maps a vector to text.")
    first = compact.ingest(str(tmp_path / "docs"))
    assert first.startswith("indexed 2 documents (2 ingested), ") and "\n" not in first
    assert compact.ingest(str(tmp_path / "docs")).startswith("indexed 2 documents (2 unchanged)")


def test_compact_ingest_caps_its_error_lines():
    from cogito_estella.mcp.server import _ingest_summary
    out = _ingest_summary([f"source=f{i} error=unreadable" for i in range(9)])
    assert out.splitlines() == ["nothing indexed", *[f"source=f{i} error=unreadable"
                                                     for i in range(5)], "+4 more errors"]


def _tool_list(server):
    return {t.name: t for t in server._tool_manager.list_tools()}


def test_the_compact_server_offers_ask_alone_when_the_corpus_was_indexed_at_startup(compact):
    server = build_server(compact.store, compact, ["README.md", "docs/ops.md"],
                          indexed_at_startup=True)
    tools = _tool_list(server)
    assert set(tools) == {"ask"}
    assert set(tools["ask"].parameters["properties"]) == {"question"}
    assert "(README.md, docs/ops.md)" in tools["ask"].description


def test_the_compact_server_keeps_ingest_when_nothing_was_indexed(compact):
    tools = _tool_list(build_server(compact.store, compact))
    assert set(tools) == {"ask", "ingest"}
    assert "indexed with `ingest`" in tools["ask"].description


def test_the_compact_surface_costs_a_fraction_of_the_full_one(tools, compact):
    build_server(tools.store)
    full = tools.store.ledger.schema_tokens
    build_server(compact.store, compact, ["a.md"], indexed_at_startup=True)
    assert 0 < compact.store.ledger.schema_tokens * 5 < full
    assert f"tool_definitions_tokens={full}" in tools.stats()


def test_corpus_label_names_the_first_sources_and_counts_the_rest():
    from cogito_estella.mcp.server import corpus_label
    assert corpus_label([]) == ""
    assert corpus_label([f"d{i}.md" for i in range(8)]) == \
        "d0.md, d1.md, d2.md, d3.md, d4.md, d5.md and 2 more"


def test_the_operator_budget_is_the_default_of_the_full_ask(fake_extractor):
    store = GraphStore(extractor=fake_extractor(TRIPLES))
    ask = _tool_list(build_server(store, Tools(store, budget=1200)))["ask"]
    assert ask.parameters["properties"]["budget"]["default"] == 1200


def test_parse_args_reads_the_small_model_flags():
    ns = parse_args([])
    assert ns.profile == "full" and ns.ingest is None and ns.budget == 600
    ns = parse_args(["--profile", "compact", "--ingest", "docs", "--ingest", "README.md",
                     "--budget", "400"])
    assert ns.profile == "compact" and ns.ingest == [Path("docs"), Path("README.md")]
    assert ns.budget == 400
    with pytest.raises(SystemExit):
        parse_args(["--profile", "tiny"])


# -- startup indexing: the model never has to call `ingest` -----------------------------

def test_startup_indexing_runs_in_the_background_and_ask_answers_after_it(compact, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text(DOC)
    compact.index_in_background([tmp_path / "docs"]).join(timeout=30)
    assert compact.ask("What is decoded by SONAR?").startswith("All of the indexed text")


def test_reading_tools_wait_for_the_startup_indexing(compact, monkeypatch, tmp_path):
    import threading

    import cogito_estella.mcp.server as srv
    gate = threading.Event()
    real = compact.store.ingest_path
    monkeypatch.setattr(compact.store, "ingest_path", lambda p: gate.wait(10) and real(p))
    monkeypatch.setattr(srv, "INDEX_WAIT", 0.05)
    (tmp_path / "a.txt").write_text(DOC)
    th = compact.index_in_background([tmp_path / "a.txt"])
    assert compact.ask("What is decoded by SONAR?").startswith("still indexing")
    assert compact.search("SONAR").startswith("still indexing")
    gate.set()
    th.join(timeout=30)
    assert "decoded by SONAR" in compact.ask("What is decoded by SONAR?")


def test_a_failed_startup_indexing_is_reported_instead_of_an_empty_answer(compact, monkeypatch):
    def boom(_p):
        raise RuntimeError("weights missing")
    monkeypatch.setattr(compact.store, "ingest_path", boom)
    compact.index_in_background(["docs"]).join(timeout=30)
    assert compact.ask("anything") == \
        "the documents could not be indexed (RuntimeError: weights missing)"


def test_source_names_match_what_ingestion_will_call_the_documents(tmp_path):
    from cogito_estella.mcp.readers import iter_sources, source_names
    (tmp_path / "d" / "sub").mkdir(parents=True)
    (tmp_path / "d" / "a.md").write_text("x")
    (tmp_path / "d" / "sub" / "b.txt").write_text("y")
    (tmp_path / "d" / "skip.py").write_text("z")
    assert source_names(tmp_path / "d") == [n for n, _ in iter_sources(tmp_path / "d")] \
        == ["a.md", "sub/b.txt"]
    assert source_names(tmp_path / "d" / "a.md") == [str(tmp_path / "d" / "a.md")]
    assert source_names(tmp_path / "missing") == []


# -- push: the client sends the context, the model never has to decide ---------------------

@pytest.mark.parametrize("profile", ["full", "compact"])
def test_the_context_prompt_puts_the_compact_reply_before_the_question(fake_extractor, profile):
    import asyncio
    t = Tools(GraphStore(extractor=fake_extractor(TRIPLES)), profile=profile)
    t.ingest(DOC)
    server = build_server(t.store, t)
    assert [p.name for p in server._prompt_manager.list_prompts()] == ["context"]
    got = asyncio.run(server.get_prompt("context", {"question": "What is decoded by SONAR?"}))
    text = got.messages[0].content.text
    assert got.messages[0].role == "user"
    assert text.startswith("Context from the indexed documents for this question:\n"
                           "All of the indexed text fits here (1 document):\n")
    assert text.endswith("\n\nQuestion: What is decoded by SONAR?")
    assert t.store.ledger.calls["context"] == 1


def test_the_context_prompt_is_not_part_of_the_per_turn_definitions(compact):
    build_server(compact.store, compact, ["a.md"], indexed_at_startup=True)
    with_prompt = compact.store.ledger.schema_tokens
    from cogito_estella.mcp.server import schema_tokens
    assert schema_tokens(build_server(compact.store, compact, ["a.md"],
                                      indexed_at_startup=True)) == with_prompt
    assert with_prompt < 120


def test_a_compact_line_is_one_tagged_passage_even_for_a_multiline_sentence(compact):
    compact.ingest("| table | row |\n| spans | lines |\n\nThe encoder maps text to a vector.")
    out = compact.ask("What does the encoder map text to?")
    assert all(line.startswith("[text1] ") for line in out.splitlines()[1:])


# -- small sessions: section paths and the lite index ---------------------------------------

MD = """# Project

## [0.9.0] - 2026-09-01

### Added
- networkx added to the core dependencies.

```
# not a heading
```

## [0.8.0]
The encoder maps text to a vector.
"""


def test_section_paths_follow_the_markdown_headings():
    from cogito_estella.mcp.store import section_paths, shown_section
    offs = [0, MD.index("networkx"), MD.index("# not"), MD.index("The encoder")]
    paths = section_paths(MD, offs)
    assert paths == ["", "0.9.0 - 2026-09-01 › Added", "0.9.0 - 2026-09-01 › Added", "0.8.0"]
    assert shown_section(paths[1]) == "0.9.0 - 2026-09-01 › Added"


def test_several_level_one_headings_are_sections_not_a_title():
    from cogito_estella.mcp.store import section_paths
    text = "# A\nfirst.\n# B\nsecond.\n"
    assert section_paths(text, [text.index("first"), text.index("second")]) == ["A", "B"]


def test_the_shown_depth_is_a_rendering_choice(monkeypatch):
    import cogito_estella.mcp.store as st
    monkeypatch.setattr(st, "SECTION_DEPTH", 1)
    assert st.shown_section("0.9.0 › Added") == "Added"
    monkeypatch.setattr(st, "SECTION_DEPTH", 0)
    assert st.shown_section("0.9.0 › Added") == ""


def test_compact_lines_name_the_section_a_passage_sits_under(compact):
    compact.ingest(MD + FILLER)
    out = compact.ask("Which version added networkx to the core dependencies?")
    # spaCy keeps the heading lines and the bullet in one sentence that starts inside
    # `## [0.9.0]`: the path is that heading, and the version reaches the reader either way
    line = next(x for x in out.splitlines() if "networkx" in x)
    assert line.startswith("[text1 § 0.9.0 - 2026-09-01] ")


def test_the_whole_corpus_names_a_section_once_per_run_of_lines(compact):
    compact.ingest("## Setup\nInstall the encoder first. Then run the encoder.\n")
    lines = compact.ask("How do I run the encoder?").splitlines()
    assert lines[1].startswith("[text1 § Setup] ") and lines[2].startswith("[text1] Then")


def test_a_document_indexed_before_sections_gets_them_without_extraction(compact):
    compact.ingest(MD)
    doc = compact.store.docs["text1"]
    del doc["sections"]
    compact.store._ex = None                      # any extraction would now fail
    compact.store._factory = None
    assert compact.store.ingest_text(MD, "text1").status == "unchanged"
    assert "0.8.0" in compact.store.docs["text1"]["sections"]


def test_the_lite_index_ranks_without_any_model(nlp, capsys):
    from cogito_estella.mcp.store import TextOnlyExtractor
    t = Tools(GraphStore(extractor_factory=TextOnlyExtractor), profile="compact")
    res = t.store.ingest_text(BIG, "doc")
    assert res.triples == 0 and t.store.docs["doc"]["lite"] is True and not t.store.emb
    out = t.ask("What does the encoder map text to?")
    assert "[doc] The encoder maps text to a vector." in out.splitlines()[1]
    assert "encoding failed" not in capsys.readouterr().err


def test_a_full_server_extracts_again_what_the_lite_index_left_without_facts(fake_extractor, nlp):
    from cogito_estella.mcp.store import TextOnlyExtractor
    store = GraphStore(extractor_factory=TextOnlyExtractor)
    store.ingest_text(DOC, "doc")
    store._ex = fake_extractor(TRIPLES)
    res = store.ingest_text(DOC, "doc")
    assert res.status == "replaced" and res.triples == 2 and "lite" not in store.docs["doc"]
    assert store.ingest_text(DOC, "doc").status == "unchanged"


def test_lite_mode_never_resolves_or_downloads_weights(monkeypatch, tmp_path, capsys):
    import cogito_estella.mcp.server as srv

    def no_weights(*_a, **_k):
        raise AssertionError("--lite must not resolve weights")

    def boom(store, *_a, **_k):
        raise ImportError("No module named 'mcp'", name="mcp")
    monkeypatch.setattr(srv, "resolve", no_weights)
    monkeypatch.setattr(srv, "ensure_spacy_model", lambda **k: None)
    monkeypatch.setattr(srv, "build_server", boom)
    with pytest.raises(SystemExit):
        srv.main(["--dir", str(tmp_path), "--lite"])
    assert "encoder=none (lite)" in capsys.readouterr().err
    assert parse_args(["--lite"]).lite and not parse_args([]).lite


def test_deferred_indexing_starts_at_the_first_tool_call(compact, tmp_path):
    (tmp_path / "a.txt").write_text(DOC)
    compact.defer_indexing([tmp_path / "a.txt"])
    assert compact.store.docs == {}
    assert "decoded by SONAR" in compact.ask("What is decoded by SONAR?")


def test_a_cold_lite_server_indexes_at_startup_and_answers_over_stdio(tmp_path, nlp):
    """The handshake must reach stdout while the startup indexing loads spaCy: indexing
    that began before the transport claimed stdout sent every reply to stderr."""
    import asyncio

    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.md").write_text("## Setup\n" + DOC + "\n")
    args = ["-m", "cogito_estella.mcp.server", "--dir", str(tmp_path / "g"), "--lite",
            "--profile", "compact", "--ingest", str(tmp_path / "docs"), "--no-download"]

    async def run():
        params = StdioServerParameters(command=sys.executable, args=args)
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await asyncio.wait_for(s.initialize(), 60)
            names = [t.name for t in (await s.list_tools()).tools]
            res = await s.call_tool("ask", {"question": "What is decoded by SONAR?"},
                                    read_timeout_seconds=120)
            return names, "\n".join(c.text for c in res.content if hasattr(c, "text"))
    names, out = asyncio.run(asyncio.wait_for(run(), 180))
    assert names == ["ask"]
    assert "[notes.md § Setup] The generated concepts are decoded by SONAR." in out


# -- quick start: one path is the whole setup ----------------------------------------------

def test_a_path_alone_selects_the_small_session_setup():
    ns = parse_args(["docs"])
    assert ns.ingest == [Path("docs")] and ns.lite and ns.profile == "compact"
    ns = parse_args(["docs", "--full"])
    assert not ns.lite and ns.profile == "compact"
    ns = parse_args(["docs", "--profile", "full"])
    assert ns.lite and ns.profile == "full"
    ns = parse_args([])
    assert not ns.lite and ns.profile == "full" and ns.ingest is None


def test_a_missing_model_package_names_the_full_extra(monkeypatch, tmp_path):
    import importlib.util

    import cogito_estella.mcp.server as srv
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        srv.main(["--dir", str(tmp_path)])
    assert "cogito-estella[full]" in str(exc.value) and "--lite" in str(exc.value)


def test_the_server_module_does_not_import_torch():
    import subprocess
    code = ("import sys, cogito_estella.mcp.server, cogito_estella.mcp.store; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=False)
    assert out.stdout.strip() == "False"


def test_estella_answers_in_three_lines(tmp_path, nlp):
    from cogito_estella import Estella
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.md").write_text("## Setup\n" + DOC + "\n")
    docs = Estella(tmp_path / "docs")
    assert "[notes.md § Setup] The generated concepts are decoded by SONAR." in \
        docs.ask("What is decoded by SONAR?")
    msg = docs.context("What is decoded by SONAR?")
    assert msg.startswith("Context from the indexed documents") and \
        msg.endswith("Question: What is decoded by SONAR?")
    docs.add("The decoder maps a vector to text.", name="extra")
    assert "extra" in docs.store.docs


def test_estella_keeps_its_index_when_asked(tmp_path, nlp):
    from cogito_estella import Estella
    (tmp_path / "a.txt").write_text(DOC)
    Estella(tmp_path / "a.txt", save_to=tmp_path / "idx")
    again = Estella(save_to=tmp_path / "idx")
    assert str(tmp_path / "a.txt") in again.store.docs


def test_estella_takes_long_raw_text_as_text(nlp):
    from cogito_estella import Estella
    docs = Estella()
    docs.add("A sentence about the encoder. " * 200, name="long")
    assert "long" in docs.store.docs
