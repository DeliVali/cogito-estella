"""`cogito-estella setup`: config entries written for each known client, safely."""
import json
import sys
from pathlib import Path

import pytest

from cogito_estella import setup as st


@pytest.fixture
def env(tmp_path, monkeypatch):
    home, project = tmp_path / "home", tmp_path / "proj"
    (project / "docs").mkdir(parents=True)
    (project / "docs" / "notes.md").write_text("## Setup\nThe encoder maps text to a vector.\n")
    home.mkdir()
    monkeypatch.setattr(st.shutil, "which", lambda name: None)    # no client on PATH
    return home, project


def run(env, *argv, answer="y"):
    home, project = env
    lines = []
    code = st.run(list(argv), home=home, project=project, reply=lambda _q: answer,
                  out=lines.append)
    return code, "\n".join(lines)


def test_detected_clients_get_an_entry_and_existing_servers_are_kept(env):
    home, project = env
    (home / ".cursor").mkdir()
    (project / ".cursor").mkdir()
    (project / ".cursor" / "mcp.json").write_text('{"mcpServers": {"other": {"command": "x"}}}')
    code, out = run(env)
    assert code == 0 and "Cursor" in out and "Claude Code" not in out
    data = json.loads((project / ".cursor" / "mcp.json").read_text())
    assert data["mcpServers"]["other"] == {"command": "x"}
    entry = data["mcpServers"]["cogito-estella"]
    assert entry["args"][0] == str((project / "docs").resolve())
    assert entry["args"][1:] == ["--dir", str((project / ".cogito").resolve())]
    assert list((project / ".cursor").glob("mcp.json.bak-*"))


@pytest.mark.parametrize("key, rel, check", [
    ("claude-code", ".mcp.json", lambda d: "command" in d["mcpServers"]["cogito-estella"]),
    ("vscode", ".vscode/mcp.json", lambda d: d["servers"]["cogito-estella"]["type"] == "stdio"),
    ("gemini", ".gemini/settings.json", lambda d: "args" in d["mcpServers"]["cogito-estella"]),
    ("opencode", "opencode.json",
     lambda d: d["mcp"]["cogito-estella"]["type"] == "local"
     and isinstance(d["mcp"]["cogito-estella"]["command"], list)),
])
def test_each_project_client_gets_its_own_layout(env, key, rel, check):
    code, _ = run(env, "--clients", key, "--yes")
    assert code == 0 and check(json.loads((env[1] / rel).read_text()))


def test_user_level_clients_write_under_home(env):
    home, _ = env
    run(env, "--clients", "windsurf,claude-desktop", "--yes")
    assert "cogito-estella" in json.loads(
        (home / ".codeium/windsurf/mcp_config.json").read_text())["mcpServers"]
    assert "cogito-estella" in json.loads(st._desktop_config(home).read_text())["mcpServers"]


def test_codex_toml_is_appended_then_replaced_not_duplicated(env):
    home, _ = env
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text('model = "x"\n\n[mcp_servers.other]\ncommand = "y"\n')
    run(env, "--clients", "codex", "--yes")
    run(env, "--clients", "codex", "--yes")
    text = cfg.read_text()
    assert text.count("[mcp_servers.cogito-estella]") == 1
    assert 'model = "x"' in text and "[mcp_servers.other]" in text
    tomllib = pytest.importorskip("tomllib")
    parsed = tomllib.loads(text)
    assert parsed["mcp_servers"]["cogito-estella"]["args"][-2] == "--dir"


def test_a_file_that_is_not_plain_json_is_left_alone(env):
    _, project = env
    cfg = project / ".mcp.json"
    cfg.write_text("{ // a comment\n}")
    code, out = run(env, "--clients", "claude-code", "--yes")
    assert code == 1 and "not plain JSON" in out and cfg.read_text() == "{ // a comment\n}"


def test_nothing_is_written_without_approval(env):
    code, out = run(env, "--clients", "claude-code", answer="n")
    assert code == 1 and "Nothing written" in out and not (env[1] / ".mcp.json").exists()


def test_rerunning_updates_the_same_entry(env):
    run(env, "--clients", "claude-code", "--yes")
    run(env, "--clients", "claude-code", "--yes")
    assert list(json.loads((env[1] / ".mcp.json").read_text())["mcpServers"]) == \
        ["cogito-estella"]


def test_any_other_client_gets_a_printed_entry(env):
    code, out = run(env, "--print")
    assert code == 0 and '"cogito-estella"' in out and "Estella(" in out
    assert "cogito-estella context" in out


def test_no_known_client_prints_the_entry_instead_of_failing(env):
    code, out = run(env)
    assert code == 0 and "No known agent" in out and '"mcpServers"' in out


def test_an_unknown_client_name_is_refused(env):
    code, out = run(env, "--clients", "notepad")
    assert code == 2 and "unknown client" in out


def test_a_missing_docs_folder_is_refused(env):
    code, out = run(env, str(env[1] / "nope"))
    assert code == 2 and "not found" in out


def test_ask_and_context_print_for_any_harness(tmp_path, monkeypatch, capsys):
    import cogito_estella.mcp.server as srv
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("## Setup\nThe encoder maps text to a vector.\n")
    with pytest.raises(SystemExit) as exc:
        srv.main(["context", "notes.md", "What does the encoder map?"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "[notes.md § Setup] The encoder maps text to a vector." in out
    assert out.rstrip().endswith("Question: What does the encoder map?")
    assert (tmp_path / ".cogito" / "graph.json").exists()


def test_the_written_entry_starts_a_server_that_answers(env):
    """The command setup writes is launched as a client would launch it."""
    import asyncio

    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    run(env, "--clients", "claude-code", "--yes")
    entry = json.loads((env[1] / ".mcp.json").read_text())["mcpServers"]["cogito-estella"]
    command = entry["command"]
    if not Path(command).exists():                       # a test venv may lack the script
        command, entry["args"] = sys.executable, ["-m", "cogito_estella.mcp.server",
                                                  *entry["args"]]

    async def go():
        p = StdioServerParameters(command=command, args=entry["args"])
        async with stdio_client(p) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool("ask", {"question": "What does the encoder map?"})
            return res.content[0].text
    out = asyncio.run(asyncio.wait_for(go(), 120))
    assert "[notes.md § Setup] The encoder maps text to a vector." in out


def test_codex_keeps_windows_paths_valid_when_rewriting(tmp_path):
    tomllib = pytest.importorskip("tomllib")
    client = next(c for c in st.CLIENTS if c.key == "codex")
    path = tmp_path / "config.toml"
    args = [r"C:\Users\me\docs", "--dir", r"C:\Users\me\.cogito"]
    path.write_text(st.planned_text(client, path, r"C:\py\cogito-mcp.exe", args))
    text = st.planned_text(client, path, r"C:\py\cogito-mcp.exe", args)   # replace path
    assert tomllib.loads(text)["mcp_servers"]["cogito-estella"]["args"] == args
