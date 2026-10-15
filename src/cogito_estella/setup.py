"""`cogito-estella setup`: register the server with the agents on this machine.

One command for people who do not want to edit a config file. Each known client is one
row of CLIENTS: where its config lives and how a server entry is written there. Detected
clients are proposed, the change is shown, nothing is written without approval, and every
file that changes is backed up first. Any other MCP client gets the same entry from
`--print`; harnesses without MCP use `cogito-estella ask` / `context` or the Python API."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

NAME = "cogito-estella"


@dataclass(frozen=True)
class Client:
    key: str
    label: str
    path: object            # (home, project) -> Path of the config file
    kind: str               # json key layout, or "toml"
    detect: object          # (home, project) -> bool: is this client on the machine?
    style: str = "plain"    # entry shape when it differs from {command, args}


def _desktop_config(home: Path) -> Path:
    if sys.platform == "darwin":
        return home / "Library/Application Support/Claude/claude_desktop_config.json"
    if sys.platform.startswith("win"):
        return Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / \
            "Claude/claude_desktop_config.json"
    return home / ".config/Claude/claude_desktop_config.json"


def _code_user(home: Path) -> Path:
    """VS Code's user directory, where its extensions keep their settings."""
    if sys.platform == "darwin":
        return home / "Library/Application Support/Code/User"
    if sys.platform.startswith("win"):
        return Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / "Code/User"
    return home / ".config/Code/User"


def _extension(home: Path, ext: str) -> bool:
    return (_code_user(home) / "globalStorage" / ext).exists() or \
        any((home / ".vscode/extensions").glob(f"{ext}-*"))


def _antigravity_config(home: Path) -> Path:
    # the shared layout (~/.gemini/config) replaced ~/.gemini/antigravity; keep an old install's file
    old = home / ".gemini/antigravity/mcp_config.json"
    return old if old.exists() and not (home / ".gemini/config").exists() \
        else home / ".gemini/config/mcp_config.json"


CLIENTS = (
    Client("claude-code", "Claude Code", lambda h, p: p / ".mcp.json", "mcpServers",
           lambda h, p: (h / ".claude").exists() or shutil.which("claude") is not None),
    Client("claude-desktop", "Claude Desktop", lambda h, p: _desktop_config(h), "mcpServers",
           lambda h, p: _desktop_config(h).parent.exists()),
    Client("cursor", "Cursor", lambda h, p: p / ".cursor/mcp.json", "mcpServers",
           lambda h, p: (h / ".cursor").exists()),
    Client("vscode", "VS Code", lambda h, p: p / ".vscode/mcp.json", "servers",
           lambda h, p: shutil.which("code") is not None or (p / ".vscode").exists()),
    Client("windsurf", "Windsurf", lambda h, p: h / ".codeium/windsurf/mcp_config.json",
           "mcpServers", lambda h, p: (h / ".codeium/windsurf").exists()),
    Client("gemini", "Gemini CLI", lambda h, p: p / ".gemini/settings.json", "mcpServers",
           lambda h, p: (h / ".gemini").exists()),
    Client("kiro", "Kiro", lambda h, p: p / ".kiro/settings/mcp.json", "mcpServers",
           lambda h, p: (h / ".kiro").exists() or shutil.which("kiro") is not None),
    Client("antigravity", "Antigravity", lambda h, p: _antigravity_config(h), "mcpServers",
           lambda h, p: (h / ".gemini/antigravity").exists() or (h / ".gemini/config").exists()
           or shutil.which("antigravity") is not None),
    Client("cline", "Cline", lambda h, p: _code_user(h) /
           "globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
           "mcpServers", lambda h, p: _extension(h, "saoudrizwan.claude-dev")),
    Client("roo", "Roo Code", lambda h, p: p / ".roo/mcp.json", "mcpServers",
           lambda h, p: _extension(h, "rooveterinaryinc.roo-cline")),
    Client("zed", "Zed", lambda h, p: h / ".config/zed/settings.json", "context_servers",
           lambda h, p: (h / ".config/zed").exists() or shutil.which("zed") is not None),
    Client("copilot-cli", "GitHub Copilot CLI", lambda h, p: h / ".copilot/mcp-config.json",
           "mcpServers", lambda h, p: shutil.which("copilot") is not None, style="copilot"),
    Client("qwen", "Qwen Code", lambda h, p: p / ".qwen/settings.json", "mcpServers",
           lambda h, p: (h / ".qwen").exists() or shutil.which("qwen") is not None),
    Client("amazon-q", "Amazon Q", lambda h, p: h / ".aws/amazonq/mcp.json", "mcpServers",
           lambda h, p: (h / ".aws/amazonq").exists() or shutil.which("q") is not None),
    Client("codex", "Codex", lambda h, p: h / ".codex/config.toml", "toml",
           lambda h, p: (h / ".codex").exists()),
    Client("opencode", "opencode", lambda h, p: p / "opencode.json", "mcp",
           lambda h, p: (h / ".config/opencode").exists() or shutil.which("opencode")),
)


def server_command(docs: list[Path], project: Path) -> tuple[str, list[str]]:
    """The command an agent will launch, in absolute terms: a GUI client does not start
    in this directory and may not share this shell's PATH. Under `uvx` the environment
    is temporary, so the entry launches through `uvx` again."""
    args = [*(str(d.resolve()) for d in docs), "--dir", str((project / ".cogito").resolve())]
    if "uv" in Path(sys.prefix).parts and any("archive" in x for x in Path(sys.prefix).parts):
        return "uvx", [NAME, *args]
    exe = Path(sys.executable).parent / ("cogito-mcp.exe" if os.name == "nt" else "cogito-mcp")
    found = str(exe) if exe.exists() else (shutil.which("cogito-mcp") or "cogito-mcp")
    return found, args


def entry_for(client: Client, command: str, args: list[str]) -> dict:
    if client.kind == "servers":                           # VS Code
        return {"type": "stdio", "command": command, "args": args}
    if client.style == "copilot":                          # Copilot CLI: "local", tool list
        return {"type": "local", "command": command, "args": args, "tools": ["*"]}
    if client.kind == "mcp":                               # opencode
        return {"type": "local", "command": [command, *args], "enabled": True}
    return {"command": command, "args": args}


def _toml_section(command: str, args: list[str]) -> str:
    q = json.dumps                                         # a JSON string is a TOML string
    return (f"[mcp_servers.{NAME}]\ncommand = {q(command)}\n"
            f"args = [{', '.join(q(a) for a in args)}]\n")


def planned_text(client: Client, path: Path, command: str, args: list[str]) -> str:
    """The whole new content of `path`; raises ValueError when the existing file cannot
    be read safely (it is then left alone)."""
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if client.kind == "toml":
        section = _toml_section(command, args)
        pattern = re.compile(rf"^\[mcp_servers\.{re.escape(NAME)}\]\n(?:(?!\[).*\n?)*", re.MULTILINE)
        if pattern.search(old):
            return pattern.sub(section, old)
        return old + ("\n" if old and not old.endswith("\n") else "") + \
            ("\n" if old else "") + section
    try:
        data = json.loads(old) if old.strip() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not plain JSON ({exc.msg}); add the entry by hand") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object; add the entry by hand")  # noqa: TRY004
    servers = data.setdefault(client.kind, {})
    if not isinstance(servers, dict):
        raise ValueError(  # noqa: TRY004 - the caller reports every refusal alike
            f"{path}: '{client.kind}' is not an object; add the entry by hand")
    servers[NAME] = entry_for(client, command, args)
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write(path: Path, text: str) -> Path | None:
    """Write `text`, backing up the previous file; returns the backup path."""
    backup = None
    if path.exists():
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return backup


def generic_snippet(command: str, args: list[str], docs: list[Path]) -> str:
    first = str(docs[0]) if docs else "docs/"
    return "\n".join([
        "Any MCP client (stdio): add this server entry",
        json.dumps({"mcpServers": {NAME: {"command": command, "args": args}}}, indent=2),
        "",
        "Any Python harness:",
        "  from cogito_estella import Estella",
        f"  docs = Estella({first!r})",
        "  prompt = docs.context(question)    # send as the user message",
        "",
        "Any other harness (any language that can run a command):",
        f"  cogito-estella context {first} \"your question\"",
    ])


def run(argv: list[str], home: Path | None = None, project: Path | None = None,
        reply=input, out=print) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog=f"{NAME} setup",
                                 description="Register Cogito Estella with your agents.")
    ap.add_argument("docs", nargs="*", type=Path, help="documents to serve (default: ./docs, "
                                                       "else the current directory)")
    ap.add_argument("--clients", help="comma-separated keys: " +
                                      ", ".join(c.key for c in CLIENTS))
    ap.add_argument("--yes", action="store_true", help="write without asking")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="only print the entry, for any other client or harness")
    ns = ap.parse_args(argv)
    home = home or Path.home()
    project = (project or Path.cwd()).resolve()
    docs = ns.docs or [project / "docs" if (project / "docs").is_dir() else project]
    missing = [d for d in docs if not d.exists()]
    if missing:
        out(f"{NAME}: not found: {', '.join(map(str, missing))}")
        return 2
    command, args = server_command(docs, project)
    if ns.print_only:
        out(generic_snippet(command, args, docs))
        return 0
    if ns.clients:
        keys = {k.strip() for k in ns.clients.split(",") if k.strip()}
        unknown = keys - {c.key for c in CLIENTS}
        if unknown:
            out(f"{NAME}: unknown client(s) {', '.join(sorted(unknown))}; "
                f"known: {', '.join(c.key for c in CLIENTS)}; others: --print")
            return 2
        chosen = [c for c in CLIENTS if c.key in keys]
    else:
        chosen = [c for c in CLIENTS if c.detect(home, project)]
    if not chosen:
        out("No known agent found on this machine. Use this entry wherever you need it:\n")
        out(generic_snippet(command, args, docs))
        return 0
    plans, skipped = [], []
    for c in chosen:
        path = c.path(home, project)
        try:
            plans.append((c, path, planned_text(c, path, command, args)))
        except (ValueError, OSError) as exc:
            skipped.append(f"  {c.label}: {exc}")
    out(f"Cogito Estella will serve: {', '.join(str(d) for d in docs)}")
    out("Changes:")
    for c, path, _ in plans:
        out(f"  {c.label:15s} {path}{' (new)' if not path.exists() else ' (backed up first)'}")
    for line in skipped:
        out(line)
    if not plans:
        out("\nNothing written. Entry to add by hand:\n" + generic_snippet(command, args, docs))
        return 1
    if not ns.yes and reply("Write these changes? [y/N] ").strip().lower() not in ("y", "yes"):
        out("Nothing written.")
        return 1
    for c, path, text in plans:
        backup = write(path, text)
        out(f"  wrote {path}" + (f" (backup: {backup.name})" if backup else ""))
    out("Done. Restart your agent; it now has the `ask` tool over your documents.")
    out(f"Other clients: `{NAME} setup --print`.")
    return 0
