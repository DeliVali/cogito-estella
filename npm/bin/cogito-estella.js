#!/usr/bin/env node
// Runs the Python package: uvx (no install), else a cogito-estella already on PATH.
const { spawn, spawnSync } = require("node:child_process");

// a command exists unless spawning it fails with ENOENT (its own exit code is irrelevant)
const has = (cmd) => spawnSync(cmd, ["--version"], { stdio: "ignore" }).error?.code !== "ENOENT";
const [cmd, args] = has("uvx") ? ["uvx", ["cogito-estella", ...process.argv.slice(2)]]
  : has("cogito-estella") ? ["cogito-estella", process.argv.slice(2)] : [null, null];

if (!cmd) {
  console.error("cogito-estella needs Python. Install uv (https://docs.astral.sh/uv/) " +
    "or run: pip install cogito-estella");
  process.exit(127);
}
const child = spawn(cmd, args, { stdio: "inherit" });       // stdio is the MCP wire
child.on("exit", (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
for (const s of ["SIGINT", "SIGTERM"]) process.on(s, () => child.kill(s));
