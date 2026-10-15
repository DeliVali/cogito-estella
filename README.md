# Cogito Estella (v0.19.4)

[![CI](https://github.com/DeliVali/cogito-estella/actions/workflows/ci.yml/badge.svg)](https://github.com/DeliVali/cogito-estella/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cogito-estella)](https://pypi.org/project/cogito-estella/)
[![npm](https://img.shields.io/npm/v/cogito-estella)](https://www.npmjs.com/package/cogito-estella)
[![Python](https://img.shields.io/pypi/pyversions/cogito-estella)](https://pypi.org/project/cogito-estella/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/weights-Hugging%20Face-yellow)](https://huggingface.co/DeliVali/cogito-estella)

![cost](https://img.shields.io/badge/cost%20vs%20Read%2FGrep-3.7%C3%97%20lower%20%28Opus%2C%20long%20sessions%29-orange)
![accuracy](https://img.shields.io/badge/accuracy-39%2F40%20vs%2024%2F40-brightgreen)
![MCP](https://img.shields.io/badge/MCP-16%20clients-blue)
![lite](https://img.shields.io/badge/default%20index-no%20model%2C%20no%20torch-teal)
![code graph](https://img.shields.io/badge/code%20graph-tree--sitter-purple)
![formats](https://img.shields.io/badge/reads-code%2C%20docx%2C%20pptx%2C%20xlsx%2C%20pdf%2C%20epub-green)

Your agent reads a few hundred tokens instead of whole files. Point Estella at a folder
of documents or code; for every question it returns only the lines that answer it, each
tagged with its file and section. Indexing spends no LLM tokens.

In long Claude Code sessions over a mixed project (code, PDF, Word, Excel, PowerPoint),
Opus answered 39 of 40 questions with Estella and 24 with its own `Read`/`Grep`, for
$4.97 instead of $18.34 ([measured](#measured)).

## Quick start

```bash
pip install cogito-estella          # Python 3.10-3.13, no GPU, no model download
cogito-estella setup docs/          # registers the server with the agents on this machine
```

`setup` finds the agents installed here (Claude Code, Claude Desktop, Cursor, VS Code,
Windsurf, Gemini CLI, Kiro, Antigravity, Cline, Roo Code, Zed, Copilot CLI, Qwen Code,
Amazon Q, Codex, opencode), shows the change, asks before writing and backs up every
file it touches. Where a client allows it, the read-only tools are pre-approved, so
`ask` runs without a confirmation prompt.

Other ways in:

| From | Command |
| :--- | :--- |
| No install | `uvx cogito-estella setup docs/` |
| Node / TypeScript | `npx -y cogito-estella setup docs/` (runs the Python package; needs `uv`) |
| Any MCP client | `cogito-estella setup --print`, or `{"command": "cogito-mcp", "args": ["docs/"]}` |
| Any language over HTTP | `cogito-mcp docs/ --http 8765`: MCP at `/mcp`, `POST /ask` and `POST /context` with `{"question": "..."}` |
| Any language, one shot | `cogito-estella context docs/ "When did the embargo end?"` |
| Python | see below |

```python
from cogito_estella import Estella

docs = Estella("docs/")                       # indexed in about a second
docs.ask("When did the embargo end?")         # the lines that answer it
docs.context("When did the embargo end?")     # the question with its lines, for any model
```

```
[1973_oil_crisis.txt] By the end of the embargo in March 1974, the price of oil had risen ...
```

The HTTP server has no authentication: keep it on localhost or behind your own proxy.

**Which tools to give the agent?** For reading a project, `ask` alone measured best: with
the agent's own file tools added next to it, accuracy stayed the same and tokens rose
14 to 28 %, and a small local model lost a quarter of its answers to reading whole files.
Keep the file tools for editing. Many small models never call a tool at all; send them
`docs.context(question)` as the user message, or use the server's `context` prompt.

## What it reads

- Text and code in any language: `.py`, `.ts`, `.kt`, `.swift`, `.java`, `.go`, `.dart`,
  `.json`, `.yaml`, `Dockerfile`, `Podfile`... Binary files are recognized by content, not
  by extension.
- Documents: Markdown, HTML, PDF (`[pdf]` extra), Word, PowerPoint, Excel, OpenDocument,
  EPUB, RTF, email (`.eml`) and Jupyter notebooks, with no extra dependency.
- A folder walk skips `.git`, `.env`, `.venv`, `node_modules`, build output and files over
  2 MB, and keeps `.github`, `.gitlab`, `.circleci` and `.devcontainer`.

Prose is split into sentences and keeps its Markdown heading path
(`[CHANGELOG.md § 0.9.0] …`). Code is split into lines, and a line is found by its words,
the parts of its identifiers (`timeoutSeconds` by "timeout") and its file path. In a
reply, code lines are grouped under their file, in file order:

```
[httpx/_config.py]
  DEFAULT_TIMEOUT_CONFIG = Timeout(timeout=5.0)
  DEFAULT_LIMITS = Limits(max_connections=100, max_keepalive_connections=20)
```

## Lite and full

The lite index is the default and needs no model download. The full server adds a local
model that turns documents into a knowledge graph.

| | Lite (default) | Full (`pip install "cogito-estella[full]"`) |
| :--- | :--- | :--- |
| Finding the answer for `ask` | IDF shortlist, then a small learned re-ranker | the same, plus an encoder similarity for prose |
| Knowledge graph | none | prose: facts extracted by the model; code: definitions, calls, imports, inheritance by syntax |
| Start-up | about a second | loads a 1.9 GB encoder (GPU optional) |
| Tools | `ask`, `context` | also `query`, `provenance`, `search`, `entities`, Neo4j export |

Both modes find answers the same way: a lexical shortlist ordered by a 12-feature learned
scorer. The large model's job is the graph, not the search; as a ranking signal it adds
under a point on prose and hurts on code, so code is never ranked with it. The agent
reading the reply does the understanding and rephrases when a reply misses.

Start full with `cogito-mcp docs/ --full`, or keep a light session and load the graph model
once, when needed:

```python
docs.ask("Who signed it?", deep=True)    # loads the model once; later calls reuse it
```

The code graph comes from tree-sitter and covers about 40 languages (checked on Python,
Kotlin, Swift, TypeScript, Java and Go):

```
w.py defines Wallet          Wallet inherits Base         Wallet contains pay
pay calls post               w.py imports os
```

Each fact points to the line it came from, keeps the case of names (`Pay` the class,
`pay` the method) and goes to Neo4j like any other fact.

On first run the full server downloads its weights from
[Hugging Face](https://huggingface.co/DeliVali/cogito-estella): Meta's M2M-100 418M encoder
(MIT) under a learned pooling, and three decoder heads (Apache-2.0). The graph lives in
`.cogito/graph.json`; add that directory to `.gitignore`.

## Tools

- `ask(question)`: start here. The lines that answer the question within a token budget
  (300 by default), best first. When the scorer doubts its best line the reply says so
  and widens itself once; a corpus that fits the budget is returned whole. For questions
  that chain two facts it searches a second time with the names its best lines bring in.
- `ingest(path_or_text)`: a file, a folder or raw text. Unchanged files are skipped.
- `query(entity)`, `provenance(edge_ids)`, `search(term)`, `entities(prefix)`, `stats()`:
  graph and text lookups for when one `ask` is not enough (full profile).

Every tool declares MCP annotations: the readers are read-only and `ingest` never
deletes, which lets clients such as Codex run them unattended.

Server flags: `--full`, `--profile compact|full`, `--ingest PATH`, `--budget`, `--http`,
`--dir`, `--device`, `--no-download`.

## Measured

**Long sessions.** One continuous Claude Code conversation per session, three sessions of
13 or 14 questions, 40 in all, over a project of 110 code files (httpx, starlette, h11,
anyio) plus a PDF paper, a Word paper, an Excel sheet and a PowerPoint deck. Questions
cover code lookups, flows across files, documents, tables and slides; answers are judged
against a reference. Tokens are all input tokens of the 40 questions, cache included.

| Agent | Own `Read`/`Grep` only | Estella `ask` only (default budget, 300) |
| :--- | :--- | :--- |
| Opus | 24/40 · 3.24M tokens · $18.34 | **39/40 · 0.99M tokens · $4.97** |
| Haiku | 24/40 · 2.22M tokens · $0.40 | **36/40 · 0.91M tokens · $0.15** |
| Qwen3.5 4B, local | 1/18 (stopped) · 364k tokens per question | **29/40** · 29k per question (budget 600) |

Without Estella, Claude Code could not read the Excel sheet or the slides and read the long
documents only in part; code questions were answered either way. At a 600-token budget
Opus also answered 39/40 for $6.44 and Haiku 37/40 for $0.18; the default is 300 because
it cost 20 to 23 % less for at most one answer.

An agent whose shell opens any format with a script answers these questions on its own;
there `ask` adds its replies to the context and saves no tokens. Estella pays where the
agent cannot read a format, or where reading whole files would fill its context.

**One question per session.** Haiku, `ask` only, 30 questions per set unless noted; the
code set is four libraries Estella was never tuned on.

| Case | Lite | Full |
| :--- | :--- | :--- |
| Research papers (5 papers, 56k tokens) | 29/30 | 29/30 |
| Unseen code (110 files) | 29/30 | 28/30 |
| Code of this repository (40 questions) | 37/40 | 37/40 |
| Wikipedia, two-step questions (HotpotQA, 60 questions) | 50/60 | 51/60 |

With every relevant Wikipedia paragraph pasted into the prompt the same agent answers
53/60: HotpotQA carries some wrong reference answers, so that is the ceiling there. On the
papers Estella read 2.6k input tokens per question against 16.4k for `Read`/`Grep`; on the
unseen code, 3.9k against 8.4k (one identifier finds its line, so `grep` is cheap there).

Small local models: on 8 Wikipedia articles and 48 questions per model (Qwen2.5 1.5B and
3B, CPU), sending `context` instead of the whole article kept accuracy within noise
(McNemar p = 0.39 and 0.65) at 3 to 5 times fewer prompt tokens. The 1.5B never calls a
tool, so `context` is its only route.

Graph extraction on prose: Triple F1 0.852 for the default three-head ensemble, on
held-out sentences whose entity and relation combinations never appeared in training.

## Python and Neo4j

```python
from cogito_estella.integrations.llamaindex_connector import CogitoGraphExtractor
from cogito_estella.mcp.weights import resolve

w = resolve(None, None)                                  # default weights, downloaded once
ex = CogitoGraphExtractor([str(c) for c in w.checkpoints], str(w.vocab), pool_path=w.pool,
                          threshold=w.operating_point[0], adj_threshold=w.operating_point[1])
ex.extract("The committee approved the new budget.")    # [(subject, relation, object), ...]
```

`to_neo4j` writes triples with their provenance, `extract_with_literals` and
`literals_to_neo4j` keep IDs, hashes and amounts verbatim. Call `ensure_schema(driver)`
once per database before concurrent writers.

## How the graph model works

Each prose head is a 38M-parameter non-autoregressive decoder on a frozen sentence
encoder: the candidate entities in a sentence query its vector, and existence plus a
76-relation adjacency come out in one forward pass. Relations are named from the
sentence's dependency syntax when a pattern is recognized; otherwise the fact keeps its
class label and is listed under `~ class-only, verify with provenance:`. Every edge points
back to its source sentence. Each checkpoint records its encoder and revision, and a
start-up canary stops the server when the encoder does not match.

## Development

```bash
uv sync --all-extras
uv run python -m spacy download en_core_web_sm
uv run pytest tests/
```

CI runs the suite on Python 3.10 to 3.13. Package layout: `mcp/` (store, readers, code
graph, scorers, server), `model/` (decoder heads), `encoders/`, `integrations/`.
`CHANGELOG.md` has the version history.

## License

Apache License 2.0 (see `LICENSE`). M2M-100 is MIT.
