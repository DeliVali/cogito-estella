# Cogito Estella (v0.19.2)

Your agent reads a few hundred tokens instead of whole documents. Point Estella at your
files; for every question it hands the model only the passages that answer it, each
tagged with its file and section. On 8 Wikipedia articles and 48 questions per model,
small local models answered as well as with the whole article pasted in the prompt
while reading 3 to 5 times fewer tokens and making fewer wrong answers
([numbers below](#small-sessions-and-small-models)). No LLM tokens are spent on indexing.

## Quick start

```bash
pip install cogito-estella        # Python 3.10-3.13, no GPU, no model download
```

**Set up your agents in one command** (Claude Code, Claude Desktop, Cursor, VS Code,
Windsurf, Gemini CLI, Kiro, Antigravity, Cline, Roo Code, Zed, Copilot CLI, Qwen Code, Amazon Q, Codex, opencode): it finds the ones on this machine, shows the
change, asks before writing and backs up every file it touches.

```bash
cogito-estella setup docs/          # or, without installing: uvx cogito-estella setup docs/
```

**From Node or TypeScript**: `npx -y cogito-estella setup docs/`, or in an MCP config
`{"command": "npx", "args": ["-y", "cogito-estella", "docs/"]}`. The npm package is a
launcher for the Python one: it needs `uv` (or `pip install cogito-estella`) on the machine.

**Over HTTP** (a harness that cannot spawn a process, any language): `cogito-mcp docs/ --http 8765`
serves MCP at `http://127.0.0.1:8765/mcp` and plain `POST /ask` / `POST /context` with
`{"question": "..."}` returning text. No authentication: keep it on localhost or behind your proxy.

**Any other MCP client**: `cogito-estella setup --print` prints the entry, or add it by
hand; the folder is indexed when the server starts.

```json
{"mcpServers": {"docs": {"command": "cogito-mcp", "args": ["docs/"]}}}
```

Or without installing: `"command": "uvx", "args": ["cogito-estella", "docs/"]`.

**Any Python harness**, no MCP needed:

```python
from cogito_estella import Estella

docs = Estella("docs/")                 # indexed in about a second
prompt = docs.context("When did the embargo end?")   # the question plus its passages
docs.ask("Who signed it?", deep=True)   # facts + encoder ranking; the model loads once, then stays
```

**Any other harness**, in any language that can run a command:
`cogito-estella context docs/ "When did the embargo end?"` prints the question with its
passages (`ask` prints the passages alone); the index is kept in `./.cogito`.

**Languages.** Passages are found in any language: on XQuAD (the same questions in 11
languages) the answer reaches the reply 96.7 % of the time in English, 90–96 % in Spanish,
German, Turkish, Vietnamese, Russian and Chinese, 82–87 % in Greek, Arabic and Hindi, and
77 % in Thai. The graph model of the full server is English-only.

**Which way?** If the model calls tools well, give it the MCP server: it calls `ask` when
it needs the documents. If it does not (many small models never call a tool), send
`docs.context(question)` as the user message, or use the server's `context` prompt: the
answer reaches the model either way.

```
[1973_oil_crisis.txt] By the end of the embargo in March 1974, the price of oil had risen ...
```

## Long sessions: the graph model

For long sessions over many documents, the full server adds a knowledge graph extracted
by a local model, the graph tools below, and a slightly better ranking:

```bash
pip install "cogito-estella[full]"      # add [pdf] for PDF input
```

```json
{"mcpServers": {"cogito": {"command": "cogito-mcp", "args": ["--dir", ".cogito"]}}}
```

On first run the server downloads the spaCy model and the default weights from
[Hugging Face](https://huggingface.co/DeliVali/cogito-estella): Meta's frozen M2M-100 418M
encoder (MIT, ~1.9 GB) under a learned pooling, and three decoder heads (Apache-2.0).
A GPU is optional (`--device cpu`). The graph lives in `.cogito/graph.json`; add that
directory to `.gitignore`.

## Tools

- `ingest(path_or_text)` — a file, a directory (txt/md/html/pdf) or raw text. Documents
  are deduplicated by content hash and replaced when they change.
- `ask(question, budget=600, use_dense=False)` — start here. The graph facts for the
  entities in the question, a `--` line, then the source sentences that answer it, within
  `budget` tokens (`use_sonar` is the older name of the flag). Sentences are shortlisted by IDF and ordered by a learned relevance
  scorer; `use_dense=True` ranks them by encoder cosine instead.
- `query(entity, hops, limit)`, `provenance(edge_ids)`, `search(term)`, `entities(prefix)`
  and `stats()` go deeper when one `ask` is not enough. Facts listed under
  `~ class-only, verify with provenance:` carry only a coarse relation label.

```
entities: blt, layer · scorer=learned
blt encode byte #12
--
2412.09871.txt s112: "We use SwiGLU activation in the feed-forward layers, as in Llama 3."
```

Flags: `--encoder sonar` for the SONAR-space heads (research only: `pip install
"cogito-estella[sonar]"`, CC-BY-NC 4.0 at runtime); `--checkpoint`, `--vocab` and `--pool`
for local weights; `--device`; `--no-download`. `COGITO_ASK_SCORER=lexical` keeps the
plain IDF order.

## Small sessions and small models

The full server pays off over long sessions: the encoder and the graph load once and
every later question is cheap. A short session over a few documents pays that start-up
in full, and a small model may not use the tools at all. Three switches make a short
session cheap:

- **`--lite`** (the default when the server is given a path) indexes sentences only, split
  by spaCy's rule-based sentencizer. No encoder, graph head or spaCy model is loaded or
  downloaded, and torch is not needed;
  `ask` keeps its learned ranking, one of whose features is the absence of an encoder
  cosine. Graph tools have no facts for documents indexed this way, and a full server
  re-extracts them the next time it ingests them. Cold session on CPU, index plus first
  answer, answer present in the compact reply:

  | corpus | full | `--lite` |
  | :--- | :--- | :--- |
  | README (2.1k tokens), 10 questions | 29.3 s · 2.6 GB · 8/10 | 3.8 s · 0.6 GB · 9/10 |
  | README + CHANGELOG (9.4k), 22 questions | 67.6 s · 2.8 GB · 20/22 | 4.3 s · 0.7 GB · 21/22 |

  End to end over the README alone (8 questions with an answer, 2 without, a cold
  `--lite` server, native list/read/grep in every branch): neither Qwen2.5 1.5B nor 3B
  read the short file by itself (0 of 8 with native tools only); the 1.5B answered 7 of
  8 with no wrong answer when the client pushed the `context` prompt (946 prompt tokens
  per question), the 3B 2 of 8 calling `ask` under `--profile compact` and 3 of 8 under
  push.

  On an independent set the full server keeps a small edge: 8 SQuAD v1.1 Wikipedia
  articles (4.4k tokens on average, each a fresh session) and their 1,140 human-written
  questions put the exact answer in the reply 96.2 % of the time with `--lite` and 96.8 %
  with the full server (7 questions only the full server finds, none the other way;
  McNemar p = 0.016), at 533 and 561 tokens per reply, against 4.4k for the article.
  Indexing all eight took 31 s and 715 MB with `--lite`, 479 s and 2.9 GB in full. Use
  `--lite` for short sessions and keep the full server for long ones.

  End to end on the same articles (48 sampled questions per branch, a fresh `--lite`
  index per article, Qwen2.5-Instruct Q4_K_M on CPU), against pasting the whole article:

  | model | branch | right | wrong | prompt tokens / question |
  | :--- | :--- | :--- | :--- | :--- |
  | 1.5B | whole article in the prompt | 39 | 9 | 4,646 |
  | 1.5B | `context` push | 35 | 8 | 894 |
  | 1.5B | `ask` tool (never called) | 4 | 12 | 506 |
  | 3B | whole article in the prompt | 35 | 11 | 4,646 |
  | 3B | `context` push | 32 | 5 | 1,004 |
  | 3B | `ask` tool (called in 41 of 48) | 34 | 5 | 1,494 |

  Paired against the whole article (McNemar), push is not significantly different for
  either model (p = 0.39 and 0.65) nor is the `ask` tool for the 3B (p = 1), at 3 to 5×
  fewer tokens; the tool is significantly worse for the 1.5B, which never calls it. The
  integrator picks the route: `ask` for a model that calls tools, `context` for one that
  does not.
- **`--profile compact`** and the **`context` prompt**, below, cut the per-turn cost and
  reach models that do not call tools.
- **Sections.** Markdown documents keep the heading path of every sentence, and compact
  lines name it (`[CHANGELOG.md § 0.9.0 - 2026-09-01] …`): a sentence ranked on its own
  no longer loses the version or section that gives it meaning (one more answer of 22
  reaches the reply above, at the same token cost). Documents indexed before this keep
  working and get their paths when re-ingested, without a new extraction.

```json
{"mcpServers": {"cogito": {"command": "cogito-mcp",
  "args": ["--dir", ".cogito", "--lite", "--profile", "compact", "--ingest", "docs"]}}}
```

A model can only use `ask` if it decides to call it, and small models often do not: in
the bench below a 1.5B model called no tool at all, under any prompt, and a 3B model
facing the full seven-tool surface never called `ask` and made up 12 answers out of 24.
Two ways to get the documents to such a model:

- **Pull, `--profile compact`.** `ask(question)` is the only tool (plus `ingest` when
  nothing was indexed at startup); its description names the indexed files, and its
  reply is one tagged passage per line under one plain-language line, which warns when
  the scorer doubts its best passage. The server widens the budget once by itself when
  it doubts, and sends the whole corpus when it fits the budget. The tool definitions
  cost 96 tokens per model turn instead of 961.
- **Push, the `context` prompt.** An MCP prompt, invoked by the client or the user,
  never by the model: it returns the question preceded by the compact `ask` reply, to be
  sent as the user message. It works with a model that calls no tools, and costs nothing
  in the per-turn definitions.

`--ingest PATH` (repeatable) indexes documents at startup, in the background, so the
model never has to call `ingest`; reading tools wait for it rather than answer from a
half-built index. `--budget` sets the token budget of one reply.

Measured on CPU with Qwen2.5-Instruct (Q4_K_M, temperature 0): 22 questions whose exact
answer is in this repository's README and CHANGELOG (9.4k tokens, the cost of reading
both before any question) plus 2 with no answer; `list_files`, `read_file` and `grep`
are available in every branch:

| branch | 1.5B right / wrong | 3B right / wrong | 3B calls `ask` | prompt tokens / question (3B) |
| :--- | :--- | :--- | :--- | :--- |
| native tools only | 0 / 0 | 0 / 0 | — | 502 |
| + full profile | 1 / 4 | 0 / 12 | 0 of 24 | 1,484 |
| + compact profile | 0 / 2 | 11 / 1 | 17 of 24 | 1,681 |
| native + `context` push | 13 / 5 | 6 / 4 | — | 1,287 |

One run per branch: a difference of three answers is within run-to-run noise (the 3B
push branch moved from 9 to 6 between two runs that differed only in line formatting).
The 1.5B never calls a tool, so its compact row measures nothing new and push is its
only route. The 3B under push declines passages that do hold the answer; the 1.5B misses
5 of the 19 answers that reach it, and 3 answers never reach it (a version number that
lives in a section heading, a table cell, a short README line).

## Measured

50 questions over five arXiv papers, answered by Claude Code headless (Haiku) under the
same rules in every branch, judged blind to the branch, with the Claude Code prefix
calibrated the same day:

| branch | accuracy | raw input tokens / question | net tokens / question |
| :--- | :--- | :--- | :--- |
| whole paper in the prompt | 1.00 | 38,231 | 12,810 |
| native `Read`/`Grep` on the file | 0.98 | 89,493 | 4,753 |
| one `ask` reply (600-token budget) | 0.78 | 26,126 | 613 |

Raw tokens are what the API bills before caching, measured end to end; the ~25k-token
Claude Code prefix dominates every branch. Net tokens subtract that calibrated prefix and
are an estimate. One `ask` reply answers 78 % of the questions (abstaining on 16 %) at
1.46× fewer raw tokens and about 21× fewer net tokens than pasting the paper; the
`Read`/`Grep` agent is accurate but costs 2.3× the raw tokens of pasting the paper.

Extraction quality: the default three-head ensemble reaches Triple F1 **0.852** on
held-out sentences whose entity/relation combinations never appeared in training; on
5,000 sentences held out under both protocols it scores 0.819 against 0.730 for the
SONAR-space ensemble.

A single `ask` is the cheapest route but abstains on some questions. The header's `p=`
says how sure the scorer is, so a caller can escalate before answering; both escalation
policies were measured on the same 50 questions, reusing the rows above:

| policy | accuracy | context tokens / question | per correct answer |
| :--- | :--- | :--- | :--- |
| one `ask` at 600 | 0.78 | 588 | 754 (15.0×) |
| `ask` → read the document when it abstains | 0.94 | 2,353 | 2,503 (4.5×) |
| `ask` → `ask` at 1200 when `p` < 0.3 → document | 0.94 | 1,380 | 1,468 (7.7×) |
| `ask` → document when `p` < 0.3 or it abstains | **0.96** | 2,486 | 2,590 (4.4×) |

A second `ask` spends no LLM tokens, so escalating the budget is far cheaper than reading
the document: it leaves 5 documents read out of 50 instead of 8. Sending low-confidence
questions straight to the document instead is the accurate end of the trade: it answers
0.96 and leaves two answers wrong-without-warning rather than three.

## Python

```python
from cogito_estella.integrations.llamaindex_connector import CogitoGraphExtractor
from cogito_estella.mcp.weights import resolve

w = resolve(None, None)                                  # the default assets, downloaded once
ex = CogitoGraphExtractor([str(c) for c in w.checkpoints], str(w.vocab), pool_path=w.pool,
                          threshold=w.operating_point[0], adj_threshold=w.operating_point[1])
ex.extract("The committee approved the new budget.")    # [(subject, relation, object), ...]
```

`extract_with_literals` keeps exact literals (IDs, hashes, amounts) verbatim and
`literals_to_neo4j` stores them with provenance; `to_neo4j` writes triples into Neo4j. Call
`ensure_schema(driver)` once per database before concurrent writers: its uniqueness
constraints make parallel `MERGE`s deterministic, and they cannot be created while the
database still holds duplicate nodes. `cogito_estella.graph_summary.GraphSummarizer`
writes community summaries over the triples and flags (`accepted=False`) the ones that
mention entities absent from their cluster.

## How it works

Each head is a compact non-autoregressive decoder (38M parameters) on a frozen sentence
encoder: the candidate entities found in the sentence act as queries over its vector, and
existence plus a 76-relation adjacency come out in one forward pass (about 0.01 ms per
sentence on an RTX 5070). Relations are named from the sentence's dependency syntax when a
pattern is recognized; otherwise the fact keeps the class label only. Every edge points
back to its source sentence and character spans.

## Weights and licensing

| Encoder | Assets on the Hub | Licence |
| :--- | :--- | :--- |
| `m2m100-pool` (default) | `m2m100-pool/cogito-prose-ontology-m2mpool{,-s2,-s3}.safetensors`, `pool.safetensors`, `vocab-onto-m2mpool.json`; manifest `encoders.json` | heads and pooling Apache-2.0; M2M-100 encoder MIT |
| `sonar` | `cogito-prose-ontology{,-s2,-s3}.pt`, `vocab-onto.json` | heads Apache-2.0; the SONAR encoder they need is CC-BY-NC 4.0 |

Every checkpoint carries its encoder, revision, width and normalization; a checkpoint
decoded in the wrong space stops the server, and a startup canary re-checks the encoder
against shipped reference cosines.

Three standalone research heads (not part of the default `ask`/`ingest` path) are also
ported to the permissive encoder, each measured above its SONAR original: tool-call
extraction (Triple F1 1.000), entity-conditioned prose on the 60-verb vocabulary
(0.878 vs 0.827, 5-model ensemble, `CogitoGraphExtractor` loads it unchanged), and code
call/import extraction (0.979 vs 0.777) — the last via a LoRA-adapted M2M-100 encoder,
so it too carries no non-commercial term. Details and a loader in the Hub model card.

## Development

```bash
uv sync --all-extras
uv run python -m spacy download en_core_web_sm
uv run pytest tests/
```

637 tests with every extra installed; modules that need the `[sonar]` extra and the tests
that load a real encoder on CUDA skip themselves without them. Package layout:
`model/` (decoder heads), `encoders/` (encoder contract, adapters, pooling, canary),
`mcp/` (store, readers, scorers, server), `integrations/` (the extractor). Training data,
experiments and logs are untracked; see `CHANGELOG.md` for the version history.

## License

Apache License 2.0 (see `LICENSE`). Weight licensing is described above.
