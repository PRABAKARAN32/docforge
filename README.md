# DocForge

[![CI](https://github.com/PRABAKARAN32/docforge/actions/workflows/ci.yml/badge.svg)](https://github.com/PRABAKARAN32/docforge/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**Keeps a documentation RAG knowledge base fresh by detecting exactly which pages changed —
instead of rebuilding everything.**

Documentation sites change constantly. Most RAG pipelines handle updates the wasteful way:
re-crawl the whole site and re-embed every page, even when only a handful changed. For a large
docs site (1,000–10,000+ pages) that means long refresh windows, wasted embedding cost, and
stale answers in between.

DocForge takes the other path: on every run it **detects exactly which pages changed**
(hashing normalized markdown, comparing against the last run) and **touches only those** —
deleting stale chunks and re-embedding just the new content. A re-sync of an unchanged site
does nothing at all.

📚 **Full documentation:** [`Docs/`](Docs/) (architecture, complete CLI/MCP reference,
configuration, roadmap) · [`GUID.md`](GUID.md) (the design decision log — the *why*).

---

## Contents

- [Why it exists](#why-it-exists)
- [Features](#features)
- [How it works](#how-it-works)
- [Install](#install)
- [Quickstart](#quickstart)
- [CLI usage](#cli-usage)
- [MCP server: connect an LLM to your docs](#mcp-server-connect-an-llm-to-your-docs)
- [Configuration](#configuration)
- [Development](#development)
- [Project status](#project-status)
- [License](#license)
- [Full documentation (`Docs/`)](Docs/)

## Why it exists

DocForge is the local, always-fresh knowledge base behind **Legendary Dev Tool**, a local-first
AI coding assistant. A local LLM is only as good as the context it has — if it doesn't know the
*current* documentation for the libraries you're using, it hallucinates outdated or wrong APIs.
DocForge keeps that context current, entirely on your own machine: no code or docs sent to a
third party, no per-request cost.

It's equally usable standalone: point it at any documentation site and get a searchable,
always-current knowledge base — from the terminal, or as MCP tools an LLM client calls directly.

## Features

- **Change detection that actually skips work** — normalized-markdown hashing + an HTTP
  conditional-request (ETag/304) pre-check mean an unchanged page is never re-crawled *or*
  re-embedded, not just re-embedded.
- **Multiple named knowledge bases** — sync as many docs sites as you want, each isolated in
  its own manifest scope and vector-store collection (`--name`, defaults to the site's host).
- **Local-first, three ways to run the vector store** — Docker container, embedded on-disk
  (no Docker, no server, like SQLite), or point at a remote/managed Qdrant Cloud cluster.
- **Local embeddings** — [fastembed](https://github.com/qdrant/fastembed) (ONNX, no PyTorch),
  with automatic GPU use when available (`--device auto|cpu|cuda`).
- **Parallel, rate-limited crawling** — concurrent with a per-domain politeness delay,
  configurable or disabled entirely (`--concurrency`, `--crawl-delay`, `--no-rate-limit`).
- **Live progress with a self-correcting ETA** — no more wondering if a multi-thousand-page
  sync has hung.
- **An MCP server** (`docforge-mcp`) — exposes your knowledge bases as tools (`list_docs`,
  `search_docs`) any MCP-capable LLM client can call: Claude Code, Claude Desktop, LM Studio,
  or your own agent. stdio and HTTP transports, HTTP secured with an auto-generated bearer
  token by default.
- **`.env` configuration** — set your Qdrant/DB/model settings once instead of repeating flags.

## How it works

```
crawl (Crawl4AI) → normalize markdown → hash per page → diff vs. last run
    → { new, changed, deleted, unchanged }
    → delete stale chunks → chunk + embed only new/changed content → upsert → update stored hash
```

Two front doors onto that same core, adding no logic of their own:

```
                     ┌─────────────┐        ┌──────────────────┐
  a terminal   ───▶  │  docforge   │  ───▶  │   change          │  ───▶  vector store
  (a human)          │  (CLI)      │        │   detection +     │        (Qdrant)
                      └─────────────┘        │   RAG sync core   │
  an LLM chat  ───▶  ┌─────────────┐  ───▶  │                   │  ───▶  SQLite manifest
  (MCP client)       │ docforge-mcp│        └──────────────────┘        (page hashes)
                      │  (MCP tools)│
                      └─────────────┘
```

Full module-by-module breakdown: [`Docs/ARCHITECTURE.md`](Docs/ARCHITECTURE.md).

## Install

Requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/PRABAKARAN32/docforge.git
cd docforge
uv sync                       # creates .venv, installs everything from uv.lock
uv run playwright install chromium   # one-time: the headless browser Crawl4AI needs
```

> **Running the commands:** `uv sync` installs the `docforge` and `docforge-mcp` commands into
> the project's `.venv`. Run them with `uv run docforge …` / `uv run docforge-mcp …` (works from
> the project directory, nothing to activate), or activate the venv once
> (`source .venv/bin/activate`) and drop the `uv run` prefix. The copy-paste Quickstart below
> uses the `uv run` form; the reference listings further down show the bare `docforge` for
> brevity — they're the same command either way.

## Quickstart

**1. Start a vector store** — pick one:

```bash
docker compose up -d          # A) Qdrant in Docker on :6333 (persists to a named volume)
```

Or skip Docker entirely (option B, embedded on disk — like SQLite, but for vectors): there's
nothing to start, just add `--qdrant-path ./vectors` to the `sync`/`search` commands below.

**2. Build a knowledge base:**

```bash
uv run docforge sync https://docs.python.org/3/
```

**3. Search it:**

```bash
uv run docforge search "how does async work"
```

**4. Run it again later** — only pages that actually changed get re-crawled and re-embedded:

```bash
uv run docforge sync https://docs.python.org/3/   # unchanged pages: skipped entirely
```

## CLI usage

```bash
docforge sync <url> [options]     # crawl, detect changes, embed into the vector store
docforge diff <url>               # preview what would change — write nothing
docforge status                   # list knowledge bases and their page counts
docforge search "<query>"         # search one KB (--name) or all of them (default)
docforge remove <name>            # drop a knowledge base (or --all to wipe everything)
```

Common flags on `sync`/`diff`:

```bash
docforge sync <url> --name docker                 # explicit knowledge-base name
docforge sync <url> --bfs                         # no sitemap? crawl page-by-page instead
docforge sync <url> --dry-run                     # preview changes, write nothing
docforge sync <url> --max-pages 100                # cap pages processed
docforge sync <url> --concurrency 10               # crawl 10 pages in parallel
docforge sync <url> --crawl-delay 0.2 0.5          # tune the per-domain politeness delay
docforge sync <url> --device cuda                  # use a GPU for embedding
docforge sync <url> --force                        # ignore ETag/304, re-crawl every page
docforge sync <url> --qdrant-path ./vectors         # embedded Qdrant, no Docker
docforge sync <url> --qdrant-url <cluster-url>      # remote/Qdrant Cloud
```

Run `docforge --help` or `docforge <command> --help` for the full, grouped flag reference, or
see [`Docs/CLI.md`](Docs/CLI.md) for every command and flag in one page.

### Multiple docs sites

Each site lives in its own **named knowledge base** — isolated manifest scope and vector-store
collection, independently searchable:

```bash
docforge sync https://docs.docker.com/   --name docker
docforge sync https://nginx.org/en/docs/ --name nginx --bfs

docforge search "<query>" --name docker    # search just one
docforge search "<query>"                  # search all of them, merged by relevance
```

## MCP server: connect an LLM to your docs

`docforge-mcp` exposes the knowledge bases you built with `docforge sync` as tools an LLM can
call mid-conversation — `list_docs()` and `search_docs(query, name=None)`. It adds no new
retrieval logic; it's a thin front door onto the exact same core the CLI uses.

**For a client that launches the server itself** (Claude Code, Claude Desktop) — stdio, the
default, no network involved at all:

```bash
claude mcp add docforge -- uv run --directory /path/to/docforge docforge-mcp
```

**For a client that connects by URL** (LM Studio, a custom agent) — HTTP, secured by default
with an auto-generated token (fresh every run, printed for you to copy — the same idea as
Jupyter Notebook's own auto-token):

```bash
uv run docforge-mcp --transport http --port 8000
```
```
DocForge MCP server (http) listening at http://127.0.0.1:8000/mcp
Auth token (generated fresh this run): 8c8JWYJyugjd2MaLgSTHOhkU_0B-HOl8B7IoithwNRA
Authorization header: Bearer 8c8JWYJyugjd2MaLgSTHOhkU_0B-HOl8B7IoithwNRA
```
Paste the URL + `Authorization` header into your client's MCP config. For a token that stays
stable across restarts instead of regenerating each time, set `DOCFORGE_MCP_TOKEN` in `.env`.

Run `docforge-mcp --help` for every flag (`--transport stdio|http|both`, `--host`, `--port`,
`--token`, `--no-auth`), or see [`Docs/MCP_SERVER.md`](Docs/MCP_SERVER.md) for the full guide —
client config examples (Claude Code, a generic `mcp.json`), `both`-transport setup, and
troubleshooting.

## Configuration

Copy `.env.example` to `.env` to set these once instead of repeating flags. Precedence for
every setting: **CLI flag > `.env`/environment variable > built-in default.**

| Variable | Meaning | Default |
|---|---|---|
| `DOCFORGE_DB` | manifest database path | `docforge.db` |
| `QDRANT_URL` | Qdrant server (Docker/native/remote) | `http://localhost:6333` |
| `QDRANT_PATH` | embedded Qdrant folder (no Docker) — wins over `QDRANT_URL` | unset |
| `QDRANT_API_KEY` | API key for a remote/managed Qdrant | unset |
| `DOCFORGE_EMBED_MODEL` | fastembed model — must match what you synced with | `BAAI/bge-small-en-v1.5` |
| `DOCFORGE_MCP_TRANSPORT` | `docforge-mcp` default transport | `stdio` |
| `DOCFORGE_MCP_TOKEN` | stable HTTP auth token (else one is generated per run) | unset |

Full reference, including the three vector-store connection modes: [`Docs/CONFIGURATION.md`](Docs/CONFIGURATION.md).

## Development

```bash
uv run ruff check .                          # lint
uv run pytest                                # fast tests (no network/browser required)
DOCFORGE_NETWORK_TESTS=1 uv run pytest       # include live-crawl integration tests
```

CI runs lint + the fast test suite on every push and PR (`.github/workflows/ci.yml`).

Contributions are welcome — open an issue to discuss a change before sending a large PR.

## Project status

Change detection, RAG sync, the CLI, and the MCP server are all built and tested — see
[`Docs/ROADMAP.md`](Docs/ROADMAP.md) for what shipped in each milestone, and
[`GUID.md`](GUID.md) for the design reasoning behind each decision. Packaging the whole tool as
a Docker image is next.

## License

Licensed under the [Apache License 2.0](LICENSE).

Built on [Crawl4AI](https://github.com/unclecode/crawl4ai) (crawling, HTML→Markdown),
[Qdrant](https://github.com/qdrant/qdrant) (vector store), [fastembed](https://github.com/qdrant/fastembed)
(local embeddings), and the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).
