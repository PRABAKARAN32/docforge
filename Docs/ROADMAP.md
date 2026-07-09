# Roadmap — what's built, milestone by milestone

DocForge is built in **vertical slices**: each milestone is complete, working, tested software
before the next begins — not five half-finished layers built in parallel. This page is the
record of what each milestone actually delivered. For *why* each major decision was made, see
[`GUID.md`](../GUID.md); for how to *use* what's here, see [`CLI.md`](CLI.md) and
[`MCP_SERVER.md`](MCP_SERVER.md).

| Milestone | Status |
|---|---|
| M0 — Scaffolding | ✅ Done |
| M1 — Change detection | ✅ Done |
| M2 — RAG sync | ✅ Done |
| M3 — CLI | ✅ Done |
| M4 — MCP server | ✅ Done |
| M5 — Docker packaging | ⏳ Next |

---

## M0 — Scaffolding

The project skeleton: `src/` layout, `pyproject.toml`, `LICENSE` (Apache-2.0), CI (lint + test
on every push), and the initial `README`. Open-source hygiene from the first commit, not
retrofitted later.

## M1 — Change detection

The core insight, built first and independently useful even with no RAG attached: crawl a
site, normalize + hash each page's markdown, store the hashes in a SQLite manifest, and diff
this run's hashes against the last run's to produce `{new, changed, deleted, unchanged}`.

- **Discovery** (`discovery.py`): sitemap-first (fast, complete, polite), with an opt-in
  breadth-first crawl fallback (`--bfs`) for sites with no sitemap.
- **Hashing** (`hashing.py`): SHA-256 of *normalized* markdown — cosmetic noise ("Last updated:
  ...", whitespace) is stripped before hashing, so only meaningful content changes register.
- **Manifest** (`manifest.py`): a single SQLite file remembering every page's hash between runs.
- **Diff** (`diff.py`): pure set comparison, plus a critical safety guard — deletions are never
  applied unless the crawl completed successfully, so a partial crawl failure can never look
  like "these pages were deleted."

## M2 — RAG sync

Wires the diff from M1 into an actual vector store, touching only what changed:

- **Chunking** (`chunking.py`): paragraph-packing with overlap, so a page's content is split
  into embeddable pieces without losing context across chunk boundaries.
- **Embedding** (`embedder.py`): local embeddings via [fastembed](https://github.com/qdrant/fastembed)
  (ONNX, no PyTorch), with automatic GPU use when available and a safe fallback to CPU.
- **Vector store** (`vectorstore.py`): [Qdrant](https://github.com/qdrant/qdrant), running three
  ways — Docker container, embedded on-disk (no Docker at all), or a remote/managed cluster.
  Every stored chunk carries its source page's URL, so "this page changed" becomes a single
  targeted delete instead of a full-collection scan.
- **Sync orchestration** (`rag.py`): delete stale chunks for changed/deleted pages, chunk +
  embed only new/changed pages, upsert. Unchanged pages: untouched.

Performance work landed within this milestone as real usage surfaced real bottlenecks:
- **Parallel, rate-limited crawling** — concurrent crawling with a per-domain politeness delay,
  replacing sequential one-page-at-a-time crawling.
- **HTTP conditional requests (ETag/304)** — a cheap pre-check that skips the full browser
  render for pages a server reports as unchanged, guarded so a first-ever sync (nothing stored
  yet) never wastes time on pointless conditional requests.
- **GPU-accelerated embedding** — measured to be the dominant cost of a sync once crawling was
  parallelized; `--device auto|cpu|cuda`.
- **Live progress bars with a self-correcting ETA** (`progress.py`) — for both the crawl and
  embed phases, so a multi-thousand-page sync doesn't look like it's hung.

## M3 — CLI

The `docforge` command (`cli.py`): `sync`, `diff`, `status`/`list`, `search`, `remove` — each a
thin, independently-testable wrapper around the M1/M2 core, wired together with dependency
injection so the whole CLI is unit-tested with no network, no browser, no real vector store.

- **Multiple named knowledge bases** — sync any number of docs sites, each isolated in its own
  manifest scope and Qdrant collection (`--name`, default derived from the site's host).
- **`.env` configuration** (`config.py`, shared with M4) — set Qdrant/DB/embed-model settings
  once instead of repeating flags on every command; precedence is flag > `.env`/env > default.
- **Fail-fast** — the vector store's reachability is confirmed *before* an expensive crawl
  starts, not after.

## M4 — MCP server

`docforge-mcp` (`mcp_server.py`): exposes the same retrieval capability as MCP tools any
MCP-capable LLM client can call — `list_docs()` and `search_docs(query, name=None)`. No new
retrieval logic; a thin wrapper around the exact same search path `docforge search` uses.
Deliberately does **not** expose crawling/ingestion as a tool (that stays a CLI-only operation —
see `GUID.md` Decision 5.16 for the full reasoning).

- **Three transports**: `stdio` (the client launches the process itself — Claude Code, Claude
  Desktop), `http` (binds a port for clients that connect by URL — LM Studio, a custom agent),
  and `both` (runs simultaneously in one process).
- **Secure by default** — the HTTP transport requires a bearer token; if none is configured, a
  fresh one is generated and printed on every start (the same pattern Jupyter Notebook uses),
  rather than defaulting to open access. `--no-auth` is the explicit, deliberate opt-out.

## M5 — Docker packaging (next)

Not yet started. The goal: package DocForge (plus a bundled Qdrant) as a single container, so
running it requires no manual Python/uv/Docker-Compose setup at all.
