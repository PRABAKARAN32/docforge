# Architecture

How DocForge is put together, and why. For the reasoning behind each major decision, see
[`GUID.md`](../GUID.md) — this page is the map; that one is the "why."

## The pipeline

```
                 discovery.py                    conditional.py
                 (sitemap / BFS)                  (304 pre-check)
                      │                                  │
                      ▼                                  ▼
   seed URL ──▶ [page URLs] ──▶ detector.py ──▶ crawler.py ──▶ [crawled pages]
                                    │ (orchestrates)                │
                                    │                                │ hashing.py
                                    ▼                                ▼
                               manifest.py ◀── diff.py ◀── {url: content_hash}
                             (SQLite: (name,url)         (new / changed / deleted /
                              → hash, → validators)       unchanged)
                                    │
                                    ▼
                       change-detection result (report + page markdown)
                                    │
                                    ▼
                                 rag.py
                     (delete stale chunks, chunk + embed new ones)
                               │        │
                    chunking.py         embedder.py
                  (markdown → chunk)   (chunk text → vector)
                               │        │
                               ▼        ▼
                          vectorstore.py (Qdrant: upsert / search / delete)
```

**Two front doors** call the same core above and add no logic of their own:

```
docforge (CLI)        → a human at a terminal → sync / diff / status / search / remove
docforge-mcp (server) → an LLM in a chat client → list_docs / search_docs (MCP tools)
```

Both read the same `.env` configuration and open the same vector store the same way, via a
shared `config.py`.

## What each module is responsible for

| Module | Responsibility |
|---|---|
| `hashing.py` | Normalize a page's markdown and hash it into a stable fingerprint. |
| `manifest.py` | SQLite: remember every page's hash between runs, scoped per knowledge base. |
| `diff.py` | Compare two hash-maps → `{new, changed, deleted, unchanged}`, with the deletion-safety guard. |
| `crawler.py` | Turn a list of URLs into clean markdown (thin wrapper around Crawl4AI). |
| `conditional.py` | A cheap "did this page change?" HTTP check, before spending a full browser render. |
| `discovery.py` | Turn one seed URL into the full list of a site's page URLs. |
| `detector.py` | Orchestrates the above into one change-detection run (crawl → hash → diff). |
| `chunking.py` | Split a page's markdown into overlapping, embeddable pieces. |
| `embedder.py` | Turn chunk text into vectors (local, via fastembed; GPU-aware). |
| `vectorstore.py` | Persist / search / delete chunk vectors (Qdrant). |
| `rag.py` | Apply a diff to the vector store — delete stale, embed new, touch nothing else. |
| `progress.py` | Live terminal progress bars with a self-correcting ETA (CLI only). |
| `config.py` | `.env`/environment resolution and vector-store opening, shared by both front doors. |
| `cli.py` | The `docforge` command. |
| `mcp_server.py` | The `docforge-mcp` server. |

## Design principles that show up throughout

- **Protocols, not base classes.** `Embedder` and `VectorStore` are Python `Protocol` types —
  the rest of the code depends on a shape (`.embed(...)`, `.search(...)`), never a concrete
  implementation. Swapping the embedding model or the vector database touches one file.
- **Dependency injection everywhere network/browser/model access happens.** Every function that
  touches the outside world takes that access as a parameter with a real production default
  (`crawl: Crawler = crawl_urls`). Tests override it with a fake; nothing else changes. This is
  why the test suite runs in seconds with no Docker, no browser, and — for most of it — no model
  download.
- **Truth vs. mutation, kept separate.** Detecting what changed (`detect_changes`,
  `diff_hashes`) never writes anything; applying it (`apply_changes`) is a separate, explicit
  step. This is what makes `--dry-run` and `docforge diff` trivial and correct by construction.
- **Guarded deletions.** A partial/failed crawl must never look like "these pages were
  deleted." One function (`deletions_to_apply`) encodes that rule, called independently by
  every subsystem that could otherwise get it wrong (the manifest, the vector store).
- **Both front doors are thin.** Every real operation is a plain, testable function
  (`run_sync`, `search_docs_text`, ...) with a real default; `main()`/`build_server()` do
  nothing but wire arguments to those functions. Neither `cli.py` nor `mcp_server.py` contains
  retrieval logic of its own.

## Storage

- **SQLite** (the "manifest") — one local file, one row per `(knowledge_base, url)`, storing
  the page's content hash and (if the server supports it) its HTTP validators. Zero setup: it's
  a library, not a service.
- **Qdrant** (the vector store) — one collection per knowledge base. Runs three ways: a Docker
  container, embedded on-disk with no server at all (`--qdrant-path`), or a remote/managed
  cluster (`--qdrant-url`).

See [`CONFIGURATION.md`](CONFIGURATION.md) for every setting, [`CLI.md`](CLI.md) for the full
command reference, and [`MCP_SERVER.md`](MCP_SERVER.md) for connecting an LLM client.
