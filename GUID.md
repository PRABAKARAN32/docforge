# DocForge — Project Guide

> **Living document.** This is the project's memory. It records not just _what_ we built, but
> _why_ each decision was made. Read it before making an architectural change; update it when a
> decision changes or a milestone completes. If you ever forget why something is the way it is,
> the answer belongs here.
>
> For **what's implemented today** — commands, flags, configuration — see [`README.md`](README.md)
> and the [`Docs/`](Docs/) folder. This file is the *decision log*, not the user manual.

---

## 0. How to read this document

This guide serves two audiences at once:

1. **A contributor**, needing the reasoning behind the architecture before changing it.
2. **A local LLM assistant**, which can load this file as context to work on DocForge without
   re-explaining the whole project every time.

Each major decision is written as **Decision → Reasoning → Trade-off**, because the reasoning
is the transferable skill. Tools change; judgment does not.

---

## 1. The big picture — where DocForge fits

DocForge is **one component of a larger project called Legendary Dev Tool.**

**Legendary Dev Tool** is an agentic AI coding assistant (think: a VS Code extension) whose
defining idea is that **the LLM runs locally**. Instead of paying per-token to cloud providers
(OpenAI, Anthropic, etc.), the model runs on the developer's own machine. Once the model is
local, the tool can cheaply and privately do things like:

- automatic test-case generation
- local security checks
- code quality and coverage analysis

...all without sending code to a third party or paying per request.

**The catch:** a local LLM is only as good as the context you can feed it. If the model does
not know the _current_ documentation for the libraries a developer uses, it hallucinates
outdated or wrong APIs. So the local model needs a **local, always-fresh knowledge base**
built from documentation.

**That knowledge base is what DocForge builds and maintains.** DocForge is the fuel line for
Legendary Dev Tool — it is not a side quest, it is what keeps the whole engine accurate. It is
also fully usable standalone, independent of Legendary Dev Tool.

> **Scope discipline:** This guide covers **DocForge only.** Legendary Dev Tool is the context,
> not the current work.

---

## 2. The problem DocForge solves

Documentation sites change constantly — new API endpoints, renamed paths, updated examples.
Most RAG (Retrieval-Augmented Generation) systems handle updates the naive way: **re-crawl the
entire site and re-embed everything**, even when only a few pages changed.

For a large docs site (1,000–10,000+ pages) this means:

- Re-downloading and re-processing thousands of unchanged pages.
- Re-paying embedding costs for content that never changed.
- Long refresh windows where the RAG answers from stale information.
- Risk of duplicate or contradictory chunks in the vector store if updates are handled carelessly.

**The core insight:** on any given refresh, only a handful of pages actually changed. If we can
**detect exactly what changed and touch only that**, we get correctness _without_ the full-rebuild
cost every time.

> **One-line summary:** Existing tools can build a documentation RAG _once_. DocForge keeps it
> _correct_ as the documentation changes — detecting exactly what changed and updating only that,
> instead of rebuilding everything.

---

## 3. Prior art — what exists and why it isn't enough

| Existing tool                                  | What it solves                                                                                     | Why it's not enough                                                                                                         |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Crawl4AI**                                   | Crawls sites, converts HTML → clean, LLM-ready Markdown.                                           | No hash-diffing or cross-run update logic.                                                                                  |
| **LlamaIndex `IngestionPipeline`**             | Hash-based dedup + `UPSERTS_AND_DELETE` strategy: updates only changed docs, removes deleted ones. | No web crawler. Expects ready-made `Document` objects. Update logic runs _within_ a run, not _across_ crawl runs over time. |
| **Commercial actors (Apify, Firecrawl, etc.)** | Some track a per-page content hash across runs.                                                    | Leave scheduling + diff logic to the user; paid/closed products.                                                            |
| **Nuclia "Agentic RAG Sync Agent"**            | Closest match — syncs a sitemap into a RAG store on a schedule.                                    | Closed, commercial, undocumented internals.                                                                                 |

**Conclusion:** the individual pieces — crawling, hashing, delete-and-upsert — all exist
_separately_, in different tools. No open-source project wires them into one simple,
purpose-built workflow for keeping a documentation RAG in sync over time.

### Honest positioning (important)

Our differentiator is **not** "nobody has solved this" — Nuclia's Sync Agent does something very
close. Our real, defensible differentiator is:

> **Open-source + local-first + one command.**

That is a thinner moat, and that is completely fine for a tooling/portfolio project. Overclaiming
novelty is a red flag reviewers catch instantly; honest scoping reads as senior. State it honestly:
we are making an integration that today is either missing entirely or locked inside closed
commercial products.

---

## 4. The solution architecture

DocForge is a lightweight open-source bridge connecting Crawl4AI's crawling to an update-aware
RAG pipeline, built specifically for documentation sites. It splits cleanly into two halves,
plus two thin front doors onto that same core.

### Change detection (M1)

```
crawl (Crawl4AI) → clean markdown → normalize → hash per page → compare to last run
    → { new, changed, deleted, unchanged }
```

### RAG sync (M2)

```
diff report → delete stale chunks for changed/deleted pages
    → chunk + embed new content → upsert to vector store → update stored hash
```

### Front doors (M3, M4) — thin, no logic of their own

```
docforge (CLI)        → a human, a terminal    → sync/diff/status/search/remove
docforge-mcp (server) → an LLM, a chat client   → list_docs/search_docs as MCP tools
```

**End result:** run `docforge sync <docs-site>` once to build the knowledge base. Run it again
months later, and only the pages that actually changed get re-processed — an unchanged site
re-syncs to a no-op.

---

## 5. Key engineering decisions (with reasoning)

### Decision 5.1 — Use Crawl4AI as-is; do not reinvent crawling

- **Reasoning:** Crawling well (JS rendering, clean markdown, anti-bot handling) is a solved,
  hard problem. Crawl4AI does it and is battle-tested. Reinventing it adds huge scope for zero
  differentiation.
- **Trade-off:** We take on a dependency, but it is permissively licensed (see §6) and it saves
  months of work.

### Decision 5.2 — Build change detection ourselves; do NOT use a framework for it

The pipeline splits into two halves. Ask "commodity or differentiator?" for each:

- **Half A — change detection** (crawl → hash → diff → new/changed/deleted): **Build it ourselves.**
  It is genuinely simple — hash each page, store hashes in a manifest, diff two sets of hashes.
  This is our core IP _and_ it is trivial. Pulling in a giant framework for a dictionary comparison
  would be absurd. LlamaIndex doesn't even help here; its dedup runs _inside_ a run, not across
  crawl runs over time.

- **Half B — RAG sync** (chunk → embed → upsert → delete stale): **Thin libraries, our own
  orchestration.** LlamaIndex's `IngestionPipeline` _would_ handle the delete/upsert bookkeeping,
  but it is a heavy dependency with a large transitive tree — which fights our design goal of a
  _lightweight, local, dockerized_ tool. So: use thin, well-scoped libraries for the commodity
  parts (a local embedding model, a vector DB with good native `delete-by-metadata` and `upsert`),
  and write the orchestration ourselves.

- **Trade-off:** Slightly more code than importing LlamaIndex wholesale — but we understand every
  line (critical, since this feeds Legendary Dev Tool), and the project stays light. We design the
  embedder and vector store as **pluggable interfaces** (realized in M2 as Python `Protocol`
  classes — see Decision 5.8) so someone can swap in a different implementation later.

### Decision 5.3 — The page/chunk metadata scheme is make-or-break

- **The problem:** our diff is at the **page** level, but the vector store operates at the **chunk**
  level. When page X changes, we must delete _every_ old chunk from page X and insert the new ones.
- **The solution:** every chunk must carry stable metadata pointing back to its page — a
  `source_url`. Then "delete stale chunks for page X" is simply: delete where `source_url == X`.
- **Reasoning:** Get this right up front and sync is easy. Get it wrong and you get orphaned and
  duplicated chunks forever. This is exactly where it pays to think before coding.

### Decision 5.4 — Hash the normalized markdown, not raw HTML

- **Reasoning:** Raw HTML is full of noise (nav menus, ads, timestamps, session tokens) that
  changes every load and would flag every page as "changed." Hash the _cleaned markdown_ from
  Crawl4AI, and **normalize first** — strip "last updated" dates, collapse whitespace — so only
  _meaningful_ changes register.
- **Trade-off:** Requires deliberately deciding what counts as a meaningful change; done once, saves
  endless false positives.

### Decision 5.5 — Deletion must be guarded

- **The danger:** we detect a deleted page by comparing this crawl's URL set against last run's.
  But if a crawl _fails halfway_, good pages look "missing" and we would wrongly delete their chunks.
- **The rule:** **Never apply deletions unless the crawl completed successfully.** Guard it
  explicitly (`deletions_to_apply`), and apply the same guard everywhere a deletion could
  happen — the manifest *and* the vector store both have to agree on it independently.

### Decision 5.6 — Idempotency from day one

- **Rule:** Running `docforge sync` twice in a row with no doc changes must do _nothing_ the second
  time — no re-embedding, no errors, no writes.
- **Reasoning:** Designing for idempotency from the start makes the whole architecture cleaner and
  is the mark of a robust tool. It's also what makes a 300-page unchanged re-sync fast: the
  expensive work (crawl, embed) is skipped, not just the write.

### Decision 5.7 — SQLite as the state/manifest store

- **What state we need:** a manifest remembering `(knowledge_base, url) → content_hash →
  last_seen`, plus HTTP validators for the conditional-request pre-check (Decision 5.13).
- **Reasoning:** SQLite is a single-file, zero-config, transactional store — a perfect fit for a
  local tool. It becomes the "memory" of DocForge.
- **Trade-off:** Not a networked/multi-writer DB, but we don't need one for a local single-process
  tool.

### Decision 5.8 — Pluggable interfaces via Python `Protocol`, not base classes

- **Reasoning:** The embedder and vector store are both defined as `Protocol` classes
  (`Embedder`: `.dimension`, `.embed(texts)`; `VectorStore`: `ensure_collection`,
  `upsert_chunks`, `delete_by_source_url`, `search`, ...) rather than abstract base classes. Any
  object with the right method shapes satisfies the interface automatically — no inheritance
  required. The rest of the codebase depends on these shapes, never on the concrete
  implementation (fastembed, Qdrant) directly.
- **Trade-off:** Slightly less IDE "jump to implementation" convenience than a concrete base
  class; in exchange, tests inject tiny fakes with zero real dependencies, and swapping either
  implementation later touches exactly one file.

### Decision 5.9 — Qdrant as the vector store; three connection modes

- **Reasoning:** Qdrant has first-class `delete-by-filter` (the exact primitive
  `delete_by_source_url` needs) and a genuinely local, no-server embedded mode — important for a
  tool whose whole premise is "local-first, zero required infrastructure." It supports three
  connection shapes from the same client library: a Docker container (`docker-compose.yml`,
  persisted to a named volume), embedded on-disk (`--qdrant-path`, in-process, no server at
  all — like SQLite), or a remote/managed cluster (`--qdrant-url` + `--qdrant-api-key`, e.g.
  Qdrant Cloud). One collection per knowledge base, so KBs are cleanly isolated from each other.
- **Trade-off:** A real dependency (vs. hand-rolling a flat-file vector index), justified by how
  much correct, fast filtered-delete behavior it buys for free.

### Decision 5.10 — fastembed for local embeddings; GPU is the real performance lever

- **Reasoning:** fastembed runs ONNX models locally with no PyTorch dependency — lightweight,
  fits the local-first goal. Default model is `BAAI/bge-small-en-v1.5` (384-dim, small, fast,
  good quality for English documentation). Measurement during development showed embedding is
  the dominant cost of a sync (~99.8% of processing time once crawling is parallelized) — so
  `--device auto|cpu|cuda` (GPU auto-detected, with an automatic, warned fallback to CPU if GPU
  setup fails) is the lever that actually matters for large syncs, not micro-optimizing batch
  sizes.
- **Trade-off:** A GPU-accelerated embedder never crashes for lack of a GPU (graceful
  degradation to CPU), at the cost of a slightly more complex constructor than "always CPU."

### Decision 5.11 — Chunking: paragraph-packing with overlap, character-based

- **Reasoning:** Split on paragraph (blank-line) boundaries, greedily pack paragraphs up to a
  character budget (default 1200 chars), hard-split any single paragraph that alone exceeds the
  budget, and carry a small overlap (default 150 chars) from each chunk into the next so a
  thought spanning a chunk boundary isn't lost entirely on either side.
- **Trade-off:** Character-based sizing is a proxy for the embedding model's actual token count,
  not exact — simple and effective in practice, with token-aware sizing left as a future
  refinement if it ever proves necessary.

### Decision 5.12 — Multiple named knowledge bases

- **Reasoning:** One user syncing several docs sites (Docker, nginx, Kubernetes...) needs them
  isolated and independently searchable, not mixed into one undifferentiated pile. Realized as:
  one SQLite manifest file with a `(name, url)` composite primary key (so many sites share one
  small file, scoped by `name`), and **one Qdrant collection per knowledge base** (a different
  isolation mechanism for a different storage engine — each the natural fit for that engine, not
  forced to match). The default name is derived from the URL's host
  (`docs.docker.com` → `docs_docker_com`), overridable with `--name`.
- **Trade-off:** A composite key is marginally more complex than a single-column primary key;
  in exchange, `docforge search` can merge results across every knowledge base by default with
  no extra bookkeeping, and removing one KB never touches another's data.

### Decision 5.13 — HTTP conditional requests as a *guarded* pre-check, not a blanket one

- **Reasoning:** A page's own HTTP server can answer "did this change?" cheaply via
  `ETag`/`If-Modified-Since`, avoiding a full headless-browser render for pages that are
  unchanged. But conditional-checking *every* URL before every crawl — including on a first-ever
  sync, where nothing is stored yet — means thousands of pointless requests that can only ever
  come back 200, turning into a multi-minute stall before the real crawl even starts. The fix:
  only issue a conditional request for a URL that has **both** been seen before **and** has a
  stored validator to send (`_preselect` in `detector.py`); every other URL skips straight to a
  real crawl, which captures fresh validators for next time.
- **Trade-off:** A site that never sends validators gets no benefit from this — the pre-check
  degrades to "no-op," never to "silently miss a real change." Reported explicitly to the user
  (`--conditional auto|on|off`) rather than failing silently.

### Decision 5.14 — Parallel, rate-limited crawling

- **Reasoning:** Sequential crawling doesn't scale to sites with hundreds or thousands of pages.
  Crawling runs concurrently (Crawl4AI's `arun_many` + a memory-aware dispatcher, capping
  concurrent browser sessions by available RAM) with a per-*domain* rate limiter (a polite random
  delay between requests to the same site, with backoff on 429/503) — since on a single big docs
  site, that per-domain delay is the real throughput ceiling, not raw concurrency.
  `--concurrency`/`--crawl-delay`/`--no-rate-limit` expose the actual knobs that matter.
- **Trade-off:** Faster crawling is also less polite; defaults are deliberately modest (5
  concurrent, 0.5–1.5s delay), with the aggressive settings opt-in for sites the user controls
  or trusts.

### Decision 5.15 — Dependency injection over a framework, for both front doors

- **Reasoning:** Both `cli.py` and `mcp_server.py` follow the same shape: a plain, injectable
  function holds the real logic (`run_sync`, `search_docs_text`, ...) with real production
  defaults for anything that touches the network/a model/a store; a thin wiring layer
  (`main()`/`build_server()`) supplies the real implementations. No web framework, no CLI
  framework beyond the stdlib's `argparse` — just functions and Protocol-typed parameters. This
  is what makes the test suite run in seconds with no Docker, no browser, and (mostly) no model
  download, and what let the MCP server be built as a *second* thin front door onto the exact
  same core with no logic duplicated.
- **Trade-off:** More explicit wiring code than a framework would generate, in exchange for the
  whole system being traceable by reading it, not by knowing a framework's conventions.

### Decision 5.16 — MCP server: retrieval-only scope, secure by default

- **Reasoning:** M4 exposes DocForge's existing search as MCP tools (`list_docs`, `search_docs`)
  for any MCP-capable LLM client — Claude Code, Claude Desktop, LM Studio, a custom local-LLM
  agent. Deliberately **out of scope**: an ingestion/crawl tool. Crawling a large site takes
  minutes, which is the wrong shape for a synchronous tool call inside a chat turn; ingestion
  stays a CLI-only operation run on the user's own schedule. Two transports: `stdio` (the client
  spawns the process itself — no network boundary, no auth needed) and `http` (a real network
  boundary — secured **by default** with a randomly generated bearer token, printed once per run
  and never persisted, the same pattern Jupyter Notebook uses for its own local server; an
  explicit `--no-auth` opts out, and `DOCFORGE_MCP_TOKEN` gives a stable token across restarts).
- **Trade-off:** No "ask the assistant to go index a new site" convenience from inside a chat —
  a deliberate line, revisitable if a real recurring need for it shows up (it would need to be a
  background job with a status-poll tool, not a blocking call, to do properly).

---

## 6. Licensing

### Can we open-source DocForge? — Yes.

"Can I open-source this?" is really three questions:

1. **Am I allowed to, given my dependencies?** Dependency licenses constrain yours. The danger is
   _copyleft_ (GPL/AGPL), which can force your whole project to adopt the same terms. We checked
   every core dependency — none are copyleft: **Crawl4AI** (Apache-2.0, permissive, commercial
   use OK — it requests attribution, honored in the README), **Qdrant** and **fastembed**
   (Apache-2.0), the **MCP Python SDK** (MIT).
2. **Is the code legally mine to give away?** Yes — we write the orchestration ourselves, no
   copy-pasted source. (Real-world checklist item: if code were written on an employer's time/
   equipment, an IP clause could claim it. This project is independent, so we're clear.)
3. **Which license do I pick?** Our choice.

### Decision: license DocForge under **Apache-2.0**

- **Reasoning:** Pairs cleanly with Crawl4AI/Qdrant/fastembed (same license, zero friction),
  includes an explicit patent grant that protects users, and reads as a deliberate, serious
  project.
- **Alternative:** MIT — shorter and even more permissive; a fine choice if brevity is valued.
- **Disclaimer:** This is engineering judgment, not legal advice. For a hobby/portfolio OSS tool
  this is fine; if it ever goes commercial, get real legal review.

---

## 7. Roadmap — vertical slices

We build in **vertical slices**: each milestone is _working software_, shipped fully before the
next begins. We do **not** build all layers at once — that is the classic failure mode (five
projects in one trench coat, none finished).

| Milestone | Deliverable | Status |
| --------- | ----------- | ------ |
| **M0** | Scaffolding: repo, venv/uv, `LICENSE`, `README`, tests + CI, linting. Open-source standard from commit one. | **Done** |
| **M1** | Change detection: crawl → normalize → hash → SQLite manifest → diff report (new/changed/deleted). No RAG yet, independently testable and already useful. | **Done** |
| **M2** | RAG sync: chunk → embed (local model) → upsert to vector DB → delete stale, wired to M1's diff. Pluggable embedder + store. | **Done** |
| **M3** | One-command CLI: `docforge sync/diff/status/search/remove`, idempotent, `.env` config, live progress. | **Done** |
| **M4** | MCP server — expose DocForge as MCP tools (`list_docs`, `search_docs`) so any MCP-capable LLM client can use it; stdio + HTTP transports, secure-by-default auth. | **Done** |
| **M5** | Docker + run-as-service for a clean, no-manual-setup user experience. | **Next** |

Full detail on what shipped in each milestone: [`Docs/ROADMAP.md`](Docs/ROADMAP.md). Command
reference: [`Docs/CLI.md`](Docs/CLI.md). MCP server reference: [`Docs/MCP_SERVER.md`](Docs/MCP_SERVER.md).

---

## 8. M0 in retrospect

M0 established the project skeleton: a `src/` layout (forces installing the package to import
it, so tests run against exactly what a user would experience — catching packaging bugs early),
Apache-2.0 from commit one, and CI (lint + test) running on every push so the project stays
provably green, not just "green on my machine."

**Tooling note:** M0 was originally built with a raw `venv` + `pip install -e`. The project has
since moved to [**uv**](https://docs.astral.sh/uv/) for environment and dependency management
(`uv sync`, `uv run ...`, `uv.lock` committed) — faster, and it unifies "create the venv" and
"install locked dependencies" into one command. See `README.md` for the current install steps;
this section is kept only as a historical record of the reasoning, not as a setup guide.

---

## 9. Decisions settled since M0

At the end of M0, three questions were still open. All three are resolved and documented as
full Decision entries above — recorded here as a single pointer so the history isn't lost:

- **Vector database** → **Qdrant** (Decision 5.9).
- **Local embedding model** → **fastembed**, default `BAAI/bge-small-en-v1.5` (Decision 5.10).
- **Chunking strategy** → **paragraph-packing with overlap** (Decision 5.11).

No open decisions remain blocking current work. The next real decision point is M5 (Docker
packaging) — how the whole tool (DocForge + a bundled Qdrant) ships as a single, no-setup
container.

---

## 10. Engineering principles we're practicing

These are the habits that separate an experienced SDE from a newbie. They apply beyond DocForge.

1. **State the problem in one sentence before coding.** If you can't, you don't understand it yet.
2. **Commodity vs. differentiator.** Never reinvent commodities; never outsource your differentiator.
3. **Scope discipline / vertical slices.** Ship one working slice fully before starting the next.
   Most projects die from building all layers at once and finishing none.
4. **Design the hard invariant first** (here: the page→chunk metadata scheme). Think before coding
   at exactly the point where it's easy to get burned.
5. **Guard the dangerous operations** (deletion only on successful crawl) — and apply the guard
   independently everywhere it matters, so it can't drift out of sync between subsystems.
6. **Idempotency and clean state** from day one.
7. **Honest positioning.** Don't overclaim novelty; state your real, defensible differentiator.
8. **Ask "who owns this code?" and "what license constrains me?"** reflexively.
9. **Document decisions and their reasoning** (this file) so the _why_ survives.
10. **Verify claims about runtime behavior live, not just via unit tests.** A real end-to-end
    smoke test (a real crawl, a real embedded vector store, a real second knowledge base) caught
    a genuine bug — embedded-mode Qdrant crashing on a second collection because a client was
    never closed — that an all-fakes unit test suite had no way to see, because the fake never
    modeled the resource in question.
11. **Security defaults should be secure without effort.** The MCP server's HTTP auth defaults to
    *on* (an auto-generated token, printed for the user) rather than requiring someone to
    remember a flag — the safe path and the path of least resistance should be the same path.

---

_Last updated: after M4 (MCP server) — see §7 for full milestone status. Update this file
whenever a decision changes or a milestone completes._
