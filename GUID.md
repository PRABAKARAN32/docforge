# DocForge — Project Guide

> **Living document.** This is the project's memory. It records not just _what_ we are
> building, but _why_ each decision was made. Read it before writing code; update it when
> a decision changes. If you ever forget why something is the way it is, the answer belongs here.

---

## 0. How to read this document

This guide is written to serve two audiences at once:

1. **You**, working in VS Code, needing a single source of truth for the project.
2. **A local LLM assistant**, which can load this file as context to help you build DocForge
   without re-explaining the whole project every time.

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
Legendary Dev Tool — it is not a side quest, it is what keeps the whole engine accurate.

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
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
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
RAG pipeline, built specifically for documentation sites. It splits cleanly into two halves.

### v1 — Change detection

```
crawl (Crawl4AI) → clean markdown → hash per page → compare to last run
    → { changed, new, deleted } list
```

### v2 — RAG sync

```
diff report → delete stale chunks for changed/deleted pages
    → chunk + embed new content → upsert to vector store → update stored hash
```

**End result:** run `docforge sync <docs-site>` once to build the knowledge base. Run it again
months later, and only the pages that actually changed get re-processed.

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
  and write the ~50 lines of orchestration ourselves.

- **Trade-off:** Slightly more code than importing LlamaIndex wholesale — but we understand every
  line (critical, since this feeds Legendary Dev Tool), and the project stays light. We design the
  embedder and vector store as **pluggable interfaces** so someone can swap in LlamaIndex later if
  they want.

### Decision 5.3 — The page/chunk metadata scheme is make-or-break

- **The problem:** our diff is at the **page** level, but the vector store operates at the **chunk**
  level. When page X changes, we must delete _every_ old chunk from page X and insert the new ones.
- **The solution:** every chunk must carry stable metadata pointing back to its page — a
  `source_url`. Then "delete stale chunks for page X" is simply: delete where `source_url == X`.
- **Reasoning:** Get this right up front and sync is easy. Get it wrong and you get orphaned and
  duplicated chunks forever. This is exactly where newbies get burned and seniors think _before_
  coding.

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
- **The rule:** **Never apply deletions unless the crawl completed successfully.** Guard it explicitly.

### Decision 5.6 — Idempotency from day one

- **Rule:** Running `docforge sync` twice in a row with no doc changes must do _nothing_ the second
  time — no re-embedding, no errors.
- **Reasoning:** Designing for idempotency from the start makes the whole architecture cleaner and
  is the mark of a robust tool.

### Decision 5.7 — SQLite as the state/manifest store

- **What state we need:** a manifest remembering `url → content_hash → chunk_ids → last_seen`.
- **Reasoning:** SQLite is a single-file, zero-config, transactional store — a perfect fit for a
  local tool. It becomes the "memory" of DocForge.
- **Trade-off:** Not a networked/multi-writer DB, but we don't need one for a local single-process tool.

---

## 6. Licensing

### Can we open-source DocForge? — Yes.

"Can I open-source this?" is really three questions:

1. **Am I allowed to, given my dependencies?** Dependency licenses constrain yours. The danger is
   _copyleft_ (GPL/AGPL), which can force your whole project to adopt the same terms. We checked
   every core dependency — none are copyleft:
   - **Crawl4AI → Apache-2.0** (permissive; commercial use OK). It _requests_ attribution (a badge
     or a text line). Honor it — one line in the README.
   - **LlamaIndex → MIT** (the most permissive common license).
2. **Is the code legally mine to give away?** Yes — we write the orchestration ourselves, no
   copy-pasted source. (Real-world checklist item: if code were written on an employer's time/
   equipment, an IP clause could claim it. This project is independent, so we're clear.)
3. **Which license do I pick?** Our choice.

### Decision: license DocForge under **Apache-2.0**

- **Reasoning:** Pairs cleanly with Crawl4AI (same license, zero friction), includes an explicit
  patent grant that protects users, and reads as a deliberate, serious project.
- **Alternative:** MIT — shorter and even more permissive; a fine choice if brevity is valued.
- **Disclaimer:** This is engineering judgment, not legal advice. For a hobby/portfolio OSS tool
  this is fine; if it ever goes commercial, get real legal review.

---

## 7. Roadmap — vertical slices

We build in **vertical slices**: each milestone is _working software_, shipped fully before the
next begins. We do **not** build all layers at once — that is the classic failure mode (five
projects in one trench coat, none finished).

| Milestone | Deliverable                                                                                                                                                   | Status          |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| **M0**    | Scaffolding: repo, venv, `LICENSE`, `README`, tests + CI, linting. Open-source standard from commit one.                                                      | **In progress** |
| **M1**    | Change detection only: crawl → normalize → hash → SQLite manifest → diff report (new/changed/deleted). No RAG yet. Independently testable and already useful. | Next            |
| **M2**    | RAG sync: chunk → embed (local model) → upsert to vector DB → delete stale, wired to M1's diff. Pluggable embedder + store.                                   | Later           |
| **M3**    | One-command CLI + config: `docforge sync <url>`, idempotent and resumable.                                                                                    | Later           |
| **M4**    | MCP server — expose DocForge as MCP tools so any MCP-capable LLM (local, Claude, etc.) can use it.                                                            | Later           |
| **M5**    | Docker + run-as-service for a clean, no-manual-setup user experience.                                                                                         | Later           |

---

## 8. M0 — scaffolding details (current work)

Assumes **Python 3.11+** (Crawl4AI needs 3.10+; use a modern version).

### Step 1 — Virtual environment

```bash
mkdir docforge && cd docforge
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

_Why:_ a venv is an isolated Python sandbox per project. Without it, conflicting dependency versions
across projects cause "dependency hell." One isolated environment per project, always. Installing
globally is the newbie tell.

### Step 2 — Git + `.gitignore` (before the first commit)

```bash
git init
```

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
.env
*.db
*.sqlite3
dist/
build/
*.egg-info/
.pytest_cache/
.ruff_cache/
```

_Why:_ commit the _recipe_, not the installed environment (`.venv/`). Never commit secrets (`.env`)
— committing secrets to a public repo is the most common catastrophic OSS mistake. `*.db` keeps the
local SQLite manifest out of the repo.

### Step 3 — Project layout (`src/` layout)

```
docforge/
├── src/
│   └── docforge/
│       ├── __init__.py
│       ├── crawler.py       # (M1) wraps Crawl4AI
│       ├── hashing.py       # (M1) normalize + hash
│       ├── manifest.py      # (M1) SQLite state
│       └── diff.py          # (M1) new/changed/deleted
├── tests/
│   └── test_hashing.py
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
└── .github/
    └── workflows/
        └── ci.yml
```

_Why `src/`:_ it forces you to install your own package to import it, so tests run against the
_installed_ package exactly as a user would experience it — catching packaging bugs early.

### Step 4 — `pyproject.toml`

Modern single source of truth for build config, dependencies, and tooling (replaces `setup.py` +
`requirements.txt`):

```toml
[project]
name = "docforge"
version = "0.1.0"
description = "Keeps a documentation RAG in sync by detecting and updating only what changed."
readme = "README.md"
requires-python = ">=3.11"
license = "Apache-2.0"
dependencies = []   # added deliberately, per milestone

[project.optional-dependencies]
dev = ["pytest", "ruff"]

[project.scripts]
docforge = "docforge.cli:main"   # wires the `docforge` command (M3)

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
```

Install in editable mode:

```bash
pip install -e ".[dev]"
```

_Why `dependencies = []`:_ we add each library _when its milestone needs it and we understand why_,
never a giant speculative list. _Why `-e`:_ code changes take effect instantly without reinstalling.

### Step 5 — Open-source-standard files

- **`LICENSE`** — full Apache-2.0 text (use GitHub's "Add file → LICENSE" picker for the canonical copy).
- **`README.md`** — the front door: what DocForge does, the problem, install steps, a usage example,
  and the Crawl4AI attribution line. A good README is a real differentiator.
- **`.github/workflows/ci.yml`** — runs linter + tests on every push (proves the project stays green
  automatically, not just "on my machine"):

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest
```

### Finish M0

```bash
git add .
git commit -m "chore: project scaffolding"
```

---

## 9. Open decisions (to settle before M1/M2)

- **[ ] Vector database choice** (needed before M2, shapes M1's manifest + chunk-metadata design):
  Chroma, Qdrant, or LanceDB? Must have good native `upsert` and `delete-by-metadata`. Decide based
  on what Legendary Dev Tool will use locally.
- **[ ] Local embedding model** (M2): which model runs locally for embeddings.
- **[ ] Chunking strategy** (M2): how documentation markdown is split into chunks.

---

## 10. Engineering principles we're practicing

These are the habits that separate an experienced SDE from a newbie. They apply beyond DocForge.

1. **State the problem in one sentence before coding.** If you can't, you don't understand it yet.
2. **Commodity vs. differentiator.** Never reinvent commodities; never outsource your differentiator.
3. **Scope discipline / vertical slices.** Ship one working slice fully before starting the next.
   Most projects die from building all layers at once and finishing none.
4. **Design the hard invariant first** (here: the page→chunk metadata scheme). Think before coding
   at exactly the point where newbies get burned.
5. **Guard the dangerous operations** (deletion only on successful crawl).
6. **Idempotency and clean state** from day one.
7. **Honest positioning.** Don't overclaim novelty; state your real, defensible differentiator.
8. **Ask "who owns this code?" and "what license constrains me?"** reflexively.
9. **Document decisions and their reasoning** (this file) so the _why_ survives.

---

_Last updated: during M0. Update this file whenever a decision changes or a milestone completes._
