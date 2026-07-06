# DocForge

**Keeps a documentation RAG in sync by detecting and updating only what changed — instead of rebuilding everything.**

Documentation sites change constantly. Most RAG pipelines handle updates the wasteful way: re-crawl the whole site and re-embed every page, even when only a handful changed. For a large docs site (1,000–10,000+ pages) that means long refresh windows, wasted embedding cost, and stale answers in between.

DocForge takes the other path: on each run it **detects exactly which pages changed** (hashing normalized markdown) and **touches only those** — deleting stale chunks and re-embedding just the new content.

> **Status:** Early development — **M0 (scaffolding)**. Change detection (M1) and RAG sync (M2) are next. See [`GUID.md`](GUID.md) for the full design and reasoning.

## Why it exists

DocForge is the local, always-fresh knowledge base behind **Legendary Dev Tool**, a local-first AI coding assistant. A local LLM is only as good as the context it has; DocForge keeps that context current without sending code to third parties or paying per-request.

## How it works

```
crawl (Crawl4AI) → normalized markdown → hash per page → diff vs. last run
    → { new, changed, deleted }
    → delete stale chunks → chunk + embed new content → upsert → update stored hash
```

## Install (development)

Requires **Python 3.11+**.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"            # editable install + dev tools (pytest, ruff)
```

Run the checks:

```bash
ruff check .
pytest
```

## Usage

```bash
# Coming in M3 — the one-command interface:
docforge sync <docs-site-url>
```

Run it once to build the knowledge base; run it again months later and only the pages that actually changed get re-processed.

## License

Licensed under the [Apache License 2.0](LICENSE).

Built on [Crawl4AI](https://github.com/unclecode/crawl4ai) (Apache-2.0) for crawling and HTML→Markdown conversion.
