"""The ``docforge`` command-line interface.

The ``docforge`` command runs the whole pipeline as one command::

    docforge sync <url> [--bfs] [--max-pages N] [--dry-run] [--db PATH] [--qdrant-url URL]

It runs discover -> detect -> embed changed pages into the vector store -> apply (manifest).

Design: :func:`run_sync` holds the logic and takes ``discover``, ``crawl``, ``embedder``,
``store``, and ``out`` as injected dependencies, so it is unit-tested with fakes (no network,
no Docker). :func:`main` wires the real functions, argument parsing, and live progress output.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from docforge.crawler import DEFAULT_CONCURRENCY, CrawledPage, crawl_urls
from docforge.detector import Crawler, apply_changes, detect_changes
from docforge.diff import deletions_to_apply
from docforge.discovery import discover_urls
from docforge.embedder import DEFAULT_DEVICE, DEFAULT_MODEL, Embedder
from docforge.manifest import Manifest
from docforge.rag import embed_changes
from docforge.vectorstore import VectorStore


def run_sync(
    seed_url: str,
    *,
    db_path: str = "docforge.db",
    allow_bfs: bool = False,
    max_pages: int | None = None,
    dry_run: bool = False,
    qdrant_url: str = "http://localhost:6333",
    qdrant_path: str | None = None,
    embed_model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    discover=discover_urls,
    crawl: Crawler = crawl_urls,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    out=print,
) -> int:
    """Run one ``sync``: discover, detect changes, embed the changed pages, update the manifest.

    Returns a process exit code (0 = success, 1 = nothing to do / no pages / vector-store error).
    """
    out(f"Discovering pages for {seed_url} ...")
    urls = discover(seed_url, allow_bfs=allow_bfs, max_pages=max_pages)

    if not urls:
        if not allow_bfs:
            out(f"No sitemap found for {seed_url}.")
            out("Re-run with --bfs to crawl the site page-by-page.")
        else:
            out("No pages discovered.")
        return 1

    out(f"Discovered {len(urls)} pages.")

    with Manifest(db_path) as manifest:
        result = detect_changes(urls, manifest, crawl=crawl)
        report = result.report
        out(
            f"Changes: {len(report.new)} new, {len(report.changed)} changed, "
            f"{len(report.deleted)} deleted, {len(report.unchanged)} unchanged."
        )
        if not result.crawl_succeeded and report.deleted:
            out("Note: some pages failed to crawl; deletions are suppressed this run.")

        if dry_run:
            out("Dry run: no changes written.")
            return 0

        needs_store = report.has_content_changes or bool(
            deletions_to_apply(report, crawl_succeeded=result.crawl_succeeded)
        )
        if needs_store:
            to_embed = len(report.new) + len(report.changed)
            out(f"Embedding {to_embed} changed page(s) into the vector store ...")
            if not _embed_into_store(
                result, qdrant_url, qdrant_path, embed_model, device, embedder, store, out
            ):
                return 1

        apply_changes(manifest, result)
        out("Manifest updated.")

    return 0


def _embed_into_store(
    result, qdrant_url, qdrant_path, embed_model, device, embedder, store, out
) -> bool:
    """Embed changes into the vector store; return False (with a helpful message) on error."""
    try:
        if embedder is None:
            from docforge.embedder import FastEmbedEmbedder

            embedder = FastEmbedEmbedder(embed_model, device=device)
            out(f"  (embedding device: {embedder.device})")
        if store is None:
            from docforge.vectorstore import QdrantVectorStore

            # An explicit --qdrant-path means embedded (no-Docker) mode; else a server URL.
            store = (
                QdrantVectorStore(path=qdrant_path)
                if qdrant_path is not None
                else QdrantVectorStore(url=qdrant_url)
            )
        embed_changes(result, embedder, store)
    except Exception as exc:  # noqa: BLE001 -- surface any store/model failure as a clean message
        out(f"Vector store error: {exc}")
        if qdrant_path is None:
            out(f"Is Qdrant running? Start it with: docker compose up -d  (expected at {qdrant_url})")
        return False
    return True


def _make_progress_crawl(concurrency: int) -> Crawler:
    """Build a real crawler (at the given concurrency) wrapped with a live progress line."""

    def _crawl(urls: Sequence[str]) -> list[CrawledPage]:
        total = len(urls)

        def on_page(done: int, _total: int, _url: str) -> None:
            print(f"\rCrawling {done}/{total} ...", end="", flush=True)

        pages = crawl_urls(urls, concurrency=concurrency, on_page=on_page)
        if total:
            print()  # end the progress line with a newline
        return pages

    return _crawl


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docforge",
        description="Keep a documentation knowledge base in sync by detecting only what changed.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="Detect and record what changed on a docs site.")
    sync.add_argument("url", help="The documentation site URL to sync.")
    sync.add_argument(
        "--bfs",
        action="store_true",
        help="If the site has no sitemap, crawl it page-by-page instead.",
    )
    sync.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Maximum pages to process (default: no limit).",
    )
    sync.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing to the manifest.",
    )
    sync.add_argument(
        "--db",
        default="docforge.db",
        metavar="PATH",
        help="Path to the manifest database file (default: docforge.db).",
    )
    sync.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        metavar="URL",
        help="Qdrant server URL: our Docker container, a native install, or remote "
        "(default: http://localhost:6333).",
    )
    sync.add_argument(
        "--qdrant-path",
        default=None,
        metavar="DIR",
        help="Run Qdrant embedded (no Docker/server), storing vectors in this local folder. "
        "Takes precedence over --qdrant-url.",
    )
    sync.add_argument(
        "--embed-model",
        default=DEFAULT_MODEL,
        metavar="NAME",
        help=f"fastembed model name for embeddings (default: {DEFAULT_MODEL}).",
    )
    sync.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        metavar="N",
        help=f"How many pages to crawl in parallel (default: {DEFAULT_CONCURRENCY}). "
        "Higher is faster but heavier on the target server.",
    )
    sync.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_DEVICE,
        help="Embedding compute device: auto (GPU if available, else CPU), cpu, or cuda "
        f"(default: {DEFAULT_DEVICE}). Falls back to CPU if a GPU isn't usable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "sync":
        return run_sync(
            args.url,
            db_path=args.db,
            allow_bfs=args.bfs,
            max_pages=args.max_pages,
            dry_run=args.dry_run,
            qdrant_url=args.qdrant_url,
            qdrant_path=args.qdrant_path,
            embed_model=args.embed_model,
            device=args.device,
            crawl=_make_progress_crawl(args.concurrency),
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
