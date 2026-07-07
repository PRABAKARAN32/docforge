"""The ``docforge`` command-line interface.

M3 exposes the M1 change-detection pipeline as one command::

    docforge sync <url> [--bfs] [--max-pages N] [--dry-run] [--db PATH]

It runs discover -> detect -> (apply). Embedding into a vector store is M2; until then
``sync`` maintains the manifest and reports what changed.

Design: :func:`run_sync` holds the logic and takes ``discover``, ``crawl``, and ``out`` as
injected dependencies, so it is unit-tested with fakes (no network). :func:`main` wires the
real functions, argument parsing, and live progress output.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from docforge.crawler import CrawledPage, crawl_urls
from docforge.detector import Crawler, apply_changes, detect_changes
from docforge.discovery import discover_urls
from docforge.manifest import Manifest


def run_sync(
    seed_url: str,
    *,
    db_path: str = "docforge.db",
    allow_bfs: bool = False,
    max_pages: int | None = None,
    dry_run: bool = False,
    discover=discover_urls,
    crawl: Crawler = crawl_urls,
    out=print,
) -> int:
    """Run one ``sync``: discover pages, detect changes, and (unless dry-run) apply them.

    Returns a process exit code (0 = success, 1 = nothing to do / no pages found).
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
        else:
            apply_changes(manifest, result)
            out("Manifest updated.")

    return 0


def _progress_crawl(urls: Sequence[str]) -> list[CrawledPage]:
    """Real crawler wrapped with a live progress line (used by :func:`main`)."""
    total = len(urls)

    def on_page(done: int, _total: int, _url: str) -> None:
        print(f"\rCrawling {done}/{total} ...", end="", flush=True)

    pages = crawl_urls(urls, on_page=on_page)
    if total:
        print()  # end the progress line with a newline
    return pages


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
            crawl=_progress_crawl,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
