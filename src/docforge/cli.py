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

from docforge import __version__
from docforge.conditional import http_conditional_get
from docforge.crawler import DEFAULT_CONCURRENCY, CrawledPage, crawl_urls
from docforge.detector import Crawler, apply_changes, detect_changes
from docforge.diff import deletions_to_apply
from docforge.discovery import discover_urls
from docforge.embedder import DEFAULT_DEVICE, DEFAULT_MODEL, Embedder
from docforge.manifest import Manifest
from docforge.rag import embed_changes
from docforge.vectorstore import VectorStore

# Sentinel so callers/tests can pass conditional=None explicitly (disable) vs. not passing it
# (resolve from the --conditional/--force flags).
_UNSET = object()

# Colorful help via rich-argparse, with a plain fallback if it isn't installed. It preserves
# the raw examples epilog and auto-disables color when output isn't a terminal (pipes/CI).
try:
    from rich_argparse import RawDescriptionRichHelpFormatter as _HelpFormatter

    _HelpFormatter.styles["argparse.prog"] = "bold cyan"
    _HelpFormatter.styles["argparse.groups"] = "bold magenta"
    _HelpFormatter.styles["argparse.args"] = "cyan"
    _HelpFormatter.styles["argparse.metavar"] = "dim cyan"
    _HelpFormatter.styles["argparse.help"] = "default"
except ImportError:  # pragma: no cover -- fallback keeps help working without the dep
    _HelpFormatter = argparse.RawDescriptionHelpFormatter


def _open_store(qdrant_url: str, qdrant_path: str | None):
    """Open the Qdrant store: embedded (--qdrant-path) if given, else the server URL."""
    from docforge.vectorstore import QdrantVectorStore

    if qdrant_path is not None:
        return QdrantVectorStore(path=qdrant_path)
    return QdrantVectorStore(url=qdrant_url)


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
    conditional_mode: str = "auto",
    force: bool = False,
    discover=discover_urls,
    crawl: Crawler = crawl_urls,
    conditional=_UNSET,
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

    if conditional is _UNSET:
        # --force or --conditional off disables the 304 pre-check; otherwise use it.
        use_conditional = not force and conditional_mode != "off"
        conditional = http_conditional_get if use_conditional else None

    with Manifest(db_path) as manifest:
        result = detect_changes(urls, manifest, crawl=crawl, conditional=conditional)
        report = result.report

        if conditional is not None:
            skipped = len(result.current_hashes) - len(result.pages)
            if result.conditional_supported is False:
                out(
                    "Note: this site doesn't send ETag/Last-Modified, so unchanged pages "
                    "can't be skipped; all pages were re-crawled."
                )
            elif skipped > 0:
                out(f"Skipped {skipped} unchanged page(s) via conditional requests (304).")

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
            store = _open_store(qdrant_url, qdrant_path)
        embed_changes(result, embedder, store)
    except Exception as exc:  # noqa: BLE001 -- surface any store/model failure as a clean message
        out(f"Vector store error: {exc}")
        if qdrant_path is None:
            out(f"Is Qdrant running? Start it with: docker compose up -d  (expected at {qdrant_url})")
        return False
    return True


def run_diff(
    seed_url: str,
    *,
    db_path: str = "docforge.db",
    allow_bfs: bool = False,
    max_pages: int | None = None,
    conditional_mode: str = "auto",
    force: bool = False,
    discover=discover_urls,
    crawl: Crawler = crawl_urls,
    conditional=_UNSET,
    out=print,
) -> int:
    """Preview what would change on a site: crawl + detect, list the changes, write nothing."""
    out(f"Discovering pages for {seed_url} ...")
    urls = discover(seed_url, allow_bfs=allow_bfs, max_pages=max_pages)
    if not urls:
        out(f"No pages discovered for {seed_url} (try --bfs if it has no sitemap).")
        return 1
    out(f"Discovered {len(urls)} pages.")

    if conditional is _UNSET:
        conditional = http_conditional_get if (not force and conditional_mode != "off") else None

    with Manifest(db_path) as manifest:
        result = detect_changes(urls, manifest, crawl=crawl, conditional=conditional)
        report = result.report

    for url in sorted(report.new):
        out(f"  + new      {url}")
    for url in sorted(report.changed):
        out(f"  ~ changed  {url}")
    for url in sorted(report.deleted):
        out(f"  - deleted  {url}")
    out(
        f"Would change: {len(report.new)} new, {len(report.changed)} changed, "
        f"{len(report.deleted)} deleted, {len(report.unchanged)} unchanged. (nothing written)"
    )
    return 0


def run_status(*, db_path: str = "docforge.db", out=print) -> int:
    """Show what the manifest is tracking: total pages, broken down by site (host)."""
    from urllib.parse import urlparse

    with Manifest(db_path) as manifest:
        urls = manifest.hashes()

    if not urls:
        out(f"No pages tracked in {db_path}.")
        return 0

    per_host: dict[str, int] = {}
    for url in urls:
        host = urlparse(url).netloc or "?"
        per_host[host] = per_host.get(host, 0) + 1

    out(f"{len(urls)} pages tracked in {db_path}, across {len(per_host)} site(s):")
    for host, count in sorted(per_host.items()):
        out(f"  {count:6d}  {host}")
    return 0


def run_search(
    query: str,
    *,
    qdrant_url: str = "http://localhost:6333",
    qdrant_path: str | None = None,
    embed_model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    limit: int = 5,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    out=print,
) -> int:
    """Search the knowledge base: embed the query and print the closest chunks."""
    try:
        if embedder is None:
            from docforge.embedder import FastEmbedEmbedder

            embedder = FastEmbedEmbedder(embed_model, device=device)
        if store is None:
            store = _open_store(qdrant_url, qdrant_path)
        vector = embedder.embed([query])[0]
        hits = store.search(vector, limit=limit)
    except Exception as exc:  # noqa: BLE001 -- clean message instead of a traceback
        out(f"Search failed: {exc}")
        out("Has anything been synced yet, and is the vector store reachable?")
        return 1

    if not hits:
        out("No results.")
        return 0

    for hit in hits:
        out(f"[{hit.score:.3f}] {hit.source_url}")
        out(f"    {hit.text[:200].strip()}")
    return 0


def run_remove(
    site: str,
    *,
    db_path: str = "docforge.db",
    qdrant_url: str = "http://localhost:6333",
    qdrant_path: str | None = None,
    store: VectorStore | None = None,
    out=print,
) -> int:
    """Remove a site's pages from the manifest and its chunks from the vector store.

    ``site`` matches any tracked URL that contains it (e.g. a host like ``docs.example.com``).
    """
    with Manifest(db_path) as manifest:
        matching = [url for url in manifest.hashes() if site in url]
        if not matching:
            out(f"No tracked pages match {site!r}.")
            return 0

        try:
            if store is None:
                store = _open_store(qdrant_url, qdrant_path)
            for url in matching:
                store.delete_by_source_url(url)
        except Exception as exc:  # noqa: BLE001 -- vector store may be down; report cleanly
            out(f"Vector store error: {exc}")
            out("Nothing was removed. Is the vector store reachable?")
            return 1

        for url in matching:
            manifest.delete_page(url)

    out(f"Removed {len(matching)} page(s) matching {site!r} (manifest + vector store).")
    return 0


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


_SYNC_EXAMPLES = """\
examples:
  docforge sync https://docs.example.com/               build / refresh the knowledge base
  docforge sync https://nginx.org/en/docs/ --bfs        crawl page-by-page (site has no sitemap)
  docforge sync <url> --dry-run                         preview changes, write nothing
  docforge sync <url> --max-pages 50 --concurrency 10   cap pages; crawl 10 in parallel
  docforge sync <url> --qdrant-path ./vectors           no Docker (Qdrant embedded on disk)
  docforge sync <url> --device cuda                     use a GPU for embedding
  docforge sync <url> --force                           ignore ETag/304 and re-crawl everything

A vector store is required: run `docker compose up -d` (Qdrant on :6333), or pass --qdrant-path DIR.
"""


def _add_crawling_flags(sub: argparse.ArgumentParser) -> None:
    group = sub.add_argument_group("crawling")
    group.add_argument(
        "--bfs", action="store_true",
        help="If the site has no sitemap, crawl it page-by-page instead.",
    )
    group.add_argument(
        "--max-pages", type=int, default=None, metavar="N",
        help="Maximum pages to process (default: no limit).",
    )
    group.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, metavar="N",
        help=f"Pages to crawl in parallel (default: {DEFAULT_CONCURRENCY}). "
        "Higher is faster but heavier on the target server.",
    )


def _add_detection_flags(sub: argparse.ArgumentParser) -> None:
    group = sub.add_argument_group("change detection")
    group.add_argument(
        "--db", default="docforge.db", metavar="PATH",
        help="Path to the manifest database file, storing page hashes (default: docforge.db).",
    )
    group.add_argument(
        "--conditional", choices=["auto", "on", "off"], default="auto",
        help="Use HTTP conditional requests (ETag/304) to skip re-crawling unchanged pages: "
        "auto/on = use when the server supports it (warn if not), off = never (default: auto).",
    )
    group.add_argument(
        "--force", action="store_true",
        help="Ignore stored validators and re-crawl every page (skips the 304 pre-check).",
    )


def _add_vector_store_flags(sub: argparse.ArgumentParser) -> None:
    group = sub.add_argument_group("vector store")
    group.add_argument(
        "--qdrant-url", default="http://localhost:6333", metavar="URL",
        help="Qdrant server URL: Docker container, native install, or remote "
        "(default: http://localhost:6333).",
    )
    group.add_argument(
        "--qdrant-path", default=None, metavar="DIR",
        help="Run Qdrant embedded (no Docker), storing vectors in this local folder. "
        "Takes precedence over --qdrant-url.",
    )


def _add_embedding_flags(sub: argparse.ArgumentParser) -> None:
    group = sub.add_argument_group("embedding")
    group.add_argument(
        "--embed-model", default=DEFAULT_MODEL, metavar="NAME",
        help=f"fastembed model name for embeddings (default: {DEFAULT_MODEL}).",
    )
    group.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE,
        help="Embedding compute device: auto (GPU if available, else CPU), cpu, or cuda "
        f"(default: {DEFAULT_DEVICE}). Falls back to CPU if a GPU isn't usable.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docforge",
        description="DocForge keeps a documentation knowledge base fresh: it detects exactly which "
        "pages changed and embeds only those into a vector store.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"docforge {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # sync -- crawl, detect, embed
    sync = subparsers.add_parser(
        "sync",
        help="Crawl a docs site, detect what changed, and embed it into the vector store.",
        description="Discover a site's pages, crawl them, detect exactly what changed since the "
        "last run, and embed only the new/changed pages into the vector store (deleting stale "
        "chunks first). Re-running with no changes does nothing.",
        epilog=_SYNC_EXAMPLES,
        formatter_class=_HelpFormatter,
    )
    sync.add_argument("url", help="The documentation site URL to sync.")
    _add_crawling_flags(sync)
    _add_detection_flags(sync)
    _add_vector_store_flags(sync)
    _add_embedding_flags(sync)
    sync.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing anything (no embedding, no manifest update).",
    )

    # diff -- preview changes, write nothing
    diff = subparsers.add_parser(
        "diff",
        help="Preview what would change on a site (crawl + detect, write nothing).",
        description="Crawl a site and list which pages are new / changed / deleted vs. the "
        "manifest, without embedding or writing anything.",
        formatter_class=_HelpFormatter,
    )
    diff.add_argument("url", help="The documentation site URL to check.")
    _add_crawling_flags(diff)
    _add_detection_flags(diff)

    # status -- what the manifest tracks
    status = subparsers.add_parser(
        "status",
        help="Show what the manifest is tracking (pages per site).",
        formatter_class=_HelpFormatter,
    )
    status.add_argument(
        "--db", default="docforge.db", metavar="PATH",
        help="Manifest database file to inspect (default: docforge.db).",
    )

    # search -- query the knowledge base
    search = subparsers.add_parser(
        "search",
        help="Search the knowledge base and print the closest chunks.",
        description="Embed a query and return the most similar chunks from the vector store, "
        "with their source page and score.",
        formatter_class=_HelpFormatter,
    )
    search.add_argument("query", help="What to search for.")
    search.add_argument(
        "--limit", type=int, default=5, metavar="N", help="Number of results (default: 5).",
    )
    _add_vector_store_flags(search)
    _add_embedding_flags(search)

    # remove -- drop a site from the KB
    remove = subparsers.add_parser(
        "remove",
        help="Remove a site's pages from the manifest and its chunks from the vector store.",
        description="Delete every tracked page whose URL contains the given text (e.g. a host), "
        "along with its chunks in the vector store.",
        formatter_class=_HelpFormatter,
    )
    remove.add_argument("site", help="Host or URL substring to remove, e.g. docs.example.com.")
    remove.add_argument(
        "--db", default="docforge.db", metavar="PATH",
        help="Manifest database file (default: docforge.db).",
    )
    _add_vector_store_flags(remove)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:  # bare `docforge` -> show help instead of an error
        parser.print_help()
        return 0

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
            conditional_mode=args.conditional,
            force=args.force,
            crawl=_make_progress_crawl(args.concurrency),
        )
    if args.command == "diff":
        return run_diff(
            args.url,
            db_path=args.db,
            allow_bfs=args.bfs,
            max_pages=args.max_pages,
            conditional_mode=args.conditional,
            force=args.force,
            crawl=_make_progress_crawl(args.concurrency),
        )
    if args.command == "status":
        return run_status(db_path=args.db)
    if args.command == "search":
        return run_search(
            args.query,
            qdrant_url=args.qdrant_url,
            qdrant_path=args.qdrant_path,
            embed_model=args.embed_model,
            device=args.device,
            limit=args.limit,
        )
    if args.command == "remove":
        return run_remove(
            args.site,
            db_path=args.db,
            qdrant_url=args.qdrant_url,
            qdrant_path=args.qdrant_path,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
