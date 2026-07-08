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
import os
import shutil
import sys
from collections.abc import Sequence

from docforge import __version__
from docforge.conditional import http_conditional_get
from docforge.crawler import DEFAULT_BASE_DELAY, DEFAULT_CONCURRENCY, CrawledPage, crawl_urls
from docforge.detector import Crawler, apply_changes, detect_changes
from docforge.diff import deletions_to_apply
from docforge.discovery import derive_kb_name, discover_urls
from docforge.embedder import DEFAULT_DEVICE, DEFAULT_MODEL, Embedder
from docforge.manifest import Manifest
from docforge.progress import ProgressBar, format_duration
from docforge.rag import embed_changes
from docforge.vectorstore import VectorStore

# Sentinel so callers/tests can pass conditional=None explicitly (disable) vs. not passing it
# (resolve from the --conditional/--force flags).
_UNSET = object()


def _load_dotenv(path: str = ".env") -> None:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into the environment (dependency-free).

    Existing environment variables win, so a real ``export`` overrides ``.env``. Blank lines,
    ``#`` comments, an optional ``export`` prefix, and surrounding quotes are handled.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return  # no .env file -> nothing to load

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)

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


def _open_store(
    qdrant_url: str,
    qdrant_path: str | None,
    api_key: str | None = None,
    timeout: float = 60.0,
    collection: str = "docforge",
):
    """Open the Qdrant store for ``collection`` (one per knowledge base).

    Embedded (--qdrant-path) if given, else the server URL. ``api_key`` authenticates to a
    remote/managed Qdrant (e.g. Qdrant Cloud); ``timeout`` (seconds) covers server requests.
    """
    from docforge.vectorstore import QdrantVectorStore

    if qdrant_path is not None:
        return QdrantVectorStore(path=qdrant_path, collection=collection)
    return QdrantVectorStore(
        url=qdrant_url, api_key=api_key, timeout=timeout, collection=collection
    )


def _store_error_hint(qdrant_url: str, qdrant_path: str | None) -> str:
    """A hint tailored to how the user is connecting (local Docker vs. remote/cloud)."""
    if qdrant_path is not None:
        return f"Could not open the embedded vector store at {qdrant_path}."
    if "localhost" in qdrant_url or "127.0.0.1" in qdrant_url:
        return f"Is Qdrant running? Start it with: docker compose up -d  (expected at {qdrant_url})"
    return (
        f"Could not reach the vector store at {qdrant_url}. Check the URL, that QDRANT_API_KEY "
        "is set/correct, your network, and try a larger --qdrant-timeout for a distant cluster."
    )


def run_sync(
    seed_url: str,
    *,
    name: str | None = None,
    db_path: str = "docforge.db",
    allow_bfs: bool = False,
    max_pages: int | None = None,
    dry_run: bool = False,
    qdrant_url: str = "http://localhost:6333",
    qdrant_path: str | None = None,
    qdrant_api_key: str | None = None,
    qdrant_timeout: float = 60.0,
    embed_model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    conditional_mode: str = "auto",
    force: bool = False,
    discover=discover_urls,
    crawl: Crawler = crawl_urls,
    conditional=_UNSET,
    embed=embed_changes,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    out=print,
) -> int:
    """Run one ``sync``: discover, detect changes, embed the changed pages, update the manifest.

    Returns a process exit code (0 = success, 1 = nothing to do / no pages / vector-store error).
    """
    kb_name = name or derive_kb_name(seed_url)
    out(f"Knowledge base: {kb_name}")
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
        # Fail-fast: confirm the vector store is reachable BEFORE the expensive crawl+embed,
        # so a bad URL / API key / down cluster doesn't waste a long crawl. (Skipped for
        # --dry-run, which never writes.)
        if not dry_run:
            prepared = _prepare_store(
                embedder, store, qdrant_url, qdrant_path, qdrant_api_key, qdrant_timeout,
                embed_model, device, kb_name, out,
            )
            if prepared is None:
                return 1
            embedder, store = prepared

        result = detect_changes(urls, manifest, name=kb_name, crawl=crawl, conditional=conditional)
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
            if not _embed_into_store(result, embedder, store, qdrant_url, qdrant_path, out, embed):
                return 1

        apply_changes(manifest, result, name=kb_name)
        out("Manifest updated.")

    return 0


def _prepare_store(
    embedder, store, qdrant_url, qdrant_path, qdrant_api_key, qdrant_timeout,
    embed_model, device, collection, out,
):
    """Build the embedder + open the vector store and confirm it's reachable, BEFORE crawling.

    This is the fail-fast: a bad URL / API key / down cluster errors here in seconds, instead
    of after a long crawl+embed. Returns (embedder, store) or None (after printing why) on error.
    """
    try:
        if embedder is None:
            from docforge.embedder import FastEmbedEmbedder

            embedder = FastEmbedEmbedder(embed_model, device=device)
            out(f"  (embedding device: {embedder.device})")
        if store is None:
            store = _open_store(
                qdrant_url, qdrant_path, qdrant_api_key, qdrant_timeout, collection
            )
        # ensure_collection makes a real round-trip -> proves the store is reachable/writable now.
        store.ensure_collection(embedder.dimension)
    except Exception as exc:  # noqa: BLE001 -- surface store/model failure as a clean message
        out(f"Vector store error: {exc}")
        out(_store_error_hint(qdrant_url, qdrant_path))
        return None
    return embedder, store


def _embed_into_store(result, embedder, store, qdrant_url, qdrant_path, out, embed=embed_changes) -> bool:
    """Embed changes into the (already-prepared) vector store; False + message on error."""
    try:
        embed(result, embedder, store)
    except Exception as exc:  # noqa: BLE001 -- surface any store failure as a clean message
        out(f"Vector store error: {exc}")
        out(_store_error_hint(qdrant_url, qdrant_path))
        return False
    return True


def run_diff(
    seed_url: str,
    *,
    name: str | None = None,
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
    kb_name = name or derive_kb_name(seed_url)
    out(f"Knowledge base: {kb_name}")
    out(f"Discovering pages for {seed_url} ...")
    urls = discover(seed_url, allow_bfs=allow_bfs, max_pages=max_pages)
    if not urls:
        out(f"No pages discovered for {seed_url} (try --bfs if it has no sitemap).")
        return 1
    out(f"Discovered {len(urls)} pages.")

    if conditional is _UNSET:
        conditional = http_conditional_get if (not force and conditional_mode != "off") else None

    with Manifest(db_path) as manifest:
        result = detect_changes(urls, manifest, name=kb_name, crawl=crawl, conditional=conditional)
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
    """List the knowledge bases in the manifest, with page counts."""
    with Manifest(db_path) as manifest:
        kbs = manifest.names()

    if not kbs:
        out(f"No knowledge bases tracked in {db_path}.")
        return 0

    total = sum(kbs.values())
    out(f"{total} pages across {len(kbs)} knowledge base(s) in {db_path}:")
    for kb_name, count in sorted(kbs.items()):
        out(f"  {count:6d}  {kb_name}")
    return 0


def run_search(
    query: str,
    *,
    name: str | None = None,
    search_all: bool = False,
    db_path: str = "docforge.db",
    qdrant_url: str = "http://localhost:6333",
    qdrant_path: str | None = None,
    qdrant_api_key: str | None = None,
    qdrant_timeout: float = 60.0,
    embed_model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    limit: int = 5,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    out=print,
) -> int:
    """Search one knowledge base (--name) or all of them (default / --all)."""
    try:
        if embedder is None:
            from docforge.embedder import FastEmbedEmbedder

            embedder = FastEmbedEmbedder(embed_model, device=device)
        vector = embedder.embed([query])[0]

        if store is not None:
            hits = list(store.search(vector, limit=limit))
        else:
            if name and not search_all:
                collections = [name]
            else:
                with Manifest(db_path) as manifest:
                    collections = sorted(manifest.names())
                if not collections:
                    out("No knowledge bases yet -- run `docforge sync <url>` first.")
                    return 0
            hits = []
            for collection in collections:
                bound = _open_store(qdrant_url, qdrant_path, qdrant_api_key, qdrant_timeout, collection)
                hits.extend(bound.search(vector, limit=limit))
            hits.sort(key=lambda hit: hit.score, reverse=True)
            hits = hits[:limit]
    except Exception as exc:  # noqa: BLE001 -- clean message instead of a traceback
        out(f"Search failed: {exc}")
        out(_store_error_hint(qdrant_url, qdrant_path))
        return 1

    if not hits:
        out("No results.")
        return 0

    for hit in hits:
        out(f"[{hit.score:.3f}] {hit.source_url}")
        out(f"    {hit.text[:200].strip()}")
    return 0


def run_remove(
    name: str | None = None,
    *,
    remove_all: bool = False,
    db_path: str = "docforge.db",
    qdrant_url: str = "http://localhost:6333",
    qdrant_path: str | None = None,
    qdrant_api_key: str | None = None,
    qdrant_timeout: float = 60.0,
    store: VectorStore | None = None,
    out=print,
) -> int:
    """Remove one knowledge base by ``name``, or ``--all`` to wipe everything.

    A single KB drops its manifest pages + its Qdrant collection. ``--all`` drops every KB's
    collection AND deletes the manifest DB file (and the embedded vectors folder, if used).
    """
    if remove_all:
        return _remove_all(db_path, qdrant_url, qdrant_path, qdrant_api_key, qdrant_timeout, store, out)

    if not name:
        out("Specify a knowledge base name (see `docforge status`), or --all to remove everything.")
        return 1

    with Manifest(db_path) as manifest:
        if name not in manifest.names():
            out(f"No knowledge base named {name!r}. Run `docforge status` to list them.")
            return 0

        try:
            bound = store if store is not None else _open_store(
                qdrant_url, qdrant_path, qdrant_api_key, qdrant_timeout, name
            )
            bound.delete_collection()
        except Exception as exc:  # noqa: BLE001 -- vector store may be down; report cleanly
            out(f"Vector store error: {exc}")
            out(_store_error_hint(qdrant_url, qdrant_path))
            return 1

        removed = manifest.delete_kb(name)

    out(f"Removed knowledge base {name!r} ({removed} pages + its vectors).")
    return 0


def _remove_all(db_path, qdrant_url, qdrant_path, qdrant_api_key, qdrant_timeout, store, out) -> int:
    """Wipe everything: every KB's collection + the manifest DB file (+ embedded vectors)."""
    with Manifest(db_path) as manifest:
        kbs = list(manifest.names())

    try:
        if qdrant_path is not None:
            # Embedded: the whole folder holds all collections -> remove it in one shot.
            if os.path.isdir(qdrant_path):
                shutil.rmtree(qdrant_path)
        else:
            for kb_name in kbs:
                bound = store if store is not None else _open_store(
                    qdrant_url, qdrant_path, qdrant_api_key, qdrant_timeout, kb_name
                )
                bound.delete_collection()
    except Exception as exc:  # noqa: BLE001 -- vector store may be down; report cleanly
        out(f"Vector store error: {exc}")
        out(_store_error_hint(qdrant_url, qdrant_path))
        return 1

    # Delete the manifest DB file (skip in-memory / missing).
    if db_path != ":memory:" and os.path.exists(db_path):
        os.remove(db_path)

    out(f"Removed everything: {len(kbs)} knowledge base(s) + the manifest at {db_path}.")
    return 0


def _make_progress_crawl(
    concurrency: int,
    base_delay: tuple[float, float] = DEFAULT_BASE_DELAY,
    rate_limit: bool = True,
) -> Crawler:
    """Build a real crawler (concurrency + rate-limit settings) with a live progress bar.

    Prints a rough upfront estimate for a rate-limited crawl -- on a single docs domain the
    rate limiter serializes to ~one page per delay, so ``pages * avg_delay`` is a fair guess --
    then a live, self-correcting ETA that refines it as pages complete.
    """

    def _crawl(urls: Sequence[str]) -> list[CrawledPage]:
        total = len(urls)
        if total and rate_limit:
            estimate = total * (base_delay[0] + base_delay[1]) / 2
            print(
                f"Estimated crawl time: ~{format_duration(estimate)} for {total} page(s) "
                "(refined live below)."
            )
        bar = ProgressBar(total, "Crawling")

        def on_page(done: int, _total: int, _url: str) -> None:
            bar.update(done)

        pages = crawl_urls(
            urls,
            concurrency=concurrency,
            base_delay=base_delay,
            rate_limit=rate_limit,
            on_page=on_page,
        )
        if total:
            bar.finish()
        return pages

    return _crawl


def _make_progress_embed():
    """Wrap :func:`embed_changes` with a live progress bar (created once the total is known)."""

    def _embed(result, embedder: Embedder, store: VectorStore) -> None:
        bar: ProgressBar | None = None

        def on_page(done: int, total: int, _url: str) -> None:
            nonlocal bar
            if bar is None:
                bar = ProgressBar(total, "Embedding")
            bar.update(done)

        embed_changes(result, embedder, store, on_page=on_page)
        if bar is not None:
            bar.finish()

    return _embed


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
    group.add_argument(
        "--crawl-delay", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Random delay range (seconds) between requests to the SAME site "
        f"(default: {DEFAULT_BASE_DELAY[0]} {DEFAULT_BASE_DELAY[1]}). Lower it to speed up a "
        "big single-domain crawl; this is the main throughput knob.",
    )
    group.add_argument(
        "--no-rate-limit", action="store_true",
        help="Remove the per-site delay entirely (fastest, least polite; may trigger 429s).",
    )


def _add_detection_flags(sub: argparse.ArgumentParser) -> None:
    group = sub.add_argument_group("change detection")
    group.add_argument(
        "--db", default=None, metavar="PATH",
        help="Path to the manifest database file, storing page hashes "
        "(default: DOCFORGE_DB env var, else docforge.db).",
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
        "--qdrant-url", default=None, metavar="URL",
        help="Qdrant server URL: Docker container, native install, or remote "
        "(default: QDRANT_URL env var, else http://localhost:6333).",
    )
    group.add_argument(
        "--qdrant-path", default=None, metavar="DIR",
        help="Run Qdrant embedded (no Docker), storing vectors in this local folder. "
        "Takes precedence over --qdrant-url.",
    )
    group.add_argument(
        "--qdrant-api-key", default=None, metavar="KEY",
        help="API key for a remote/managed Qdrant (e.g. Qdrant Cloud). Prefer the "
        "QDRANT_API_KEY environment variable to keep the key out of shell history.",
    )
    group.add_argument(
        "--qdrant-timeout", type=float, default=60.0, metavar="SECONDS",
        help="Timeout for vector-store requests, in seconds (default: 60). Raise it for a "
        "slow or distant/remote cluster.",
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


def _add_name_flag(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--name", default=None, metavar="NAME",
        help="Knowledge base name -- its own manifest scope + Qdrant collection. "
        "Default: derived from the site's host (e.g. docs.docker.com -> docs_docker_com).",
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
    _add_name_flag(sync)
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
    _add_name_flag(diff)
    _add_crawling_flags(diff)
    _add_detection_flags(diff)

    # status / list -- the knowledge bases tracked (aliases)
    for command in ("status", "list"):
        sub = subparsers.add_parser(
            command,
            help="List the knowledge bases (docs sites) and their page counts.",
            formatter_class=_HelpFormatter,
        )
        sub.add_argument(
            "--db", default=None, metavar="PATH",
            help="Manifest database file to inspect (default: DOCFORGE_DB env var, else docforge.db).",
        )

    # search -- query one or all knowledge bases
    search = subparsers.add_parser(
        "search",
        help="Search a knowledge base and print the closest chunks.",
        description="Embed a query and return the most similar chunks. Use --name to search one "
        "knowledge base, or --all (the default) to search across all of them.",
        formatter_class=_HelpFormatter,
    )
    search.add_argument("query", help="What to search for.")
    search.add_argument(
        "--name", default=None, metavar="NAME", help="Search only this knowledge base.",
    )
    search.add_argument(
        "--all", action="store_true", dest="search_all",
        help="Search across all knowledge bases (the default when --name is omitted).",
    )
    search.add_argument(
        "--limit", type=int, default=5, metavar="N", help="Number of results (default: 5).",
    )
    search.add_argument(
        "--db", default=None, metavar="PATH",
        help="Manifest database file (used to list KBs for --all; default: DOCFORGE_DB, else docforge.db).",
    )
    _add_vector_store_flags(search)
    _add_embedding_flags(search)

    # remove -- drop a whole knowledge base
    remove = subparsers.add_parser(
        "remove",
        help="Remove a knowledge base (or --all): its manifest pages and its Qdrant collection.",
        description="Delete a knowledge base by name (its manifest pages + its vector-store "
        "collection), or --all to wipe every knowledge base AND the manifest DB file. See "
        "`docforge status` for the names.",
        formatter_class=_HelpFormatter,
    )
    remove.add_argument(
        "name", nargs="?", help="The knowledge base name to remove (e.g. docs_docker_com).",
    )
    remove.add_argument(
        "--all", action="store_true", dest="remove_all",
        help="Remove EVERYTHING: all collections + the manifest DB file (and embedded vectors).",
    )
    remove.add_argument(
        "--db", default=None, metavar="PATH",
        help="Manifest database file (default: DOCFORGE_DB env var, else docforge.db).",
    )
    _add_vector_store_flags(remove)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _load_dotenv()  # let a local .env supply QDRANT_URL / QDRANT_API_KEY / DOCFORGE_DB / QDRANT_PATH
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:  # bare `docforge` -> show help instead of an error
        parser.print_help()
        return 0

    # Config resolution, in precedence order: explicit flag > .env/env var > built-in default.
    db_path = getattr(args, "db", None) or os.getenv("DOCFORGE_DB") or "docforge.db"
    qdrant_url = getattr(args, "qdrant_url", None) or os.getenv("QDRANT_URL") or "http://localhost:6333"
    qdrant_path = getattr(args, "qdrant_path", None) or os.getenv("QDRANT_PATH")
    api_key = getattr(args, "qdrant_api_key", None) or os.getenv("QDRANT_API_KEY")
    timeout = getattr(args, "qdrant_timeout", 60.0)

    if args.command in ("sync", "diff"):
        base_delay = tuple(args.crawl_delay) if args.crawl_delay else DEFAULT_BASE_DELAY
        crawl = _make_progress_crawl(args.concurrency, base_delay, not args.no_rate_limit)

    if args.command == "sync":
        return run_sync(
            args.url,
            name=args.name,
            db_path=db_path,
            allow_bfs=args.bfs,
            max_pages=args.max_pages,
            dry_run=args.dry_run,
            qdrant_url=qdrant_url,
            qdrant_path=qdrant_path,
            qdrant_api_key=api_key,
            qdrant_timeout=timeout,
            embed_model=args.embed_model,
            device=args.device,
            conditional_mode=args.conditional,
            force=args.force,
            crawl=crawl,
            embed=_make_progress_embed(),
        )
    if args.command == "diff":
        return run_diff(
            args.url,
            name=args.name,
            db_path=db_path,
            allow_bfs=args.bfs,
            max_pages=args.max_pages,
            conditional_mode=args.conditional,
            force=args.force,
            crawl=crawl,
        )
    if args.command in ("status", "list"):
        return run_status(db_path=db_path)
    if args.command == "search":
        return run_search(
            args.query,
            name=args.name,
            search_all=args.search_all,
            db_path=db_path,
            qdrant_url=qdrant_url,
            qdrant_path=qdrant_path,
            qdrant_api_key=api_key,
            qdrant_timeout=timeout,
            embed_model=args.embed_model,
            device=args.device,
            limit=args.limit,
        )
    if args.command == "remove":
        return run_remove(
            args.name,
            remove_all=args.remove_all,
            db_path=db_path,
            qdrant_url=qdrant_url,
            qdrant_path=qdrant_path,
            qdrant_api_key=api_key,
            qdrant_timeout=timeout,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
