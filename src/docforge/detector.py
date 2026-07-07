"""Orchestration: wire crawl -> hash -> manifest -> diff into one change-detection run.

This is the piece that makes M1 usable as a unit. The individual components stay small
and single-purpose; this module just sequences them:

    1. crawl the URLs                     -> {url: markdown}   (crawler.py)
    2. hash each page                     -> {url: hash}       (hashing.py)
    3. read last run's hashes             -> {url: hash}       (manifest.py)
    4. diff previous vs current           -> DiffReport        (diff.py)

Two functions, kept separate on purpose (truth vs. mutation):
  * :func:`detect_changes` computes what changed and changes *nothing*.
  * :func:`apply_changes`  writes the result into the manifest (guarded deletions).

The crawler is injected (the ``crawl`` parameter) so the whole flow is unit-testable with
a fake crawler -- no network, no browser. That's the payoff of ``CrawledPage`` being our
own small type rather than a Crawl4AI object.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from docforge.crawler import CrawledPage, crawl_urls
from docforge.diff import DiffReport, deletions_to_apply, diff_hashes
from docforge.hashing import content_hash
from docforge.manifest import Manifest

# A crawler is anything that takes URLs and returns crawled pages. The real one is
# crawl_urls; tests pass a fake. Typing it this way is what makes injection explicit.
Crawler = Callable[[Sequence[str]], list[CrawledPage]]


@dataclass(frozen=True)
class ChangeDetectionResult:
    """The outcome of a detection run, before anything is written to the manifest."""

    report: DiffReport
    current_hashes: dict[str, str]
    crawl_succeeded: bool
    # url -> markdown for the pages crawled this run. The RAG sync (M2) needs the content
    # of new/changed pages to chunk + embed them; unchanged pages' markdown is unused.
    pages: dict[str, str]


def detect_changes(
    urls: Sequence[str],
    manifest: Manifest,
    *,
    crawl: Crawler = crawl_urls,
) -> ChangeDetectionResult:
    """Crawl ``urls``, fingerprint them, and diff against the manifest. No mutation.

    ``crawl_succeeded`` is True only if every requested URL came back as a page. A
    shortfall means some pages failed to crawl, so downstream deletion is suppressed
    (Decision 5.5) -- we never treat a failed fetch as "the page was deleted." (This is
    a conservative signal; site discovery in a later milestone can refine it.)
    """
    pages = crawl(urls)
    markdown_by_url = {page.url: page.markdown for page in pages}
    current_hashes = {url: content_hash(md) for url, md in markdown_by_url.items()}
    previous_hashes = manifest.hashes()

    report = diff_hashes(previous_hashes, current_hashes)
    crawl_succeeded = len(pages) == len(urls)

    return ChangeDetectionResult(
        report=report,
        current_hashes=current_hashes,
        crawl_succeeded=crawl_succeeded,
        pages=markdown_by_url,
    )


def apply_changes(manifest: Manifest, result: ChangeDetectionResult) -> None:
    """Persist a detection result into the manifest.

    Upserts the hash of every new/changed page, and removes deleted pages -- but only
    the deletions that are safe to apply given whether the crawl fully succeeded.
    Unchanged pages are left untouched, so a no-change run writes nothing (idempotency,
    Decision 5.6).
    """
    for url in result.report.new | result.report.changed:
        manifest.upsert_page(url, result.current_hashes[url])

    for url in deletions_to_apply(result.report, crawl_succeeded=result.crawl_succeeded):
        manifest.delete_page(url)
