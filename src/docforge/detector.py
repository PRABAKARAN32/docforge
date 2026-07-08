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
from dataclasses import dataclass, field

from docforge.conditional import ConditionalFetcher
from docforge.crawler import CrawledPage, crawl_urls
from docforge.diff import DiffReport, deletions_to_apply, diff_hashes
from docforge.hashing import content_hash
from docforge.manifest import Manifest

# A crawler is anything that takes URLs and returns crawled pages. The real one is
# crawl_urls; tests pass a fake. Typing it this way is what makes injection explicit.
Crawler = Callable[[Sequence[str]], list[CrawledPage]]

Validators = dict[str, tuple[str | None, str | None]]  # url -> (etag, last_modified)


@dataclass(frozen=True)
class ChangeDetectionResult:
    """The outcome of a detection run, before anything is written to the manifest."""

    report: DiffReport
    current_hashes: dict[str, str]
    crawl_succeeded: bool
    # url -> markdown for the pages crawled this run. The RAG sync (M2) needs the content
    # of new/changed pages to chunk + embed them; unchanged pages' markdown is unused.
    pages: dict[str, str]
    # Fresh HTTP validators captured this run (for changed/new pages), to store for next time.
    new_validators: Validators = field(default_factory=dict)
    # None if the 304 pre-check was off; else whether the site sent any validators at all.
    conditional_supported: bool | None = None


def _preselect(
    urls: Sequence[str],
    previous_hashes: dict[str, str],
    validators: Validators,
    conditional: ConditionalFetcher | None,
) -> tuple[list[str], dict[str, str], bool]:
    """Split URLs into those to crawl vs. those the server confirmed unchanged (304).

    A conditional request is only worth making for a URL we've **seen before and have a stored
    validator for** -- only those can possibly come back 304. New pages (and *every* page on a
    first sync, when nothing is stored yet) skip straight to crawling: no wasted requests, and
    the crawl itself captures fresh validators for next time. This is why a first sync no longer
    stalls doing thousands of pointless conditional GETs.
    """
    if conditional is None:
        return list(urls), {}, False

    to_crawl: list[str] = []
    unchanged_304: dict[str, str] = {}
    any_validators = False

    for url in urls:
        etag, last_modified = validators.get(url, (None, None))
        if (etag or last_modified) and url in previous_hashes:
            response = conditional(url, etag, last_modified)
            any_validators = any_validators or response.has_validators
            if response.not_modified:
                unchanged_304[url] = previous_hashes[url]
                continue
        to_crawl.append(url)

    return to_crawl, unchanged_304, any_validators


def detect_changes(
    urls: Sequence[str],
    manifest: Manifest,
    *,
    name: str = "default",
    crawl: Crawler = crawl_urls,
    conditional: ConditionalFetcher | None = None,
) -> ChangeDetectionResult:
    """Crawl ``urls``, fingerprint them, and diff against the manifest. No mutation.

    ``name`` is the knowledge base being synced -- the manifest is compared/scoped to it.

    If ``conditional`` is given, a cheap HTTP pre-check first skips pages the server reports
    as unchanged (304), so only new/changed pages are browser-crawled.

    ``crawl_succeeded`` is True only if every URL we *tried to crawl* came back. A shortfall
    means some pages failed, so downstream deletion is suppressed (Decision 5.5) -- we never
    treat a failed fetch as "the page was deleted." 304-skipped pages count as present.
    """
    previous_hashes = manifest.hashes(name)
    stored_validators = manifest.validators(name)

    to_crawl, unchanged_304, precheck_saw_validators = _preselect(
        urls, previous_hashes, stored_validators, conditional
    )

    pages = crawl(to_crawl)
    markdown_by_url = {page.url: page.markdown for page in pages}
    crawled_hashes = {url: content_hash(md) for url, md in markdown_by_url.items()}
    current_hashes = {**unchanged_304, **crawled_hashes}

    # Validators come from the crawl responses themselves (captured for next sync's pre-check).
    new_validators: Validators = {
        page.url: (page.etag, page.last_modified)
        for page in pages
        if page.etag or page.last_modified
    }

    report = diff_hashes(previous_hashes, current_hashes)
    crawl_succeeded = len(pages) == len(to_crawl)

    conditional_supported = None
    if conditional is not None:
        conditional_supported = bool(new_validators) or precheck_saw_validators

    return ChangeDetectionResult(
        report=report,
        current_hashes=current_hashes,
        crawl_succeeded=crawl_succeeded,
        pages=markdown_by_url,
        new_validators=new_validators,
        conditional_supported=conditional_supported,
    )


def apply_changes(manifest: Manifest, result: ChangeDetectionResult, *, name: str = "default") -> None:
    """Persist a detection result into knowledge base ``name`` in the manifest.

    Upserts the hash of every new/changed page (with any fresh HTTP validators), and removes
    deleted pages -- but only the deletions that are safe to apply given whether the crawl
    fully succeeded. Unchanged pages are left untouched, so a no-change run writes nothing
    (idempotency, Decision 5.6).
    """
    for url in result.report.new | result.report.changed:
        etag, last_modified = result.new_validators.get(url, (None, None))
        manifest.upsert_page(
            name, url, result.current_hashes[url], etag=etag, last_modified=last_modified
        )

    for url in deletions_to_apply(result.report, crawl_succeeded=result.crawl_succeeded):
        manifest.delete_page(name, url)
