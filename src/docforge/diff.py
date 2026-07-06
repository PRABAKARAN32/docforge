"""Diffing: compare this crawl's hashes against the manifest's to find what changed.

This is the culmination of M1's change detection. Given two ``{url -> content_hash}``
maps -- ``previous`` (from the manifest, i.e. last run) and ``current`` (this crawl) --
it sorts every URL into one of four buckets:

    new        present now, absent before        -> embed it
    changed    present in both, hash differs      -> re-embed it
    deleted    present before, absent now         -> remove it (but see the guard)
    unchanged  present in both, same hash          -> do nothing (the common case)

The whole point of DocForge is that on a typical refresh, almost everything lands in
``unchanged`` and only a handful of pages need work.

Two separate concerns, kept separate on purpose:
  * :func:`diff_hashes` computes the *truth* -- a pure comparison of the two maps.
  * :func:`deletions_to_apply` applies *policy* -- Decision 5.5: never act on deletions
    unless the crawl fully succeeded, because a half-failed crawl makes good pages look
    "missing" and would otherwise wrongly delete their content.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class DiffReport:
    """The result of comparing two crawls: every URL sorted into one bucket."""

    new: frozenset[str]
    changed: frozenset[str]
    deleted: frozenset[str]
    unchanged: frozenset[str]

    @property
    def has_content_changes(self) -> bool:
        """True if anything needs processing (new or changed). Useful for idempotency:
        if this is False and there are no deletions, a sync can safely do nothing."""
        return bool(self.new or self.changed)


def diff_hashes(previous: Mapping[str, str], current: Mapping[str, str]) -> DiffReport:
    """Compare last run's hashes (``previous``) with this run's (``current``).

    Pure function: no I/O, deterministic. On a first-ever run ``previous`` is empty, so
    every page is ``new`` and nothing is ``deleted``.
    """
    previous_urls = set(previous)
    current_urls = set(current)

    new = current_urls - previous_urls
    deleted = previous_urls - current_urls
    common = previous_urls & current_urls

    changed = {url for url in common if previous[url] != current[url]}
    unchanged = common - changed

    return DiffReport(
        new=frozenset(new),
        changed=frozenset(changed),
        deleted=frozenset(deleted),
        unchanged=frozenset(unchanged),
    )


def deletions_to_apply(report: DiffReport, *, crawl_succeeded: bool) -> frozenset[str]:
    """Which deletions are safe to actually apply (Decision 5.5).

    If the crawl did not fully succeed, we return *no* deletions: pages that appear
    "missing" may simply not have been crawled, and deleting their content would be
    data loss. Only a clean, complete crawl is trusted to mean "these pages are truly
    gone."
    """
    if not crawl_succeeded:
        return frozenset()
    return report.deleted
