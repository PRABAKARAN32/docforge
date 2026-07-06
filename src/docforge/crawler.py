"""Crawling: turn documentation URLs into clean markdown.

This is a *thin wrapper* around Crawl4AI (Decision 5.1 in GUID.md: crawling is a
solved problem — we don't reinvent it). Its only job is to adapt Crawl4AI's rich
result objects into a small, stable shape (`CrawledPage`) that the rest of DocForge
depends on. If we ever swap the crawler, only this file changes.

Scope (M1, slice 1): crawl an explicit list of URLs. Whole-site discovery
(sitemap / deep crawl) is a deliberate follow-up, kept out to keep this slice small.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from crawl4ai import AsyncWebCrawler


@dataclass(frozen=True)
class CrawledPage:
    """One successfully crawled page, reduced to just what DocForge needs.

    `url` is the page's address (used later as the stable `source_url` that links
    every chunk back to its page — Decision 5.3). `markdown` is Crawl4AI's cleaned
    markdown, before DocForge's own normalization/hashing.
    """

    url: str
    markdown: str


async def crawl_urls_async(urls: Sequence[str]) -> list[CrawledPage]:
    """Crawl each URL and return the pages that succeeded.

    Failed pages are skipped rather than raising, so one bad URL doesn't abort the
    whole run. Callers that need to guard deletions (Decision 5.5: never delete on a
    partial crawl) should compare the returned URL set against what they expected.
    """
    pages: list[CrawledPage] = []
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            result = await crawler.arun(url=url)
            if result.success:
                pages.append(CrawledPage(url=result.url, markdown=str(result.markdown)))
    return pages


def crawl_urls(urls: Sequence[str]) -> list[CrawledPage]:
    """Synchronous convenience wrapper around :func:`crawl_urls_async`."""
    return asyncio.run(crawl_urls_async(urls))
