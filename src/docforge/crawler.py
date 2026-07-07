"""Crawling: turn documentation URLs into clean markdown.

This is a *thin wrapper* around Crawl4AI (Decision 5.1 in GUID.md: crawling is a
solved problem — we don't reinvent it). Its only job is to adapt Crawl4AI's rich
result objects into a small, stable shape (`CrawledPage`) that the rest of DocForge
depends on. If we ever swap the crawler, only this file changes.

Crawls an explicit list of URLs concurrently (via Crawl4AI's ``arun_many`` + a memory-aware
dispatcher and rate limiter). Whole-site discovery lives in ``discovery.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)

# We identify ourselves honestly instead of pretending to be a human browser.
# Ethical crawling 101: say who you are so site owners can see/contact the bot.
DEFAULT_USER_AGENT = "DocForge/0.1 (documentation sync bot; +https://github.com/DocForge)"

# Modest default so we're fast but polite -- a handful of pages in flight, not hundreds.
DEFAULT_CONCURRENCY = 5

# Called after each page is attempted, as on_page(done, total, url) -> so a caller (the CLI)
# can show live progress. Kept as an injected callback so the crawler stays UI-agnostic.
ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class CrawledPage:
    """One successfully crawled page, reduced to just what DocForge needs.

    `url` is the page's address (used later as the stable `source_url` that links
    every chunk back to its page — Decision 5.3). `markdown` is Crawl4AI's cleaned
    markdown, before DocForge's own normalization/hashing.
    """

    url: str
    markdown: str


async def crawl_urls_async(
    urls: Sequence[str],
    *,
    respect_robots_txt: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    concurrency: int = DEFAULT_CONCURRENCY,
    on_page: ProgressCallback | None = None,
) -> list[CrawledPage]:
    """Crawl the URLs concurrently and return the pages that succeeded.

    Uses Crawl4AI's ``arun_many`` with a ``MemoryAdaptiveDispatcher`` (caps concurrency by
    RAM so many browser tabs can't OOM the machine) and a ``RateLimiter`` (polite delays +
    exponential backoff on 429/503). ``concurrency`` is the max pages in flight — kept modest
    by default so we're fast but not abusive. Streaming is on, so results arrive as each page
    finishes, driving ``on_page(done, total, url)`` live even while others crawl.

    Ethical defaults (see Notes/M1 on crawling ethics):
      * ``respect_robots_txt=True`` — robots-disallowed URLs are reported failed and skipped.
      * an honest ``user_agent`` identifying DocForge.

    Failed pages (including robots-blocked ones) are skipped rather than raising, so one bad
    URL doesn't abort the run. Callers that guard deletions (Decision 5.5) should compare the
    returned URL set against what they expected.
    """
    browser_config = BrowserConfig(user_agent=user_agent, verbose=False)
    run_config = CrawlerRunConfig(check_robots_txt=respect_robots_txt, verbose=False, stream=True)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        max_session_permit=concurrency,
        rate_limiter=RateLimiter(
            base_delay=(1.0, 2.0),          # polite random pause between requests
            max_delay=30.0,                 # backoff ceiling
            max_retries=3,
            rate_limit_codes=[429, 503],    # slow down when the server pushes back
        ),
    )

    total = len(urls)
    done = 0
    pages: list[CrawledPage] = []
    async with AsyncWebCrawler(config=browser_config) as crawler:
        async for result in await crawler.arun_many(
            list(urls), config=run_config, dispatcher=dispatcher
        ):
            done += 1
            if result.success:
                pages.append(CrawledPage(url=result.url, markdown=str(result.markdown)))
            if on_page is not None:
                on_page(done, total, getattr(result, "url", ""))
    return pages


def crawl_urls(
    urls: Sequence[str],
    *,
    respect_robots_txt: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    concurrency: int = DEFAULT_CONCURRENCY,
    on_page: ProgressCallback | None = None,
) -> list[CrawledPage]:
    """Synchronous convenience wrapper around :func:`crawl_urls_async`."""
    return asyncio.run(
        crawl_urls_async(
            urls,
            respect_robots_txt=respect_robots_txt,
            user_agent=user_agent,
            concurrency=concurrency,
            on_page=on_page,
        )
    )


def _internal_hrefs(links: object) -> list[str]:
    """Pull the 'internal' link hrefs out of a Crawl4AI result's links.

    Robust to the links being either a model (``.internal`` -> objects with ``.href``)
    or a plain dict (``{"internal": [{"href": ...}]}``).
    """
    internal = getattr(links, "internal", None)
    if internal is None and isinstance(links, dict):
        internal = links.get("internal", [])
    hrefs: list[str] = []
    for item in internal or []:
        href = getattr(item, "href", None)
        if href is None and isinstance(item, dict):
            href = item.get("href")
        if href:
            hrefs.append(href)
    return hrefs


async def fetch_page_links_async(
    url: str,
    *,
    respect_robots_txt: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[str]:
    """Crawl one page and return the internal (same-doc-site) links it contains.

    Used by BFS discovery to walk a site that has no sitemap. Returns ``[]`` if the
    page fails to crawl (including robots-blocked), so a bad page doesn't abort the walk.
    """
    browser_config = BrowserConfig(user_agent=user_agent, verbose=False)
    run_config = CrawlerRunConfig(check_robots_txt=respect_robots_txt, verbose=False)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        if not result.success:
            return []
        return _internal_hrefs(result.links)


def fetch_page_links(
    url: str,
    *,
    respect_robots_txt: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[str]:
    """Synchronous wrapper around :func:`fetch_page_links_async`."""
    return asyncio.run(
        fetch_page_links_async(url, respect_robots_txt=respect_robots_txt, user_agent=user_agent)
    )
