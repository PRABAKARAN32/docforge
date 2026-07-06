"""Tests for the crawler.

Crawling only does anything by driving a real browser against the real network, so
this is an *integration* test, not a unit test. It is opt-in: it runs only when the
env var DOCFORGE_NETWORK_TESTS is set, so CI (no browser system libs, restricted
network) skips it and stays green. Run it locally with:

    DOCFORGE_NETWORK_TESTS=1 uv run pytest -k crawler
"""

import os

import pytest

from docforge.crawler import CrawledPage, crawl_urls

_NETWORK = os.getenv("DOCFORGE_NETWORK_TESTS")


@pytest.mark.skipif(not _NETWORK, reason="set DOCFORGE_NETWORK_TESTS=1 to run network tests")
def test_crawl_urls_returns_markdown_for_a_real_page() -> None:
    pages = crawl_urls(["https://example.com"])

    assert len(pages) == 1
    page = pages[0]
    assert isinstance(page, CrawledPage)
    assert page.url.startswith("https://example.com")
    # example.com always contains this heading; proves we got real cleaned markdown.
    assert "Example Domain" in page.markdown
