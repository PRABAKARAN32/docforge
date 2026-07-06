"""Tests for sitemap-based discovery.

The parser is pure (no I/O). For discover_urls we inject a fake fetcher -- a dict of
url -> xml -- so the discovery logic (including sitemap-index recursion and scope
filtering) is tested with no network. One opt-in test hits a real sitemap.
"""

import os
from urllib.parse import urlparse

import pytest

from docforge.discovery import Fetcher, discover_urls, parse_sitemap

_NETWORK = os.getenv("DOCFORGE_NETWORK_TESTS")

_NS = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'


def _urlset(*urls: str) -> str:
    locs = "".join(f"<url><loc>{u}</loc></url>" for u in urls)
    return f"<urlset {_NS}>{locs}</urlset>"


def _index(*sitemaps: str) -> str:
    locs = "".join(f"<sitemap><loc>{u}</loc></sitemap>" for u in sitemaps)
    return f"<sitemapindex {_NS}>{locs}</sitemapindex>"


def make_fetcher(pages: dict[str, str]) -> Fetcher:
    def _fetch(url: str) -> str:
        if url not in pages:
            raise FileNotFoundError(url)  # simulate a 404
        return pages[url]

    return _fetch


# --- parse_sitemap (pure) ---

def test_parse_urlset_returns_pages() -> None:
    pages, nested = parse_sitemap(_urlset("https://d/a", "https://d/b"))
    assert pages == ["https://d/a", "https://d/b"]
    assert nested == []


def test_parse_sitemapindex_returns_nested() -> None:
    pages, nested = parse_sitemap(_index("https://d/sm-1.xml", "https://d/sm-2.xml"))
    assert pages == []
    assert nested == ["https://d/sm-1.xml", "https://d/sm-2.xml"]


# --- discover_urls (fake fetcher) ---

def test_discover_from_simple_sitemap() -> None:
    site = {"https://d/sitemap.xml": _urlset("https://d/a", "https://d/b")}
    assert discover_urls("https://d/", fetch=make_fetcher(site)) == ["https://d/a", "https://d/b"]


def test_discover_follows_sitemap_index() -> None:
    site = {
        "https://d/sitemap.xml": _index("https://d/sm-1.xml", "https://d/sm-2.xml"),
        "https://d/sm-1.xml": _urlset("https://d/a"),
        "https://d/sm-2.xml": _urlset("https://d/b", "https://d/c"),
    }
    assert discover_urls("https://d/", fetch=make_fetcher(site)) == [
        "https://d/a",
        "https://d/b",
        "https://d/c",
    ]


def test_discover_filters_out_other_hosts() -> None:
    site = {
        "https://d/sitemap.xml": _urlset("https://d/a", "https://evil.com/x", "https://d/b"),
    }
    # Only same-host pages survive the scope boundary.
    assert discover_urls("https://d/", fetch=make_fetcher(site)) == ["https://d/a", "https://d/b"]


def test_discover_returns_empty_when_no_sitemap() -> None:
    assert discover_urls("https://d/", fetch=make_fetcher({})) == []


def test_discover_deduplicates() -> None:
    site = {
        "https://d/sitemap.xml": _index("https://d/sm-1.xml", "https://d/sm-2.xml"),
        "https://d/sm-1.xml": _urlset("https://d/a", "https://d/b"),
        "https://d/sm-2.xml": _urlset("https://d/b"),  # b appears twice
    }
    assert discover_urls("https://d/", fetch=make_fetcher(site)) == ["https://d/a", "https://d/b"]


# --- opt-in: hits a real sitemap over the network ---

@pytest.mark.skipif(not _NETWORK, reason="set DOCFORGE_NETWORK_TESTS=1 to run network tests")
def test_discover_from_a_real_sitemap() -> None:
    urls = discover_urls("https://www.sitemaps.org/")
    assert len(urls) > 10  # the real site has many pages
    assert all(urlparse(u).netloc == "www.sitemaps.org" for u in urls)  # scope held
