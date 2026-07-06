"""Discovery: turn one seed URL into the full list of a site's page URLs.

`detect_changes` needs the list of pages to crawl. In the real world a user gives us
*one* URL (``docforge sync https://docs.docker.com/``) and expects us to find the rest.
This module does that discovery.

Strategy: **sitemap-first.** Most docs sites publish ``/sitemap.xml`` -- a machine-readable
list of every page. Reading it is faster, more complete (it catches "orphan" pages nothing
links to), and more polite (a couple of requests instead of rendering the whole site) than
crawling link-by-link. (BFS link-following, for sites with no sitemap, is a documented
follow-up -- see Notes/M1/02.)

Design: the XML parser is a *pure function* (testable with no network); the fetching is
injected via the ``fetch`` parameter so the discovery logic is unit-testable with a fake.
Kept dependency-free (stdlib ``urllib`` + ``xml.etree``), so importing this module does not
drag in the heavy browser stack.
"""

from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from urllib.parse import urlparse

# A fetcher takes a URL and returns the response body as text. Real one below; fake in tests.
Fetcher = Callable[[str], str]

DEFAULT_USER_AGENT = "DocForge/0.1 (documentation sync bot; +https://github.com/DocForge)"


def _localname(tag: str) -> str:
    """Return an XML tag's local name, ignoring any ``{namespace}`` prefix.

    Sitemap elements are namespaced (``{http://www.sitemaps.org/...}loc``); we match on the
    local name so the parser doesn't depend on the exact namespace URL.
    """
    return tag.rsplit("}", 1)[-1]


def parse_sitemap(xml_text: str) -> tuple[list[str], list[str]]:
    """Parse sitemap XML into ``(page_urls, nested_sitemap_urls)``.

    A ``<sitemapindex>`` points to *other* sitemaps -> its ``<loc>`` entries are nested
    sitemaps. A ``<urlset>`` lists pages -> its ``<loc>`` entries are page URLs. Returns
    both lists so the caller can recurse into nested sitemaps.
    """
    root = ET.fromstring(xml_text)
    locs = [e.text.strip() for e in root.iter() if _localname(e.tag) == "loc" and e.text]
    if _localname(root.tag) == "sitemapindex":
        return [], locs
    return locs, []


def _http_get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 (http(s) only)
        return response.read().decode("utf-8")


def discover_urls(
    seed_url: str,
    *,
    fetch: Fetcher = _http_get,
    max_sitemaps: int = 100,
) -> list[str]:
    """Discover a site's page URLs from its sitemap, starting at ``seed_url``.

    Fetches ``<host>/sitemap.xml``, following ``<sitemapindex>`` entries into nested
    sitemaps (bounded by ``max_sitemaps``). Results are restricted to the seed's host
    (the scope boundary, so we don't wander onto other domains) and returned sorted and
    de-duplicated. Sitemaps that fail to fetch or parse are skipped rather than aborting
    the whole discovery. Returns ``[]`` if no readable sitemap is found.
    """
    seed = urlparse(seed_url)
    base = f"{seed.scheme}://{seed.netloc}"

    to_fetch = [f"{base}/sitemap.xml"]
    seen_sitemaps: set[str] = set()
    pages: set[str] = set()

    while to_fetch and len(seen_sitemaps) < max_sitemaps:
        sitemap_url = to_fetch.pop()
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)

        try:
            page_urls, nested = parse_sitemap(fetch(sitemap_url))
        except Exception:  # noqa: BLE001 -- a bad sitemap must not kill discovery
            continue

        pages.update(page_urls)
        to_fetch.extend(url for url in nested if url not in seen_sitemaps)

    # Scope boundary: keep only pages on the same host as the seed.
    in_scope = {url for url in pages if urlparse(url).netloc == seed.netloc}
    return sorted(in_scope)
