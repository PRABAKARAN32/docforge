"""Discovery: turn one seed URL into the full list of a site's page URLs.

`detect_changes` needs the list of pages to crawl. In the real world a user gives us
*one* URL (``docforge sync https://docs.docker.com/``) and expects us to find the rest.
This module does that discovery.

Strategy: **sitemap-first, BFS fallback.** Most docs sites publish ``/sitemap.xml`` -- a
machine-readable list of every page. Reading it is faster, more complete (it catches
"orphan" pages nothing links to), and more polite (a couple of requests instead of rendering
the whole site) than crawling link-by-link. For sites with *no* sitemap (e.g. nginx.org), we
fall back to a BFS crawl that walks the site's links -- but only when the caller opts in
(``allow_bfs``), since it is much heavier.

Design: the XML parser is a *pure function* (testable with no network); the fetching is
injected via the ``fetch`` parameter so the discovery logic is unit-testable with a fake.
Kept dependency-free (stdlib ``urllib`` + ``xml.etree``), so importing this module does not
drag in the heavy browser stack.
"""

from __future__ import annotations

import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Callable
from urllib.parse import urldefrag, urlparse

# A fetcher takes a URL and returns the response body as text. Real one below; fake in tests.
Fetcher = Callable[[str], str]

# A link-fetcher takes a page URL and returns the links found on it. The real one renders
# the page (crawler.fetch_page_links); tests pass a fake link-graph. This is the injected
# commodity that makes the BFS walk itself testable without a network/browser.
LinkFetcher = Callable[[str], list[str]]

DEFAULT_USER_AGENT = "DocForge/0.1 (documentation sync bot; +https://github.com/DocForge)"


def derive_kb_name(url: str) -> str:
    """Derive a default knowledge-base / collection name from a URL's host.

    ``https://docs.docker.com/`` -> ``docs_docker_com``. Sanitized to a safe collection name
    (lowercase, ``[a-z0-9_]``), so it's valid for both SQLite scoping and Qdrant.
    """
    host = urlparse(url).netloc or url
    name = re.sub(r"[^a-z0-9]+", "_", host.lower()).strip("_")
    return name or "docs"


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


def discover_via_sitemap(
    seed_url: str,
    *,
    fetch: Fetcher = _http_get,
    max_sitemaps: int = 100,
) -> list[str]:
    """Discover page URLs from the site's sitemap. Returns ``[]`` if there's no sitemap.

    Fetches ``<host>/sitemap.xml``, following ``<sitemapindex>`` entries into nested
    sitemaps (bounded by ``max_sitemaps``). Results are restricted to the seed's host and
    returned sorted and de-duplicated. Sitemaps that fail to fetch or parse are skipped
    rather than aborting discovery.
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

    in_scope = {url for url in pages if urlparse(url).netloc == seed.netloc}
    return sorted(in_scope)


def bfs_discover_urls(
    seed_url: str,
    *,
    fetch_links: LinkFetcher,
    max_pages: int | None = None,
) -> list[str]:
    """Discover page URLs by walking links from the seed (BFS), for sites with no sitemap.

    This is DocForge's own graph-walk (Decision 5.2): a queue + a visited set, scoped to
    the seed's host, bounded by ``max_pages`` (``None`` = unbounded). The "get links on a
    page" step is injected (``fetch_links``) -- the real one renders the page, but tests
    pass a fake link-graph, so this logic is verified with no network.

    The visited set makes it terminate (A->B->A loops can't spin forever); the host scope
    stops it wandering onto other domains; ``max_pages`` caps runaway crawls. URLs are
    compared with their ``#fragment`` stripped so ``/a`` and ``/a#section`` aren't treated
    as two pages.
    """
    host = urlparse(seed_url).netloc
    queue: deque[str] = deque([urldefrag(seed_url).url])
    visited: set[str] = set()
    discovered: list[str] = []

    while queue:
        if max_pages is not None and len(discovered) >= max_pages:
            break
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        if urlparse(url).netloc != host:
            continue
        discovered.append(url)

        try:
            links = fetch_links(url)
        except Exception:  # noqa: BLE001 -- one bad page must not kill the walk
            links = []
        for raw_link in links:
            link = urldefrag(raw_link).url
            if link not in visited and urlparse(link).netloc == host:
                queue.append(link)

    return sorted(discovered)


def discover_urls(
    seed_url: str,
    *,
    fetch: Fetcher = _http_get,
    allow_bfs: bool = False,
    fetch_links: LinkFetcher | None = None,
    max_pages: int | None = None,
    max_sitemaps: int = 100,
) -> list[str]:
    """Discover a site's page URLs: sitemap first, BFS crawl as a fallback.

    Order:
      1. Try the sitemap (fast, complete, polite). If it yields pages, use them.
      2. Otherwise, only if ``allow_bfs`` is True, crawl page-by-page from the seed.
         ``allow_bfs`` defaults False so we never silently launch a heavy crawl -- the
         caller (the CLI, via ``--bfs``) must opt in.

    ``max_pages`` caps the result for *either* path (``None`` = no limit). ``fetch_links``
    lets tests inject a fake link-graph; in real use it defaults to the Crawl4AI-backed
    page link fetcher (imported lazily so this module stays light unless BFS is used).
    """
    sitemap_pages = discover_via_sitemap(seed_url, fetch=fetch, max_sitemaps=max_sitemaps)
    if sitemap_pages:
        return sitemap_pages[:max_pages] if max_pages is not None else sitemap_pages

    if not allow_bfs:
        return []

    if fetch_links is None:
        from docforge.crawler import fetch_page_links  # lazy: avoids the heavy import path

        fetch_links = fetch_page_links
    return bfs_discover_urls(seed_url, fetch_links=fetch_links, max_pages=max_pages)
