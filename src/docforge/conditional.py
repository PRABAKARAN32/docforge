"""Conditional HTTP requests (ETag / If-Modified-Since) for the 304 pre-check.

Before browser-crawling a page we've seen before, we ask the server cheaply (a plain HTTP
request, not the headless browser) whether it changed. If the server answers ``304 Not
Modified``, we skip the expensive render entirely.

This is the HTTP-layer step the browser can't do; keeping it here (stdlib ``urllib``) means
it's light and, via the injectable fetcher, testable without a network.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_USER_AGENT = "DocForge/0.1 (documentation sync bot; +https://github.com/DocForge)"


@dataclass(frozen=True)
class ConditionalResponse:
    """Outcome of a conditional request: the status and any validators the server sent."""

    status: int  # 304, 200, or 0 on network error
    etag: str | None
    last_modified: str | None

    @property
    def not_modified(self) -> bool:
        """True if the server confirmed the page is unchanged (304)."""
        return self.status == 304

    @property
    def has_validators(self) -> bool:
        """True if the server sent an ETag/Last-Modified (so 304 is possible next time)."""
        return self.etag is not None or self.last_modified is not None


# (url, etag, last_modified) -> ConditionalResponse. Real one below; tests inject a fake.
ConditionalFetcher = Callable[[str, "str | None", "str | None"], ConditionalResponse]


def http_conditional_get(
    url: str,
    etag: str | None,
    last_modified: str | None,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 30.0,
) -> ConditionalResponse:
    """Send a conditional GET; return the status + any validators from the response.

    Sends ``If-None-Match`` / ``If-Modified-Since`` when we have stored validators. A ``304``
    comes back as an ``HTTPError`` (urllib treats non-2xx that way) whose headers we still
    read. Any network error yields ``status=0`` so the caller falls back to a normal crawl.
    """
    request = urllib.request.Request(url, method="GET")
    request.add_header("User-Agent", user_agent)
    if etag:
        request.add_header("If-None-Match", etag)
    if last_modified:
        request.add_header("If-Modified-Since", last_modified)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return ConditionalResponse(
                response.status, response.headers.get("ETag"), response.headers.get("Last-Modified")
            )
    except urllib.error.HTTPError as exc:  # 304 (and other statuses) land here
        return ConditionalResponse(
            exc.code, exc.headers.get("ETag"), exc.headers.get("Last-Modified")
        )
    except Exception:  # noqa: BLE001 -- unknown -> let the caller crawl normally
        return ConditionalResponse(0, None, None)
