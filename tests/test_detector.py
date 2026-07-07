"""Tests for the change-detection orchestration.

A fake crawler (just a dict of url -> markdown) is injected, so these exercise the whole
crawl->hash->manifest->diff flow with zero network or browser. This is the payoff of
making the crawler injectable and CrawledPage our own type.
"""

from docforge.conditional import ConditionalResponse
from docforge.crawler import CrawledPage
from docforge.detector import Crawler, apply_changes, detect_changes
from docforge.manifest import Manifest


def const_conditional(mapping: dict[str, ConditionalResponse]):
    """Fake conditional fetcher: returns a canned response per URL (default 200, no validators)."""

    def _fetch(url: str, etag, last_modified) -> ConditionalResponse:
        return mapping.get(url, ConditionalResponse(200, None, None))

    return _fetch


def make_crawler(site: dict[str, str]) -> Crawler:
    """A fake crawler that returns pages for whichever requested URLs it 'has'."""

    def _crawl(urls):
        return [CrawledPage(url=u, markdown=site[u]) for u in urls if u in site]

    return _crawl


def crawler_with_validators(site: dict[str, str], *, etag=None, last_modified=None) -> Crawler:
    """Like make_crawler, but the returned pages carry HTTP validators (as a real crawl would)."""

    def _crawl(urls):
        return [
            CrawledPage(url=u, markdown=site[u], etag=etag, last_modified=last_modified)
            for u in urls
            if u in site
        ]

    return _crawl


def test_first_run_marks_everything_new_and_populates_manifest() -> None:
    site = {"https://d/a": "# A\n\nAlpha.", "https://d/b": "# B\n\nBeta."}
    with Manifest(":memory:") as m:
        result = detect_changes(list(site), m, crawl=make_crawler(site))
        assert result.report.new == frozenset(site)
        assert result.crawl_succeeded

        apply_changes(m, result)
        assert set(m.hashes()) == set(site)  # manifest now knows both pages


def test_second_identical_run_is_a_no_op() -> None:
    site = {"https://d/a": "# A\n\nAlpha."}
    with Manifest(":memory:") as m:
        apply_changes(m, detect_changes(list(site), m, crawl=make_crawler(site)))
        before = m.hashes()

        result = detect_changes(list(site), m, crawl=make_crawler(site))
        assert result.report.unchanged == frozenset(site)
        assert not result.report.has_content_changes

        apply_changes(m, result)
        assert m.hashes() == before  # nothing written the second time


def test_changed_page_is_detected_and_updated() -> None:
    with Manifest(":memory:") as m:
        v1 = {"https://d/a": "# A\n\nOriginal."}
        apply_changes(m, detect_changes(list(v1), m, crawl=make_crawler(v1)))
        original_hash = m.hashes()["https://d/a"]

        v2 = {"https://d/a": "# A\n\nEdited content."}
        result = detect_changes(list(v2), m, crawl=make_crawler(v2))
        assert result.report.changed == frozenset({"https://d/a"})

        apply_changes(m, result)
        assert m.hashes()["https://d/a"] != original_hash  # hash updated


def test_deleted_page_removed_when_crawl_succeeded() -> None:
    with Manifest(":memory:") as m:
        v1 = {"https://d/a": "A", "https://d/b": "B"}
        apply_changes(m, detect_changes(list(v1), m, crawl=make_crawler(v1)))

        # The site now only has page a; we crawl only [a], and it fully succeeds.
        v2 = {"https://d/a": "A"}
        result = detect_changes(list(v2), m, crawl=make_crawler(v2))
        assert result.report.deleted == frozenset({"https://d/b"})
        assert result.crawl_succeeded

        apply_changes(m, result)
        assert set(m.hashes()) == {"https://d/a"}  # b removed


def test_deletion_guard_keeps_pages_when_crawl_partially_failed() -> None:
    with Manifest(":memory:") as m:
        v1 = {"https://d/a": "A", "https://d/b": "B"}
        apply_changes(m, detect_changes(list(v1), m, crawl=make_crawler(v1)))

        # We still ASK for both [a, b], but the crawler only returns a (b failed).
        # b must NOT be deleted -- it only looks missing.
        broken_site = {"https://d/a": "A"}  # crawler 'has' only a
        result = detect_changes(["https://d/a", "https://d/b"], m, crawl=make_crawler(broken_site))
        assert result.report.deleted == frozenset({"https://d/b"})
        assert not result.crawl_succeeded  # asked for 2, got 1

        apply_changes(m, result)
        assert set(m.hashes()) == {"https://d/a", "https://d/b"}  # b preserved


# --- conditional (304) pre-check ---

def test_first_sync_makes_no_conditional_requests() -> None:
    # The whole point of the fix: with an empty manifest nothing can 304, so the pre-check
    # must issue ZERO requests (no silent wait before crawling).
    calls: list[str] = []

    def counting_conditional(url, etag, last_modified) -> ConditionalResponse:
        calls.append(url)
        return ConditionalResponse(200, None, None)

    site = {f"https://d/{i}": "x" for i in range(50)}
    with Manifest(":memory:") as m:
        detect_changes(list(site), m, crawl=make_crawler(site), conditional=counting_conditional)

    assert calls == []  # first sync -> no stored validators -> no pre-check requests at all




def test_304_page_is_skipped_and_stays_present() -> None:
    with Manifest(":memory:") as m:
        site = {"https://d/a": "aaa", "https://d/b": "bbb"}
        # First sync captures validators (as a real crawl would) so a 304 can happen next time.
        apply_changes(m, detect_changes(list(site), m, crawl=crawler_with_validators(site, etag='"v1"')))

        # Re-sync: server says page a is unchanged (304); page b is crawled and changed.
        cond = const_conditional({"https://d/a": ConditionalResponse(304, '"Ea"', None)})
        crawled: dict[str, list[str]] = {}

        def crawl(urls):
            crawled["urls"] = list(urls)
            return [CrawledPage(url="https://d/b", markdown="bbb-new")]

        result = detect_changes(list(site), m, crawl=crawl, conditional=cond)

        assert "https://d/a" not in crawled["urls"]  # 304 -> never crawled
        assert "https://d/b" in crawled["urls"]
        assert "https://d/a" in result.report.unchanged  # present & unchanged
        assert "https://d/a" not in result.report.deleted  # guard: not deleted
        assert "https://d/b" in result.report.changed


def test_conditional_supported_flag_reflects_validators() -> None:
    with Manifest(":memory:") as m:
        site = {"https://d/a": "aaa"}
        no_validators = const_conditional({})  # every response is 200 with no validators
        result = detect_changes(list(site), m, crawl=make_crawler(site), conditional=no_validators)
        assert result.conditional_supported is False


def test_new_validators_are_captured_from_the_crawl_and_stored() -> None:
    with Manifest(":memory:") as m:
        site = {"https://d/a": "aaa"}
        # Validators now come from the crawl response headers, not a separate pre-check.
        crawl = crawler_with_validators(site, etag='"E1"', last_modified="Mon")
        result = detect_changes(list(site), m, crawl=crawl, conditional=const_conditional({}))

        assert result.conditional_supported is True
        assert result.new_validators["https://d/a"] == ('"E1"', "Mon")
        apply_changes(m, result)
        assert m.validators()["https://d/a"] == ('"E1"', "Mon")
