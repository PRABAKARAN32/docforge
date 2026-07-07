"""Tests for the CLI's run_sync, driven by injected fakes (no network/browser)."""

from docforge.cli import run_sync
from docforge.crawler import CrawledPage
from docforge.manifest import Manifest


def fake_discover(urls: list[str]):
    def _discover(seed: str, *, allow_bfs: bool = False, max_pages: int | None = None) -> list[str]:
        return urls

    return _discover


def fake_crawl(site: dict[str, str]):
    def _crawl(urls) -> list[CrawledPage]:
        return [CrawledPage(url=u, markdown=site[u]) for u in urls if u in site]

    return _crawl


def test_sync_reports_changes_and_updates_manifest(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    site = {"https://d/a": "# A", "https://d/b": "# B"}
    lines: list[str] = []

    code = run_sync(
        "https://d/",
        db_path=db,
        discover=fake_discover(list(site)),
        crawl=fake_crawl(site),
        out=lines.append,
    )

    assert code == 0
    assert any("Discovered 2 pages." in line for line in lines)
    assert any("2 new" in line for line in lines)
    assert any("Manifest updated." in line for line in lines)
    with Manifest(db) as m:
        assert set(m.hashes()) == set(site)


def test_dry_run_writes_nothing(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    site = {"https://d/a": "# A"}
    lines: list[str] = []

    code = run_sync(
        "https://d/",
        db_path=db,
        dry_run=True,
        discover=fake_discover(list(site)),
        crawl=fake_crawl(site),
        out=lines.append,
    )

    assert code == 0
    assert any("Dry run: no changes written." in line for line in lines)
    with Manifest(db) as m:
        assert m.hashes() == {}  # nothing persisted


def test_no_sitemap_without_bfs_warns_and_exits() -> None:
    lines: list[str] = []
    code = run_sync(
        "https://d/",
        db_path=":memory:",
        allow_bfs=False,
        discover=fake_discover([]),  # no pages found (no sitemap)
        crawl=fake_crawl({}),
        out=lines.append,
    )

    assert code == 1
    assert any("No sitemap found" in line for line in lines)
    assert any("--bfs" in line for line in lines)


def test_bfs_enabled_but_no_pages_found() -> None:
    lines: list[str] = []
    code = run_sync(
        "https://d/",
        db_path=":memory:",
        allow_bfs=True,
        discover=fake_discover([]),
        crawl=fake_crawl({}),
        out=lines.append,
    )

    assert code == 1
    assert any("No pages discovered." in line for line in lines)
