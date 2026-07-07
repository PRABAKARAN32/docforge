"""Tests for the CLI's run_sync, driven by injected fakes (no network/browser).

The vector store is a real in-process Qdrant (`location=":memory:"`) and the embedder is a
tiny fake, so the full sync path runs with no Docker and no model download.
"""

from collections.abc import Sequence

import pytest

from docforge.chunking import Chunk
from docforge.cli import _build_parser, run_diff, run_remove, run_search, run_status, run_sync
from docforge.crawler import CrawledPage
from docforge.manifest import Manifest
from docforge.vectorstore import QdrantVectorStore


class FakeEmbedder:
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


def memory_store() -> QdrantVectorStore:
    from qdrant_client import QdrantClient

    return QdrantVectorStore(client=QdrantClient(location=":memory:"), collection="test")


def fake_discover(urls: list[str]):
    def _discover(seed: str, *, allow_bfs: bool = False, max_pages: int | None = None) -> list[str]:
        return urls

    return _discover


def fake_crawl(site: dict[str, str]):
    def _crawl(urls) -> list[CrawledPage]:
        return [CrawledPage(url=u, markdown=site[u]) for u in urls if u in site]

    return _crawl


def test_sync_reports_changes_updates_manifest_and_embeds(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    site = {"https://d/a": "# A\n\nAlpha.", "https://d/b": "# B\n\nBeta."}
    store = memory_store()
    lines: list[str] = []

    code = run_sync(
        "https://d/",
        db_path=db,
        discover=fake_discover(list(site)),
        crawl=fake_crawl(site),
        conditional=None,  # disable the 304 pre-check (no network in tests)
        embedder=FakeEmbedder(),
        store=store,
        out=lines.append,
    )

    assert code == 0
    assert any("Discovered 2 pages." in line for line in lines)
    assert any("2 new" in line for line in lines)
    assert any("Embedding 2 changed page(s)" in line for line in lines)
    assert any("Manifest updated." in line for line in lines)
    with Manifest(db) as m:
        assert set(m.hashes()) == set(site)
    assert store.count() > 0  # chunks were embedded into the vector store


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
        conditional=None,  # disable the 304 pre-check (no network in tests)
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


def test_parser_accepts_qdrant_path_and_embed_model() -> None:
    args = _build_parser().parse_args(
        ["sync", "https://d/", "--qdrant-path", "./vec", "--embed-model", "BAAI/bge-base-en-v1.5"]
    )
    assert args.qdrant_path == "./vec"
    assert args.embed_model == "BAAI/bge-base-en-v1.5"
    assert args.qdrant_url == "http://localhost:6333"  # default still present


def test_parser_accepts_concurrency() -> None:
    args = _build_parser().parse_args(["sync", "https://d/", "--concurrency", "12"])
    assert args.concurrency == 12


def test_parser_accepts_device() -> None:
    args = _build_parser().parse_args(["sync", "https://d/", "--device", "cpu"])
    assert args.device == "cpu"


def test_parser_rejects_unknown_device() -> None:
    with pytest.raises(SystemExit):  # argparse choices= rejects invalid values
        _build_parser().parse_args(["sync", "https://d/", "--device", "tpu"])


def test_parser_accepts_conditional_and_force() -> None:
    args = _build_parser().parse_args(["sync", "https://d/", "--conditional", "off", "--force"])
    assert args.conditional == "off"
    assert args.force is True
    # defaults
    assert _build_parser().parse_args(["sync", "https://d/"]).conditional == "auto"


def test_parser_supports_all_subcommands() -> None:
    parser = _build_parser()
    assert parser.parse_args(["diff", "https://d/"]).command == "diff"
    assert parser.parse_args(["status"]).command == "status"
    assert parser.parse_args(["search", "hello"]).command == "search"
    assert parser.parse_args(["remove", "d.com"]).command == "remove"


# --- status ---

def test_status_reports_pages_per_site(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    with Manifest(db) as m:
        m.upsert_page("https://a.com/x", "h1")
        m.upsert_page("https://a.com/y", "h2")
        m.upsert_page("https://b.com/z", "h3")
    lines: list[str] = []

    assert run_status(db_path=db, out=lines.append) == 0
    assert any("3 pages tracked" in line for line in lines)
    assert any("a.com" in line for line in lines)
    assert any("b.com" in line for line in lines)


def test_status_empty_db(tmp_path) -> None:
    lines: list[str] = []
    assert run_status(db_path=str(tmp_path / "empty.db"), out=lines.append) == 0
    assert any("No pages tracked" in line for line in lines)


# --- diff ---

def test_diff_lists_changes_and_writes_nothing(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    site = {"https://d/a": "A", "https://d/b": "B"}
    lines: list[str] = []

    code = run_diff(
        "https://d/",
        db_path=db,
        discover=fake_discover(list(site)),
        crawl=fake_crawl(site),
        conditional=None,
        out=lines.append,
    )

    assert code == 0
    assert any("Would change: 2 new" in line for line in lines)
    assert any("+ new" in line and "https://d/a" in line for line in lines)
    with Manifest(db) as m:
        assert m.hashes() == {}  # diff never writes


# --- search ---

def test_search_prints_hits() -> None:
    store = memory_store()
    store.ensure_collection(4)
    store.upsert_chunks([Chunk("https://d/a", 0, "install with pip")], [[1.0, 0.0, 0.0, 0.0]])
    lines: list[str] = []

    code = run_search("how to install", embedder=FakeEmbedder(), store=store, out=lines.append)

    assert code == 0
    assert any("https://d/a" in line for line in lines)


# --- remove ---

def test_remove_deletes_from_manifest_and_store(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    store = memory_store()
    store.ensure_collection(4)
    with Manifest(db) as m:
        m.upsert_page("https://a.com/x", "h1")
        m.upsert_page("https://b.com/y", "h2")
    store.upsert_chunks([Chunk("https://a.com/x", 0, "t")], [[1.0, 0.0, 0.0, 0.0]])
    lines: list[str] = []

    code = run_remove("a.com", db_path=db, store=store, out=lines.append)

    assert code == 0
    with Manifest(db) as m:
        assert set(m.hashes()) == {"https://b.com/y"}  # a.com removed, b.com kept
    assert store.count() == 0  # a.com's chunk removed
    assert any("Removed 1 page" in line for line in lines)


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
