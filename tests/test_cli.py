"""Tests for the CLI's run_sync, driven by injected fakes (no network/browser).

The vector store is a real in-process Qdrant (`location=":memory:"`) and the embedder is a
tiny fake, so the full sync path runs with no Docker and no model download.
"""

import os
from collections.abc import Sequence

import pytest

from docforge.chunking import Chunk
from docforge.cli import (
    _build_parser,
    _load_dotenv,
    run_diff,
    run_remove,
    run_search,
    run_status,
    run_sync,
)
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
        name="docker",
        db_path=db,
        discover=fake_discover(list(site)),
        crawl=fake_crawl(site),
        conditional=None,  # disable the 304 pre-check (no network in tests)
        embedder=FakeEmbedder(),
        store=store,
        out=lines.append,
    )

    assert code == 0
    assert any("Knowledge base: docker" in line for line in lines)
    assert any("Discovered 2 pages." in line for line in lines)
    assert any("2 new" in line for line in lines)
    assert any("Embedding 2 changed page(s)" in line for line in lines)
    assert any("Manifest updated." in line for line in lines)
    with Manifest(db) as m:
        assert set(m.hashes("docker")) == set(site)
    assert store.count() > 0  # chunks were embedded into the vector store


def test_dry_run_writes_nothing(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    site = {"https://d/a": "# A"}
    lines: list[str] = []

    code = run_sync(
        "https://d/",
        name="docker",
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
        assert m.names() == {}  # nothing persisted


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
    assert args.qdrant_url is None  # parser default is None; main resolves flag>env>localhost


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


def test_parser_accepts_crawl_delay_and_no_rate_limit() -> None:
    args = _build_parser().parse_args(
        ["sync", "https://d/", "--crawl-delay", "0.2", "0.6", "--no-rate-limit"]
    )
    assert args.crawl_delay == [0.2, 0.6]
    assert args.no_rate_limit is True
    # defaults
    plain = _build_parser().parse_args(["sync", "https://d/"])
    assert plain.crawl_delay is None
    assert plain.no_rate_limit is False


def test_parser_accepts_qdrant_api_key() -> None:
    args = _build_parser().parse_args(["sync", "https://d/", "--qdrant-api-key", "secret"])
    assert args.qdrant_api_key == "secret"
    # search and remove also accept it
    assert _build_parser().parse_args(
        ["search", "q", "--qdrant-api-key", "k"]
    ).qdrant_api_key == "k"


def test_parser_supports_all_subcommands() -> None:
    parser = _build_parser()
    assert parser.parse_args(["diff", "https://d/"]).command == "diff"
    assert parser.parse_args(["status"]).command == "status"
    assert parser.parse_args(["list"]).command == "list"
    assert parser.parse_args(["search", "hello"]).command == "search"
    assert parser.parse_args(["remove", "docker"]).command == "remove"


def test_parser_sync_accepts_name() -> None:
    args = _build_parser().parse_args(["sync", "https://d/", "--name", "docker"])
    assert args.name == "docker"
    assert _build_parser().parse_args(["sync", "https://d/"]).name is None


# --- status / list ---

def test_status_lists_knowledge_bases(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    with Manifest(db) as m:
        m.upsert_page("docker", "https://a.com/x", "h1")
        m.upsert_page("docker", "https://a.com/y", "h2")
        m.upsert_page("nginx", "https://b.com/z", "h3")
    lines: list[str] = []

    assert run_status(db_path=db, out=lines.append) == 0
    assert any("3 pages across 2 knowledge base(s)" in line for line in lines)
    assert any("docker" in line for line in lines)
    assert any("nginx" in line for line in lines)


def test_status_empty_db(tmp_path) -> None:
    lines: list[str] = []
    assert run_status(db_path=str(tmp_path / "empty.db"), out=lines.append) == 0
    assert any("No knowledge bases tracked" in line for line in lines)


# --- diff ---

def test_diff_lists_changes_and_writes_nothing(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    site = {"https://d/a": "A", "https://d/b": "B"}
    lines: list[str] = []

    code = run_diff(
        "https://d/",
        name="docker",
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
        assert m.names() == {}  # diff never writes


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

def test_remove_deletes_a_whole_knowledge_base(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    store = memory_store()
    store.ensure_collection(4)
    with Manifest(db) as m:
        m.upsert_page("docker", "https://a.com/x", "h1")
        m.upsert_page("nginx", "https://b.com/y", "h2")
    store.upsert_chunks([Chunk("https://a.com/x", 0, "t")], [[1.0, 0.0, 0.0, 0.0]])
    lines: list[str] = []

    code = run_remove("docker", db_path=db, store=store, out=lines.append)

    assert code == 0
    with Manifest(db) as m:
        assert m.names() == {"nginx": 1}  # docker KB removed, nginx kept
    assert any("Removed knowledge base 'docker'" in line for line in lines)


def test_remove_unknown_kb_is_a_noop(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    lines: list[str] = []
    code = run_remove("nope", db_path=db, store=memory_store(), out=lines.append)
    assert code == 0
    assert any("No knowledge base named" in line for line in lines)


def test_remove_without_name_or_all_errors() -> None:
    lines: list[str] = []
    code = run_remove(None, db_path=":memory:", store=memory_store(), out=lines.append)
    assert code == 1
    assert any("--all" in line for line in lines)


def test_remove_all_wipes_manifest_and_deletes_db(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    store = memory_store()
    store.ensure_collection(4)
    with Manifest(db) as m:
        m.upsert_page("docker", "https://d/x", "h1")
        m.upsert_page("nginx", "https://n/y", "h2")
    lines: list[str] = []

    code = run_remove(remove_all=True, db_path=db, store=store, out=lines.append)

    assert code == 0
    assert not os.path.exists(db)  # the manifest DB file is deleted
    assert any("Removed everything" in line for line in lines)


def test_parser_remove_accepts_name_or_all() -> None:
    assert _build_parser().parse_args(["remove", "docker"]).name == "docker"
    all_args = _build_parser().parse_args(["remove", "--all"])
    assert all_args.name is None
    assert all_args.remove_all is True


# --- .env loading ---

def test_load_dotenv_sets_missing_but_not_existing(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        'QDRANT_URL="https://x.cloud.qdrant.io:6333"\n'
        "export DOCFORGE_DB=SqlDB/docforge.db\n"
        "QDRANT_API_KEY=from_file\n"
    )
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("DOCFORGE_DB", raising=False)
    monkeypatch.setenv("QDRANT_API_KEY", "real_env")  # a real env var must WIN over .env

    _load_dotenv(str(env_file))

    assert os.environ["QDRANT_URL"] == "https://x.cloud.qdrant.io:6333"  # quotes stripped
    assert os.environ["DOCFORGE_DB"] == "SqlDB/docforge.db"  # export prefix handled
    assert os.environ["QDRANT_API_KEY"] == "real_env"  # existing env not overridden


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
