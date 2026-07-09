"""Tests for the MCP server: pure logic (list_docs_text/search_docs_text) with fakes, plus a
build_server smoke test using the real (in-process) mcp SDK -- no network, no model download.
"""

import asyncio
from collections.abc import Sequence

from docforge.chunking import Chunk
from docforge.manifest import Manifest
from docforge.mcp_server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    _build_parser,
    build_server,
    list_docs_text,
    search_docs_text,
)
from docforge.vectorstore import QdrantVectorStore


class FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


def memory_store(collection: str = "test") -> QdrantVectorStore:
    from qdrant_client import QdrantClient

    return QdrantVectorStore(client=QdrantClient(location=":memory:"), collection=collection)


# --- list_docs_text ---


def test_list_docs_text_lists_knowledge_bases(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    with Manifest(db) as m:
        m.upsert_page("docker", "https://a.com/x", "h1")
        m.upsert_page("docker", "https://a.com/y", "h2")
        m.upsert_page("nginx", "https://b.com/z", "h3")

    text = list_docs_text(db)

    assert "docker" in text
    assert "nginx" in text
    assert "2 pages" in text
    assert "1 pages" in text


def test_list_docs_text_empty_db(tmp_path) -> None:
    text = list_docs_text(str(tmp_path / "empty.db"))
    assert "No knowledge bases tracked" in text


# --- search_docs_text ---


def test_search_docs_text_returns_hits() -> None:
    store = memory_store()
    store.ensure_collection(4)
    store.upsert_chunks([Chunk("https://d/a", 0, "install with pip")], [[1.0, 0.0, 0.0, 0.0]])

    text = search_docs_text(
        "docker", "how to install",
        qdrant_url="unused", qdrant_path=None, qdrant_api_key=None, qdrant_timeout=60.0,
        embed_model="unused", embedder=FakeEmbedder(), store=store,
    )

    assert "https://d/a" in text
    assert "install with pip" in text


def test_search_docs_text_no_results() -> None:
    store = memory_store()
    store.ensure_collection(4)

    text = search_docs_text(
        "docker", "anything",
        qdrant_url="unused", qdrant_path=None, qdrant_api_key=None, qdrant_timeout=60.0,
        embed_model="unused", embedder=FakeEmbedder(), store=store,
    )

    assert "No results" in text


def test_search_docs_text_reports_store_errors_cleanly() -> None:
    class BrokenStore:
        def search(self, vector, limit):
            raise RuntimeError("connection refused")

    text = search_docs_text(
        "docker", "anything",
        qdrant_url="http://localhost:6333", qdrant_path=None, qdrant_api_key=None,
        qdrant_timeout=60.0, embed_model="unused", embedder=FakeEmbedder(), store=BrokenStore(),
    )

    assert "Search failed" in text
    assert "connection refused" in text


# --- build_server ---


def test_build_server_registers_both_tools(tmp_path) -> None:
    server = build_server(
        db_path=str(tmp_path / "d.db"), qdrant_url="http://localhost:6333", qdrant_path=None,
        qdrant_api_key=None, qdrant_timeout=60.0, embed_model="unused",
    )

    tools = {tool.name for tool in asyncio.run(server.list_tools())}

    assert tools == {"list_docs", "search_docs"}


def test_build_server_list_docs_tool_call_reads_the_manifest(tmp_path) -> None:
    db = str(tmp_path / "d.db")
    with Manifest(db) as m:
        m.upsert_page("docker", "https://a.com/x", "h1")

    server = build_server(
        db_path=db, qdrant_url="http://localhost:6333", qdrant_path=None,
        qdrant_api_key=None, qdrant_timeout=60.0, embed_model="unused",
    )

    content, _structured = asyncio.run(server.call_tool("list_docs", {}))

    assert any("docker" in block.text for block in content)


def test_build_server_binds_the_given_host_and_port(tmp_path) -> None:
    server = build_server(
        db_path=str(tmp_path / "d.db"), qdrant_url="http://localhost:6333", qdrant_path=None,
        qdrant_api_key=None, qdrant_timeout=60.0, embed_model="unused",
        host="0.0.0.0", port=9001,
    )

    assert server.settings.host == "0.0.0.0"
    assert server.settings.port == 9001


def test_build_server_defaults_host_and_port(tmp_path) -> None:
    server = build_server(
        db_path=str(tmp_path / "d.db"), qdrant_url="http://localhost:6333", qdrant_path=None,
        qdrant_api_key=None, qdrant_timeout=60.0, embed_model="unused",
    )

    assert server.settings.host == DEFAULT_HOST
    assert server.settings.port == DEFAULT_PORT


# --- CLI parser (--transport/--host/--port) ---


def test_parser_defaults_to_no_explicit_transport() -> None:
    args = _build_parser().parse_args([])
    assert args.transport is None  # main() resolves None -> env -> "stdio"
    assert args.host is None
    assert args.port is None


def test_parser_accepts_transport_host_port() -> None:
    args = _build_parser().parse_args(
        ["--transport", "http", "--host", "0.0.0.0", "--port", "9001"]
    )
    assert args.transport == "http"
    assert args.host == "0.0.0.0"
    assert args.port == 9001


def test_parser_rejects_unknown_transport() -> None:
    import pytest

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--transport", "carrier-pigeon"])


def test_main_stdio_never_prints(tmp_path, monkeypatch, capsys) -> None:
    """Regression guard: stdio transport must not print -- it would corrupt the JSON-RPC stream."""
    import docforge.mcp_server as mcp_server_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCFORGE_MCP_TRANSPORT", raising=False)
    fake_server = type("FakeServer", (), {"run": lambda self, transport="stdio": None})()
    monkeypatch.setattr(mcp_server_module, "build_server", lambda **kw: fake_server)

    mcp_server_module.main([])

    assert capsys.readouterr().out == ""


def test_main_http_prints_the_url(tmp_path, monkeypatch, capsys) -> None:
    import docforge.mcp_server as mcp_server_module

    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    fake_server = type(
        "FakeServer", (), {"run": lambda self, transport="stdio": calls.append(transport)}
    )()
    monkeypatch.setattr(mcp_server_module, "build_server", lambda **kw: fake_server)

    mcp_server_module.main(["--transport", "http", "--port", "9001"])

    assert calls == ["streamable-http"]
    out = capsys.readouterr().out
    assert "http://127.0.0.1:9001/mcp" in out
