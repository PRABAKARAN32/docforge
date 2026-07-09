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
        "how to install", name="docker", db_path="unused",
        qdrant_url="unused", qdrant_path=None, qdrant_api_key=None, qdrant_timeout=60.0,
        embed_model="unused", embedder=FakeEmbedder(), store=store,
    )

    assert "https://d/a" in text
    assert "install with pip" in text


def test_search_docs_text_no_results() -> None:
    store = memory_store()
    store.ensure_collection(4)

    text = search_docs_text(
        "anything", name="docker", db_path="unused",
        qdrant_url="unused", qdrant_path=None, qdrant_api_key=None, qdrant_timeout=60.0,
        embed_model="unused", embedder=FakeEmbedder(), store=store,
    )

    assert "No results" in text


def test_search_docs_text_reports_store_errors_cleanly() -> None:
    class BrokenStore:
        def search(self, vector, limit):
            raise RuntimeError("connection refused")

    text = search_docs_text(
        "anything", name="docker", db_path="unused",
        qdrant_url="http://localhost:6333", qdrant_path=None, qdrant_api_key=None,
        qdrant_timeout=60.0, embed_model="unused", embedder=FakeEmbedder(), store=BrokenStore(),
    )

    assert "Search failed" in text
    assert "connection refused" in text


def test_search_docs_text_searches_all_kbs_when_name_omitted(tmp_path) -> None:
    """Mirrors docforge search's default (--all) behavior, and is a regression test for a real
    bug: embedded Qdrant (--qdrant-path) locks its storage folder per client, so searching a
    2nd+ collection without closing the previous one crashed with 'already accessed by another
    instance of Qdrant client'."""
    db = str(tmp_path / "d.db")
    qdrant_path = str(tmp_path / "vectors")
    with Manifest(db) as m:
        m.upsert_page("docker", "https://docker/x", "h1")
        m.upsert_page("nginx", "https://nginx/y", "h2")

    for collection, url, text in [
        ("docker", "https://docker/x", "install docker with apt"),
        ("nginx", "https://nginx/y", "install nginx with apt"),
    ]:
        store = QdrantVectorStore(path=qdrant_path, collection=collection)
        store.ensure_collection(4)
        store.upsert_chunks([Chunk(url, 0, text)], [[1.0, 0.0, 0.0, 0.0]])
        store.close()

    text = search_docs_text(
        "install", db_path=db,
        qdrant_url="unused", qdrant_path=qdrant_path, qdrant_api_key=None, qdrant_timeout=60.0,
        embed_model="unused", embedder=FakeEmbedder(),
    )

    assert "https://docker/x" in text
    assert "https://nginx/y" in text


def test_search_docs_text_no_kbs_at_all(tmp_path) -> None:
    text = search_docs_text(
        "anything", db_path=str(tmp_path / "empty.db"),
        qdrant_url="unused", qdrant_path=None, qdrant_api_key=None, qdrant_timeout=60.0,
        embed_model="unused", embedder=FakeEmbedder(),
    )

    assert "No knowledge bases tracked" in text


# --- build_server ---


def test_build_server_registers_both_tools(tmp_path) -> None:
    server = build_server(
        db_path=str(tmp_path / "d.db"), qdrant_url="http://localhost:6333", qdrant_path=None,
        qdrant_api_key=None, qdrant_timeout=60.0, embed_model="unused",
    )

    tools = {tool.name for tool in asyncio.run(server.list_tools())}

    assert tools == {"list_docs", "search_docs"}


def test_build_server_search_docs_name_is_optional_in_the_tool_schema(tmp_path) -> None:
    """A model must be able to call search_docs(query=...) alone -- name is a refinement,
    not a required first step (that's the whole point of the search-all default)."""
    server = build_server(
        db_path=str(tmp_path / "d.db"), qdrant_url="http://localhost:6333", qdrant_path=None,
        qdrant_api_key=None, qdrant_timeout=60.0, embed_model="unused",
    )

    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    schema = tools["search_docs"].inputSchema

    assert schema["required"] == ["query"]


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
    assert args.token is None
    assert args.no_auth is False


def test_parser_accepts_transport_host_port_token() -> None:
    args = _build_parser().parse_args(
        ["--transport", "http", "--host", "0.0.0.0", "--port", "9001", "--token", "secret"]
    )
    assert args.transport == "http"
    assert args.host == "0.0.0.0"
    assert args.port == 9001
    assert args.token == "secret"


def test_parser_accepts_both_transport() -> None:
    assert _build_parser().parse_args(["--transport", "both"]).transport == "both"


def test_parser_accepts_no_auth_flag() -> None:
    assert _build_parser().parse_args(["--no-auth"]).no_auth is True


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

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_main_http_generates_a_token_by_default(tmp_path, monkeypatch, capsys) -> None:
    """Secure by default: no --token/env/--no-auth given -> a fresh token is minted and printed."""
    import docforge.mcp_server as mcp_server_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCFORGE_MCP_TOKEN", raising=False)
    calls: list[tuple] = []

    async def fake_serve_http(server, host, port, token):
        calls.append((host, port, token))

    monkeypatch.setattr(mcp_server_module, "build_server", lambda **kw: object())
    monkeypatch.setattr(mcp_server_module, "_serve_http_async", fake_serve_http)

    mcp_server_module.main(["--transport", "http", "--port", "9001"])

    assert len(calls) == 1
    host, port, token = calls[0]
    assert (host, port) == ("127.0.0.1", 9001)
    assert isinstance(token, str) and len(token) > 20  # a real generated token, not empty

    captured = capsys.readouterr()
    assert captured.out == ""  # nothing ever goes to stdout for http/both
    assert "http://127.0.0.1:9001/mcp" in captured.err
    assert "generated fresh this run" in captured.err
    assert token in captured.err  # printed so the user can copy it into their client
    assert "No token configured" not in captured.err


def test_main_http_no_auth_gives_open_access(tmp_path, monkeypatch, capsys) -> None:
    import docforge.mcp_server as mcp_server_module

    monkeypatch.chdir(tmp_path)
    calls: list[tuple] = []

    async def fake_serve_http(server, host, port, token):
        calls.append((host, port, token))

    monkeypatch.setattr(mcp_server_module, "build_server", lambda **kw: object())
    monkeypatch.setattr(mcp_server_module, "_serve_http_async", fake_serve_http)

    mcp_server_module.main(["--transport", "http", "--no-auth"])

    assert calls == [("127.0.0.1", 8000, None)]
    assert "No token configured (--no-auth) -- open access." in capsys.readouterr().err


def test_main_http_with_token_skips_the_open_access_message(tmp_path, monkeypatch, capsys) -> None:
    import docforge.mcp_server as mcp_server_module

    monkeypatch.chdir(tmp_path)
    calls: list[tuple] = []

    async def fake_serve_http(server, host, port, token):
        calls.append((host, port, token))

    monkeypatch.setattr(mcp_server_module, "build_server", lambda **kw: object())
    monkeypatch.setattr(mcp_server_module, "_serve_http_async", fake_serve_http)

    mcp_server_module.main(["--transport", "http", "--token", "secret"])

    assert calls == [("127.0.0.1", 8000, "secret")]
    err = capsys.readouterr().err
    assert "Authorization required" in err
    assert "No token configured" not in err
    assert "WARNING" not in err  # loopback host -> no nudge needed


def test_main_warns_when_binding_non_loopback_with_no_auth(tmp_path, monkeypatch, capsys) -> None:
    """A token is always generated by default, so the WARNING only fires when --no-auth is
    explicitly combined with a non-loopback host -- that combination is the real risk."""
    import docforge.mcp_server as mcp_server_module

    monkeypatch.chdir(tmp_path)

    async def fake_serve_http(server, host, port, token):
        pass

    monkeypatch.setattr(mcp_server_module, "build_server", lambda **kw: object())
    monkeypatch.setattr(mcp_server_module, "_serve_http_async", fake_serve_http)

    mcp_server_module.main(["--transport", "http", "--host", "0.0.0.0", "--no-auth"])

    assert "WARNING" in capsys.readouterr().err


def test_main_no_warning_on_non_loopback_when_token_generated(tmp_path, monkeypatch, capsys) -> None:
    """Binding non-loopback is fine without a WARNING as long as a token is actually required."""
    import docforge.mcp_server as mcp_server_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCFORGE_MCP_TOKEN", raising=False)

    async def fake_serve_http(server, host, port, token):
        pass

    monkeypatch.setattr(mcp_server_module, "build_server", lambda **kw: object())
    monkeypatch.setattr(mcp_server_module, "_serve_http_async", fake_serve_http)

    mcp_server_module.main(["--transport", "http", "--host", "0.0.0.0"])

    assert "WARNING" not in capsys.readouterr().err


def test_main_both_dispatches_to_serve_both_async(tmp_path, monkeypatch, capsys) -> None:
    import docforge.mcp_server as mcp_server_module

    monkeypatch.chdir(tmp_path)
    calls: list[tuple] = []

    async def fake_serve_both(server, host, port, token):
        calls.append((host, port, token))

    monkeypatch.setattr(mcp_server_module, "build_server", lambda **kw: object())
    monkeypatch.setattr(mcp_server_module, "_serve_both_async", fake_serve_both)

    mcp_server_module.main(["--transport", "both", "--token", "secret"])

    assert calls == [("127.0.0.1", 8000, "secret")]


# --- _resolve_token ---


def test_resolve_token_no_auth_wins_even_over_an_explicit_token() -> None:
    from docforge.mcp_server import _resolve_token

    token, generated = _resolve_token("explicit-secret", no_auth=True)

    assert token is None
    assert generated is False


def test_resolve_token_explicit_wins_over_generation() -> None:
    from docforge.mcp_server import _resolve_token

    token, generated = _resolve_token("explicit-secret", no_auth=False)

    assert token == "explicit-secret"
    assert generated is False


def test_resolve_token_generates_a_real_token_when_nothing_given() -> None:
    from docforge.mcp_server import _resolve_token

    token, generated = _resolve_token(None, no_auth=False)

    assert generated is True
    assert isinstance(token, str)
    assert len(token) > 20  # a real random token, not a placeholder


def test_resolve_token_generates_a_different_token_each_call() -> None:
    """Regression guard for the 'fresh every run, never persisted' requirement."""
    from docforge.mcp_server import _resolve_token

    first, _ = _resolve_token(None, no_auth=False)
    second, _ = _resolve_token(None, no_auth=False)

    assert first != second


# --- BearerAuthMiddleware (raw ASGI, no starlette test client needed) ---


def _http_scope(auth_header: str | None) -> dict:
    headers = [(b"authorization", auth_header.encode())] if auth_header is not None else []
    return {"type": "http", "headers": headers}


def test_bearer_auth_rejects_missing_header() -> None:
    from docforge.mcp_server import BearerAuthMiddleware

    async def app(scope, receive, send):
        raise AssertionError("inner app must not run when unauthorized")

    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(BearerAuthMiddleware(app, "secret")(_http_scope(None), None, send))

    assert sent[0]["status"] == 401


def test_bearer_auth_rejects_wrong_token() -> None:
    from docforge.mcp_server import BearerAuthMiddleware

    async def app(scope, receive, send):
        raise AssertionError("inner app must not run when unauthorized")

    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(BearerAuthMiddleware(app, "secret")(_http_scope("Bearer wrong"), None, send))

    assert sent[0]["status"] == 401


def test_bearer_auth_allows_correct_token() -> None:
    from docforge.mcp_server import BearerAuthMiddleware

    called = []

    async def app(scope, receive, send):
        called.append(True)

    asyncio.run(BearerAuthMiddleware(app, "secret")(_http_scope("Bearer secret"), None, None))

    assert called == [True]


def test_bearer_auth_passes_through_non_http_scopes() -> None:
    """Lifespan/other ASGI scopes aren't HTTP requests -- must never be auth-gated."""
    from docforge.mcp_server import BearerAuthMiddleware

    called = []

    async def app(scope, receive, send):
        called.append(True)

    asyncio.run(BearerAuthMiddleware(app, "secret")({"type": "lifespan"}, None, None))

    assert called == [True]
