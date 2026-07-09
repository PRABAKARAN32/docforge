"""The DocForge MCP server (M4): exposes the already-built RAG as tools for an LLM client
(Claude Desktop, Claude Code, or a local-LLM agent runtime).

This is a thin wrapper -- no new retrieval logic. ``search_docs`` reuses the same
embed-query-then-search path as ``docforge search``; ``list_docs`` reads the same manifest as
``docforge status``. See Notes/Discussion/01-mcp-server-what-and-why.md for why this is scoped
to retrieval only (crawling stays a CLI-only operation).

Two tools:
  * ``list_docs()``                     -- which knowledge bases exist, with page counts.
  * ``search_docs(query, name=None)``   -- the closest chunks; ``name`` scopes to one
    knowledge base, omitted searches all of them at once (mirrors ``docforge search``).

Three transport modes, picked with ``--transport``:
  * ``stdio`` (default) -- the client launches ``docforge-mcp`` itself as a subprocess and
    talks over its stdin/stdout. No port, no URL, no token. What Claude Desktop/Code expect
    (`claude mcp add ...`). This is inherently local and trusted -- the "client" is whoever
    started the process -- so there's no network boundary to protect and no auth is applied.
  * ``http`` -- the server instead binds a port and serves plain HTTP, so anything that can
    make an HTTP request (a custom local-LLM agent, a different machine on the LAN) can
    connect to it by URL instead of spawning it. THIS is a real network boundary -- see auth
    below.
  * ``both`` -- runs stdio and http at once, in the same process against the same knowledge
    bases. Useful if this process is itself spawned by a stdio client (Claude Code) but you
    also want an HTTP port open for something else (LM Studio, a custom agent) at the same
    time, instead of running two separate ``docforge-mcp`` processes.

Nothing is ever printed to **stdout** for any mode that includes stdio (``stdio`` or ``both``)
-- it would corrupt the JSON-RPC stream sent over stdout, which is why running plain ``stdio``
by hand just sits there silently. Startup/status messages always go to **stderr** instead,
which every MCP client treats as ordinary logs, never protocol.

**Auth (http only) is secure by default.** If you don't set ``--token``/``DOCFORGE_MCP_TOKEN``,
the server generates a fresh random token itself on every start (like Jupyter Notebook does)
and prints it -- ``Authorization: Bearer <token>`` is then required on every HTTP request
(checked with a constant-time comparison to avoid timing attacks). The generated token is never
written anywhere; restarting the server gets a *new* one. For a token that stays stable across
restarts (so a client config doesn't need updating every time), set ``DOCFORGE_MCP_TOKEN``
yourself. To run with no auth at all (open access), pass ``--no-auth`` explicitly.

Run it with ``docforge-mcp`` (registered in ``pyproject.toml``).

Design mirrors ``cli.py``: pure, injectable functions (``list_docs_text``,
``search_docs_text``) hold the logic and are unit-tested with fakes; :func:`build_server`
wires them into MCP tools; :func:`main` wires real config (``.env``/env, via
``docforge.config``, plus ``--transport``/``--host``/``--port``/``--token``) and starts the
server.
"""

from __future__ import annotations

import argparse
import hmac
import os
import sys
from collections.abc import Sequence

from docforge.config import load_dotenv, open_store, resolve_settings, store_error_hint
from docforge.embedder import DEFAULT_DEVICE, Embedder
from docforge.manifest import Manifest
from docforge.vectorstore import VectorStore

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_TRANSPORT = "stdio"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class BearerAuthMiddleware:
    """ASGI middleware requiring ``Authorization: Bearer <token>`` on every HTTP request.

    Only wraps the ``http``/``both`` transport's web app -- stdio has no network boundary
    (see the module docstring), so nothing to check there. Uses a constant-time comparison
    (:func:`hmac.compare_digest`) so a mistyped token can't be brute-forced via response-time
    differences.
    """

    def __init__(self, app, token: str) -> None:
        self._app = app
        self._expected = f"Bearer {token}"

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or ())
        presented = headers.get(b"authorization", b"").decode("latin-1")
        if not hmac.compare_digest(presented, self._expected):
            from starlette.responses import JSONResponse

            response = JSONResponse({"error": "Unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        await self._app(scope, receive, send)


def list_docs_text(db_path: str) -> str:
    """List the tracked knowledge bases and their page counts, as text for the model."""
    with Manifest(db_path) as manifest:
        kbs = manifest.names()

    if not kbs:
        return f"No knowledge bases tracked in {db_path}. Run `docforge sync <url>` first."

    lines = [f"{count} pages -- {name}" for name, count in sorted(kbs.items())]
    return "\n".join(lines)


def search_docs_text(
    query: str,
    *,
    name: str | None = None,
    limit: int = 5,
    db_path: str,
    qdrant_url: str,
    qdrant_path: str | None,
    qdrant_api_key: str | None,
    qdrant_timeout: float,
    embed_model: str,
    device: str = DEFAULT_DEVICE,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> str:
    """Search one knowledge base (``name``) or all of them (``name=None``), as text.

    Mirrors ``run_search`` in ``cli.py`` exactly (same embed-then-search path, same
    search-all-by-default behavior) so a model doesn't need to call ``list_docs`` first just
    to search -- only when it wants to scope to one specific knowledge base.
    """
    try:
        if embedder is None:
            from docforge.embedder import FastEmbedEmbedder

            embedder = FastEmbedEmbedder(embed_model, device=device)
        vector = embedder.embed([query])[0]

        if store is not None:
            hits = list(store.search(vector, limit=limit))
        else:
            if name:
                collections = [name]
            else:
                with Manifest(db_path) as manifest:
                    collections = sorted(manifest.names())
                if not collections:
                    return f"No knowledge bases tracked in {db_path}. Run `docforge sync <url>` first."
            hits = []
            for collection in collections:
                bound = open_store(qdrant_url, qdrant_path, qdrant_api_key, qdrant_timeout, collection)
                try:
                    hits.extend(bound.search(vector, limit=limit))
                finally:
                    # Embedded (--qdrant-path) mode locks the whole storage folder per client --
                    # must close before opening the next collection, or the 2nd+ one crashes.
                    bound.close()
            hits.sort(key=lambda hit: hit.score, reverse=True)
            hits = hits[:limit]
    except Exception as exc:  # noqa: BLE001 -- surface a clean message to the model, not a traceback
        return f"Search failed: {exc}\n{store_error_hint(qdrant_url, qdrant_path)}"

    if not hits:
        return f"No results in knowledge base {name!r}." if name else "No results."

    parts = [f"[{hit.score:.3f}] {hit.source_url}\n{hit.text.strip()}" for hit in hits]
    return "\n\n".join(parts)


def build_server(
    *,
    db_path: str,
    qdrant_url: str,
    qdrant_path: str | None,
    qdrant_api_key: str | None,
    qdrant_timeout: float,
    embed_model: str,
    device: str = DEFAULT_DEVICE,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
):
    """Build the MCP server with ``list_docs``/``search_docs`` tools bound to this config.

    ``host``/``port`` only matter for the ``http`` transport (:func:`main`) -- ignored for
    stdio, but harmless to always set.
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "docforge",
        instructions="Search locally-indexed documentation knowledge bases. Call list_docs "
        "first if you don't already know the exact knowledge base name.",
        host=host,
        port=port,
    )

    @server.tool()
    def list_docs() -> str:
        """List the knowledge bases (docs sites) DocForge has indexed, with page counts."""
        return list_docs_text(db_path)

    @server.tool()
    def search_docs(query: str, name: str | None = None, limit: int = 5) -> str:
        """Search documentation and return the closest matching chunks.

        Args:
            query: What to search for, in natural language.
            name: Restrict the search to one knowledge base (see list_docs for exact names).
                Omit to search across every knowledge base at once.
            limit: Maximum number of chunks to return (default 5).
        """
        return search_docs_text(
            query,
            name=name,
            limit=limit,
            db_path=db_path,
            qdrant_url=qdrant_url,
            qdrant_path=qdrant_path,
            qdrant_api_key=qdrant_api_key,
            qdrant_timeout=qdrant_timeout,
            embed_model=embed_model,
            device=device,
        )

    return server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docforge-mcp",
        description="Run the DocForge MCP server, exposing list_docs/search_docs to an LLM client.",
    )
    parser.add_argument(
        "--transport", choices=["stdio", "http", "both"], default=None,
        help="stdio (default): the client launches this process itself, no URL, no auth. "
        "http: bind a port and print a URL + auth token a client connects to directly. "
        "both: run stdio and http at once, in the same process "
        "(default: DOCFORGE_MCP_TRANSPORT env var, else stdio).",
    )
    parser.add_argument(
        "--host", default=None, metavar="HOST",
        help=f"Bind address for the http side (default: {DEFAULT_HOST}). Use 0.0.0.0 to accept "
        "connections from other machines -- strongly discouraged with --no-auth.",
    )
    parser.add_argument(
        "--port", type=int, default=None, metavar="PORT",
        help=f"Bind port for the http side (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--token", default=None, metavar="TOKEN",
        help="Require 'Authorization: Bearer TOKEN' on http requests (default: "
        "DOCFORGE_MCP_TOKEN env var, else a fresh token is generated and printed each run). "
        "Ignored for pure stdio.",
    )
    parser.add_argument(
        "--no-auth", action="store_true",
        help="Run http with no authorization check at all (open access). Only the explicit "
        "opt-out for this -- by default a token is always required.",
    )
    return parser


async def _serve_http_async(server, host: str, port: int, token: str | None) -> None:
    """Serve the http side: :func:`FastMCP.streamable_http_app`, wrapped with auth if configured."""
    import uvicorn

    app = server.streamable_http_app()
    if token:
        app = BearerAuthMiddleware(app, token)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    await uvicorn.Server(config).serve()


async def _serve_both_async(server, host: str, port: int, token: str | None) -> None:
    """Run stdio and http concurrently in this process, sharing the same server/tools."""
    import anyio

    async with anyio.create_task_group() as tg:
        tg.start_soon(server.run_stdio_async)
        tg.start_soon(_serve_http_async, server, host, port, token)


def _resolve_token(explicit: str | None, *, no_auth: bool) -> tuple[str | None, bool]:
    """Resolve the http auth token: explicit > generated > none (--no-auth).

    Returns ``(token, generated)`` -- ``generated`` is True only when a token was minted here
    (not user-supplied), so the caller knows whether to print "generate a new one on restart".
    """
    if no_auth:
        return None, False
    if explicit:
        return explicit, False
    import secrets

    return secrets.token_urlsafe(32), True


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()  # let a local .env supply QDRANT_URL / DOCFORGE_DB / DOCFORGE_EMBED_MODEL / ...
    args = _build_parser().parse_args(argv)
    settings = resolve_settings()
    transport = args.transport or os.getenv("DOCFORGE_MCP_TRANSPORT") or DEFAULT_TRANSPORT
    host = args.host or DEFAULT_HOST
    port = args.port or DEFAULT_PORT

    server = build_server(
        db_path=settings["db_path"],
        qdrant_url=settings["qdrant_url"],
        qdrant_path=settings["qdrant_path"],
        qdrant_api_key=settings["qdrant_api_key"],
        qdrant_timeout=settings["qdrant_timeout"],
        embed_model=settings["embed_model"],
        host=host,
        port=port,
    )

    if transport == "stdio":
        server.run()  # never print here -- would corrupt the stdio JSON-RPC stream
        return

    # Only http/both ever need a token -- resolved (and possibly generated) here, not for stdio.
    token, token_generated = _resolve_token(
        args.token or os.getenv("DOCFORGE_MCP_TOKEN"), no_auth=args.no_auth
    )

    # http or both: stdout is safe to use for stdio-only, but "both" also runs a stdio side
    # in this same process -- so everything user-facing goes to stderr, never stdout, always.
    if token is None and host not in _LOOPBACK_HOSTS:
        print(
            f"WARNING: binding to {host} with --no-auth -- anyone who can reach this port "
            "can read your knowledge bases.",
            file=sys.stderr,
        )
    print(f"DocForge MCP server (http) listening at http://{host}:{port}/mcp", file=sys.stderr)
    if token is None:
        print("No token configured (--no-auth) -- open access.", file=sys.stderr)
    elif token_generated:
        print(f"Auth token (generated fresh this run): {token}", file=sys.stderr)
        print(f"Authorization header: Bearer {token}", file=sys.stderr)
        print(
            "This token is NOT saved -- restarting the server generates a new one. Set "
            "DOCFORGE_MCP_TOKEN in .env for a token that stays the same across restarts.",
            file=sys.stderr,
        )
    else:
        print("Authorization required: Bearer <token>", file=sys.stderr)
    print("Ctrl+C to stop.", file=sys.stderr)

    import anyio

    if transport == "http":
        anyio.run(_serve_http_async, server, host, port, token)
    else:
        anyio.run(_serve_both_async, server, host, port, token)


if __name__ == "__main__":
    main()
