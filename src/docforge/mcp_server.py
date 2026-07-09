"""The DocForge MCP server (M4): exposes the already-built RAG as tools for an LLM client
(Claude Desktop, Claude Code, or a local-LLM agent runtime).

This is a thin wrapper -- no new retrieval logic. ``search_docs`` reuses the same
embed-query-then-search path as ``docforge search``; ``list_docs`` reads the same manifest as
``docforge status``. See Notes/Discussion/01-mcp-server-what-and-why.md for why this is scoped
to retrieval only (crawling stays a CLI-only operation).

Two tools:
  * ``list_docs()``               -- which knowledge bases exist, with page counts.
  * ``search_docs(name, query)``  -- the closest chunks in one knowledge base.

Two transports, picked with ``--transport``:
  * ``stdio`` (default) -- the client launches ``docforge-mcp`` itself as a subprocess and
    talks over its stdin/stdout. No port, no URL. What Claude Desktop/Code expect
    (`claude mcp add ...`). Nothing may ever be printed to stdout in this mode -- it would
    corrupt the JSON-RPC stream -- which is why running it by hand just sits there silently.
  * ``http`` -- the server instead binds a port and serves plain HTTP, so anything that can
    make an HTTP request (a custom local-LLM agent, a different machine on the LAN) can
    connect to it by URL instead of spawning it. Prints the URL to stdout on startup, since
    stdout isn't part of the wire protocol in this mode.

Run it with ``docforge-mcp`` (registered in ``pyproject.toml``).

Design mirrors ``cli.py``: pure, injectable functions (``list_docs_text``,
``search_docs_text``) hold the logic and are unit-tested with fakes; :func:`build_server`
wires them into MCP tools; :func:`main` wires real config (``.env``/env, via
``docforge.config``, plus ``--transport``/``--host``/``--port``) and starts the server.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from docforge.config import load_dotenv, open_store, resolve_settings, store_error_hint
from docforge.embedder import DEFAULT_DEVICE, Embedder
from docforge.manifest import Manifest
from docforge.vectorstore import VectorStore

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def list_docs_text(db_path: str) -> str:
    """List the tracked knowledge bases and their page counts, as text for the model."""
    with Manifest(db_path) as manifest:
        kbs = manifest.names()

    if not kbs:
        return f"No knowledge bases tracked in {db_path}. Run `docforge sync <url>` first."

    lines = [f"{count} pages -- {name}" for name, count in sorted(kbs.items())]
    return "\n".join(lines)


def search_docs_text(
    name: str,
    query: str,
    *,
    limit: int = 5,
    qdrant_url: str,
    qdrant_path: str | None,
    qdrant_api_key: str | None,
    qdrant_timeout: float,
    embed_model: str,
    device: str = DEFAULT_DEVICE,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> str:
    """Search one knowledge base and return the closest chunks, formatted as text.

    Mirrors ``run_search`` in ``cli.py`` (same embed-then-search path), scoped to a single
    named collection since an LLM tool call names the knowledge base explicitly.
    """
    try:
        if embedder is None:
            from docforge.embedder import FastEmbedEmbedder

            embedder = FastEmbedEmbedder(embed_model, device=device)
        vector = embedder.embed([query])[0]
        bound = (
            store
            if store is not None
            else open_store(qdrant_url, qdrant_path, qdrant_api_key, qdrant_timeout, name)
        )
        hits = list(bound.search(vector, limit=limit))
    except Exception as exc:  # noqa: BLE001 -- surface a clean message to the model, not a traceback
        return f"Search failed: {exc}\n{store_error_hint(qdrant_url, qdrant_path)}"

    if not hits:
        return f"No results in knowledge base {name!r}."

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
    def search_docs(name: str, query: str, limit: int = 5) -> str:
        """Search a knowledge base and return the closest matching documentation chunks.

        Args:
            name: The knowledge base to search (see list_docs for exact names).
            query: What to search for, in natural language.
            limit: Maximum number of chunks to return (default 5).
        """
        return search_docs_text(
            name,
            query,
            limit=limit,
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
        "--transport", choices=["stdio", "http"], default=None,
        help="stdio (default): the client launches this process itself, no URL. "
        "http: bind a port and print a URL a client connects to directly "
        "(default: DOCFORGE_MCP_TRANSPORT env var, else stdio).",
    )
    parser.add_argument(
        "--host", default=None, metavar="HOST",
        help=f"Bind address for --transport http (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port", type=int, default=None, metavar="PORT",
        help=f"Bind port for --transport http (default: {DEFAULT_PORT}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()  # let a local .env supply QDRANT_URL / DOCFORGE_DB / DOCFORGE_EMBED_MODEL / ...
    args = _build_parser().parse_args(argv)
    settings = resolve_settings()
    transport = args.transport or os.getenv("DOCFORGE_MCP_TRANSPORT") or "stdio"
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

    print(f"DocForge MCP server (http) listening at http://{host}:{port}/mcp")
    print("Connect your MCP client to that URL. Ctrl+C to stop.")
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
