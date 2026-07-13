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

**A second, opt-in auth mode: OAuth 2.1**, for hosted "custom connector" clients (Claude.ai,
ChatGPT) that don't support pasting in a static bearer token and instead speak a real OAuth
authorization-code flow with PKCE. Setting ``--client-id``/``--client-secret`` (or
``DOCFORGE_MCP_CLIENT_ID``/``DOCFORGE_MCP_CLIENT_SECRET``) switches http/both into this mode
instead -- see :mod:`docforge.oauth` and GUID.md Decision 5.17 for the full design (no Dynamic
Client Registration; a pre-registered client id/secret; in-memory-only codes/tokens, the same
"regenerates on restart" trade-off as the static token). Mutually exclusive with
``--token``/``--no-auth`` -- pick one auth mode per run.

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
from dataclasses import dataclass
from urllib.parse import urlparse

from docforge.config import load_dotenv, open_store, resolve_settings, store_error_hint
from docforge.embedder import DEFAULT_DEVICE, Embedder
from docforge.manifest import Manifest
from docforge.oauth import TokenStore, build_oauth_app, resolve_client_credentials
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


class OAuthAuthMiddleware:
    """Dispatches to the OAuth surface (discovery/``/authorize``/``/token``, unauthenticated by
    design -- those *are* the auth endpoints) or the MCP app (bearer-token gated against a
    :class:`~docforge.oauth.TokenStore`).

    Unlike :class:`BearerAuthMiddleware`'s single static secret, a ``401`` here carries a
    ``WWW-Authenticate: Bearer resource_metadata="..."`` header pointing at the protected
    resource metadata document -- required for Claude/ChatGPT to discover where to start the
    OAuth flow (see GUID.md Decision 5.17). That URL is resolved the same way
    :func:`docforge.oauth.build_oauth_app`'s own endpoints resolve theirs (``public_url``
    override, else per-request ``X-Forwarded-Proto``/``Host`` detection via
    :func:`docforge.oauth.external_base_url`) so it stays correct behind a reverse proxy or
    tunnel (ngrok, Cloudflare Tunnel), not just for direct ``host:port`` access.
    """

    _OAUTH_PATH_PREFIXES = ("/.well-known/", "/authorize", "/token")

    def __init__(
        self,
        mcp_app,
        oauth_app,
        store: TokenStore,
        *,
        default_host: str,
        public_url: str | None = None,
    ) -> None:
        self._mcp_app = mcp_app
        self._oauth_app = oauth_app
        self._store = store
        self._default_host = default_host
        self._public_url = public_url

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._mcp_app(scope, receive, send)
            return

        if scope["path"].startswith(self._OAUTH_PATH_PREFIXES):
            await self._oauth_app(scope, receive, send)
            return

        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers") or ()}
        presented = headers.get("authorization", "")
        token = presented.removeprefix("Bearer ") if presented.startswith("Bearer ") else ""
        if not token or not self._store.validate_access_token(token):
            from starlette.responses import JSONResponse

            from docforge.oauth import external_base_url

            base_url = (self._public_url or "").rstrip("/") or external_base_url(
                headers, default_scheme=scope.get("scheme", "http"), default_host=self._default_host
            )
            response = JSONResponse(
                {"error": "Unauthorized"},
                status_code=401,
                headers={
                    "WWW-Authenticate": (
                        f'Bearer resource_metadata="{base_url}/.well-known/oauth-protected-resource"'
                    )
                },
            )
            await response(scope, receive, send)
            return

        await self._mcp_app(scope, receive, send)


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
    public_host: str | None = None,
):
    """Build the MCP server with ``list_docs``/``search_docs`` tools bound to this config.

    ``host``/``port`` only matter for the ``http`` transport (:func:`main`) -- ignored for
    stdio, but harmless to always set.

    ``public_host`` (``netloc`` of ``--public-url``/``DOCFORGE_MCP_PUBLIC_URL``, e.g.
    ``"abcd1234.ngrok-free.app"``) extends the MCP SDK's own DNS-rebinding ``Host``-header
    allowlist to include it. Without this, the SDK auto-enables that protection for the default
    loopback host and rejects *any* other ``Host`` header outright -- which silently breaks a
    reverse-proxied/tunneled server (ngrok, Cloudflare Tunnel) even after our own auth checks
    already passed: the SDK's check runs downstream, inside ``streamable_http_app()``. See
    GUID.md Decision 5.17's follow-up note. ``None`` leaves the SDK's own default behavior
    untouched (loopback-only allowlist for ``host in (127.0.0.1, localhost, ::1)``, no
    allowlist at all otherwise -- e.g. ``--host 0.0.0.0``), so plain non-tunneled use is
    unaffected.
    """
    from mcp.server.fastmcp import FastMCP

    transport_security = None
    if public_host:
        from mcp.server.transport_security import TransportSecuritySettings

        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", public_host],
            allowed_origins=[
                "http://127.0.0.1:*",
                "http://localhost:*",
                "http://[::1]:*",
                f"https://{public_host}",
                f"http://{public_host}",
            ],
        )

    server = FastMCP(
        "docforge",
        instructions="Search locally-indexed documentation knowledge bases. Call list_docs "
        "first if you don't already know the exact knowledge base name.",
        host=host,
        port=port,
        transport_security=transport_security,
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
    parser.add_argument(
        "--client-id", default=None, metavar="CLIENT_ID",
        help="Pre-registered OAuth client id for the http side (default: DOCFORGE_MCP_CLIENT_ID "
        "env var, else generated fresh and printed each run). Setting this (or "
        "--client-secret) switches http/both into a full OAuth 2.1 flow -- the mode Claude.ai/ "
        "ChatGPT custom connectors need -- instead of the static --token. Mutually exclusive "
        "with --token/--no-auth.",
    )
    parser.add_argument(
        "--client-secret", default=None, metavar="CLIENT_SECRET",
        help="Pre-registered OAuth client secret (default: DOCFORGE_MCP_CLIENT_SECRET env var, "
        "else generated fresh and printed each run). Ignored unless OAuth mode is active.",
    )
    parser.add_argument(
        "--public-url", default=None, metavar="URL",
        help="Externally-reachable base URL for this server (e.g. an ngrok/Cloudflare Tunnel "
        "URL), for when it's reached through a reverse proxy rather than directly at "
        "--host:--port (default: DOCFORGE_MCP_PUBLIC_URL env var). Required for tunneled/"
        "proxied setups: it's added to the MCP SDK's own Host-header allowlist (otherwise "
        "every tunneled request is rejected before reaching this server's own auth check), "
        "and in OAuth mode it's also used for metadata/redirects (though those alone would "
        "auto-detect from X-Forwarded-Proto/Host without this flag).",
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


@dataclass
class _OAuthRuntime:
    """Bundles what the OAuth http/both serve path needs, resolved once in ``main()``."""

    client_id: str
    client_secret: str
    store: TokenStore
    extra_redirect_uris: frozenset[str]
    public_url: str | None = None


async def _serve_http_async_oauth(server, host: str, port: int, oauth: _OAuthRuntime) -> None:
    """Serve the http side in OAuth mode: mount the discovery/authorize/token endpoints
    alongside the MCP app, gated by :class:`OAuthAuthMiddleware` instead of the static
    :class:`BearerAuthMiddleware`.

    Metadata/redirect URLs are *not* fixed at ``http://{host}:{port}`` -- that's only the local
    bind address, and would be wrong (typically an unreachable ``127.0.0.1``) for anyone
    connecting through a reverse proxy or tunnel. See ``OAuthAuthMiddleware`` and
    :func:`docforge.oauth.build_oauth_app` for the per-request/``public_url`` resolution.
    """
    import uvicorn

    default_host = f"{host}:{port}"
    mcp_app = server.streamable_http_app()
    oauth_app = build_oauth_app(
        client_id=oauth.client_id,
        client_secret=oauth.client_secret,
        store=oauth.store,
        default_host=default_host,
        public_url=oauth.public_url,
        extra_redirect_uris=oauth.extra_redirect_uris,
    )
    app = OAuthAuthMiddleware(
        mcp_app, oauth_app, oauth.store, default_host=default_host, public_url=oauth.public_url
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    await uvicorn.Server(config).serve()


async def _serve_both_async_oauth(server, host: str, port: int, oauth: _OAuthRuntime) -> None:
    """Run stdio and OAuth-mode http concurrently in this process."""
    import anyio

    async with anyio.create_task_group() as tg:
        tg.start_soon(server.run_stdio_async)
        tg.start_soon(_serve_http_async_oauth, server, host, port, oauth)


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
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = resolve_settings()
    transport = args.transport or os.getenv("DOCFORGE_MCP_TRANSPORT") or DEFAULT_TRANSPORT
    host = args.host or DEFAULT_HOST
    port = args.port or DEFAULT_PORT

    oauth_client_id = args.client_id or os.getenv("DOCFORGE_MCP_CLIENT_ID")
    oauth_client_secret = args.client_secret or os.getenv("DOCFORGE_MCP_CLIENT_SECRET")
    oauth_mode = bool(oauth_client_id or oauth_client_secret)
    token_flags_given = bool(args.token or os.getenv("DOCFORGE_MCP_TOKEN") or args.no_auth)
    if oauth_mode and token_flags_given:
        parser.error(
            "--client-id/--client-secret (OAuth mode) can't be combined with "
            "--token/--no-auth/DOCFORGE_MCP_TOKEN -- pick one auth mode."
        )

    # Resolved before build_server() -- the MCP SDK's own DNS-rebinding Host-header allowlist
    # is fixed at server construction time, so a tunneled/reverse-proxied public host has to be
    # known before the FastMCP() call, not just when advertising OAuth metadata later.
    public_url = args.public_url or os.getenv("DOCFORGE_MCP_PUBLIC_URL")
    public_host = urlparse(public_url).netloc if public_url else None

    server = build_server(
        db_path=settings["db_path"],
        qdrant_url=settings["qdrant_url"],
        qdrant_path=settings["qdrant_path"],
        qdrant_api_key=settings["qdrant_api_key"],
        qdrant_timeout=settings["qdrant_timeout"],
        embed_model=settings["embed_model"],
        host=host,
        port=port,
        public_host=public_host,
    )

    if transport == "stdio":
        server.run()  # never print here -- would corrupt the stdio JSON-RPC stream
        return

    import anyio

    if oauth_mode:
        client_id, client_secret, credentials_generated = resolve_client_credentials(
            oauth_client_id, oauth_client_secret
        )
        extra_redirect_uris = frozenset(
            uri.strip()
            for uri in (os.getenv("DOCFORGE_MCP_REDIRECT_URIS") or "").split(",")
            if uri.strip()
        )
        oauth_runtime = _OAuthRuntime(
            client_id=client_id,
            client_secret=client_secret,
            store=TokenStore(),
            extra_redirect_uris=extra_redirect_uris,
            public_url=public_url,
        )
        # The URL printed below is only a best-effort hint for you to eyeball -- the actual
        # metadata served to clients is resolved per-request (see OAuthAuthMiddleware /
        # build_oauth_app), not fixed at startup, precisely so a reverse proxy or tunnel in
        # front of --host:--port doesn't need this to be exactly right.
        display_url = (public_url or f"http://{host}:{port}").rstrip("/")

        # http or both: stdout is safe to use for stdio-only, but "both" also runs a stdio side
        # in this same process -- so everything user-facing goes to stderr, never stdout, always.
        if host not in _LOOPBACK_HOSTS:
            print(
                f"WARNING: binding to {host} -- anyone who can reach this port can start the "
                "OAuth flow (though completing it still requires the client secret).",
                file=sys.stderr,
            )
        print(f"DocForge MCP server (http, OAuth) listening at {display_url}/mcp", file=sys.stderr)
        print(
            f"Discovery: {display_url}/.well-known/oauth-authorization-server", file=sys.stderr
        )
        if not public_url:
            print(
                "Behind a reverse proxy or tunnel (ngrok, Cloudflare Tunnel)? Pass "
                "--public-url/DOCFORGE_MCP_PUBLIC_URL -- required so the MCP SDK's own "
                "Host-header check accepts those requests at all (metadata/redirects alone "
                "would auto-detect the host without it, but that check won't).",
                file=sys.stderr,
            )
        print(f"Client ID: {client_id}", file=sys.stderr)
        if credentials_generated:
            print(f"Client secret (generated fresh this run): {client_secret}", file=sys.stderr)
            print(
                "Neither is saved -- restarting generates new ones, and any live OAuth session "
                "will need to reconnect. Set DOCFORGE_MCP_CLIENT_ID and "
                "DOCFORGE_MCP_CLIENT_SECRET in .env for values that stay stable across "
                "restarts.",
                file=sys.stderr,
            )
        else:
            print("Client secret: <as configured>", file=sys.stderr)
        print("Ctrl+C to stop.", file=sys.stderr)

        if transport == "http":
            anyio.run(_serve_http_async_oauth, server, host, port, oauth_runtime)
        else:
            anyio.run(_serve_both_async_oauth, server, host, port, oauth_runtime)
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
    if not public_url and host in _LOOPBACK_HOSTS:
        print(
            "Exposing this through a reverse proxy or tunnel (ngrok, Cloudflare Tunnel)? Pass "
            "--public-url/DOCFORGE_MCP_PUBLIC_URL, or the MCP SDK's own Host-header check will "
            "reject those requests before they reach this server's auth check.",
            file=sys.stderr,
        )
    print("Ctrl+C to stop.", file=sys.stderr)

    if transport == "http":
        anyio.run(_serve_http_async, server, host, port, token)
    else:
        anyio.run(_serve_both_async, server, host, port, token)


if __name__ == "__main__":
    main()
