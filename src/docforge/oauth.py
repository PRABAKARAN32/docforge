"""OAuth 2.1 authorization server for docforge-mcp's http transport (M4.1: opt-in second auth
mode, alongside the static bearer token in ``mcp_server.py``).

**Why this exists.** The static bearer token (``--token``/``DOCFORGE_MCP_TOKEN``) is fine for a
client you configure by hand -- Claude Code, Claude Desktop, LM Studio. It is *not* what
Claude.ai or ChatGPT require when a user adds a server as a hosted "custom connector": those
clients speak OAuth 2.1 and expect the server to front an authorization server -- Protected
Resource Metadata (RFC 9728), Authorization Server Metadata (RFC 8414), a
``401 WWW-Authenticate`` discovery handshake, an ``/authorize`` consent step, and a ``/token``
endpoint with PKCE. See GUID.md Decision 5.17 for the full reasoning.

**What's deliberately not here: Dynamic Client Registration.** DCR causes a new OAuth client to
be registered on every fresh connection -- fine for a multi-tenant SaaS, wasteful for a
single-user local tool. Instead the client is *pre-registered*: a stable Client ID + Client
Secret, resolved the same way the static token is (:func:`resolve_client_credentials` mirrors
``mcp_server._resolve_token``'s precedence and "generated" semantics exactly).

**What's deliberately not here: persistence.** Authorization codes, access tokens, and refresh
tokens live only in :class:`TokenStore`, in memory, for the lifetime of one server process --
the same "not saved, restarting generates a new one" trade-off already documented for the
static token. Restarting invalidates any live OAuth session; the client reconnects.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse

DEFAULT_AUTH_CODE_TTL = 60.0
DEFAULT_ACCESS_TOKEN_TTL = 3600.0
DEFAULT_REFRESH_TOKEN_TTL = 30 * 24 * 3600.0

# Fixed callbacks for the hosted connector surfaces that don't support DCR-based discovery of
# their own redirect URI -- see GUID.md Decision 5.17 for the sources.
FIXED_REDIRECT_URIS = frozenset(
    {
        "https://claude.ai/api/mcp/auth_callback",
        "https://chatgpt.com/connector_platform_oauth_redirect",
        "https://chatgpt.com/oauth/callback",
        "https://chat.openai.com/oauth/callback",
    }
)
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}


def resolve_client_credentials(
    client_id: str | None, client_secret: str | None
) -> tuple[str, str, bool]:
    """Resolve the pre-registered OAuth client: explicit > freshly generated.

    Mirrors :func:`docforge.mcp_server._resolve_token`'s precedence and "generated" flag, so
    ``main()`` can print the same "won't survive a restart, set the env var for stability"
    warning it already prints for the static token.
    """
    generated = False
    if not client_id:
        client_id = secrets.token_urlsafe(16)
        generated = True
    if not client_secret:
        client_secret = secrets.token_urlsafe(32)
        generated = True
    return client_id, client_secret, generated


def verify_pkce(code_verifier: str, code_challenge: str, method: str) -> bool:
    """Check a PKCE ``code_verifier`` against the ``code_challenge`` stored at ``/authorize``
    time. OAuth 2.1 requires S256 -- ``"plain"`` (or anything else) is rejected outright, not
    silently downgraded to."""
    if method != "S256" or not code_verifier:
        return False
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return hmac.compare_digest(computed, code_challenge)


def is_allowed_redirect_uri(uri: str, extra_allowed: frozenset[str] = frozenset()) -> bool:
    """Is ``uri`` a redirect URI we'll issue an authorization code to?

    An allowlist, not freeform: the fixed hosted Claude.ai/ChatGPT callbacks
    (:data:`FIXED_REDIRECT_URIS`), any loopback address with any port (RFC 8252 SS7.3, for
    native clients like Claude Code -- the port varies per session and the hostname is what
    matters), plus whatever the operator added via ``DOCFORGE_MCP_REDIRECT_URIS``.
    """
    if not uri:
        return False
    if uri in FIXED_REDIRECT_URIS or uri in extra_allowed:
        return True
    parsed = urlparse(uri)
    return parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS


def external_base_url(headers: dict[str, str], *, default_scheme: str, default_host: str) -> str:
    """Best-effort externally-reachable base URL for this server (``scheme://host``, no
    trailing slash), so OAuth metadata/redirects are correct when the server is reached through
    a reverse proxy or tunnel (ngrok, Cloudflare Tunnel, a LAN reverse proxy) rather than
    directly at its bind address.

    Honors ``X-Forwarded-Proto``/``X-Forwarded-Host`` (or plain ``Host``) if present -- these
    only affect what URL *we advertise back*, never an authorization decision, so a spoofed
    header here can't grant access it wouldn't otherwise have. Falls back to
    ``default_scheme://default_host`` (the local bind address) when no such header is present,
    i.e. plain local/LAN access with no proxy in front.
    """
    scheme = headers.get("x-forwarded-proto", "").split(",")[0].strip() or default_scheme
    host = (
        headers.get("x-forwarded-host", "").split(",")[0].strip()
        or headers.get("host", "").strip()
        or default_host
    )
    return f"{scheme}://{host}"


def protected_resource_metadata(resource_url: str, issuer_url: str) -> dict:
    """RFC 9728 Protected Resource Metadata -- tells the client where the authorization server
    is, served at ``/.well-known/oauth-protected-resource``."""
    return {"resource": resource_url, "authorization_servers": [issuer_url]}


def authorization_server_metadata(issuer_url: str) -> dict:
    """RFC 8414 Authorization Server Metadata, served at
    ``/.well-known/oauth-authorization-server``. No ``registration_endpoint`` -- DCR is
    intentionally not supported (see module docstring)."""
    return {
        "issuer": issuer_url,
        "authorization_endpoint": f"{issuer_url}/authorize",
        "token_endpoint": f"{issuer_url}/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post"],
        "scopes_supported": ["docforge"],
    }


@dataclass
class AuthCode:
    client_id: str
    redirect_uri: str
    code_challenge: str
    code_challenge_method: str
    scope: str
    expires_at: float


@dataclass
class _Token:
    scope: str
    expires_at: float


class TokenStore:
    """In-memory authorization codes / access tokens / refresh tokens for one server run.

    Deliberately not persisted -- see the module docstring. A durable store would need its own
    SQLite table, mirroring :class:`docforge.manifest.Manifest`'s pattern; out of scope until
    restart-during-a-live-connector-session proves to be a real recurring problem.
    """

    def __init__(self) -> None:
        self._codes: dict[str, AuthCode] = {}
        self._access_tokens: dict[str, _Token] = {}
        self._refresh_tokens: dict[str, _Token] = {}

    def create_auth_code(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
        code_challenge_method: str,
        scope: str,
    ) -> str:
        code = secrets.token_urlsafe(32)
        self._codes[code] = AuthCode(
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            scope=scope,
            expires_at=time.time() + DEFAULT_AUTH_CODE_TTL,
        )
        return code

    def consume_auth_code(self, code: str, *, client_id: str, redirect_uri: str) -> AuthCode | None:
        """Single-use: pops and returns the code's record, or ``None`` if missing, expired, or
        issued for a different client/redirect_uri. A second call for the same code always
        returns ``None`` -- the pop already removed it."""
        record = self._codes.pop(code, None)
        if record is None:
            return None
        if record.expires_at < time.time():
            return None
        if record.client_id != client_id or record.redirect_uri != redirect_uri:
            return None
        return record

    def issue_tokens(self, scope: str) -> tuple[str, str]:
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)
        self._access_tokens[access_token] = _Token(scope, time.time() + DEFAULT_ACCESS_TOKEN_TTL)
        self._refresh_tokens[refresh_token] = _Token(scope, time.time() + DEFAULT_REFRESH_TOKEN_TTL)
        return access_token, refresh_token

    def validate_access_token(self, token: str) -> bool:
        record = self._access_tokens.get(token)
        return record is not None and record.expires_at >= time.time()

    def rotate_refresh_token(self, refresh_token: str) -> tuple[str, str] | None:
        """Redeem ``refresh_token`` for a new access+refresh token pair, invalidating the old
        refresh token (rotation, not reuse) -- required for public clients by the MCP
        authorization spec; harmless and strictly safer for our confidential client too."""
        record = self._refresh_tokens.pop(refresh_token, None)
        if record is None or record.expires_at < time.time():
            return None
        return self.issue_tokens(record.scope)


def build_oauth_app(
    *,
    client_id: str,
    client_secret: str,
    store: TokenStore,
    default_host: str,
    public_url: str | None = None,
    extra_redirect_uris: frozenset[str] = frozenset(),
):
    """Build the OAuth surface: discovery metadata + ``/authorize`` + ``/token``.

    ``default_host`` is the local bind address (``"127.0.0.1:8000"``) used when nothing else
    tells us how the client actually reached this server. If ``public_url`` is given (explicit
    ``--public-url``/``DOCFORGE_MCP_PUBLIC_URL`` override) it always wins; otherwise the base
    URL advertised in metadata is detected per-request from ``X-Forwarded-Proto``/``Host``
    (see :func:`external_base_url`) -- so it's correct automatically behind a reverse proxy or
    tunnel (ngrok, Cloudflare Tunnel) without any configuration, and falls back to
    ``http://{default_host}`` for plain local/LAN access.

    This module only owns the OAuth endpoints, not request auth on ``/mcp`` itself -- that's
    ``OAuthAuthMiddleware`` in ``mcp_server.py``, which checks ``store.validate_access_token``
    and dispatches either here or to the MCP app.
    """
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
    from starlette.routing import Route

    def _base_url(request: Request) -> str:
        if public_url:
            return public_url.rstrip("/")
        return external_base_url(
            dict(request.headers), default_scheme=request.url.scheme, default_host=default_host
        )

    def _validate_authorize_params(params) -> str | None:
        """Return an OAuth error code, or ``None`` if every required param is valid."""
        if params.get("response_type") != "code":
            return "unsupported_response_type"
        if params.get("client_id") != client_id:
            return "unauthorized_client"
        if not is_allowed_redirect_uri(params.get("redirect_uri", ""), extra_redirect_uris):
            return "invalid_redirect_uri"
        if params.get("code_challenge_method", "S256") != "S256":
            return "invalid_request"
        if not params.get("code_challenge"):
            return "invalid_request"
        return None

    async def protected_resource_metadata_endpoint(request: Request) -> JSONResponse:
        base_url = _base_url(request)
        return JSONResponse(protected_resource_metadata(f"{base_url}/mcp", base_url))

    async def authorization_server_metadata_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(authorization_server_metadata(_base_url(request)))

    async def authorize_get(request: Request):
        params = request.query_params
        error = _validate_authorize_params(params)
        if error:
            return JSONResponse({"error": error}, status_code=400)
        hidden = "".join(
            f'<input type="hidden" name="{key}" value="{value}">' for key, value in params.items()
        )
        return HTMLResponse(
            "<!doctype html><html><body>"
            "<h1>DocForge MCP</h1>"
            "<p>Allow this application to search your local DocForge knowledge bases?</p>"
            f"<form method='post'>{hidden}"
            "<button type='submit'>Approve</button></form>"
            "</body></html>"
        )

    async def authorize_post(request: Request):
        form = await request.form()
        error = _validate_authorize_params(form)
        if error:
            return JSONResponse({"error": error}, status_code=400)
        code = store.create_auth_code(
            client_id=form["client_id"],
            redirect_uri=form["redirect_uri"],
            code_challenge=form["code_challenge"],
            code_challenge_method=form.get("code_challenge_method", "S256"),
            scope=form.get("scope", "docforge"),
        )
        query = {"code": code}
        if form.get("state"):
            query["state"] = form["state"]
        return RedirectResponse(f"{form['redirect_uri']}?{urlencode(query)}", status_code=302)

    def _authenticate_client(form) -> bool:
        return form.get("client_id") == client_id and hmac.compare_digest(
            form.get("client_secret", ""), client_secret
        )

    async def token_endpoint(request: Request):
        form = await request.form()
        grant_type = form.get("grant_type")

        if grant_type == "authorization_code":
            if not _authenticate_client(form):
                return JSONResponse({"error": "invalid_client"}, status_code=401)
            record = store.consume_auth_code(
                form.get("code", ""),
                client_id=client_id,
                redirect_uri=form.get("redirect_uri", ""),
            )
            if record is None or not verify_pkce(
                form.get("code_verifier", ""), record.code_challenge, record.code_challenge_method
            ):
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            access_token, refresh_token = store.issue_tokens(record.scope)
            return JSONResponse(
                {
                    "access_token": access_token,
                    "token_type": "Bearer",
                    "expires_in": int(DEFAULT_ACCESS_TOKEN_TTL),
                    "refresh_token": refresh_token,
                    "scope": record.scope,
                }
            )

        if grant_type == "refresh_token":
            if not _authenticate_client(form):
                return JSONResponse({"error": "invalid_client"}, status_code=401)
            result = store.rotate_refresh_token(form.get("refresh_token", ""))
            if result is None:
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            access_token, refresh_token = result
            return JSONResponse(
                {
                    "access_token": access_token,
                    "token_type": "Bearer",
                    "expires_in": int(DEFAULT_ACCESS_TOKEN_TTL),
                    "refresh_token": refresh_token,
                }
            )

        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    return Starlette(
        routes=[
            Route(
                "/.well-known/oauth-protected-resource", protected_resource_metadata_endpoint
            ),
            Route(
                "/.well-known/oauth-authorization-server",
                authorization_server_metadata_endpoint,
            ),
            Route("/authorize", authorize_get, methods=["GET"]),
            Route("/authorize", authorize_post, methods=["POST"]),
            Route("/token", token_endpoint, methods=["POST"]),
        ]
    )
