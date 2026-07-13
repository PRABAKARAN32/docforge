"""Tests for the OAuth 2.1 authorization surface (docforge.oauth) -- pure-function unit tests
plus raw-ASGI endpoint tests, mirroring the style already used for BearerAuthMiddleware in
tests/test_mcp_server.py (no starlette TestClient, no pytest-asyncio -- asyncio.run(...) driving
a hand-built scope/receive/send).
"""

import asyncio
import base64
import hashlib
import secrets
from urllib.parse import parse_qs, urlparse

from docforge.oauth import (
    TokenStore,
    authorization_server_metadata,
    build_oauth_app,
    is_allowed_redirect_uri,
    protected_resource_metadata,
    resolve_client_credentials,
    verify_pkce,
)

CLAUDE_CALLBACK = "https://claude.ai/api/mcp/auth_callback"
CHATGPT_CALLBACK = "https://chatgpt.com/connector_platform_oauth_redirect"


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


# --- resolve_client_credentials ---


def test_resolve_client_credentials_explicit_wins() -> None:
    client_id, client_secret, generated = resolve_client_credentials("my-id", "my-secret")
    assert (client_id, client_secret, generated) == ("my-id", "my-secret", False)


def test_resolve_client_credentials_generates_both_when_nothing_given() -> None:
    client_id, client_secret, generated = resolve_client_credentials(None, None)
    assert generated is True
    assert len(client_id) > 10
    assert len(client_secret) > 20
    assert client_id != client_secret


def test_resolve_client_credentials_generates_only_the_missing_half() -> None:
    client_id, client_secret, generated = resolve_client_credentials("my-id", None)
    assert client_id == "my-id"
    assert generated is True
    assert len(client_secret) > 20


def test_resolve_client_credentials_generates_a_different_pair_each_call() -> None:
    first_id, first_secret, _ = resolve_client_credentials(None, None)
    second_id, second_secret, _ = resolve_client_credentials(None, None)
    assert first_id != second_id
    assert first_secret != second_secret


# --- verify_pkce ---


def test_verify_pkce_accepts_a_matching_verifier() -> None:
    verifier, challenge = _pkce_pair()
    assert verify_pkce(verifier, challenge, "S256") is True


def test_verify_pkce_rejects_a_wrong_verifier() -> None:
    _verifier, challenge = _pkce_pair()
    assert verify_pkce("wrong-verifier", challenge, "S256") is False


def test_verify_pkce_rejects_plain_method() -> None:
    verifier, challenge = _pkce_pair()
    assert verify_pkce(verifier, challenge, "plain") is False


def test_verify_pkce_rejects_empty_verifier() -> None:
    _verifier, challenge = _pkce_pair()
    assert verify_pkce("", challenge, "S256") is False


# --- is_allowed_redirect_uri ---


def test_redirect_uri_allows_claude_callback() -> None:
    assert is_allowed_redirect_uri(CLAUDE_CALLBACK) is True


def test_redirect_uri_allows_chatgpt_callback() -> None:
    assert is_allowed_redirect_uri(CHATGPT_CALLBACK) is True


def test_redirect_uri_allows_loopback_with_any_port() -> None:
    assert is_allowed_redirect_uri("http://127.0.0.1:54231/callback") is True
    assert is_allowed_redirect_uri("http://localhost:1234/callback") is True


def test_redirect_uri_rejects_arbitrary_https_uri() -> None:
    assert is_allowed_redirect_uri("https://evil.example.com/callback") is False


def test_redirect_uri_rejects_empty() -> None:
    assert is_allowed_redirect_uri("") is False


def test_redirect_uri_accepts_configured_extras() -> None:
    extra = frozenset({"https://my-agent.example.com/oauth/callback"})
    assert is_allowed_redirect_uri("https://my-agent.example.com/oauth/callback", extra) is True
    assert is_allowed_redirect_uri("https://other.example.com/callback", extra) is False


# --- metadata builders ---


def test_protected_resource_metadata_shape() -> None:
    meta = protected_resource_metadata("http://127.0.0.1:8000/mcp", "http://127.0.0.1:8000")
    assert meta["resource"] == "http://127.0.0.1:8000/mcp"
    assert meta["authorization_servers"] == ["http://127.0.0.1:8000"]


def test_authorization_server_metadata_has_no_registration_endpoint() -> None:
    meta = authorization_server_metadata("http://127.0.0.1:8000")
    assert meta["authorization_endpoint"] == "http://127.0.0.1:8000/authorize"
    assert meta["token_endpoint"] == "http://127.0.0.1:8000/token"
    assert meta["code_challenge_methods_supported"] == ["S256"]
    assert "registration_endpoint" not in meta  # no DCR, by design


# --- TokenStore ---


def test_token_store_auth_code_is_single_use() -> None:
    store = TokenStore()
    code = store.create_auth_code(
        client_id="cid", redirect_uri="http://localhost/cb",
        code_challenge="chal", code_challenge_method="S256", scope="docforge",
    )
    first = store.consume_auth_code(code, client_id="cid", redirect_uri="http://localhost/cb")
    assert first is not None
    second = store.consume_auth_code(code, client_id="cid", redirect_uri="http://localhost/cb")
    assert second is None


def test_token_store_auth_code_rejects_client_or_redirect_mismatch() -> None:
    store = TokenStore()
    code = store.create_auth_code(
        client_id="cid", redirect_uri="http://localhost/cb",
        code_challenge="chal", code_challenge_method="S256", scope="docforge",
    )
    assert store.consume_auth_code(code, client_id="wrong", redirect_uri="http://localhost/cb") is None


def test_token_store_auth_code_expires(monkeypatch) -> None:
    import docforge.oauth as oauth_module

    monkeypatch.setattr(oauth_module, "DEFAULT_AUTH_CODE_TTL", -1.0)  # already expired
    store = TokenStore()
    code = store.create_auth_code(
        client_id="cid", redirect_uri="http://localhost/cb",
        code_challenge="chal", code_challenge_method="S256", scope="docforge",
    )
    assert store.consume_auth_code(code, client_id="cid", redirect_uri="http://localhost/cb") is None


def test_token_store_validates_issued_access_tokens() -> None:
    store = TokenStore()
    access_token, _refresh_token = store.issue_tokens("docforge")
    assert store.validate_access_token(access_token) is True
    assert store.validate_access_token("not-a-real-token") is False


def test_token_store_refresh_rotation_invalidates_the_old_refresh_token() -> None:
    store = TokenStore()
    _access_token, refresh_token = store.issue_tokens("docforge")
    result = store.rotate_refresh_token(refresh_token)
    assert result is not None
    new_access_token, new_refresh_token = result
    assert new_refresh_token != refresh_token
    assert store.validate_access_token(new_access_token) is True
    assert store.rotate_refresh_token(refresh_token) is None  # old one is dead


def test_token_store_rejects_unknown_refresh_token() -> None:
    store = TokenStore()
    assert store.rotate_refresh_token("never-issued") is None


# --- build_oauth_app (raw ASGI, no test client needed) ---


def _scope(method: str, path: str, *, query_string: bytes = b"", headers: dict | None = None) -> dict:
    header_list = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query_string,
        "headers": header_list,
        "scheme": "http",
        "server": ("127.0.0.1", 8000),
    }


def _call(app, scope: dict, body: bytes = b"") -> tuple[int, dict[str, str], bytes]:
    sent = []
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))

    start = next(m for m in sent if m["type"] == "http.response.start")
    response_body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in start["headers"]}
    return start["status"], headers, response_body


def _app(**overrides):
    defaults = dict(
        client_id="cid",
        client_secret="csecret",
        store=TokenStore(),
        default_host="127.0.0.1:8000",
    )
    defaults.update(overrides)
    return build_oauth_app(**defaults), defaults["store"]


def test_protected_resource_metadata_endpoint() -> None:
    app, _store = _app()
    status, _headers, body = _call(app, _scope("GET", "/.well-known/oauth-protected-resource"))
    assert status == 200
    assert b'"resource"' in body


def test_authorization_server_metadata_endpoint() -> None:
    app, _store = _app()
    status, _headers, body = _call(app, _scope("GET", "/.well-known/oauth-authorization-server"))
    assert status == 200
    assert b'"authorization_endpoint"' in body


def test_authorize_get_renders_a_consent_page_for_valid_params() -> None:
    app, _store = _app()
    _verifier, challenge = _pkce_pair()
    query = (
        f"response_type=code&client_id=cid&redirect_uri={CLAUDE_CALLBACK}"
        f"&code_challenge={challenge}&code_challenge_method=S256&state=xyz"
    ).encode()
    status, _headers, body = _call(app, _scope("GET", "/authorize", query_string=query))
    assert status == 200
    assert b"Approve" in body
    assert b'value="cid"' in body


def test_authorize_get_rejects_wrong_client_id() -> None:
    app, _store = _app()
    _verifier, challenge = _pkce_pair()
    query = (
        f"response_type=code&client_id=someone-else&redirect_uri={CLAUDE_CALLBACK}"
        f"&code_challenge={challenge}&code_challenge_method=S256"
    ).encode()
    status, _headers, body = _call(app, _scope("GET", "/authorize", query_string=query))
    assert status == 400
    assert b"unauthorized_client" in body


def test_authorize_get_rejects_disallowed_redirect_uri() -> None:
    app, _store = _app()
    _verifier, challenge = _pkce_pair()
    query = (
        "response_type=code&client_id=cid&redirect_uri=https://evil.example.com/cb"
        f"&code_challenge={challenge}&code_challenge_method=S256"
    ).encode()
    status, _headers, body = _call(app, _scope("GET", "/authorize", query_string=query))
    assert status == 400
    assert b"invalid_redirect_uri" in body


def test_authorize_post_redirects_with_a_code() -> None:
    app, store = _app()
    _verifier, challenge = _pkce_pair()
    body = (
        f"response_type=code&client_id=cid&redirect_uri={CLAUDE_CALLBACK}"
        f"&code_challenge={challenge}&code_challenge_method=S256&state=xyz"
    ).encode()
    status, headers, _body = _call(
        app,
        _scope(
            "POST", "/authorize", headers={"content-type": "application/x-www-form-urlencoded"}
        ),
        body=body,
    )
    assert status == 302
    location = urlparse(headers["location"])
    params = parse_qs(location.query)
    assert params["state"] == ["xyz"]
    assert len(params["code"][0]) > 10


def test_full_authorization_code_and_refresh_flow() -> None:
    app, store = _app()
    verifier, challenge = _pkce_pair()
    authorize_body = (
        f"response_type=code&client_id=cid&redirect_uri={CLAUDE_CALLBACK}"
        f"&code_challenge={challenge}&code_challenge_method=S256&state=xyz"
    ).encode()
    status, headers, _body = _call(
        app,
        _scope(
            "POST", "/authorize", headers={"content-type": "application/x-www-form-urlencoded"}
        ),
        body=authorize_body,
    )
    assert status == 302
    code = parse_qs(urlparse(headers["location"]).query)["code"][0]

    token_body = (
        f"grant_type=authorization_code&code={code}&redirect_uri={CLAUDE_CALLBACK}"
        f"&client_id=cid&client_secret=csecret&code_verifier={verifier}"
    ).encode()
    status, _headers, body = _call(
        app,
        _scope("POST", "/token", headers={"content-type": "application/x-www-form-urlencoded"}),
        body=token_body,
    )
    assert status == 200
    assert b'"access_token"' in body
    import json

    payload = json.loads(body)
    assert store.validate_access_token(payload["access_token"]) is True

    # the code is single-use -- replaying the same exchange must fail
    status, _headers, body = _call(
        app,
        _scope("POST", "/token", headers={"content-type": "application/x-www-form-urlencoded"}),
        body=token_body,
    )
    assert status == 400
    assert b"invalid_grant" in body

    # refresh rotation
    refresh_body = (
        f"grant_type=refresh_token&refresh_token={payload['refresh_token']}"
        "&client_id=cid&client_secret=csecret"
    ).encode()
    status, _headers, body = _call(
        app,
        _scope("POST", "/token", headers={"content-type": "application/x-www-form-urlencoded"}),
        body=refresh_body,
    )
    assert status == 200
    new_payload = json.loads(body)
    assert new_payload["access_token"] != payload["access_token"]


def test_token_endpoint_rejects_wrong_client_secret() -> None:
    app, store = _app()
    code = store.create_auth_code(
        client_id="cid", redirect_uri=CLAUDE_CALLBACK,
        code_challenge="chal", code_challenge_method="S256", scope="docforge",
    )
    body = (
        f"grant_type=authorization_code&code={code}&redirect_uri={CLAUDE_CALLBACK}"
        "&client_id=cid&client_secret=WRONG&code_verifier=whatever"
    ).encode()
    status, _headers, response_body = _call(
        app,
        _scope("POST", "/token", headers={"content-type": "application/x-www-form-urlencoded"}),
        body=body,
    )
    assert status == 401
    assert b"invalid_client" in response_body


def test_token_endpoint_rejects_mismatched_pkce_verifier() -> None:
    app, store = _app()
    _verifier, challenge = _pkce_pair()
    code = store.create_auth_code(
        client_id="cid", redirect_uri=CLAUDE_CALLBACK,
        code_challenge=challenge, code_challenge_method="S256", scope="docforge",
    )
    body = (
        f"grant_type=authorization_code&code={code}&redirect_uri={CLAUDE_CALLBACK}"
        "&client_id=cid&client_secret=csecret&code_verifier=wrong-verifier"
    ).encode()
    status, _headers, response_body = _call(
        app,
        _scope("POST", "/token", headers={"content-type": "application/x-www-form-urlencoded"}),
        body=body,
    )
    assert status == 400
    assert b"invalid_grant" in response_body


def test_token_endpoint_rejects_unsupported_grant_type() -> None:
    app, _store = _app()
    body = b"grant_type=client_credentials&client_id=cid&client_secret=csecret"
    status, _headers, response_body = _call(
        app,
        _scope("POST", "/token", headers={"content-type": "application/x-www-form-urlencoded"}),
        body=body,
    )
    assert status == 400
    assert b"unsupported_grant_type" in response_body
