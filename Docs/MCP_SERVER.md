# MCP server: connecting an LLM to your docs

`docforge-mcp` exposes the knowledge bases you built with `docforge sync` as tools an LLM can
call mid-conversation. It adds no new retrieval logic — it's a thin front door onto the same
core `docforge search` uses. See [`GUID.md`](../GUID.md) Decision 5.16 for why ingestion/crawl
is deliberately *not* exposed as a tool.

## The two tools

### `list_docs()`
No arguments. Returns every tracked knowledge base and its page count.

### `search_docs(query, name=None, limit=5)`
- `query` *(required)* — natural-language search text.
- `name` *(optional)* — restrict the search to one exact knowledge base (from `list_docs`).
  **Omit it and every knowledge base is searched at once**, merged and ranked together — this
  mirrors `docforge search`'s own default, so a model doesn't need a `list_docs` round-trip
  just to ask a question.
- `limit` *(optional, default 5)* — maximum chunks returned.

You don't call these directly — you ask the LLM a question in chat, and the model decides
whether a tool call is relevant. That's the point of MCP: it's the model's tool call, not
yours. Small/low-capability models are often unreliable at deciding *when* to call a tool at
all — if nothing gets called, that's usually a model-capability issue, not a server issue.

## Two transports — pick based on who's connecting

`docforge-mcp` speaks MCP two different ways (a third, `both`, runs them simultaneously).

| | `stdio` (default) | `http` |
|---|---|---|
| Who starts the server | the client, as a subprocess | you, once, standalone |
| How the client finds it | a command to run | a URL |
| Needs auth? | No — there's no network involved at all | **Yes, on by default** |
| Use for | Claude Code / Claude Desktop | LM Studio, a custom agent, a different machine |

### `stdio` — for Claude Code / Claude Desktop

```bash
docforge-mcp
```
Sits with no output — correct, it's waiting for a client to spawn it and speak MCP over its
stdin/stdout. Register it with an MCP client rather than running it by hand:

**Claude Code:**
```bash
claude mcp add docforge -- uv run --directory /path/to/docforge docforge-mcp
claude mcp list          # confirm it's connected
```

**Any other stdio-capable client** (add to its `mcpServers` config):
```json
{
  "mcpServers": {
    "docforge": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/docforge", "docforge-mcp"]
    }
  }
}
```

### `http` — for LM Studio, your own agent, or anything that connects by URL

```bash
docforge-mcp --transport http --port 8000
```
```
DocForge MCP server (http) listening at http://127.0.0.1:8000/mcp
Auth token (generated fresh this run): 8c8JWYJyugjd2MaLgSTHOhkU_0B-HOl8B7IoithwNRA
Authorization header: Bearer 8c8JWYJyugjd2MaLgSTHOhkU_0B-HOl8B7IoithwNRA
This token is NOT saved -- restarting the server generates a new one. Set DOCFORGE_MCP_TOKEN
in .env for a token that stays the same across restarts.
```
**No separate token-generation step needed** — a fresh token is minted and printed automatically
(the same idea Jupyter Notebook uses for its own local server). Copy it into your client:

```json
{
  "mcpServers": {
    "docforge": {
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

For a token that stays stable across restarts instead of regenerating each time, set
`DOCFORGE_MCP_TOKEN` in `.env`, or pass `--token` directly — either always wins over
auto-generation. `--no-auth` is the explicit opt-out for a fully trusted, local-only setup.

### OAuth — for Claude.ai / ChatGPT custom connectors

The static token above works for any client you configure by hand, but Claude.ai's and
ChatGPT's *hosted* "custom connector" flows don't take a pasted-in token — they speak a real
OAuth 2.1 authorization-code flow with PKCE, and expect the server to act as its own
authorization server. Setting a Client ID (or Secret) switches `http`/`both` into this mode
instead of the static token — see [GUID.md](../GUID.md) Decision 5.17 for the full design
rationale.

```bash
docforge-mcp --transport http --port 8000 --client-id my-docforge
```
```
DocForge MCP server (http, OAuth) listening at http://127.0.0.1:8000/mcp
Discovery: http://127.0.0.1:8000/.well-known/oauth-authorization-server
Client ID: my-docforge
Client secret (generated fresh this run): 8c8JWYJyugjd2MaLgSTHOhkU_0B-HOl8B7IoithwNRA
Neither is saved -- restarting generates new ones, and any live OAuth session will need to
reconnect. Set DOCFORGE_MCP_CLIENT_ID and DOCFORGE_MCP_CLIENT_SECRET in .env for values that
stay stable across restarts.
```

In Claude.ai or ChatGPT's "Add custom connector" form, enter the server URL
(`http://host:8000/mcp`, or a public URL if you've exposed one) plus the printed Client ID and
Client Secret. The client discovers everything else (the authorization/token endpoints, PKCE
requirements) from `/.well-known/oauth-authorization-server`, redirects you through a one-click
consent page DocForge serves at `/authorize`, and exchanges the resulting code for an access
token at `/token`.

**What this is *not*:** Dynamic Client Registration (a new client per connection) and durable
token storage. The Client ID/Secret are pre-registered once (by you, via flag or `.env`), and
authorization codes/access/refresh tokens live only in memory for that server process — the
same "not saved, restarting generates a new one" trade-off already true of the static token
above. Restarting `docforge-mcp` invalidates any live OAuth session; reconnect the connector in
Claude/ChatGPT's UI to get a fresh one.

Like `--token`, `--client-id`/`--client-secret` are mutually exclusive with `--token`/
`--no-auth` — pick one auth mode per run.

**Exposing this to Claude.ai/ChatGPT almost always means running behind a tunnel** (ngrok,
Cloudflare Tunnel) or a reverse proxy, since those are hosted services and can't reach your
`127.0.0.1`. The OAuth metadata/redirect URLs are *not* fixed to `--host:--port` at startup —
they're detected per request from the `X-Forwarded-Proto`/`Host` headers your tunnel already
sends, so this works with no extra configuration for ngrok and most reverse proxies. If that
detection is ever wrong (a proxy that strips those headers, or you want a fixed value), set
`--public-url`/`DOCFORGE_MCP_PUBLIC_URL` to override it explicitly:

```bash
docforge-mcp --transport http --client-id my-docforge --public-url https://abcd1234.ngrok-free.app
```

### `both` — one process serving stdio and http at once

Useful when a client launches the process over stdio (Claude Code) but you also want a URL
another client can dial into (LM Studio), sharing the same running knowledge bases:

```bash
claude mcp add docforge -- uv run --directory /path/to/docforge docforge-mcp \
  --transport both --port 8000 --token <a-stable-token>
```
Prefer a stable, explicit token here — with `both`, the client (not you) launches the process,
so an auto-generated token would print into a log you'd have to go find rather than a terminal
you're watching.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--transport {stdio,http,both}` | `stdio` (or `DOCFORGE_MCP_TRANSPORT`) | Which mode to run. |
| `--host HOST` | `127.0.0.1` | Bind address for the http side. `0.0.0.0` accepts connections from other machines — strongly discouraged with `--no-auth`. |
| `--port PORT` | `8000` | Bind port for the http side. |
| `--token TOKEN` | unset (or `DOCFORGE_MCP_TOKEN`) | Require this bearer token on http requests. If unset (and `--no-auth` isn't given), one is generated and printed each run. |
| `--no-auth` | off | Run http with no authorization check at all. The explicit, deliberate opt-out — auth is on by default. |
| `--client-id ID` | unset (or `DOCFORGE_MCP_CLIENT_ID`) | Switches http/both into OAuth mode with this pre-registered Client ID. If unset here but `--client-secret` is given, one is generated and printed. Mutually exclusive with `--token`/`--no-auth`. |
| `--client-secret SECRET` | unset (or `DOCFORGE_MCP_CLIENT_SECRET`) | Pre-registered OAuth Client Secret. Generated and printed if unset while OAuth mode is active. |
| `--public-url URL` | unset (or `DOCFORGE_MCP_PUBLIC_URL`); auto-detected per-request otherwise | Externally-reachable base URL to advertise in OAuth metadata, for reverse proxy/tunnel setups (ngrok, Cloudflare Tunnel). OAuth mode only. |

## Troubleshooting

- **"No knowledge bases tracked in ..."** — run `docforge sync <url>` first, or the server is
  reading the wrong `.env`/DB path (it reads `.env` from its *launch* working directory).
- **`401 Unauthorized`** (static token mode) — the client's `Authorization` header doesn't
  match the server's current token. Most common cause: the server restarted since you copied
  the token (an auto-generated token is fresh every run, by design) — either re-copy it, or
  switch to a stable `DOCFORGE_MCP_TOKEN`.
- **`401 Unauthorized`** (OAuth mode) — same root cause, OAuth flavor: the server restarted, so
  every previously issued access/refresh token is gone (in-memory only, by design — see the
  OAuth section above). Reconnect the connector in Claude/ChatGPT's UI to run `/authorize`
  again. If the client never got past discovery at all (no `/authorize` request ever hits the
  server), check that its `WWW-Authenticate` header on the `401` correctly points at
  `/.well-known/oauth-protected-resource` and that the URL is reachable from the client (a
  `127.0.0.1`-only bind won't be reachable from Claude.ai's hosted servers — you'd need a
  public URL, e.g. via a tunnel, for that surface specifically; Claude Code's native OAuth flow
  runs locally and doesn't have this restriction).
- **"127.0.0.1 refused to connect" in a browser, or Claude/ChatGPT can't complete the OAuth
  flow, even though `/.well-known/...` returned 200 through your tunnel** — the server minted
  the correct discovery response but the *client* ended up with a `127.0.0.1` URL somewhere
  (this was a real bug prior to the `X-Forwarded-Proto`/`Host` detection described above; it's
  now automatic for ngrok and most reverse proxies). If you still see it: check your proxy
  actually forwards those headers (some strip them by default), or just set
  `--public-url`/`DOCFORGE_MCP_PUBLIC_URL` explicitly to remove the guesswork.
- **Empty/irrelevant results** — check `DOCFORGE_EMBED_MODEL` matches what you synced with; a
  mismatched model either errors or returns garbage.
- **Client shows the server as disconnected** — run `docforge-mcp` manually with the exact
  same command/args/working-directory the client uses; an error there is the real cause, just
  easier to see outside the client.
