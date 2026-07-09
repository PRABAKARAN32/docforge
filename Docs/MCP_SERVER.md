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

## Troubleshooting

- **"No knowledge bases tracked in ..."** — run `docforge sync <url>` first, or the server is
  reading the wrong `.env`/DB path (it reads `.env` from its *launch* working directory).
- **`401 Unauthorized`** — the client's `Authorization` header doesn't match the server's
  current token. Most common cause: the server restarted since you copied the token (an
  auto-generated token is fresh every run, by design) — either re-copy it, or switch to a
  stable `DOCFORGE_MCP_TOKEN`.
- **Empty/irrelevant results** — check `DOCFORGE_EMBED_MODEL` matches what you synced with; a
  mismatched model either errors or returns garbage.
- **Client shows the server as disconnected** — run `docforge-mcp` manually with the exact
  same command/args/working-directory the client uses; an error there is the real cause, just
  easier to see outside the client.
