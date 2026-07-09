# Configuration

Copy [`.env.example`](../.env.example) to `.env` in the project root to set these once instead
of repeating CLI flags. `.env` is gitignored — never commit it.

**Precedence for every setting: CLI flag > `.env`/environment variable > built-in default.**

| Variable | Used by | Meaning | Default |
|---|---|---|---|
| `DOCFORGE_DB` | `docforge`, `docforge-mcp` | Manifest database path (page hashes, per knowledge base). | `docforge.db` |
| `QDRANT_URL` | `docforge`, `docforge-mcp` | Qdrant server URL (Docker, native, or remote). | `http://localhost:6333` |
| `QDRANT_PATH` | `docforge`, `docforge-mcp` | Embedded Qdrant folder (no Docker/server at all). Wins over `QDRANT_URL` if both are set. | unset |
| `QDRANT_API_KEY` | `docforge`, `docforge-mcp` | API key for a remote/managed Qdrant (e.g. Qdrant Cloud). | unset |
| `DOCFORGE_EMBED_MODEL` | `docforge`, `docforge-mcp` | fastembed model name. Must stay the same across syncs of one knowledge base — switching models changes the vector dimension. | `BAAI/bge-small-en-v1.5` |
| `DOCFORGE_MCP_TRANSPORT` | `docforge-mcp` | Default transport if `--transport` isn't passed: `stdio`, `http`, or `both`. | `stdio` |
| `DOCFORGE_MCP_TOKEN` | `docforge-mcp` | A stable HTTP auth token, so it doesn't regenerate (and need re-pasting into your client config) on every restart. | unset — a fresh token is generated per run |

## Where to run commands from

`.env` is read from the **current working directory at launch**, not the project folder
automatically. If you're running `docforge`/`docforge-mcp` from somewhere else (a different
directory, or a client that spawns the process, like an MCP client), make sure the launch
command sets the working directory to the project root — e.g. `uv run --directory
/path/to/docforge docforge-mcp` — or your `.env` won't be found.

## The vector store: three ways to run it

Pick one — `QDRANT_PATH`/`--qdrant-path` always wins if set alongside `QDRANT_URL`.

| Mode | Setting | When to use it |
|---|---|---|
| **Docker** | `docker compose up -d` (see [`docker-compose.yml`](../docker-compose.yml)), default `QDRANT_URL` | You already use Docker; data persists in a named volume. |
| **Embedded** | `QDRANT_PATH=./vectors` | No Docker, no server at all — like SQLite, but for vectors. Simplest for personal/local use. |
| **Remote / Cloud** | `QDRANT_URL=https://your-cluster...` + `QDRANT_API_KEY` | A managed cluster (e.g. Qdrant Cloud) or any Qdrant server you run yourself. |

## Secrets

Never commit `QDRANT_API_KEY` or `DOCFORGE_MCP_TOKEN` — both belong in `.env` (gitignored) or a
real environment variable, never in a command you might paste into a shared chat or commit
history. Prefer the environment variable/`.env` form over the equivalent CLI flag for both —
flags land in shell history.
