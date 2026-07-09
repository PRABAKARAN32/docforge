"""Config loading and vector-store opening, shared by the CLI (``cli.py``) and the MCP server
(``mcp_server.py``).

Kept as its own module (not CLI-specific) because both entry points need the same ``.env``
loading and the same "open a Qdrant collection, embedded or remote" logic -- there's exactly
one way DocForge connects to its store, regardless of which front door called it.
"""

from __future__ import annotations

import os


def load_dotenv(path: str = ".env") -> None:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into the environment (dependency-free).

    Existing environment variables win, so a real ``export`` overrides ``.env``. Blank lines,
    ``#`` comments, an optional ``export`` prefix, and surrounding quotes are handled.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return  # no .env file -> nothing to load

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def open_store(
    qdrant_url: str,
    qdrant_path: str | None,
    api_key: str | None = None,
    timeout: float = 60.0,
    collection: str = "docforge",
):
    """Open the Qdrant store for ``collection`` (one per knowledge base).

    Embedded (--qdrant-path) if given, else the server URL. ``api_key`` authenticates to a
    remote/managed Qdrant (e.g. Qdrant Cloud); ``timeout`` (seconds) covers server requests.
    """
    from docforge.vectorstore import QdrantVectorStore

    if qdrant_path is not None:
        return QdrantVectorStore(path=qdrant_path, collection=collection)
    return QdrantVectorStore(
        url=qdrant_url, api_key=api_key, timeout=timeout, collection=collection
    )


def store_error_hint(qdrant_url: str, qdrant_path: str | None) -> str:
    """A hint tailored to how the user is connecting (local Docker vs. remote/cloud)."""
    if qdrant_path is not None:
        return f"Could not open the embedded vector store at {qdrant_path}."
    if "localhost" in qdrant_url or "127.0.0.1" in qdrant_url:
        return f"Is Qdrant running? Start it with: docker compose up -d  (expected at {qdrant_url})"
    return (
        f"Could not reach the vector store at {qdrant_url}. Check the URL, that QDRANT_API_KEY "
        "is set/correct, your network, and try a larger --qdrant-timeout for a distant cluster."
    )


def resolve_settings(
    *,
    db_path: str | None = None,
    qdrant_url: str | None = None,
    qdrant_path: str | None = None,
    qdrant_api_key: str | None = None,
    qdrant_timeout: float | None = None,
    embed_model: str | None = None,
) -> dict[str, object]:
    """Resolve settings in precedence order: explicit arg (e.g. a CLI flag) > env/.env > default.

    Call :func:`load_dotenv` first so a local ``.env`` has already populated ``os.environ``.
    """
    from docforge.embedder import DEFAULT_MODEL

    return {
        "db_path": db_path or os.getenv("DOCFORGE_DB") or "docforge.db",
        "qdrant_url": qdrant_url or os.getenv("QDRANT_URL") or "http://localhost:6333",
        "qdrant_path": qdrant_path or os.getenv("QDRANT_PATH"),
        "qdrant_api_key": qdrant_api_key or os.getenv("QDRANT_API_KEY"),
        "qdrant_timeout": qdrant_timeout if qdrant_timeout is not None else 60.0,
        "embed_model": embed_model or os.getenv("DOCFORGE_EMBED_MODEL") or DEFAULT_MODEL,
    }
