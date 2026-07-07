"""The manifest: DocForge's persistent memory of what it has seen.

Change detection compares this crawl to the last one -- but "the last one" may have
been months ago, so that knowledge must survive between runs. The manifest is that
memory: a single SQLite file mapping ``url -> content_hash -> last_seen`` (Decision 5.7).

SQLite is not a server; it's the stdlib ``sqlite3`` module reading/writing one local
file. Zero install, zero services -- ideal for a local tool.

Design choices baked in:
  * **Parameterized queries** (the ``?`` placeholders) everywhere -- never string-format
    a URL into SQL. URLs are untrusted input; this prevents SQL injection.
  * **UPSERT** (``INSERT ... ON CONFLICT DO UPDATE``) so re-recording a page updates it
    instead of duplicating or erroring -- the basis of idempotency (Decision 5.6).

(``chunk_ids`` -- linking a page to its vector-store chunks -- is added in M2 when the
RAG sync needs it. We add columns when a milestone needs them, not speculatively.)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from types import TracebackType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    url           TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    etag          TEXT,
    last_modified TEXT
)
"""

# Columns added after the original schema shipped -> migrate old DBs on open.
_OPTIONAL_COLUMNS = {"etag": "TEXT", "last_modified": "TEXT"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Manifest:
    """A SQLite-backed record of ``url -> content_hash -> last_seen``.

    Use as a context manager so the connection is always closed::

        with Manifest("docforge.db") as m:
            m.upsert_page(url, content_hash)
            previous = m.hashes()

    Pass ``":memory:"`` for an ephemeral in-process database (handy in tests).
    """

    def __init__(self, db_path: str) -> None:
        # row_factory lets us read columns by name (row["url"]) instead of by index.
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after the first schema, so old DBs upgrade cleanly."""
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(pages)")}
        for column, col_type in _OPTIONAL_COLUMNS.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE pages ADD COLUMN {column} {col_type}")

    def hashes(self) -> dict[str, str]:
        """Return the whole manifest as ``{url: content_hash}`` -- what diffing compares."""
        rows = self._conn.execute("SELECT url, content_hash FROM pages")
        return {row["url"]: row["content_hash"] for row in rows}

    def validators(self) -> dict[str, tuple[str | None, str | None]]:
        """Return ``{url: (etag, last_modified)}`` -- the HTTP validators for 304 pre-checks."""
        rows = self._conn.execute("SELECT url, etag, last_modified FROM pages")
        return {row["url"]: (row["etag"], row["last_modified"]) for row in rows}

    def upsert_page(
        self,
        url: str,
        content_hash: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        last_seen: str | None = None,
    ) -> None:
        """Insert a page, or update it if the URL already exists.

        ``etag``/``last_modified`` use COALESCE on conflict: passing ``None`` keeps whatever
        was already stored, so an ordinary hash update never wipes existing validators.
        """
        self._conn.execute(
            """
            INSERT INTO pages (url, content_hash, last_seen, etag, last_modified)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                content_hash  = excluded.content_hash,
                last_seen     = excluded.last_seen,
                etag          = COALESCE(excluded.etag, pages.etag),
                last_modified = COALESCE(excluded.last_modified, pages.last_modified)
            """,
            (url, content_hash, last_seen or _utc_now_iso(), etag, last_modified),
        )
        self._conn.commit()

    def delete_page(self, url: str) -> None:
        """Remove a page from the manifest (used when a page is deleted upstream)."""
        self._conn.execute("DELETE FROM pages WHERE url = ?", (url,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Manifest:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
