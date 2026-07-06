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
    url          TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    last_seen    TEXT NOT NULL
)
"""


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
        self._conn.commit()

    def hashes(self) -> dict[str, str]:
        """Return the whole manifest as ``{url: content_hash}`` -- what diffing compares."""
        rows = self._conn.execute("SELECT url, content_hash FROM pages")
        return {row["url"]: row["content_hash"] for row in rows}

    def upsert_page(self, url: str, content_hash: str, last_seen: str | None = None) -> None:
        """Insert a page, or update its hash/timestamp if the URL already exists."""
        self._conn.execute(
            """
            INSERT INTO pages (url, content_hash, last_seen)
            VALUES (?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                content_hash = excluded.content_hash,
                last_seen    = excluded.last_seen
            """,
            (url, content_hash, last_seen or _utc_now_iso()),
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
