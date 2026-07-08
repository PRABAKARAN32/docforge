"""The manifest: DocForge's persistent memory of what it has seen.

Change detection compares this crawl to the last one -- but "the last one" may have
been months ago, so that knowledge must survive between runs. The manifest is that
memory: a single SQLite file mapping ``(name, url) -> content_hash -> last_seen`` (Decision
5.7). ``name`` is the knowledge base a page belongs to, so one file can track many docs
sites (Docker, nginx, ...) side by side, each isolated.

SQLite is not a server; it's the stdlib ``sqlite3`` module reading/writing one local
file. Zero install, zero services -- ideal for a local tool.

Design choices baked in:
  * **Parameterized queries** (the ``?`` placeholders) everywhere -- never string-format a
    URL into SQL. URLs are untrusted input; this prevents SQL injection.
  * **UPSERT** (``INSERT ... ON CONFLICT DO UPDATE``) so re-recording a page updates it
    instead of duplicating or erroring -- the basis of idempotency (Decision 5.6).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from types import TracebackType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    name          TEXT NOT NULL,
    url           TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    etag          TEXT,
    last_modified TEXT,
    PRIMARY KEY (name, url)
)
"""

# Pages from a pre-multi-collection DB (no `name` column) are migrated under this name.
_LEGACY_NAME = "default"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Manifest:
    """A SQLite-backed record of ``(name, url) -> content_hash -> last_seen``.

    ``name`` scopes each knowledge base (one docs site). Use as a context manager so the
    connection is always closed::

        with Manifest("docforge.db") as m:
            m.upsert_page("docker", url, content_hash)
            previous = m.hashes("docker")

    Pass ``":memory:"`` for an ephemeral in-process database (handy in tests).
    """

    def __init__(self, db_path: str) -> None:
        # row_factory lets us read columns by name (row["url"]) instead of by index.
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._migrate()
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def _migrate(self) -> None:
        """Upgrade an old single-collection DB (url PK, no `name`) to the scoped schema.

        Existing rows are moved under the ``default`` knowledge base so nothing is lost.
        """
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(pages)")}
        if not columns or "name" in columns:
            return  # fresh DB, or already migrated

        has_etag = "etag" in columns
        has_last_modified = "last_modified" in columns
        etag_select = "etag" if has_etag else "NULL"
        last_modified_select = "last_modified" if has_last_modified else "NULL"

        self._conn.execute("ALTER TABLE pages RENAME TO pages_legacy")
        self._conn.execute(_SCHEMA)
        self._conn.execute(
            f"""
            INSERT INTO pages (name, url, content_hash, last_seen, etag, last_modified)
            SELECT ?, url, content_hash, last_seen, {etag_select}, {last_modified_select}
            FROM pages_legacy
            """,
            (_LEGACY_NAME,),
        )
        self._conn.execute("DROP TABLE pages_legacy")

    def names(self) -> dict[str, int]:
        """Return ``{knowledge_base_name: page_count}`` for every KB tracked in this DB."""
        rows = self._conn.execute("SELECT name, COUNT(*) AS n FROM pages GROUP BY name")
        return {row["name"]: row["n"] for row in rows}

    def hashes(self, name: str) -> dict[str, str]:
        """Return one KB's ``{url: content_hash}`` -- what diffing compares."""
        rows = self._conn.execute(
            "SELECT url, content_hash FROM pages WHERE name = ?", (name,)
        )
        return {row["url"]: row["content_hash"] for row in rows}

    def validators(self, name: str) -> dict[str, tuple[str | None, str | None]]:
        """Return one KB's ``{url: (etag, last_modified)}`` for 304 pre-checks."""
        rows = self._conn.execute(
            "SELECT url, etag, last_modified FROM pages WHERE name = ?", (name,)
        )
        return {row["url"]: (row["etag"], row["last_modified"]) for row in rows}

    def upsert_page(
        self,
        name: str,
        url: str,
        content_hash: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        last_seen: str | None = None,
    ) -> None:
        """Insert a page in KB ``name``, or update it if ``(name, url)`` already exists.

        ``etag``/``last_modified`` use COALESCE on conflict: passing ``None`` keeps whatever
        was already stored, so an ordinary hash update never wipes existing validators.
        """
        self._conn.execute(
            """
            INSERT INTO pages (name, url, content_hash, last_seen, etag, last_modified)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, url) DO UPDATE SET
                content_hash  = excluded.content_hash,
                last_seen     = excluded.last_seen,
                etag          = COALESCE(excluded.etag, pages.etag),
                last_modified = COALESCE(excluded.last_modified, pages.last_modified)
            """,
            (name, url, content_hash, last_seen or _utc_now_iso(), etag, last_modified),
        )
        self._conn.commit()

    def delete_page(self, name: str, url: str) -> None:
        """Remove a page from KB ``name`` (used when a page is deleted upstream)."""
        self._conn.execute("DELETE FROM pages WHERE name = ? AND url = ?", (name, url))
        self._conn.commit()

    def delete_kb(self, name: str) -> int:
        """Remove an entire knowledge base's pages. Returns how many were deleted."""
        cursor = self._conn.execute("DELETE FROM pages WHERE name = ?", (name,))
        self._conn.commit()
        return cursor.rowcount

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
