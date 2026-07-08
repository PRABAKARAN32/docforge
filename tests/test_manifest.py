"""Unit tests for the SQLite manifest.

Uses a real file under pytest's tmp_path so we can prove data survives across
separate connections -- the whole point of a persistent manifest. Everything is
scoped to a knowledge-base ``name`` so one file can track many docs sites.
"""

from docforge.manifest import Manifest

KB = "docker"  # a knowledge base name used across these tests


def test_empty_manifest_returns_no_hashes() -> None:
    # A brand-new manifest (first run ever) knows nothing yet.
    with Manifest(":memory:") as m:
        assert m.hashes(KB) == {}
        assert m.names() == {}


def test_upsert_then_read_back() -> None:
    with Manifest(":memory:") as m:
        m.upsert_page(KB, "https://x/a", "hash-a")
        m.upsert_page(KB, "https://x/b", "hash-b")
        assert m.hashes(KB) == {"https://x/a": "hash-a", "https://x/b": "hash-b"}


def test_upsert_same_url_updates_not_duplicates() -> None:
    with Manifest(":memory:") as m:
        m.upsert_page(KB, "https://x/a", "old")
        m.upsert_page(KB, "https://x/a", "new")  # same URL again
        assert m.hashes(KB) == {"https://x/a": "new"}  # one row, updated hash


def test_delete_removes_a_page() -> None:
    with Manifest(":memory:") as m:
        m.upsert_page(KB, "https://x/a", "hash-a")
        m.delete_page(KB, "https://x/a")
        assert m.hashes(KB) == {}


def test_different_knowledge_bases_are_isolated() -> None:
    with Manifest(":memory:") as m:
        m.upsert_page("docker", "https://d/a", "h1")
        m.upsert_page("nginx", "https://n/a", "h2")
        assert m.hashes("docker") == {"https://d/a": "h1"}
        assert m.hashes("nginx") == {"https://n/a": "h2"}
        assert m.names() == {"docker": 1, "nginx": 1}


def test_delete_kb_removes_only_that_kb() -> None:
    with Manifest(":memory:") as m:
        m.upsert_page("docker", "https://d/a", "h1")
        m.upsert_page("docker", "https://d/b", "h2")
        m.upsert_page("nginx", "https://n/a", "h3")
        removed = m.delete_kb("docker")
        assert removed == 2
        assert m.names() == {"nginx": 1}


def test_manifest_persists_across_reopen(tmp_path) -> None:
    db = str(tmp_path / "docforge.db")

    with Manifest(db) as first_run:
        first_run.upsert_page(KB, "https://x/a", "hash-a")

    # A completely separate connection/run reads the same file back.
    with Manifest(db) as later_run:
        assert later_run.hashes(KB) == {"https://x/a": "hash-a"}


def test_rerun_with_same_data_is_idempotent(tmp_path) -> None:
    db = str(tmp_path / "docforge.db")
    with Manifest(db) as m:
        m.upsert_page(KB, "https://x/a", "hash-a")
    with Manifest(db) as m:
        m.upsert_page(KB, "https://x/a", "hash-a")  # identical second run
        assert m.hashes(KB) == {"https://x/a": "hash-a"}  # still exactly one row


# --- HTTP validators (for 304 conditional requests) ---

def test_upsert_stores_and_reads_validators() -> None:
    with Manifest(":memory:") as m:
        m.upsert_page(KB, "https://x/a", "h", etag='"E1"', last_modified="Mon, 01 Jul 2026")
        assert m.validators(KB) == {"https://x/a": ('"E1"', "Mon, 01 Jul 2026")}


def test_upsert_without_validators_keeps_existing() -> None:
    # A plain hash update must NOT wipe stored validators (COALESCE).
    with Manifest(":memory:") as m:
        m.upsert_page(KB, "https://x/a", "h1", etag='"E1"', last_modified="Mon")
        m.upsert_page(KB, "https://x/a", "h2")  # no validators passed
        assert m.hashes(KB)["https://x/a"] == "h2"
        assert m.validators(KB)["https://x/a"] == ('"E1"', "Mon")


def test_migration_upgrades_an_old_single_collection_db(tmp_path) -> None:
    import sqlite3

    db = str(tmp_path / "old.db")
    # Simulate a pre-multi-collection DB: url PK, no `name` column.
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE pages (url TEXT PRIMARY KEY, content_hash TEXT NOT NULL, last_seen TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO pages VALUES ('https://x/a', 'h', 't')")
    conn.commit()
    conn.close()

    with Manifest(db) as m:  # opening should migrate old rows under the 'default' KB
        assert m.names() == {"default": 1}
        assert m.hashes("default") == {"https://x/a": "h"}
        m.upsert_page("default", "https://x/a", "h2", etag='"E"', last_modified="Mon")
        assert m.validators("default")["https://x/a"] == ('"E"', "Mon")
