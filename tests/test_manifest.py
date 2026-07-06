"""Unit tests for the SQLite manifest.

Uses a real file under pytest's tmp_path so we can prove data survives across
separate connections -- the whole point of a persistent manifest.
"""

from docforge.manifest import Manifest


def test_empty_manifest_returns_no_hashes() -> None:
    # A brand-new manifest (first run ever) knows nothing yet.
    with Manifest(":memory:") as m:
        assert m.hashes() == {}


def test_upsert_then_read_back() -> None:
    with Manifest(":memory:") as m:
        m.upsert_page("https://x/a", "hash-a")
        m.upsert_page("https://x/b", "hash-b")
        assert m.hashes() == {"https://x/a": "hash-a", "https://x/b": "hash-b"}


def test_upsert_same_url_updates_not_duplicates() -> None:
    with Manifest(":memory:") as m:
        m.upsert_page("https://x/a", "old")
        m.upsert_page("https://x/a", "new")  # same URL again
        result = m.hashes()
        assert result == {"https://x/a": "new"}  # one row, updated hash


def test_delete_removes_a_page() -> None:
    with Manifest(":memory:") as m:
        m.upsert_page("https://x/a", "hash-a")
        m.delete_page("https://x/a")
        assert m.hashes() == {}


def test_manifest_persists_across_reopen(tmp_path) -> None:
    db = str(tmp_path / "docforge.db")

    with Manifest(db) as first_run:
        first_run.upsert_page("https://x/a", "hash-a")

    # A completely separate connection/run reads the same file back.
    with Manifest(db) as later_run:
        assert later_run.hashes() == {"https://x/a": "hash-a"}


def test_rerun_with_same_data_is_idempotent(tmp_path) -> None:
    db = str(tmp_path / "docforge.db")
    with Manifest(db) as m:
        m.upsert_page("https://x/a", "hash-a")
    with Manifest(db) as m:
        m.upsert_page("https://x/a", "hash-a")  # identical second run
        assert m.hashes() == {"https://x/a": "hash-a"}  # still exactly one row
