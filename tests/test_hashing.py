"""Unit tests for normalization + hashing.

These are pure functions: fast, deterministic, no network or browser. The headline
tests are the two that prove DocForge's core promise -- cosmetic noise does NOT change
the hash, but real content edits DO.
"""

from docforge.hashing import content_hash, normalize_markdown


def test_normalize_unifies_line_endings() -> None:
    assert normalize_markdown("a\r\nb\rc") == "a\nb\nc\n"


def test_normalize_strips_trailing_whitespace() -> None:
    assert normalize_markdown("hello   \nworld\t") == "hello\nworld\n"


def test_normalize_collapses_blank_runs() -> None:
    assert normalize_markdown("a\n\n\n\n\nb") == "a\n\nb\n"


def test_normalize_drops_volatile_last_updated_line() -> None:
    md = "# Title\n\nReal content.\n\nLast updated: July 5, 2026\n"
    assert "Last updated" not in normalize_markdown(md)
    assert "Real content." in normalize_markdown(md)


def test_content_hash_is_deterministic() -> None:
    md = "# Docs\n\nSome content.\n"
    assert content_hash(md) == content_hash(md)


def test_content_hash_is_sha256_hex() -> None:
    h = content_hash("anything")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_cosmetic_changes_do_not_change_the_hash() -> None:
    # Same meaning; differ only in line endings, trailing spaces, blank lines,
    # and a volatile "last updated" date. The hash must be identical.
    v1 = "# Install\n\nRun the command.\n\nLast updated: 2026-07-01\n"
    v2 = "# Install\r\n\r\n\r\nRun the command.   \n\nLast updated: 2026-07-06\n"
    assert content_hash(v1) == content_hash(v2)


def test_real_content_change_changes_the_hash() -> None:
    before = "# Install\n\nRun `pip install docforge`.\n"
    after = "# Install\n\nRun `uv add docforge`.\n"
    assert content_hash(before) != content_hash(after)
