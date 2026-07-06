"""Unit tests for diffing. Pure functions -> fast and deterministic."""

from docforge.diff import DiffReport, deletions_to_apply, diff_hashes


def test_first_run_everything_is_new() -> None:
    report = diff_hashes({}, {"a": "1", "b": "2"})
    assert report.new == frozenset({"a", "b"})
    assert report.changed == frozenset()
    assert report.deleted == frozenset()
    assert report.unchanged == frozenset()


def test_four_buckets_classified_correctly() -> None:
    previous = {"keep": "h1", "edit": "h2", "gone": "h3"}
    current = {"keep": "h1", "edit": "CHANGED", "fresh": "h4"}
    report = diff_hashes(previous, current)

    assert report.new == frozenset({"fresh"})
    assert report.changed == frozenset({"edit"})
    assert report.deleted == frozenset({"gone"})
    assert report.unchanged == frozenset({"keep"})


def test_no_changes_yields_all_unchanged() -> None:
    same = {"a": "1", "b": "2"}
    report = diff_hashes(same, dict(same))
    assert report.unchanged == frozenset({"a", "b"})
    assert not report.has_content_changes


def test_has_content_changes_true_when_new_or_changed() -> None:
    assert diff_hashes({}, {"a": "1"}).has_content_changes
    assert diff_hashes({"a": "1"}, {"a": "2"}).has_content_changes


# --- the deletion guard (Decision 5.5) ---

def test_deletions_applied_when_crawl_succeeded() -> None:
    report = diff_hashes({"gone": "h"}, {})
    assert deletions_to_apply(report, crawl_succeeded=True) == frozenset({"gone"})


def test_deletions_suppressed_when_crawl_failed() -> None:
    # A half-failed crawl makes good pages look "missing" -- we must NOT delete them.
    report = diff_hashes({"gone": "h", "also": "h2"}, {})
    assert report.deleted == frozenset({"gone", "also"})  # the truth is computed...
    assert deletions_to_apply(report, crawl_succeeded=False) == frozenset()  # ...but not acted on


def test_diff_report_is_immutable() -> None:
    report = DiffReport(frozenset(), frozenset(), frozenset(), frozenset())
    # frozen dataclass -> attributes can't be reassigned
    import dataclasses

    try:
        report.new = frozenset({"x"})  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:  # pragma: no cover
        raise AssertionError("DiffReport should be immutable")
