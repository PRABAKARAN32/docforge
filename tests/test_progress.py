"""Tests for the progress bar + ETA, driven by an injected clock (no real time, no terminal)."""

from docforge.progress import ProgressBar, format_duration


def test_format_duration_scales() -> None:
    assert format_duration(0) == "0s"
    assert format_duration(5) == "5s"
    assert format_duration(59.6) == "1m00s"  # rounds up into the next minute
    assert format_duration(65) == "1m05s"
    assert format_duration(3725) == "1h02m"


def _clock(*ticks: float):
    it = iter(ticks)
    return lambda: next(it)


def test_eta_extrapolates_from_measured_rate() -> None:
    lines: list[str] = []
    # start=0 (init), update at t=2 -> 2 of 10 done in 2s => 1s/page => 8 left => ETA 8s.
    bar = ProgressBar(
        10, "Crawling", out=lines.append, clock=_clock(0.0, 2.0), min_interval=0.0
    )
    bar.update(2)

    assert "2/10" in lines[-1]
    assert " 20%" in lines[-1]
    assert "ETA 8s" in lines[-1]
    assert lines[-1].startswith("\r")


def test_finish_reports_total_and_ends_line() -> None:
    lines: list[str] = []
    bar = ProgressBar(4, "Embedding", out=lines.append, clock=_clock(0.0, 10.0, 10.0), min_interval=0.0)
    bar.update(4)  # complete
    bar.finish()

    assert any("4/4" in line and "done in 10s" in line for line in lines)
    assert lines[-1] == "\n"  # the line is capped with a newline


def test_render_is_throttled_but_final_tick_always_shows() -> None:
    lines: list[str] = []
    # init=0.0; update(1)@0.05 renders (first ever); update(2)@0.08 throttled (<0.1 since last);
    # update(3)@2.0 is the final tick of total=3 -> always renders.
    bar = ProgressBar(
        3, "Crawling", out=lines.append, clock=_clock(0.0, 0.05, 0.08, 2.0), min_interval=0.1
    )
    bar.update(1)  # first render
    bar.update(2)  # suppressed by throttle
    bar.update(3)  # final tick -> forced render

    assert len(lines) == 2
    assert "1/3" in lines[0]
    assert "3/3" in lines[1]
