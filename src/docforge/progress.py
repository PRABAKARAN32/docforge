"""Live progress bars with a self-correcting ETA, for the crawl and embed phases.

This is UI only and dependency-free. A :class:`ProgressBar` renders a single, overwriting
terminal line (via a carriage return) showing a bar, the count, percent, elapsed time, and an
ETA.

The ETA is *measured, not guessed*: it extrapolates from the average time per completed item
(``elapsed / done * remaining``). So it starts rough and tightens -- and generally shrinks --
as the run proceeds, instead of promising a number we can't know up front.

Kept separate from the CLI and crawler so the pipeline stays UI-agnostic: the bar is wired in
only at ``main()`` and both the writer (``out``) and the ``clock`` are injectable for testing.
"""

from __future__ import annotations

import sys
from collections.abc import Callable


def format_duration(seconds: float) -> str:
    """Human, compact duration: ``'45s'``, ``'3m52s'``, ``'1h02m'`` (rounded to whole seconds)."""
    total = int(max(0.0, round(seconds)))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    hours, rem = divmod(total, 3600)
    return f"{hours}h{rem // 60:02d}m"


def _stdout_write(text: str) -> None:
    """Default writer: to stdout, flushed so the line paints immediately (not line-buffered)."""
    sys.stdout.write(text)
    sys.stdout.flush()


class ProgressBar:
    """A single-line, overwriting progress bar with a self-correcting ETA.

    Call :meth:`update` with the running done-count as work completes, then :meth:`finish`
    once to cap the line with a newline. Rendering is throttled to ``min_interval`` seconds so
    a fast, chatty loop doesn't spend its time drawing (the final tick always renders).
    """

    def __init__(
        self,
        total: int,
        label: str,
        *,
        out: Callable[[str], None] | None = None,
        clock: Callable[[], float] | None = None,
        width: int = 24,
        min_interval: float = 0.1,
    ) -> None:
        import time

        self.total = max(0, int(total))
        self.label = label
        self._out = out if out is not None else _stdout_write
        self._clock = clock if clock is not None else time.monotonic
        self._width = width
        self._min_interval = min_interval
        self._start = self._clock()
        self._last_render: float | None = None
        self._maxlen = 0
        self._done = 0

    def update(self, done: int) -> None:
        """Record progress (``done`` items complete) and repaint the line if not throttled."""
        self._done = done
        now = self._clock()
        final = bool(self.total) and done >= self.total
        if (
            not final
            and self._last_render is not None
            and (now - self._last_render) < self._min_interval
        ):
            return
        self._last_render = now
        self._render(done, now)

    def finish(self) -> None:
        """Paint the final state and end the line with a newline."""
        self._render(self._done, self._clock())
        self._out("\n")

    def _render(self, done: int, now: float) -> None:
        total = self.total
        elapsed = now - self._start
        frac = (done / total) if total else 0.0
        frac = min(1.0, max(0.0, frac))
        filled = int(frac * self._width)
        bar = "█" * filled + "░" * (self._width - filled)  # full block / light shade

        if 0 < done < total:
            eta = (elapsed / done) * (total - done)  # measured rate -> extrapolated remainder
            tail = f"ETA {format_duration(eta)}"
        elif total and done >= total:
            tail = f"done in {format_duration(elapsed)}"
        else:
            tail = "ETA --"  # nothing measured yet

        pct = int(frac * 100)
        line = (
            f"{self.label} ▕{bar}▏ {done}/{total}  {pct:3d}%  "
            f"•  {format_duration(elapsed)} elapsed  •  {tail}"
        )
        # Pad to the widest line seen so a shrinking line (ETA -> "done in") leaves no leftovers.
        pad = " " * max(0, self._maxlen - len(line))
        self._maxlen = max(self._maxlen, len(line))
        self._out("\r" + line + pad)
