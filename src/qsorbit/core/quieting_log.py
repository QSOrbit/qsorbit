"""A CSV record of what each receive branch was hearing, block by block.

**Why this exists.** A combiner that selects between branches needs a
*switching margin*: how far one branch's quieting must exceed the
other's before switching is worth doing. Below that margin the two
readings are indistinguishable and a selector would chatter, changing
branch on measurement noise rather than on signal. The margin is
therefore a property of the metric, not of the sky, and it has to be
measured rather than assumed.

**It is not the ~2 dB from Session 22.** That number is a *fade depth* —
how far a signal dropped. What a margin has to clear is a *noise floor
on the metric* — how far ``quieting_A - quieting_B`` wanders when the
two branches are hearing the same thing. Different quantities, and using
the first as the second would size the margin against the wrong
phenomenon entirely.

**Why per block rather than a summary.**
:class:`~qsorbit.core.dsp.squelch.SquelchStats` already reports min, max
and last per branch, and none of those can give the difference. A
branch's own spread mixes two things: the signal's real variation, which
is *common* to both branches and cancels in a difference, and the
metric's noise, which is independent and does not. Only a paired,
time-aligned series separates them.

**Two writers, so two things differ from**
:class:`~qsorbit.core.track_log.TrackLog`, which this otherwise mirrors:

*The log owns the time origin.* It takes the absolute instant a block
was on the air and does the subtraction itself, rather than taking an
elapsed figure a caller computed. Two demodulating threads cannot each
hold their own idea of when the run began, and a difference measured
between two series on two origins measures the origins.

*Writes are locked.* Roughly 16 rows per second per branch, so the lock
is uncontended in practice; without it two threads would interleave
inside one row.

**Rows are flushed as they are written**, for the same reason the track
log flushes: the run that ends in a fault is the one whose data is most
worth having.
"""

from __future__ import annotations

import csv
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Final

#: The header row. ``branch`` is the label from station config, so the
#: two series can be told apart and paired; ``gate_open`` is the squelch's
#: own decision at that block, carried because behaviour *near* the
#: threshold is exactly where a selector would chatter.
CSV_COLUMNS: Final = (
    "t_s",
    "branch",
    "quieting_db",
    "gate_open",
)


class QuietingLog:
    """Writes one CSV row per block per branch, flushing as it goes.

    Usage::

        with QuietingLog(path) as log:
            log.record(block.midpoint, "A - Arrow V", 17.4, gate_open=True)

    Args:
        path: Where to write. Opened on entering the context, and an
            existing file is **replaced** -- a log with two runs
            concatenated into it is worse than no log, because the time
            column restarts in the middle and nothing says so.
        now: The clock, injected for tests. Read once, at :meth:`open`,
            to fix the time origin.
    """

    def __init__(self, path: Path | str, *, now: object = None) -> None:
        self._path = Path(path)
        self._now = now if now is not None else _utc_now
        self._handle = None
        self._writer: csv.writer = None  # type: ignore[valid-type]
        self._started_at: datetime | None = None
        self._lock = threading.Lock()
        self._rows = 0
        self._per_branch: dict[str, int] = {}

    @property
    def path(self) -> Path:
        """Where this log is being written."""
        return self._path

    @property
    def rows(self) -> int:
        """How many rows have been written, across every branch."""
        with self._lock:
            return self._rows

    @property
    def rows_per_branch(self) -> dict[str, int]:
        """How many rows each branch wrote, in first-seen order."""
        with self._lock:
            return dict(self._per_branch)

    def open(self) -> None:
        """Create the file, write the header, and fix the time origin."""
        # newline="" is the csv module's documented requirement, not a
        # style choice: without it the writer's own \r\n meets the text
        # layer's newline translation on Windows and every row ends
        # \r\r\n.
        self._handle = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(CSV_COLUMNS)
        self._handle.flush()
        # Taken here rather than from the first row, so that both
        # branches share one origin no matter which of them reads a
        # block first.
        self._started_at = self._now()

    def record(
        self,
        when: datetime,
        branch: str,
        quieting_db: float,
        *,
        gate_open: bool,
    ) -> None:
        """Write one block's measurement.

        Args:
            when: The instant the block was on the air -- its midpoint,
                which is the instant the quieting was measured from.
                **Absolute, not elapsed**: see the module docstring for
                why the log and not the caller owns the origin.
            branch: The branch's label, from station config.
            quieting_db: How far this block's noise sits below full
                deviation.
            gate_open: Whether the squelch was open at this block.

        A ``t_s`` that comes out negative is written as it is rather
        than clamped: it means a block predates the log, which is a
        thing worth seeing rather than hiding.
        """
        if self._writer is None:  # pragma: no cover - guards a misuse
            raise RuntimeError("This log has not been opened.")
        assert self._started_at is not None
        elapsed_s = (when - self._started_at).total_seconds()
        with self._lock:
            self._writer.writerow(
                [
                    f"{elapsed_s:.3f}",
                    branch,
                    f"{quieting_db:.2f}",
                    "1" if gate_open else "0",
                ]
            )
            self._handle.flush()
            self._rows += 1
            self._per_branch[branch] = self._per_branch.get(branch, 0) + 1

    def close(self) -> None:
        """Close the file. Safe to call twice."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None  # type: ignore[assignment]

    def describe(self) -> str:
        """One line for the end-of-run report.

        Broken down per branch rather than reported as a total, because
        a total is exactly what would let one branch write nothing and
        still look right -- and a branch that wrote nothing is the
        failure this whole measurement would be silently built on.
        """
        with self._lock:
            total = self._rows
            per_branch = dict(self._per_branch)
        if not per_branch:
            return f"quieting log: no rows written to {self._path}"
        breakdown = ", ".join(f"{name} {count:,}" for name, count in per_branch.items())
        return f"quieting log: {total:,} row(s) ({breakdown}) written to {self._path}"

    def __enter__(self) -> QuietingLog:
        """Open on entering a ``with`` block."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close on leaving, whether or not the body raised."""
        self.close()


def _utc_now() -> datetime:
    """The current instant, timezone-aware. Matches the rest of the project."""
    return datetime.now(UTC)
