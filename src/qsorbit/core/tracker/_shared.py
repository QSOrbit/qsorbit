"""Internal helpers shared across the tracker module.

Not part of the public API — see :mod:`qsorbit.core.tracker` for that.
"""

from __future__ import annotations

from datetime import datetime

from skyfield.api import load

#: A single skyfield timescale shared by every satellite and observer
#: computation in the process. Built with ``builtin=True`` so it uses
#: skyfield's bundled leap-second/delta-T tables instead of downloading
#: fresh ones — no network access required, and identical results on
#: every machine and in CI, at the cost of losing sub-second delta-T
#: precision for dates far from when skyfield's bundled tables were
#: generated. An acceptable trade for this project.
ts = load.timescale(builtin=True)


def require_timezone_aware(time: datetime) -> None:
    """Raise ``ValueError`` if ``time`` has no ``tzinfo``.

    Every tracker computation that asks skyfield for a position at a
    particular instant needs to know which UTC instant is meant —
    naive datetimes are ambiguous, and guessing would be exactly the
    kind of bug that's invisible until it isn't.
    """
    if time.tzinfo is None:
        raise ValueError(
            "time must be timezone-aware (e.g. datetime.now(UTC)). Naive "
            "datetimes are ambiguous about which UTC instant is meant."
        )
