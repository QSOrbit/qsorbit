"""The Target protocol — anything that can be tracked across the sky.

:class:`Satellite <qsorbit.core.tracker.satellite.Satellite>` is the only
implementation today, but nothing in the pointing path actually needs a
satellite specifically. It needs *something that can say where it is in
the sky at a given time for a given observer*. Stars, planets, and the
Moon all fit that description, and skyfield already computes positions
for all of them.

Keeping this as a protocol rather than hard-typing ``Satellite``
everywhere buys two things QSOrbit wants later:

* **Camera-based alignment calibration.** Pointing at a known star and
  measuring where it actually lands in frame gives rotor misalignment
  directly, with no TLE error, timing error, or signal-strength
  estimation in the way — a substantially better calibration reference
  than a satellite pass.
* **General sky tracking.** Pointing the mast at anything, not just
  amateur radio satellites.

This is a :class:`typing.Protocol`, so conformance is structural:
``Satellite`` satisfies it without importing or subclassing anything,
and so will future target types.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.state import TopocentricState


@runtime_checkable
class Target(Protocol):
    """Something whose sky position can be computed for an observer and time."""

    @property
    def name(self) -> str:
        """A human-readable name, for logs and display."""
        ...

    def topocentric_state(self, observer: ObserverLocation, time: datetime) -> TopocentricState:
        """Compute where this target appears from ``observer`` at ``time``.

        Args:
            observer: The ground observer's location.
            time: The instant to compute, as a timezone-aware datetime.

        Returns:
            The target's sky position, range, and range rate.
        """
        ...
