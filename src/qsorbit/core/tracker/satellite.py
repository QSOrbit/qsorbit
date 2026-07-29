"""TLE loading and satellite position/velocity propagation.

Uses `skyfield <https://rhodesmill.org/skyfield/>`_ (which wraps the SGP4
propagator) to turn a two-line element set into position and velocity at
an arbitrary time. Skyfield handles the orbital mechanics and time-scale
plumbing (UTC/TT/TDB, leap seconds); this module's job is a small,
QSOrbit-shaped API around it — see :class:`Satellite`.

A single skyfield timescale is shared by every ``Satellite`` in the
process, built with ``builtin=True`` so it uses skyfield's bundled
leap-second/delta-T tables instead of downloading fresh ones at import
time. That means no network access is required and results are
identical on every machine and in CI, at the cost of losing sub-second
delta-T precision for dates far from when skyfield's bundled tables
were generated — an acceptable trade for this project.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from skyfield.api import EarthSatellite, load

from qsorbit.core.tracker.exceptions import PropagationError, TleError
from qsorbit.core.tracker.state import EciState

_ts = load.timescale(builtin=True)


class Satellite:
    """A satellite tracked from a two-line element set (TLE).

    Construct via :meth:`from_tle` or :meth:`from_file` rather than
    calling ``__init__`` directly.

    Args:
        line1: The first TLE line.
        line2: The second TLE line.
        name: Satellite name, if known. Defaults to ``"UNKNOWN"``.

    Raises:
        TleError: If the lines cannot be parsed as a TLE.
    """

    def __init__(self, line1: str, line2: str, name: str = "UNKNOWN") -> None:
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            raise TleError(
                "TLE lines must start with '1 ' and '2 ' (the line-number "
                f"field), got {line1[:2]!r} and {line2[:2]!r}. This doesn't "
                "look like a TLE."
            )
        try:
            self._sat = EarthSatellite(line1, line2, name, _ts)
        except ValueError as exc:
            raise TleError(f"Could not parse TLE lines: {exc}") from exc

    @classmethod
    def from_tle(cls, text: str) -> Satellite:
        """Build a ``Satellite`` from TLE text.

        Accepts either the standard 3-line form (a name line followed by
        the two element lines) or a bare 2-line form (just the element
        lines, in which case the name is ``"UNKNOWN"``). Blank lines and
        surrounding whitespace are tolerated.

        Args:
            text: TLE text — 2 or 3 non-blank lines.

        Returns:
            The constructed satellite.

        Raises:
            TleError: If ``text`` doesn't contain exactly 2 or 3
                non-blank lines, or the lines don't parse as a TLE.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) == 2:
            name = "UNKNOWN"
            line1, line2 = lines
        elif len(lines) == 3:
            name, line1, line2 = lines
        else:
            raise TleError(
                "Expected 2 non-blank lines (the TLE lines) or 3 (a name line "
                f"plus the TLE lines), got {len(lines)}."
            )
        return cls(line1, line2, name)

    @classmethod
    def from_file(cls, path: str | Path) -> Satellite:
        """Build a ``Satellite`` by reading a TLE from a file.

        Args:
            path: Path to a file containing TLE text (see :meth:`from_tle`
                for the accepted forms).

        Returns:
            The constructed satellite.

        Raises:
            TleError: If the file's contents don't parse as a TLE.
            OSError: If the file cannot be read.
        """
        return cls.from_tle(Path(path).read_text())

    @property
    def name(self) -> str:
        """The satellite's name, or ``"UNKNOWN"`` if none was given."""
        return self._sat.name

    @property
    def epoch(self) -> datetime:
        """The TLE's epoch — the moment its elements are most accurate — as UTC.

        Element sets are only reliable for a week or two either side of
        this moment; see the "Notes for the work" on this chunk for why
        that matters.
        """
        return self._sat.epoch.utc_datetime()

    @property
    def skyfield_satellite(self) -> EarthSatellite:
        """The underlying skyfield ``EarthSatellite``.

        An escape hatch for advanced use — topocentric conversion (a
        later chunk) needs the raw skyfield object to subtract an
        observer position from. Prefer :meth:`state_at` for everyday use.
        """
        return self._sat

    def state_at(self, time: datetime) -> EciState:
        """Compute this satellite's position and velocity at ``time``.

        Args:
            time: The instant to compute, as a timezone-aware datetime.
                Any timezone is accepted and converted to UTC internally,
                but naive datetimes are rejected — satellite math is
                meaningless without knowing which UTC instant is meant,
                and silently assuming one would be exactly the kind of
                bug that's invisible until it isn't.

        Returns:
            The satellite's position and velocity at ``time``, in the
            GCRS inertial frame (see :class:`~qsorbit.core.tracker.state.EciState`).

        Raises:
            ValueError: If ``time`` is naive (has no ``tzinfo``).
            PropagationError: If SGP4 cannot compute a valid position at
                ``time`` — typically because ``time`` is too far from
                this TLE's epoch for the elements to still describe a
                physically sensible orbit.
        """
        if time.tzinfo is None:
            raise ValueError(
                "time must be timezone-aware (e.g. datetime.now(UTC)). Naive "
                "datetimes are ambiguous about which UTC instant is meant."
            )
        t = _ts.from_datetime(time)
        geocentric = self._sat.at(t)
        if geocentric.message is not None:
            raise PropagationError(
                f"SGP4 could not compute a valid position at {time.isoformat()}: "
                f"{geocentric.message}"
            )
        x, y, z = geocentric.xyz.km
        vx, vy, vz = geocentric.velocity.km_per_s
        return EciState(
            time=time.astimezone(UTC),
            position_km=(x, y, z),
            velocity_km_s=(vx, vy, vz),
        )
