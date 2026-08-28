"""TLE loading and satellite position/velocity propagation.

Uses `skyfield <https://rhodesmill.org/skyfield/>`_ (which wraps the SGP4
propagator) to turn a two-line element set into position and velocity at
an arbitrary time. Skyfield handles the orbital mechanics and time-scale
plumbing (UTC/TT/TDB, leap seconds); this module's job is a small,
QSOrbit-shaped API around it — see :class:`Satellite`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from skyfield.api import EarthSatellite

from qsorbit.core.geometry import AzEl
from qsorbit.core.tracker._shared import require_timezone_aware, ts
from qsorbit.core.tracker.exceptions import PropagationError, TleError
from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.state import EciState, TopocentricState


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
            self._sat = EarthSatellite(line1, line2, name, ts)
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
    def norad_id(self) -> int:
        """The satellite's NORAD catalog number, taken from the TLE's line 1/2.

        The stable key for matching a loaded TLE against other data
        that identifies satellites the same way -- curated profiles
        (:mod:`qsorbit.core.profiles`) chief among them. Unlike
        :attr:`name`, which is free text and can differ between TLE
        sources, the catalog number is assigned once by Space-Track and
        never reused.
        """
        return self._sat.model.satnum

    @property
    def epoch(self) -> datetime:
        """The TLE's epoch — the moment its elements are most accurate — as UTC.

        SGP4's accuracy degrades as the requested time moves away from
        the epoch, because a TLE is a snapshot of orbital elements that
        real perturbations steadily invalidate. Element sets are
        generally only reliable for a week or two either side of this
        moment, which is why tracking software refreshes them often.
        """
        return self._sat.epoch.utc_datetime()

    @property
    def skyfield_satellite(self) -> EarthSatellite:
        """The underlying skyfield ``EarthSatellite``.

        An escape hatch for advanced use — :meth:`topocentric_state`
        uses this internally to subtract an observer position; reach
        for it directly only if you need something this class doesn't
        already expose. Prefer :meth:`state_at` and
        :meth:`topocentric_state` for everyday use.
        """
        return self._sat

    def state_at(self, time: datetime) -> EciState:
        """Compute this satellite's position and velocity at ``time``.

        Args:
            time: The instant to compute, as a timezone-aware datetime.
                Any timezone is accepted and converted to UTC internally,
                but naive datetimes are rejected — see
                :func:`~qsorbit.core.tracker._shared.require_timezone_aware`.

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
        require_timezone_aware(time)
        t = ts.from_datetime(time)
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

    def topocentric_state(self, observer: ObserverLocation, time: datetime) -> TopocentricState:
        """Compute where this satellite appears from ``observer`` at ``time``.

        This is what a rotor needs to know where to point, and what a
        Doppler correction needs to know how fast the range is changing.

        Args:
            observer: The ground observer's location.
            time: The instant to compute, as a timezone-aware datetime
                (see :meth:`state_at`).

        Returns:
            The satellite's sky position, range, and range rate as seen
            from ``observer``. The position is an
            :class:`~qsorbit.core.geometry.AzEl` — where the satellite
            *is*, not a rotor command; see :mod:`qsorbit.core.pointing`
            to turn it into one.

        Raises:
            ValueError: If ``time`` is naive (has no ``tzinfo``).
            PropagationError: If SGP4 cannot compute a valid position at
                ``time``.
        """
        require_timezone_aware(time)
        t = ts.from_datetime(time)
        difference = self._sat - observer.skyfield_position
        topocentric = difference.at(t)
        message = getattr(topocentric, "message", None)
        if message is not None:
            raise PropagationError(
                f"SGP4 could not compute a valid position at {time.isoformat()}: {message}"
            )
        alt, az, distance = topocentric.altaz()
        *_, range_rate = topocentric.frame_latlon_and_rates(observer.skyfield_position)
        # az.degrees is always meant to be in [0.0, 360.0) — skyfield's own
        # contract for a compass bearing — but azimuth is geometrically
        # undefined at zenith, so floating-point noise can occasionally land
        # exactly on the 360.0/0.0 boundary or a hair below zero. The `%` here
        # is specifically compensating for that representation edge, not
        # forgiving a genuinely out-of-range value the way AzEl's own strict
        # validation is designed to catch.
        azimuth = az.degrees % 360.0
        return TopocentricState(
            sky_position=AzEl(azimuth=azimuth, elevation=alt.degrees),
            range_km=distance.km,
            range_rate_km_s=range_rate.km_per_s,
        )
