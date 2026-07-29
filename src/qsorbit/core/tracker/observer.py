"""Ground observer location."""

from __future__ import annotations

from dataclasses import dataclass

from skyfield.api import wgs84
from skyfield.toposlib import GeographicPosition


@dataclass(frozen=True)
class ObserverLocation:
    """A ground observer's position on Earth, on the WGS84 ellipsoid.

    ``ObserverLocation`` is a value object: immutable, validated on
    construction, and comparable by value.

    Args:
        latitude: Degrees, ``-90.0 <= latitude <= 90.0``. The north pole
            is +90.
        longitude: Degrees, ``-180.0 <= longitude <= 180.0``. East is
            positive; west is negative.
        elevation_m: Height above the WGS84 ellipsoid, in meters.
            Defaults to sea level (0.0). Can be negative (e.g. Death
            Valley) — there's no lower bound worth enforcing.

    Raises:
        ValueError: If ``latitude`` or ``longitude`` is out of range.
    """

    latitude: float
    longitude: float
    elevation_m: float = 0.0

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"latitude must be in [-90.0, 90.0], got {self.latitude}.")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"longitude must be in [-180.0, 180.0], got {self.longitude}.")

    @property
    def skyfield_position(self) -> GeographicPosition:
        """The underlying skyfield ``GeographicPosition``.

        An escape hatch for advanced use, mirroring
        :attr:`Satellite.skyfield_satellite
        <qsorbit.core.tracker.satellite.Satellite.skyfield_satellite>`.
        """
        return wgs84.latlon(self.latitude, self.longitude, elevation_m=self.elevation_m)
