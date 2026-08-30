"""Coarse orbital geometry: can a satellite ever be seen from a given latitude at all.

Pure math, no TLE parsing and no skyfield -- every function here takes
plain numbers (an inclination, an altitude, a latitude) and returns
plain numbers or a bool, the same reasoning :mod:`qsorbit.core.geometry`
gives for staying a leaf module: this is worth testing with hand-picked
values, not TLE fixtures, and it should not need one to run.
:func:`~qsorbit.core.picker.build_picker_entries` is the glue that pulls
:attr:`~qsorbit.core.tracker.satellite.Satellite.inclination_deg` and
:attr:`~qsorbit.core.tracker.satellite.Satellite.mean_altitude_km` off a
real satellite and hands them in here.

**The question this module answers is permanent, not tonight's.** The
roadmap's own example is the reason it exists: "IO-86 is near-equatorial
and never rises from 40 degrees N" is a fact about the orbit and the
station's latitude, true on every night there will ever be, not a
prediction that changes with the clock -- :mod:`qsorbit.core.tracker.
pass_prediction` already answers "is there a pass in the next N hours,"
and this module answers the coarser question underneath it: is a pass
even geometrically possible, ever, from here.

**The geometry, in one picture.** A satellite's ground track -- the path
the sub-satellite point traces over the rotating Earth -- covers every
longitude at every latitude between plus and minus its orbit's
inclination (or, past a 90-degree inclination, between plus and minus
``180 - inclination``; a sun-synchronous bird at 98 degrees reaches only
82 degrees north and south, not 98). A station beyond that latitude band
never has the satellite pass directly overhead, but it can still see it
near the horizon if the satellite is high enough that its visibility
footprint -- the patch of Earth it is above the geometric horizon for --
reaches that far. That footprint's angular radius, as Earth-central
angle, is the closed form :func:`footprint_radius_deg` computes:
``rho = arccos(Re / (Re + h))``. So the station can ever see the
satellite exactly when its own latitude is no farther from the equator
than the ground track's own reach, plus that footprint radius --
:func:`is_ever_visible_from_latitude` puts the two together.

**Spherical Earth, one radius throughout, matching the mockup.** The
tracker elsewhere (:mod:`qsorbit.core.tracker.satellite`,
:mod:`qsorbit.core.tracker.sun`) uses WGS84/skyfield's own more precise
Earth model, because pointing a rotor and correcting Doppler are jobs
precision earns its keep on. This filter's job is coarser -- "is it even
worth trying," not "where exactly is it right now" -- and
``qsorbit-shell-mockup.html``'s own footprint-drawing JS already picked
a single mean Earth radius (6371 km) for exactly this kind of
approximation. Matching it here means this module's answer for a given
inclination and altitude agrees with what the eventual map (Chunk D
PR3) draws as that satellite's footprint, rather than the two silently
disagreeing by the few kilometres separating 6371 from WGS84's
6378.137.
"""

from __future__ import annotations

import math
from typing import Final

#: Mean (volumetric) Earth radius, in kilometres -- matching
#: ``qsorbit-shell-mockup.html``'s own ``Re=6371`` rather than
#: :data:`qsorbit.core.tracker.sun.EARTH_RADIUS_KM`'s WGS84 equatorial
#: value. See the module docstring's "spherical Earth" note for why the
#: two constants deliberately differ.
MEAN_EARTH_RADIUS_KM: Final = 6371.0


def max_ground_track_latitude_deg(inclination_deg: float) -> float:
    """The highest (and by symmetry, lowest) latitude an orbit's ground track ever reaches.

    Args:
        inclination_deg: Orbital inclination in degrees,
            ``0.0 <= inclination_deg <= 180.0`` -- a TLE's own declared
            range, 0 equatorial-prograde, 90 polar, near 180
            equatorial-retrograde.

    Returns:
        The ground track's reach either side of the equator, in
        degrees. Equal to ``inclination_deg`` for a prograde orbit
        (``<= 90``); a retrograde orbit's ground track reaches only
        ``180 - inclination_deg``, which is why a 98-degree
        sun-synchronous bird tops out at 82 degrees rather than passing
        over the poles.

    Raises:
        ValueError: If ``inclination_deg`` is outside ``[0.0, 180.0]``.
    """
    if not 0.0 <= inclination_deg <= 180.0:
        raise ValueError(f"inclination_deg must be in [0.0, 180.0], got {inclination_deg}.")
    return inclination_deg if inclination_deg <= 90.0 else 180.0 - inclination_deg


def footprint_radius_deg(altitude_km: float) -> float:
    """The angular radius of a satellite's flat-horizon visibility footprint.

    The Earth-central angle, from the sub-satellite point, out to where
    the satellite sits exactly on a flat (0-degree-elevation) horizon --
    a local obstruction is not part of this; see
    :class:`~qsorbit.core.horizon.HorizonMask` for that, applied
    downstream in a real pass search rather than in this coarse gate.

    Args:
        altitude_km: Height above :data:`MEAN_EARTH_RADIUS_KM`'s
            sphere, in kilometres. Must be positive -- see
            :attr:`~qsorbit.core.tracker.satellite.Satellite.mean_altitude_km`.

    Returns:
        The footprint's angular radius, in degrees.

    Raises:
        ValueError: If ``altitude_km`` is not positive.
    """
    if altitude_km <= 0.0:
        raise ValueError(f"altitude_km must be positive, got {altitude_km}.")
    return math.degrees(math.acos(MEAN_EARTH_RADIUS_KM / (MEAN_EARTH_RADIUS_KM + altitude_km)))


def footprint_circle(
    center_latitude_deg: float,
    center_longitude_deg: float,
    radius_deg: float,
    *,
    points: int = 36,
) -> tuple[tuple[float, float], ...]:
    """A ring of ``(latitude_deg, longitude_deg)`` points tracing a footprint's edge.

    The great-circle destination-point construction: starting at the
    center and walking ``radius_deg`` of angular distance along each of
    ``points`` evenly spaced bearings traces the actual circle on the
    sphere's surface, not an ellipse that only looks right near the
    equator -- the map draws this by projecting each returned point
    individually, so a footprint near either pole or crossing the
    antimeridian comes out the right shape without this function
    needing to know anything about the map's projection.

    Args:
        center_latitude_deg: The footprint's center latitude, in
            degrees -- typically a satellite's current
            :class:`~qsorbit.core.tracker.state.Subpoint`.
        center_longitude_deg: The footprint's center longitude, in
            degrees.
        radius_deg: The footprint's angular radius, in degrees -- see
            :func:`footprint_radius_deg`.
        points: How many points to trace the ring with. 36 (one every
            10 degrees of bearing) draws smoothly at any on-screen map
            size this is used for.

    Returns:
        ``points`` ``(latitude_deg, longitude_deg)`` pairs tracing the
        footprint's edge, in bearing order starting from due north. Not
        closed -- the last point does not repeat the first; a caller
        drawing a closed polygon closes it itself.

    Raises:
        ValueError: If ``radius_deg`` is not positive, or ``points`` is
            less than 3.
    """
    if radius_deg <= 0.0:
        raise ValueError(f"radius_deg must be positive, got {radius_deg}.")
    if points < 3:
        raise ValueError(f"points must be at least 3, got {points}.")

    center_lat = math.radians(center_latitude_deg)
    center_lon = math.radians(center_longitude_deg)
    angular_radius = math.radians(radius_deg)

    ring = []
    for i in range(points):
        bearing = math.radians(360.0 * i / points)
        edge_lat = math.asin(
            math.sin(center_lat) * math.cos(angular_radius)
            + math.cos(center_lat) * math.sin(angular_radius) * math.cos(bearing)
        )
        edge_lon = center_lon + math.atan2(
            math.sin(bearing) * math.sin(angular_radius) * math.cos(center_lat),
            math.cos(angular_radius) - math.sin(center_lat) * math.sin(edge_lat),
        )
        # Wrap longitude back into (-180, 180] -- edge_lon drifts outside
        # that range whenever the footprint straddles the antimeridian.
        wrapped_lon = (math.degrees(edge_lon) + 540.0) % 360.0 - 180.0
        ring.append((math.degrees(edge_lat), wrapped_lon))
    return tuple(ring)


def is_ever_visible_from_latitude(
    inclination_deg: float, altitude_km: float, station_latitude_deg: float
) -> bool:
    """Whether an orbit can ever put its satellite above the flat horizon from a given latitude.

    Args:
        inclination_deg: Orbital inclination in degrees -- see
            :func:`max_ground_track_latitude_deg`.
        altitude_km: Orbital altitude in kilometres -- see
            :func:`footprint_radius_deg`.
        station_latitude_deg: The observing station's latitude, degrees,
            ``-90.0 <= station_latitude_deg <= 90.0`` -- matching
            :attr:`~qsorbit.core.tracker.observer.ObserverLocation.latitude`'s
            own range.

    Returns:
        ``True`` if the station's latitude is within the ground track's
        reach plus the footprint radius -- ``False`` means every pass
        prediction this station will ever run against this orbit comes
        back empty, not just tonight's.

    Raises:
        ValueError: If ``inclination_deg`` or ``altitude_km`` is out of
            range (see :func:`max_ground_track_latitude_deg` and
            :func:`footprint_radius_deg`), or ``station_latitude_deg``
            is outside ``[-90.0, 90.0]``.
    """
    if not -90.0 <= station_latitude_deg <= 90.0:
        raise ValueError(
            f"station_latitude_deg must be in [-90.0, 90.0], got {station_latitude_deg}."
        )
    reach = max_ground_track_latitude_deg(inclination_deg) + footprint_radius_deg(altitude_km)
    return abs(station_latitude_deg) <= reach
