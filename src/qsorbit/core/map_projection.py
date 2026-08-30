"""Turning a latitude/longitude into a flat picture: the two projections the map offers.

Pure math, matching :mod:`qsorbit.core.orbit_geometry`'s own reasoning
for staying a leaf module -- no Qt, no skyfield, worth testing against
hand-derived reference points rather than a rendered picture. Both
functions here use the conventional right-handed map convention (x
increases east, y increases north) rather than a screen convention (y
increases downward) -- turning one into the other is a single sign
flip, and it belongs to whichever widget is actually painting pixels,
not to this module. Keeping that flip out of here means the same
projection math stays correct however many different screens or PDFs
someday draw from it.

**Two projections, one map.** :func:`equirectangular` is the simplest
possible projection -- longitude and latitude plotted directly, so a
straight line of longitude is a straight vertical line on the page.
Everywhere is drawn, including the far side of the world, and shapes
near the poles stretch badly -- the tradeoff every flat world map
makes. :func:`orthographic`, centered on the station, is what a
satellite would see looking straight down from directly overhead:
only one hemisphere is visible at all, drawn the way it would actually
look from space, so a great-circle path -- a satellite's ground track
among them -- looks like the curve it geometrically is instead of the
sawtooth an equirectangular map draws it as near either pole. Point
selection for the toggle between them was Chunk D's own roadmap entry;
the math for each is a standard closed form, not something this
project invented.

**A polyline needs one more thing neither projection gives it alone:
where to break.** A ground track that crosses the antimeridian on the
flat map, or passes to the far hemisphere on the globe, is still one
continuous physical path -- but drawing it as one continuous *line*
would draw a chord straight across the picture that was never really
there. :func:`project_polyline` is the seam between the two single-point
functions above and a widget's own polyline drawing: it projects a
whole track at once and returns it pre-split into the pieces that are
actually safe to connect with a straight line.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum


def equirectangular(latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
    """Plot a point directly by its own coordinates: ``x = longitude``, ``y = latitude``.

    Args:
        latitude_deg: Degrees, -90 to 90.
        longitude_deg: Degrees, -180 to 180.

    Returns:
        ``(x, y)`` in the same degree units as the input -- ``x`` in
        ``[-180, 180]``, ``y`` in ``[-90, 90]``. Every point on Earth
        projects somewhere; there is no "not visible" case for this
        projection, unlike :func:`orthographic`.
    """
    return (longitude_deg, latitude_deg)


def orthographic(
    latitude_deg: float,
    longitude_deg: float,
    *,
    center_latitude_deg: float,
    center_longitude_deg: float,
) -> tuple[float, float] | None:
    """Plot a point as it would look from space, looking straight down at a center point.

    The standard closed-form orthographic projection (Snyder, *Map
    Projections: A Working Manual*, formulas 20-3 and 20-4): the point
    is projected onto the plane tangent to the sphere at
    ``(center_latitude_deg, center_longitude_deg)``, as seen from
    infinitely far away along that tangent's own normal -- which is
    exactly "looking straight down from directly overhead the center
    point," the same view a satellite has of its own subpoint.

    Args:
        latitude_deg: The point's latitude, in degrees.
        longitude_deg: The point's longitude, in degrees.
        center_latitude_deg: The projection's center latitude, in
            degrees -- typically the station's own.
        center_longitude_deg: The projection's center longitude, in
            degrees.

    Returns:
        ``(x, y)`` on the unit disk (``x**2 + y**2 <= 1``), with the
        center point at the origin, or ``None`` if the point is on the
        far side of the globe from the center and so is not visible in
        this view at all -- unlike :func:`equirectangular`, where every
        point always projects somewhere.
    """
    lat = math.radians(latitude_deg)
    center_lat = math.radians(center_latitude_deg)
    delta_lon = math.radians(longitude_deg - center_longitude_deg)

    cos_c = math.sin(center_lat) * math.sin(lat) + math.cos(center_lat) * math.cos(lat) * math.cos(
        delta_lon
    )
    if cos_c < 0.0:
        return None

    x = math.cos(lat) * math.sin(delta_lon)
    y = math.cos(center_lat) * math.sin(lat) - math.sin(center_lat) * math.cos(lat) * math.cos(
        delta_lon
    )
    return (x, y)


class Projection(Enum):
    """Which of the map's two projections to draw a polyline for."""

    FLAT = "flat"
    GLOBE = "globe"


def project_polyline(
    points: Sequence[tuple[float, float]],
    projection: Projection,
    *,
    center_latitude_deg: float = 0.0,
    center_longitude_deg: float = 0.0,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Project a sequence of ``(latitude_deg, longitude_deg)`` points into drawable line segments.

    Splits the track wherever continuing it would draw something that
    was not actually there: a jump across the antimeridian on
    :data:`Projection.FLAT` (a satellite crossing +/-180 degrees
    longitude between two samples, drawn naively, would streak clean
    across the map), or a crossing to the far hemisphere on
    :data:`Projection.GLOBE` (a point :func:`orthographic` returns
    ``None`` for breaks the line rather than being skipped and silently
    reconnecting the points either side of it).

    Args:
        points: ``(latitude_deg, longitude_deg)`` pairs in order --
            typically a :func:`~qsorbit.core.tracker.ground_track.ground_track`
            result.
        projection: Which projection to draw for.
        center_latitude_deg: The view center's latitude, in degrees.
            Only meaningful for :data:`Projection.GLOBE`; ignored for
            :data:`Projection.FLAT`, which has no view center.
        center_longitude_deg: The view center's longitude, in degrees.
            Same caveat as ``center_latitude_deg``.

    Returns:
        Zero or more segments, each a tuple of ``(x, y)`` points in that
        projection's own native units (degrees for
        :data:`Projection.FLAT`, unit-disk coordinates for
        :data:`Projection.GLOBE`) -- safe to draw as a straight
        polyline within each segment, with no line drawn between
        segments. A segment always has at least two points; a single
        isolated visible point between two breaks is dropped rather
        than drawn as a line with nothing to connect to.
    """
    if projection is Projection.FLAT:
        projected = [equirectangular(lat, lon) for lat, lon in points]
        segments = []
        current = [projected[0]] if projected else []
        for (prev_x, _prev_y), (x, y) in zip(projected, projected[1:], strict=False):
            if abs(x - prev_x) > 180.0:
                segments.append(current)
                current = []
            current.append((x, y))
        segments.append(current)
        return tuple(tuple(segment) for segment in segments if len(segment) >= 2)

    segments = []
    current: list[tuple[float, float]] = []
    for lat, lon in points:
        projected_point = orthographic(
            lat,
            lon,
            center_latitude_deg=center_latitude_deg,
            center_longitude_deg=center_longitude_deg,
        )
        if projected_point is None:
            if len(current) >= 2:
                segments.append(current)
            current = []
            continue
        current.append(projected_point)
    if len(current) >= 2:
        segments.append(current)
    return tuple(tuple(segment) for segment in segments)
