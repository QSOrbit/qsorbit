"""Loading the shipped coastline outlines the map draws as its background.

The data itself is Natural Earth's public-domain 1:110m coastline
layer (see ``resources/NATURAL-EARTH-LICENSE.txt``), a GeoJSON
``FeatureCollection`` of ``LineString`` features. This module's only
job is turning that on-disk format into the plain
``(latitude_deg, longitude_deg)`` vocabulary the rest of this project
already speaks -- :class:`~qsorbit.core.tracker.state.Subpoint`,
:func:`~qsorbit.core.orbit_geometry.footprint_circle` -- rather than
GeoJSON's own ``[longitude, latitude]`` axis order leaking into code
that has to remember which convention it's reading. No Qt: parsing
JSON off disk needs no display, matching every other "load shipped
data" module in this project (:mod:`qsorbit.core.profiles.catalog`,
:mod:`qsorbit.ui.theme`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

#: Where the shipped coastline data lives, relative to this module.
DEFAULT_COASTLINES_PATH: Final[Path] = (
    Path(__file__).parent.parent / "resources" / "ne_110m_coastline.geojson"
)


def load_coastlines(
    path: str | Path = DEFAULT_COASTLINES_PATH,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Load coastline outlines as plain latitude/longitude polylines.

    Args:
        path: Path to a coastline GeoJSON file, in the same
            ``FeatureCollection`` of ``LineString`` shape as the
            shipped data. Defaults to :data:`DEFAULT_COASTLINES_PATH`.

    Returns:
        One tuple of ``(latitude_deg, longitude_deg)`` points per
        coastline segment -- draw each as an open polyline, not a
        closed shape; a segment reaching the edge of the data (an
        island fully enclosed is the exception) does not imply the
        line should close back on itself.

    Raises:
        OSError: If ``path`` cannot be read.
        ValueError: If the file is not valid JSON, is not a GeoJSON
            ``FeatureCollection``, or contains a feature geometry other
            than ``LineString``.
    """
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if data.get("type") != "FeatureCollection":
        raise ValueError(
            f"{path} is not a GeoJSON FeatureCollection (got type={data.get('type')!r})."
        )

    segments = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry", {})
        geometry_type = geometry.get("type")
        if geometry_type != "LineString":
            raise ValueError(
                f"{path} contains a {geometry_type!r} feature; only LineString is supported."
            )
        segments.append(
            tuple((latitude, longitude) for longitude, latitude in geometry["coordinates"])
        )
    return tuple(segments)
