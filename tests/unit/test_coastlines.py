"""Unit tests for the shipped-coastline loader.

Covers both the shipped file itself (it should load, and look like
coastline data) and the loader's own contract against small synthetic
GeoJSON fixtures (order conversion, error handling) -- same split
``test_catalog.py`` uses for the shipped profile catalogue versus its
own loader logic.
"""

from __future__ import annotations

import json

import pytest

from qsorbit.core.coastlines import DEFAULT_COASTLINES_PATH, load_coastlines


class TestShippedCoastlines:
    def test_the_default_path_exists(self):
        assert DEFAULT_COASTLINES_PATH.is_file()

    def test_loads_more_than_a_hundred_segments(self):
        # Natural Earth's 110m coastline layer ships 134 LineString
        # features; not asserting the exact count keeps this test from
        # breaking if the vendored file is ever refreshed, but a
        # drastically different count would mean the wrong file landed.
        segments = load_coastlines()

        assert len(segments) > 100

    def test_every_segment_has_at_least_two_points(self):
        for segment in load_coastlines():
            assert len(segment) >= 2

    def test_coordinates_land_in_valid_latitude_longitude_ranges(self):
        for segment in load_coastlines():
            for latitude, longitude in segment:
                assert -90.0 <= latitude <= 90.0
                assert -180.0 <= longitude <= 180.0


class TestLoadCoastlines:
    def test_converts_geojsons_lon_lat_order_to_lat_lon(self, tmp_path):
        # GeoJSON coordinates are [longitude, latitude] -- the opposite
        # order from every other lat/lon pair in this project
        # (Subpoint, footprint_circle). The loader's whole job is not
        # leaking that convention past this module.
        geojson_path = tmp_path / "coastline.geojson"
        geojson_path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[-83.0, 40.0], [-82.0, 41.0]],
                            },
                        }
                    ],
                }
            )
        )

        segments = load_coastlines(geojson_path)

        assert segments == (((40.0, -83.0), (41.0, -82.0)),)

    def test_multiple_features_become_multiple_segments(self, tmp_path):
        geojson_path = tmp_path / "coastline.geojson"
        geojson_path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0]]},
                        },
                        {
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": [[10.0, 10.0]]},
                        },
                    ],
                }
            )
        )

        segments = load_coastlines(geojson_path)

        assert len(segments) == 2

    def test_not_json_raises_value_error(self, tmp_path):
        bad_path = tmp_path / "bad.geojson"
        bad_path.write_text("not json at all {")

        with pytest.raises(ValueError, match="not valid JSON"):
            load_coastlines(bad_path)

    def test_not_a_feature_collection_raises_value_error(self, tmp_path):
        bad_path = tmp_path / "bad.geojson"
        bad_path.write_text(json.dumps({"type": "Feature"}))

        with pytest.raises(ValueError, match="FeatureCollection"):
            load_coastlines(bad_path)

    def test_a_non_linestring_geometry_raises_value_error(self, tmp_path):
        bad_path = tmp_path / "bad.geojson"
        bad_path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
                            },
                        }
                    ],
                }
            )
        )

        with pytest.raises(ValueError, match="Polygon"):
            load_coastlines(bad_path)

    def test_missing_file_raises_os_error(self, tmp_path):
        with pytest.raises(OSError):
            load_coastlines(tmp_path / "does-not-exist.geojson")
