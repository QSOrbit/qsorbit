"""Unit tests for load_satellites_by_norad_id().

Reuses the "TEME EXAMPLE" TLE (catalog number 5) test_satellite.py and
test_picker.py already trust.
"""

from __future__ import annotations

import textwrap

from qsorbit.core.tracker import load_satellites_by_norad_id

_TEME_EXAMPLE_TLE = """\
TEME EXAMPLE
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
"""


class TestLoadSatellitesByNoradId:
    def _tle_dir(self, tmp_path):
        directory = tmp_path / "tles"
        directory.mkdir()
        (directory / "teme.tle").write_text(textwrap.dedent(_TEME_EXAMPLE_TLE), encoding="utf-8")
        return directory

    def test_loads_a_requested_satellite(self, tmp_path):
        tle_dir = self._tle_dir(tmp_path)

        satellites = load_satellites_by_norad_id(tle_dir, frozenset({5}))

        assert set(satellites) == {5}
        assert satellites[5].name == "TEME EXAMPLE"

    def test_an_unrequested_satellite_is_not_loaded(self, tmp_path):
        tle_dir = self._tle_dir(tmp_path)

        satellites = load_satellites_by_norad_id(tle_dir, frozenset({99999}))

        assert satellites == {}

    def test_an_empty_request_returns_empty_without_touching_the_directory(self, tmp_path):
        # A nonexistent directory would raise if this actually globbed
        # it -- an empty request should short-circuit before that.
        missing = tmp_path / "does-not-exist"

        assert load_satellites_by_norad_id(missing, frozenset()) == {}

    def test_an_unparseable_tle_is_skipped(self, tmp_path):
        tle_dir = tmp_path / "tles"
        tle_dir.mkdir()
        (tle_dir / "garbage.tle").write_text("not a tle\nat all\n", encoding="utf-8")

        satellites = load_satellites_by_norad_id(tle_dir, frozenset({5}))

        assert satellites == {}
