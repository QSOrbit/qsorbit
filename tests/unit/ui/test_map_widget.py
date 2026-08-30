"""Tests for the ground-track map widget.

Split the same way :mod:`qsorbit.core.map_projection` and
:mod:`qsorbit.core.orbit_geometry` earned their own test files: the
small pure-ish coordinate helpers (:func:`~qsorbit.ui.map_widget._fit_transform`,
:func:`~qsorbit.ui.map_widget._to_screen`,
:func:`~qsorbit.ui.map_widget._project_point`,
:func:`~qsorbit.ui.map_widget._segment_path`) get direct, hand-derived
checks; :class:`~qsorbit.ui.map_widget.MapWidget` itself gets
construction, feed-wiring, and toggle behaviour, plus one ``grab()``
smoke test per projection matching
:mod:`tests.unit.ui.test_waterfall_widget`'s own convention for "does
this actually paint without raising" -- there is no assertion this
project can make about pixel output that would be worth more than that
smoke test and the geometry tests already covering the math underneath
it.
"""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime

import pytest

# A submodule, not the package: `import PySide6` succeeds on a machine
# with no Qt system libraries, and only `PySide6.QtWidgets` fails --
# with an ImportError for libEGL.so.1 rather than anything mentioning
# Qt. Guarding the package alone let CI die at collection (see
# test_waterfall_widget.py's own note).
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QRectF  # noqa: E402

from qsorbit.core.map_projection import Projection  # noqa: E402
from qsorbit.core.orbit_geometry import footprint_radius_deg  # noqa: E402
from qsorbit.core.picker import PickerEntry  # noqa: E402
from qsorbit.core.profiles import (  # noqa: E402
    AliveRecord,
    AliveStatus,
    SatelliteProfile,
)
from qsorbit.core.tracker import ObserverLocation  # noqa: E402
from qsorbit.ui.map_widget import (  # noqa: E402
    MapWidget,
    _fit_transform,
    _project_point,
    _segment_path,
    _to_screen,
)
from qsorbit.ui.theme import DEFAULT_THEMES_DIR, discover_themes  # noqa: E402
from qsorbit.ui.theme_manager import ThemeManager, accent_bar_color  # noqa: E402

TEME_EXAMPLE_TLE = """\
TEME EXAMPLE
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
"""

OBSERVER = ObserverLocation(latitude=40.0, longitude=-83.0, altitude_m=250.0)
NOW = datetime(2026, 8, 28, 0, 0, 0, tzinfo=UTC)


def _profile(norad_id=5, name="TEME EXAMPLE"):
    return SatelliteProfile(
        norad_id=norad_id,
        name=name,
        transmitters=(),
        alive=AliveRecord(status=AliveStatus.ACTIVE, as_of=NOW.date(), source="test"),
    )


@pytest.fixture(scope="session")
def qapp():
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def manager(qapp):
    return ThemeManager(discover_themes((DEFAULT_THEMES_DIR,)))


@pytest.fixture
def empty_tle_dir(tmp_path):
    """An empty, real TLE directory -- enough for construction, nothing to load."""
    tle_dir = tmp_path / "tles"
    tle_dir.mkdir()
    return tle_dir


@pytest.fixture
def populated_tle_dir(tmp_path):
    """A real TLE directory holding the one TLE this project already trusts."""
    tle_dir = tmp_path / "tles"
    tle_dir.mkdir()
    (tle_dir / "teme.tle").write_text(textwrap.dedent(TEME_EXAMPLE_TLE), encoding="utf-8")
    return tle_dir


@pytest.fixture
def widget(manager, empty_tle_dir):
    return MapWidget(themes=manager, tle_dir=empty_tle_dir, observer=OBSERVER, now=lambda: NOW)


class TestFitTransform:
    def test_flat_fits_the_360x180_extent_width_constrained(self):
        scale, center_x, center_y = _fit_transform(QRectF(0, 0, 400, 400), Projection.FLAT)

        assert scale == pytest.approx(400 / 360)
        assert (center_x, center_y) == pytest.approx((200.0, 200.0))

    def test_flat_fits_the_360x180_extent_height_constrained(self):
        scale, _center_x, _center_y = _fit_transform(QRectF(0, 0, 720, 180), Projection.FLAT)

        assert scale == pytest.approx(1.0)

    def test_globe_fits_the_unit_disk(self):
        scale, center_x, center_y = _fit_transform(QRectF(0, 0, 400, 400), Projection.GLOBE)

        assert scale == pytest.approx(200.0)
        assert (center_x, center_y) == pytest.approx((200.0, 200.0))

    def test_center_follows_the_rects_own_origin(self):
        _scale, center_x, center_y = _fit_transform(QRectF(10, 20, 400, 400), Projection.FLAT)

        assert (center_x, center_y) == pytest.approx((210.0, 220.0))


class TestToScreen:
    def test_origin_maps_to_the_given_center(self):
        point = _to_screen((0.0, 0.0), scale=10.0, center_x=100.0, center_y=50.0)

        assert (point.x(), point.y()) == pytest.approx((100.0, 50.0))

    def test_x_grows_east_same_as_screen_x(self):
        point = _to_screen((1.0, 0.0), scale=10.0, center_x=100.0, center_y=50.0)

        assert point.x() == pytest.approx(110.0)

    def test_y_flips_because_north_is_up_but_screen_y_grows_down(self):
        point = _to_screen((0.0, 1.0), scale=10.0, center_x=100.0, center_y=50.0)

        assert point.y() == pytest.approx(40.0)


class TestProjectPoint:
    def test_flat_matches_equirectangular_directly(self):
        result = _project_point(
            10.0, 20.0, Projection.FLAT, center_latitude_deg=0.0, center_longitude_deg=0.0
        )

        assert result == (20.0, 10.0)

    def test_globe_center_point_projects_to_the_origin(self):
        result = _project_point(
            40.0, -83.0, Projection.GLOBE, center_latitude_deg=40.0, center_longitude_deg=-83.0
        )

        assert result == pytest.approx((0.0, 0.0), abs=1e-9)

    def test_globe_far_side_is_not_visible(self):
        result = _project_point(
            -40.0, 97.0, Projection.GLOBE, center_latitude_deg=40.0, center_longitude_deg=-83.0
        )

        assert result is None


class TestSegmentPath:
    def test_traces_every_point_in_order(self):
        segment = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))

        path = _segment_path(segment, scale=10.0, center_x=100.0, center_y=100.0)

        assert path.elementCount() == 3
        first = path.elementAt(0)
        assert (first.x, first.y) == pytest.approx((100.0, 100.0))
        assert path.elementAt(1).isLineTo()
        assert path.elementAt(2).isLineTo()


class TestConstruction:
    def test_defaults_to_flat_with_the_flat_chip_checked(self, widget):
        assert widget._projection is Projection.FLAT
        assert widget._flat_chip.isChecked()
        assert not widget._globe_chip.isChecked()

    def test_loads_the_shipped_coastlines(self, widget):
        assert len(widget._coastlines) > 0

    def test_starts_with_no_entries_and_no_satellites(self, widget):
        assert widget._entries == ()
        assert widget._satellites == {}


class TestSetVisibleEntries:
    def test_loads_matching_satellites_from_the_tle_directory(self, manager, populated_tle_dir):
        subject = MapWidget(
            themes=manager, tle_dir=populated_tle_dir, observer=OBSERVER, now=lambda: NOW
        )
        entry = PickerEntry(profile=_profile(), next_pass=None, visible_from_latitude=True)

        subject.set_visible_entries((entry,))

        assert 5 in subject._satellites
        assert subject._entries == (entry,)

    def test_an_unmatched_norad_id_leaves_no_satellite_but_does_not_raise(self, widget):
        entry = PickerEntry(
            profile=_profile(norad_id=99999), next_pass=None, visible_from_latitude=True
        )

        widget.set_visible_entries((entry,))

        assert widget._satellites == {}

    def test_the_same_norad_id_set_reuses_the_loaded_dict(self, manager, populated_tle_dir):
        subject = MapWidget(
            themes=manager, tle_dir=populated_tle_dir, observer=OBSERVER, now=lambda: NOW
        )
        entry = PickerEntry(profile=_profile(), next_pass=None, visible_from_latitude=True)
        subject.set_visible_entries((entry,))
        loaded = subject._satellites

        # A second entry naming the same satellite -- a different
        # PickerEntry instance, same NORAD id -- should not trigger a
        # second directory walk: identity, not just equality.
        second_entry = PickerEntry(profile=_profile(), next_pass=None, visible_from_latitude=True)
        subject.set_visible_entries((second_entry,))

        assert subject._satellites is loaded


class TestProjectionToggle:
    def test_checking_globe_switches_the_projection(self, widget):
        widget._globe_chip.setChecked(True)

        assert widget._projection is Projection.GLOBE

    def test_checking_flat_again_switches_back(self, widget):
        widget._globe_chip.setChecked(True)
        widget._flat_chip.setChecked(True)

        assert widget._projection is Projection.FLAT

    def test_the_two_chips_are_mutually_exclusive(self, widget):
        widget._globe_chip.setChecked(True)

        assert not widget._flat_chip.isChecked()


class TestPaintSmoke:
    """Does it actually paint without raising -- nothing pixel-level.

    The geometry each paint call depends on (projection math, polyline
    splitting, footprint circles) already has its own hand-derived
    tests in ``test_map_projection.py`` and ``test_orbit_geometry.py``;
    this only proves ``paintEvent`` reaches the end of every branch it
    can reach without an exception, the same bar
    ``test_waterfall_widget.py`` sets with its own ``grab()`` calls.
    """

    def test_paints_with_no_entries_flat(self, widget):
        widget.resize(400, 300)
        widget.show()

        widget.grab()

    def test_paints_with_no_entries_globe(self, widget):
        widget._globe_chip.setChecked(True)
        widget.resize(400, 300)
        widget.show()

        widget.grab()

    def test_paints_a_real_satellites_track_and_footprint_flat(self, manager, populated_tle_dir):
        subject = MapWidget(
            themes=manager, tle_dir=populated_tle_dir, observer=OBSERVER, now=lambda: NOW
        )
        subject.set_visible_entries(
            (PickerEntry(profile=_profile(), next_pass=None, visible_from_latitude=True),)
        )
        subject.resize(400, 300)
        subject.show()

        subject.grab()

    def test_paints_a_real_satellites_track_and_footprint_globe(self, manager, populated_tle_dir):
        subject = MapWidget(
            themes=manager, tle_dir=populated_tle_dir, observer=OBSERVER, now=lambda: NOW
        )
        subject.set_visible_entries(
            (PickerEntry(profile=_profile(), next_pass=None, visible_from_latitude=True),)
        )
        subject._globe_chip.setChecked(True)
        subject.resize(400, 300)
        subject.show()

        subject.grab()


class TestFootprintIsStroked:
    """The footprint ring is an outline, and the inside of it is not painted.

    This exists because the map reached a real screen drawing solid
    blobs. ``_paint_satellite`` set a fill brush for the satellite's
    marker dot and never unset it, so the footprint ring below it was
    filled, and so was the *next* satellite's ground track on the
    following iteration.

    **Every one of the four paint smoke tests above passed the whole
    time.** They assert that painting does not raise, which is worth
    having and is a different claim from painting the right thing. The
    only way to tell a ring from a disc is to look at a pixel inside it
    -- Session 28's lesson about verifying a rendered result rather than
    the code that was supposed to produce it.

    Filling also defeats the ring's stated design: it is stroked
    precisely so a segment split by the antimeridian or the globe's limb
    degrades to the pieces that are really there, instead of closing
    itself into a polygon that never existed.
    """

    def _interior_samples(self, subject):
        """Screen points strictly inside the footprint ring, derived not guessed.

        Uses the widget's own transform helpers, so the sample points
        follow the projection rather than depending on a hand-measured
        pixel that a layout change would invalidate. Nothing is sampled
        to the *right* of the marker: the satellite's name label is
        drawn there.
        """
        canvas = subject._canvas
        scale, center_x, center_y = _fit_transform(QRectF(canvas.rect()), Projection.FLAT)

        satellite = next(iter(subject._satellites.values()))
        current = satellite.subpoint_at(NOW)
        point = _project_point(
            current.latitude_deg, current.longitude_deg, Projection.FLAT, 0.0, 0.0
        )
        assert point is not None, "the flat projection drops no points"
        base = _to_screen(point, scale, center_x, center_y)

        radius_deg = footprint_radius_deg(satellite.mean_altitude_km)
        offsets_deg = (
            (-0.5, 0.0),
            (-0.3, 0.0),
            (-0.35, 0.35),
            (0.0, 0.4),
            (0.0, -0.4),
        )
        points = []
        for dx, dy in offsets_deg:
            x = int(base.x() + dx * radius_deg * scale)
            y = int(base.y() - dy * radius_deg * scale)
            if 0 <= x < canvas.width() and 0 <= y < canvas.height():
                points.append((x, y))
        assert points, "no interior sample landed on the canvas"
        return points

    def test_the_inside_of_a_footprint_is_not_painted_in_the_track_colour(
        self, manager, populated_tle_dir
    ):
        """The defect, as an assertion.

        Before the fix every one of these samples came back as the
        satellite's own accent colour, because the ring was a filled
        disc. The colour is read from the theme rather than typed here,
        so this cannot drift out of step with the palette.
        """
        subject = MapWidget(
            themes=manager, tle_dir=populated_tle_dir, observer=OBSERVER, now=lambda: NOW
        )
        subject.set_visible_entries(
            (PickerEntry(profile=_profile(), next_pass=None, visible_from_latitude=True),)
        )
        subject.resize(400, 300)
        subject.show()

        image = subject._canvas.grab().toImage()
        filled = accent_bar_color(manager.current, 0)

        painted = [
            (x, y)
            for x, y in self._interior_samples(subject)
            if image.pixelColor(x, y).rgb() == filled.rgb()
        ]
        assert painted == [], (
            f"the footprint interior is filled with the track colour at {painted} -- "
            "the ring is being drawn as a disc"
        )
