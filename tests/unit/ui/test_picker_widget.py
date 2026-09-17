"""Tests for the target picker widget.

Two different strategies, deliberately kept apart:

* **Filter-chip and rendering behaviour** injects hand-built
  :class:`~qsorbit.core.picker.PickerEntry` objects directly into the
  widget's ``_entries`` and calls its private ``_render_table()`` --
  the same reasoning ``test_picker.py``'s own ``TestSortKey`` gives for
  testing a private surface directly with hand-built values: proving
  what the widget *does with* entries doesn't need a second real TLE,
  and fabricating one by editing ``TEME_EXAMPLE_TLE`` risks the
  checksum-validation trap already hit once in this PR (see
  ``test_picker.py``'s module docstring).
* **The refresh() wiring itself** -- does it actually call
  ``build_picker_entries`` and reach the table -- gets one end-to-end
  test against the one TLE this project already trusts:
  ``TEME_EXAMPLE_TLE``, the same observer and instant
  ``test_picker.py`` and ``test_cli.py``'s ``TestPlan`` already prove
  produce a real pass.
"""

from __future__ import annotations

import textwrap
from datetime import UTC, date, datetime, timedelta

import pytest

# A submodule, not the package: `import PySide6` succeeds on a machine
# with no Qt system libraries, and only `PySide6.QtWidgets` fails --
# with an ImportError for libEGL.so.1 rather than anything mentioning
# Qt. Guarding the package alone let CI die at collection (see
# test_waterfall_widget.py's own note).
pytest.importorskip("PySide6.QtWidgets")

from qsorbit.core.geometry import AzEl  # noqa: E402
from qsorbit.core.horizon import HorizonMask  # noqa: E402
from qsorbit.core.picker import Band, ModeGroup, PickerEntry  # noqa: E402
from qsorbit.core.profiles import (  # noqa: E402
    AliveRecord,
    AliveStatus,
    CatalogManifest,
    Mode,
    ProfileCatalog,
    ReliabilityClass,
    SatelliteProfile,
    Transmitter,
)
from qsorbit.core.tracker import (  # noqa: E402
    ObserverLocation,
    Pass,
    PassEvent,
    VisibleWindow,
)
from qsorbit.ui.picker_widget import _COLUMN_HEADERS, PickerWidget  # noqa: E402


def _column(header: str) -> int:
    """Which table column a header occupies.

    Tests used to spell these as literal indexes, which went quietly
    wrong the moment Chunk F PR4b inserted a column in the middle: the
    tier assertion below started reading the mode cell and failed with
    a confusing message about ``"SSB"``. Deriving the index from the
    widget's own header tuple means inserting a column moves the test
    with it, and a *renamed* column fails here with a clear
    ``KeyError`` rather than an assertion about the wrong cell.
    """
    return _COLUMN_HEADERS.index(header)


TEME_EXAMPLE_TLE = """\
TEME EXAMPLE
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
"""

OBSERVER = ObserverLocation(latitude=40.0, longitude=-83.0, altitude_m=250.0)
NOW = datetime(2026, 8, 28, 0, 0, 0, tzinfo=UTC)


def _alive(status=AliveStatus.ACTIVE, as_of=date(2026, 8, 25)):
    return AliveRecord(status=status, as_of=as_of, source="test")


def _transmitter(
    downlink_hz=435_640_000.0, mode=Mode.SSB, reliability=ReliabilityClass.UNCONDITIONAL
):
    return Transmitter(downlink_hz=downlink_hz, mode=mode, reliability=reliability)


def _profile(norad_id=5, name="RS-44", transmitters=(), alive=None):
    return SatelliteProfile(
        norad_id=norad_id,
        name=name,
        transmitters=transmitters,
        alive=alive if alive is not None else _alive(),
    )


def _pass(*, with_window: bool):
    """A hand-built pass, optionally carrying a naked-eye window.

    Built here rather than propagated, for the reason the module
    docstring gives: what the widget *does with* a window is the
    subject, and whether the sky was really dark at that instant is
    ``test_picker.py``'s and ``test_visible_window.py``'s job.
    """
    aos = PassEvent(time=NOW + timedelta(minutes=30), sky_position=AzEl(180.0, 5.0))
    tca = PassEvent(time=NOW + timedelta(minutes=45), sky_position=AzEl(90.0, 62.0))
    los = PassEvent(time=NOW + timedelta(minutes=60), sky_position=AzEl(20.0, 5.0))
    window = (
        VisibleWindow(
            begins=PassEvent(time=NOW + timedelta(minutes=35), sky_position=AzEl(150.0, 20.0)),
            ends=PassEvent(time=NOW + timedelta(minutes=52), sky_position=AzEl(50.0, 18.0)),
        )
        if with_window
        else None
    )
    return Pass(
        aos=aos,
        los=los,
        tca=tca,
        max_elevation_deg=62.0,
        az_track=(aos, tca, los),
        visible_window=window,
    )


@pytest.fixture
def widget(qapp, tmp_path):
    """A picker widget over an empty, real (but empty) TLE directory.

    Empty rather than absent: this fixture is for the entry-injection
    tests, which overwrite ``_entries`` themselves and never call
    ``refresh()`` again -- what ``refresh()`` found at construction
    doesn't matter to them, only that construction doesn't hit the
    "directory not found" branch.
    """
    tle_dir = tmp_path / "tles"
    tle_dir.mkdir()
    catalog = ProfileCatalog([])
    return PickerWidget(catalog, None, tle_dir, OBSERVER, HorizonMask(), now=lambda: NOW)


class TestFilterChips:
    def test_no_chips_checked_shows_everything(self, widget):
        widget._entries = (
            PickerEntry(
                profile=_profile(name="A", transmitters=(_transmitter(),)),
                next_pass=None,
                visible_from_latitude=True,
            ),
            PickerEntry(
                profile=_profile(name="B", transmitters=()),
                next_pass=None,
                visible_from_latitude=True,
            ),
        )
        widget._render_table()

        assert widget._table.rowCount() == 2

    def test_needs_transmitter_hides_profiles_with_none(self, widget):
        widget._entries = (
            PickerEntry(
                profile=_profile(name="A", transmitters=(_transmitter(),)),
                next_pass=None,
                visible_from_latitude=True,
            ),
            PickerEntry(
                profile=_profile(name="B", transmitters=()),
                next_pass=None,
                visible_from_latitude=True,
            ),
        )

        widget._needs_transmitter_chip.setChecked(True)

        assert widget._table.rowCount() == 1
        assert widget._table.item(0, _column("satellite")).text() == "A"

    def test_band_chip_keeps_only_matching_profiles(self, widget):
        seventy_cm = _profile(
            name="SEVENTY", transmitters=(_transmitter(downlink_hz=435_640_000.0),)
        )
        two_m = _profile(name="TWO", transmitters=(_transmitter(downlink_hz=145_825_000.0),))
        widget._entries = (
            PickerEntry(profile=seventy_cm, next_pass=None, visible_from_latitude=True),
            PickerEntry(profile=two_m, next_pass=None, visible_from_latitude=True),
        )

        widget._band_chips[Band.SEVENTY_CM].setChecked(True)

        assert widget._table.rowCount() == 1
        assert widget._table.item(0, _column("satellite")).text() == "SEVENTY"

    def test_mode_chip_keeps_only_matching_profiles(self, widget):
        fm = _profile(name="FM-SAT", transmitters=(_transmitter(mode=Mode.FM),))
        cw = _profile(name="CW-SAT", transmitters=(_transmitter(mode=Mode.CW),))
        widget._entries = (
            PickerEntry(profile=fm, next_pass=None, visible_from_latitude=True),
            PickerEntry(profile=cw, next_pass=None, visible_from_latitude=True),
        )

        widget._mode_chips[ModeGroup.FM].setChecked(True)

        assert widget._table.rowCount() == 1
        assert widget._table.item(0, _column("satellite")).text() == "FM-SAT"

    def test_reliability_chip_keeps_only_matching_profiles(self, widget):
        beacon = _profile(
            name="BEACON",
            transmitters=(_transmitter(reliability=ReliabilityClass.UNCONDITIONAL),),
        )
        transponder = _profile(
            name="TRANSPONDER",
            transmitters=(_transmitter(reliability=ReliabilityClass.DEPENDENT),),
        )
        widget._entries = (
            PickerEntry(profile=beacon, next_pass=None, visible_from_latitude=True),
            PickerEntry(profile=transponder, next_pass=None, visible_from_latitude=True),
        )

        widget._reliability_chips[ReliabilityClass.UNCONDITIONAL].setChecked(True)

        assert widget._table.rowCount() == 1
        assert widget._table.item(0, _column("satellite")).text() == "BEACON"

    def test_two_band_chips_together_are_an_or_not_an_and(self, widget):
        seventy_cm = _profile(
            name="SEVENTY", transmitters=(_transmitter(downlink_hz=435_640_000.0),)
        )
        two_m = _profile(name="TWO", transmitters=(_transmitter(downlink_hz=145_825_000.0),))
        widget._entries = (
            PickerEntry(profile=seventy_cm, next_pass=None, visible_from_latitude=True),
            PickerEntry(profile=two_m, next_pass=None, visible_from_latitude=True),
        )

        widget._band_chips[Band.SEVENTY_CM].setChecked(True)
        widget._band_chips[Band.TWO_METERS].setChecked(True)

        assert widget._table.rowCount() == 2

    def test_visible_from_latitude_chip_keeps_only_matching_entries(self, widget):
        reachable = _profile(name="REACHABLE", transmitters=(_transmitter(),))
        unreachable = _profile(name="UNREACHABLE", transmitters=(_transmitter(),))
        widget._entries = (
            PickerEntry(profile=reachable, next_pass=None, visible_from_latitude=True),
            PickerEntry(profile=unreachable, next_pass=None, visible_from_latitude=False),
        )

        widget._visible_from_latitude_chip.setChecked(True)

        assert widget._table.rowCount() == 1
        assert widget._table.item(0, _column("satellite")).text() == "REACHABLE"

    def test_naked_eye_chip_keeps_only_entries_whose_pass_has_a_window(self, widget):
        widget._entries = (
            PickerEntry(
                profile=_profile(name="TONIGHT", transmitters=(_transmitter(),)),
                next_pass=_pass(with_window=True),
                visible_from_latitude=True,
            ),
            PickerEntry(
                profile=_profile(name="DAYLIGHT", transmitters=(_transmitter(),)),
                next_pass=_pass(with_window=False),
                visible_from_latitude=True,
            ),
        )

        widget._naked_eye_chip.setChecked(True)

        assert widget._table.rowCount() == 1
        assert widget._table.item(0, _column("satellite")).text() == "TONIGHT"

    def test_the_naked_eye_chip_does_not_say_visible(self, widget):
        # "visible from here" next door already means something else
        # entirely -- whether this orbit can rise at this latitude at
        # all. Two chips both saying "visible" about different
        # questions is the confusion this wording exists to avoid.
        assert widget._naked_eye_chip.text() == "naked-eye"
        assert "visible" not in widget._naked_eye_chip.text()


class TestRowRendering:
    def test_the_naked_eye_column_shows_the_window(self, widget):
        widget._entries = (
            PickerEntry(
                profile=_profile(name="TONIGHT", transmitters=(_transmitter(),)),
                next_pass=_pass(with_window=True),
                visible_from_latitude=True,
            ),
        )
        widget._render_table()

        cell = widget._table.item(0, _column("naked-eye")).text()

        assert cell != "-"
        assert cell.endswith("min")

    def test_a_pass_with_no_window_shows_the_placeholder(self, widget):
        widget._entries = (
            PickerEntry(
                profile=_profile(name="DAYLIGHT", transmitters=(_transmitter(),)),
                next_pass=_pass(with_window=False),
                visible_from_latitude=True,
            ),
        )
        widget._render_table()

        assert widget._table.item(0, _column("naked-eye")).text() == "-"

    def test_the_naked_eye_column_sits_beside_the_pass_it_qualifies(self):
        # Deliberate placement, not incidental: the two cells are only
        # useful read together ("is that whole pass worth going out
        # for, or a slice of it?"). Pinned so that separating them
        # later has to be a decision rather than a side effect.
        assert _column("naked-eye") == _column("next pass (local)") + 1

    def test_a_dead_satellite_gets_a_dim_status_dot(self, widget):
        dead = _profile(
            name="DEAD-SAT",
            transmitters=(_transmitter(),),
            alive=_alive(status=AliveStatus.INACTIVE, as_of=date(2025, 6, 1)),
        )
        widget._entries = (PickerEntry(profile=dead, next_pass=None, visible_from_latitude=True),)
        widget._render_table()

        dot = widget._table.cellWidget(0, 0)
        assert dot.property("role") == "dim"
        assert widget._table.item(0, _column("tier")).text() == "dead 2025-06"


class TestColumnWidths:
    """Every column wide enough for what is in it.

    Found by rendering the widget rather than by a test: the pass
    column had been eliding to ``"19:37 → ..."`` since Chunk D
    shipped it, because six of the seven columns were left in Qt's
    default ``Interactive`` mode at a flat 100 px. Nothing failed,
    nothing warned -- the text is intact in the model and only the
    *painted* cell is cut, so every assertion about cell contents kept
    passing while the operator saw half a pass time.

    ``sizeHintForColumn`` is not what this asserts against: it reports
    the item's own hint (94 px here) and misses the view's margins, so
    a test written against it passes on the broken widget. The header's
    ``sectionSizeHint`` is the number Qt actually needs (117 px), and
    comparing against it is what makes this test able to fail.

    The naked-eye column raised the stake rather than changing the
    argument: a clipped ``"21:03 · 9 min"`` loses the duration,
    which is the whole payload of the cell.
    """

    def _row(self):
        return PickerEntry(
            profile=_profile(name="RS-44", transmitters=(_transmitter(),)),
            next_pass=_pass(with_window=True),
            visible_from_latitude=True,
        )

    def test_no_column_is_narrower_than_the_width_qt_asks_for(self, widget):
        widget._entries = (self._row(),)
        widget._render_table()
        widget.resize(1000, 320)

        header = widget._table.horizontalHeader()
        too_narrow = [
            _COLUMN_HEADERS[column]
            for column in range(len(_COLUMN_HEADERS))
            if widget._table.columnWidth(column) < header.sectionSizeHint(column)
        ]

        assert too_narrow == []

    def test_the_satellite_column_still_absorbs_the_slack(self, widget):
        # The other half of the same layout decision: sizing every
        # column to its contents and stopping there leaves a ragged gap
        # on the right. The test above would pass perfectly well with
        # the Stretch mode deleted, so this pins it: `satellite` is the
        # one column given more room than its own contents need.
        widget._entries = (self._row(),)
        widget._render_table()
        widget.resize(1000, 320)

        header = widget._table.horizontalHeader()
        satellite = _column("satellite")

        assert widget._table.columnWidth(satellite) > header.sectionSizeHint(satellite)


class TestVisibleEntriesAndSignal:
    """The map's own feed: what PR3's ``MapWidget`` is wired to.

    ``visible_entries`` and ``entries_changed`` carry exactly the same
    data ``_render_table()`` already computes for the table -- these
    tests prove that surface independently of the table itself, the
    same "prove what the widget does with entries" reasoning
    ``TestFilterChips`` above gives for injecting hand-built entries
    directly.
    """

    def test_visible_entries_starts_empty_before_any_render(self, qapp, tmp_path):
        tle_dir = tmp_path / "tles"
        tle_dir.mkdir()
        catalog = ProfileCatalog([])

        subject = PickerWidget(catalog, None, tle_dir, OBSERVER, HorizonMask(), now=lambda: NOW)

        # refresh() already ran once at construction (see
        # TestRefreshEndToEnd below), so an empty directory means
        # visible_entries is the empty tuple, not unset.
        assert subject.visible_entries == ()

    def test_visible_entries_reflects_the_active_filters(self, widget):
        matching = _profile(name="MATCH", transmitters=(_transmitter(),))
        filtered_out = _profile(name="OTHER", transmitters=())
        widget._entries = (
            PickerEntry(profile=matching, next_pass=None, visible_from_latitude=True),
            PickerEntry(profile=filtered_out, next_pass=None, visible_from_latitude=True),
        )

        widget._needs_transmitter_chip.setChecked(True)

        assert len(widget.visible_entries) == 1
        assert widget.visible_entries[0].profile.name == "MATCH"

    def test_entries_changed_emits_the_same_entries_as_the_property(self, widget):
        widget._entries = (
            PickerEntry(
                profile=_profile(name="A", transmitters=(_transmitter(),)),
                next_pass=None,
                visible_from_latitude=True,
            ),
        )
        received = []
        widget.entries_changed.connect(received.append)

        widget._render_table()

        assert len(received) == 1
        assert received[0] == widget.visible_entries

    def test_toggling_a_chip_re_emits_the_narrowed_set(self, widget):
        widget._entries = (
            PickerEntry(
                profile=_profile(name="A", transmitters=(_transmitter(),)),
                next_pass=None,
                visible_from_latitude=True,
            ),
            PickerEntry(
                profile=_profile(name="B", transmitters=()),
                next_pass=None,
                visible_from_latitude=True,
            ),
        )
        widget._render_table()
        received = []
        widget.entries_changed.connect(received.append)

        widget._needs_transmitter_chip.setChecked(True)

        assert len(received) == 1
        assert len(received[0]) == 1
        assert received[0][0].profile.name == "A"


class TestMissingTleDirectory:
    def test_shows_a_warning_and_stays_empty(self, qapp, tmp_path):
        missing = tmp_path / "does-not-exist"
        catalog = ProfileCatalog([])

        subject = PickerWidget(catalog, None, missing, OBSERVER, HorizonMask(), now=lambda: NOW)

        assert subject._table.rowCount() == 0
        assert "TLE directory not found" in subject._status_label.text()
        assert subject._status_label.property("role") == "warn"


class TestRefreshEndToEnd:
    def test_refresh_matches_the_tle_and_populates_the_table(self, qapp, tmp_path):
        tle_dir = tmp_path / "tles"
        tle_dir.mkdir()
        (tle_dir / "teme.tle").write_text(textwrap.dedent(TEME_EXAMPLE_TLE), encoding="utf-8")
        catalog = ProfileCatalog([_profile(norad_id=5, name="TEME EXAMPLE")])
        manifest = CatalogManifest(shipped=date(2026, 8, 25))

        subject = PickerWidget(
            catalog, manifest, tle_dir, OBSERVER, HorizonMask(), hours=48.0, now=lambda: NOW
        )

        assert subject._table.rowCount() == 1
        assert subject._table.item(0, _column("satellite")).text() == "TEME EXAMPLE"
        # A real pass, not the placeholder.
        assert subject._table.item(0, _column("next pass (local)")).text() != "-"
        assert "catalogue: shipped 2026-08-25 (3 d)" == subject._status_label.text()
        assert subject._status_label.property("role") == "dim"
