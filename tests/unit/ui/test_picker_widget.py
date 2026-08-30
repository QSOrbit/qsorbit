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
from datetime import UTC, date, datetime

import pytest

# A submodule, not the package: `import PySide6` succeeds on a machine
# with no Qt system libraries, and only `PySide6.QtWidgets` fails --
# with an ImportError for libEGL.so.1 rather than anything mentioning
# Qt. Guarding the package alone let CI die at collection (see
# test_waterfall_widget.py's own note).
pytest.importorskip("PySide6.QtWidgets")

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
from qsorbit.core.tracker import ObserverLocation  # noqa: E402
from qsorbit.ui.picker_widget import PickerWidget  # noqa: E402

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
        assert widget._table.item(0, 1).text() == "A"

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
        assert widget._table.item(0, 1).text() == "SEVENTY"

    def test_mode_chip_keeps_only_matching_profiles(self, widget):
        fm = _profile(name="FM-SAT", transmitters=(_transmitter(mode=Mode.FM),))
        cw = _profile(name="CW-SAT", transmitters=(_transmitter(mode=Mode.CW),))
        widget._entries = (
            PickerEntry(profile=fm, next_pass=None, visible_from_latitude=True),
            PickerEntry(profile=cw, next_pass=None, visible_from_latitude=True),
        )

        widget._mode_chips[ModeGroup.FM].setChecked(True)

        assert widget._table.rowCount() == 1
        assert widget._table.item(0, 1).text() == "FM-SAT"

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
        assert widget._table.item(0, 1).text() == "BEACON"

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
        assert widget._table.item(0, 1).text() == "REACHABLE"


class TestRowRendering:
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
        assert widget._table.item(0, 6).text() == "dead 2025-06"


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
        assert subject._table.item(0, 1).text() == "TEME EXAMPLE"
        assert subject._table.item(0, 2).text() != "-"  # a real pass, not the placeholder
        assert "catalogue: shipped 2026-08-25 (3 d)" == subject._status_label.text()
        assert subject._status_label.property("role") == "dim"
