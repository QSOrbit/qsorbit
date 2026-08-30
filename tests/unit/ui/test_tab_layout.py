"""Every label in a built tab must be wide enough for its own text.

This file exists because of a defect that reached a real screen: with a
real rotor attached, four of the Rotor tab's six readout rows were
clipped mid-word -- ``"39131 km, approaching at 0."`` with the rate
itself missing, which is a readout silently dropping the number it
exists to show.

**Nothing in the unit suite could have caught it, and the reason is
worth stating.** The shell was built and rendered against a *mocked*
rotor that had never ticked, so every value was the one-character
placeholder ``"-"`` and no label was ever too narrow. The strings only
get long when real hardware produces real numbers. So the check here is
deliberately not "does this tab look right" -- it is arithmetic that
holds whatever the text says: **a label is clipped when it is given less
width than its own ``sizeHint``.**

That generalises past the two columns being fixed today. PR3's Custom
tab builds widgets from a config file into grid cells nobody has sized
by hand, and Chunk D adds a picker and a map; any of those can squeeze a
label, and this will say so.

**And Chunk D's map did exactly that, in the other direction.** The
prediction above was right about the tab and wrong about the mechanism:
nothing clipped a *label*, the Plan tab starved a *canvas*. A widget
that only implements ``paintEvent`` states no size, so
``minimumSizeHint()`` comes back empty and a non-stretch
:class:`~qsorbit.ui.cards.Card` -- policy ``(Preferred, Minimum)`` --
gave it nothing: a titled card with a working flat/globe toggle above a
sliver of map, on a real screen, while twenty ``test_map_widget.py``
tests passed because they ``grab()`` the widget standalone and supply
its size themselves. So this file now carries a second check with the
same shape as the first: **a widget is starved when it is given less
than its own stated minimum.**
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QLabel, QWidget  # noqa: E402

from qsorbit.core.horizon import HorizonMask  # noqa: E402
from qsorbit.core.pointing import AlignmentOffset, TrackingLoop  # noqa: E402
from qsorbit.core.profiles import (  # noqa: E402
    AliveRecord,
    AliveStatus,
    ProfileCatalog,
    SatelliteProfile,
)
from qsorbit.core.rotor import Rotor  # noqa: E402
from qsorbit.core.rotor.capabilities import AzimuthWrap, RotorCapabilities  # noqa: E402
from qsorbit.core.rotor.position import Position  # noqa: E402
from qsorbit.core.tracker.observer import ObserverLocation  # noqa: E402
from qsorbit.core.tracker.satellite import Satellite  # noqa: E402
from qsorbit.ui.feed_hub import FeedHub  # noqa: E402
from qsorbit.ui.map_widget import _MapCanvas  # noqa: E402
from qsorbit.ui.readout_widget import ReadoutWidget  # noqa: E402
from qsorbit.ui.tabs import PlanTab, RadioTab, RotorTab  # noqa: E402
from qsorbit.ui.theme import DEFAULT_THEMES_DIR, discover_themes  # noqa: E402
from qsorbit.ui.theme_manager import ThemeManager  # noqa: E402

#: A geostationary TLE, so the readout's values are the long ones a real
#: pass produces rather than placeholders. Geostationary specifically
#: because the range string is at its widest -- five digits of
#: kilometres plus a signed rate -- which is the worst case the column
#: has to fit.
GEO_TLE = """TDRS 3
1 19548U 88091B   26240.17524428 -.00000293  00000+0  00000+0 0  9993
2 19548  12.5485 340.5136 0036793 352.6995  17.1818  1.00264911126114
"""

#: Generous, so a genuinely too-small window is never what the assertion
#: is measuring. The defect this file exists for happened at a size far
#: larger than this.
TAB_SIZE = (1200, 700)


class FakeRadio:
    """A receive session's three live levels, with realistic values."""

    live_quieting_db: float | None = -6.2
    live_squelch_open: bool | None = True
    live_tracked_frequency_hz: float | None = 435_605_213.0


@pytest.fixture
def themes(qapp) -> ThemeManager:
    manager = ThemeManager(discover_themes((DEFAULT_THEMES_DIR,)))
    manager.apply()
    return manager


@pytest.fixture
def ticked_loop(tmp_path) -> TrackingLoop:
    """A real tracking loop over a mocked rotor, ticked once.

    Ticked on purpose. An unticked loop reports no sample and every
    readout row renders as ``"-"`` -- which is exactly the blind spot
    that let this defect ship, so a fixture that skipped the tick would
    reproduce the blind spot rather than the bug.
    """
    path = tmp_path / "geo.tle"
    path.write_text(GEO_TLE, encoding="utf-8")

    rotor = MagicMock(spec=Rotor)
    rotor.capabilities = RotorCapabilities(
        azimuth_min_deg=0.0,
        azimuth_max_deg=360.0,
        elevation_min_deg=0.0,
        elevation_max_deg=90.0,
        azimuth_wrap=AzimuthWrap.EXTRA_ROTATION,
        acceptance_window_deg=1.0,
        rs485_turnaround_s=0.15,
    )
    rotor.read_position.return_value = Position(126.4, 22.5)

    loop = TrackingLoop(
        Satellite.from_file(path),
        ObserverLocation(latitude=43.0, longitude=-79.0),
        rotor,
        interval_s=1.0,
        alignment_offset=AlignmentOffset(),
    )
    loop.tick()
    return loop


def clipped_labels(widget) -> list[tuple[str, int, int]]:
    """Every label in ``widget`` narrower than the text it holds.

    Word-wrapped labels are exempt: their whole purpose is to be
    narrower than their text and grow taller instead, so ``sizeHint``
    means something different for them and comparing against it would
    report a permanent false positive.
    """
    found = []
    for label in widget.findChildren(QLabel):
        if not label.text() or label.wordWrap():
            continue
        wanted = label.sizeHint().width()
        if label.width() < wanted:
            found.append((label.text(), label.width(), wanted))
    return found


def realise(widget) -> None:
    """Show and lay out a tab, then force every readout to poll once."""
    from PySide6.QtWidgets import QApplication

    widget.resize(*TAB_SIZE)
    widget.show()
    for readout in widget.findChildren(ReadoutWidget):
        readout._on_timer()
    QApplication.processEvents()


def test_the_rotor_tab_shows_every_value_in_full(themes, ticked_loop):
    """The defect, as an assertion.

    Before the fix this reported four clipped labels: the UTC/local time
    missing its closing bracket, both rotor-axis rows missing their
    parenthetical, and the range row missing the range rate entirely.
    """
    tab = RotorTab(FeedHub(tracking=ticked_loop), themes=themes)
    realise(tab)
    assert clipped_labels(tab) == []


def test_the_radio_tab_shows_every_value_in_full(themes):
    """The other fixed-width column, checked the same way.

    No spectrum stream: the right-hand column is what
    ``SIDE_COLUMN_WIDTH`` sizes, and it is driven by the frequency and
    quieting cards, which need only the live levels.
    """
    tab = RadioTab(FeedHub(radio=FakeRadio()), themes=themes, nominal_hz=435_600_000.0)
    realise(tab)
    assert clipped_labels(tab) == []


def test_the_custom_tab_shows_every_value_in_full(themes, ticked_loop):
    """The same defect class, one PR later, in cells nobody sized by hand.

    Custom-tab cells are Cards built at whatever width the grid gives
    them -- nobody picked a column width for these, unlike the two
    fixed-turned-minimum columns the rest of this file exists for. If a
    real rotor's or receiver's strings can clip anywhere, it is here.
    """
    from qsorbit.ui.custom_tab import CustomTabConfig
    from qsorbit.ui.tabs import CustomTab

    hub = FeedHub(tracking=ticked_loop, radio=FakeRadio())
    config = CustomTabConfig(columns=2, widgets=("rotor_readout", "frequency", "quieting"))
    tab = CustomTab(hub, themes=themes, config=config)
    realise(tab)
    assert clipped_labels(tab) == []


def test_a_tab_stays_readable_when_its_window_is_wide(themes, ticked_loop):
    """Growing the window must not be what makes text fit.

    A column that is exactly wide enough at one size and clipped at
    another is the same bug wearing a different number, so this checks
    the geometry holds where there is room to spare.
    """
    tab = RotorTab(FeedHub(tracking=ticked_loop), themes=themes)
    tab.resize(1920, 1080)
    realise(tab)
    assert clipped_labels(tab) == []


def test_a_tab_shown_at_its_own_preferred_size_clips_nothing(themes, ticked_loop):
    """The guarantee the fix actually provides, stated honestly.

    An earlier version of this test claimed something stronger -- that
    the minimum width made clipping *unreachable* by resizing -- and
    failed, which is the test doing its job on its author. It does not:
    :data:`~qsorbit.ui.tabs.ROTOR_COLUMN_WIDTH` is a floor of 320, the
    content wants about 403, and a window squeezed to the floor clips
    again. Every interface clips if you make it small enough.

    What is true, and worth pinning down, is that the tab's *own*
    preferred size is wide enough for everything in it. That is the
    property that was broken before: the fixed width made the tab's
    preferred size a number chosen in advance rather than one derived
    from its contents, so opening at the preferred size still clipped.
    """
    tab = RotorTab(FeedHub(tracking=ticked_loop), themes=themes)
    realise(tab)
    tab.resize(tab.sizeHint())
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()

    assert clipped_labels(tab) == []


def test_the_check_can_actually_fail(qapp):
    """Canary, aimed at the helper rather than at a tab.

    The tabs can no longer be squeezed into clipping, which is the point
    of the fix -- so proving the *tab* clips is no longer possible, and
    a canary that tried would be asserting the fix does not work. What
    still has to be shown is that ``clipped_labels`` reports a label
    that genuinely is too narrow. Without this every assertion above
    would keep passing if the helper were quietly broken, which is the
    failure ``test_no_hardcoded_colours`` was written after: a check too
    narrow to see the defect it was supposed to catch.
    """
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    label = QLabel("39459 km, approaching at 0.027 km/s", host)
    label.setFixedWidth(40)
    host.show()
    qapp.processEvents()

    found = clipped_labels(host)
    assert found, "a 40 px label holding a long string was not reported as clipped"
    text, width, wanted = found[0]
    assert width == 40
    assert wanted > width

    # And it does not fire on a label with room to spare.
    label.setFixedWidth(wanted + 20)
    qapp.processEvents()
    assert clipped_labels(host) == []


def starved_canvases(widget) -> list[tuple[str, int, int]]:
    """Every visible widget given less height than its own minimum asks for.

    Deliberately narrower than "every widget that paints". Four other
    widgets in :mod:`qsorbit.ui` implement ``paintEvent`` and state no
    size either, and none of them is defective today -- they all live in
    ``stretch=True`` cards that hand them the slack. Flagging them here
    would be scope creep dressed as a finding.

    What this enforces is the contract that matters: **a widget that
    does state a minimum must actually be given it.** The map's fix is
    to state one; this is what keeps it honest afterwards, and what will
    fire if some future container starves it again.

    **Stock Qt widgets are excluded, and that is a finding rather than a
    convenience.** The first version of this helper checked every child
    and reported ``QHeaderView`` at 39 px against a 62 px minimum -- the
    picker table's own header, given less than it asks for by the table
    that owns it, working exactly as intended. A stock widget's
    internals are its own business and this project neither controls nor
    should police them. So the check covers widgets defined outside
    PySide6: ours, and anything a test defines to prove this helper
    still bites.
    """
    found = []
    for child in widget.findChildren(QWidget):
        if type(child).__module__.startswith("PySide6"):
            continue
        wanted = child.minimumSizeHint().height()
        if wanted > 0 and child.isVisible() and child.height() < wanted:
            found.append((type(child).__name__, child.height(), wanted))
    return found


@pytest.fixture
def plan_tab_parts(tmp_path):
    """A TLE directory and a catalogue that actually match each other.

    Matched on purpose, for the same reason ``ticked_loop`` ticks: a
    picker with no rows would emit no entries, the map would draw no
    tracks, and the tab would lay out around content that isn't there --
    reproducing the blind spot rather than the bug.
    """
    tle_dir = tmp_path / "tle"
    tle_dir.mkdir()
    (tle_dir / "geo.tle").write_text(GEO_TLE, encoding="utf-8")

    catalog = ProfileCatalog(
        (
            SatelliteProfile(
                norad_id=19548,
                name="TDRS 3",
                transmitters=(),
                alive=AliveRecord(
                    status=AliveStatus.ACTIVE,
                    as_of=date(2026, 8, 28),
                    source="test",
                ),
            ),
        )
    )
    return str(tle_dir), catalog


def test_the_plan_tab_gives_its_map_room_to_be_a_map(themes, plan_tab_parts):
    """The defect, as an assertion.

    Before the fix the canvas came out a handful of pixels tall inside a
    full-width card, because it asked for nothing and the card's
    ``Minimum`` policy obliged. The assertion is against the canvas's
    own declared floor rather than a number typed here, so the two
    cannot drift apart.
    """
    tle_dir, catalog = plan_tab_parts
    tab = PlanTab(
        themes=themes,
        catalog=catalog,
        catalog_manifest=None,
        tle_dir=tle_dir,
        observer=ObserverLocation(latitude=43.0, longitude=-79.0),
        horizon=HorizonMask(points=()),
    )
    realise(tab)

    canvas = tab.findChild(_MapCanvas)
    assert canvas is not None, "the Plan tab built no map canvas at all"
    assert canvas.height() >= _MapCanvas.MINIMUM_SIZE.height()
    assert canvas.width() >= _MapCanvas.MINIMUM_SIZE.width()
    assert starved_canvases(tab) == []


def test_the_plan_tab_clips_no_labels_either(themes, plan_tab_parts):
    """The original check, pointed at the tab this file predicted in writing."""
    tle_dir, catalog = plan_tab_parts
    tab = PlanTab(
        themes=themes,
        catalog=catalog,
        catalog_manifest=None,
        tle_dir=tle_dir,
        observer=ObserverLocation(latitude=43.0, longitude=-79.0),
        horizon=HorizonMask(points=()),
    )
    realise(tab)
    assert clipped_labels(tab) == []


def test_the_starvation_check_can_actually_fail(qapp):
    """Canary, aimed at the helper.

    ``test_no_hardcoded_colours`` shipped once as a check too narrow to
    see its own defect, so every helper in this file gets one of these.
    A canvas squeezed below its stated minimum must be reported, and a
    canvas with room must not be.
    """
    from PySide6.QtWidgets import QVBoxLayout

    host = QWidget()
    QVBoxLayout(host)

    class _Starved(QWidget):
        def minimumSizeHint(self):  # noqa: N802 - Qt's spelling
            from PySide6.QtCore import QSize

            return QSize(360, 180)

    victim = _Starved(host)
    host.layout().addWidget(victim)
    victim.setFixedHeight(9)
    host.resize(800, 400)
    host.show()
    qapp.processEvents()

    found = starved_canvases(host)
    assert found, "a 9 px canvas declaring a 180 px minimum was not reported"
    name, height, wanted = found[0]
    assert height == 9
    assert wanted == 180

    victim.setFixedHeight(200)
    qapp.processEvents()
    assert starved_canvases(host) == []
