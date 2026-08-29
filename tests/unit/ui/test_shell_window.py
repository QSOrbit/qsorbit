"""Tests for the tabbed shell and its tabs.

Runs under the ``offscreen`` platform plugin, so construction, wiring,
signal delivery, chrome application and teardown all really execute.
What it does not buy is anything visual -- no window manager, no real
DPI, not the operator's font stack -- so "it looks right" stays a bench
check on a real monitor and is claimed by nothing here.

The guard is on ``PySide6.QtWidgets`` rather than on ``PySide6``,
because ``import PySide6`` succeeds on a machine with no Qt system
libraries and only the submodule import fails. Guarding the package
alone is what let CI die at collection in Session 27.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

from qsorbit.core.dsp.iq import IQ_ZERO_OFFSET  # noqa: E402
from qsorbit.core.dsp.spectrum import SpectrumConfig  # noqa: E402
from qsorbit.core.dsp.spectrum_stream import SpectrumStream  # noqa: E402
from qsorbit.ui.cards import Card, Placeholder  # noqa: E402
from qsorbit.ui.feed_hub import FeedHub  # noqa: E402
from qsorbit.ui.frequency_widget import FrequencyWidget  # noqa: E402
from qsorbit.ui.quieting_widget import QuietingWidget  # noqa: E402
from qsorbit.ui.shell_window import TAB_TITLES, ShellWindow, TopBar  # noqa: E402
from qsorbit.ui.spectrum_line_widget import SpectrumLineWidget  # noqa: E402
from qsorbit.ui.tabs import RadioTab  # noqa: E402
from qsorbit.ui.theme import DEFAULT_THEMES_DIR, SHIPPED_THEME_SLUGS, discover_themes  # noqa: E402
from qsorbit.ui.theme_manager import ThemeManager  # noqa: E402
from qsorbit.ui.waterfall_widget import WaterfallWidget  # noqa: E402

FFT_SIZE = 64
SAMPLE_RATE = 64_000.0


def silent_block(n_samples: int) -> bytes:
    return np.full(n_samples * 2, int(IQ_ZERO_OFFSET), dtype=np.uint8).tobytes()


def make_stream(blocks: int = 6) -> SpectrumStream:
    return SpectrumStream(
        [silent_block(FFT_SIZE)] * blocks,
        SpectrumConfig(fft_size=FFT_SIZE, sample_rate_hz=SAMPLE_RATE),
        frame_rate_hz=1_000.0,
    )


def endless_blocks(delay_s: float = 0.005) -> Iterator[bytes]:
    """A source that never runs out, for testing what outlives a close.

    A finite source is the wrong instrument for that question: the
    worker reaches the end and stops on its own, so "the stream is no
    longer running" afterwards says nothing about whether the shell
    stopped it. The first version of the test below made exactly that
    mistake and failed for a reason that had nothing to do with the code
    under test. Throttled rather than free-running so it does not spin a
    core computing FFTs nobody reads.
    """
    while True:
        time.sleep(delay_s)
        yield silent_block(FFT_SIZE)


class FakeRadio:
    def __init__(self) -> None:
        self.live_quieting_db: float | None = -3.0
        self.live_squelch_open: bool | None = True
        self.live_tracked_frequency_hz: float | None = 435_605_000.0


@pytest.fixture
def themes(qapp) -> ThemeManager:
    manager = ThemeManager(discover_themes((DEFAULT_THEMES_DIR,)))
    manager.apply()
    return manager


@pytest.fixture
def full_hub() -> FeedHub:
    return FeedHub(spectrum=make_stream(), radio=FakeRadio())


def running_timers(widget) -> int:
    """How many QTimers under ``widget`` are still ticking."""
    return sum(1 for timer in widget.findChildren(QTimer) if timer.isActive())


# ----------------------------------------------------------------------
# The shell exists and holds the tabs the mockup names
# ----------------------------------------------------------------------


def test_the_shell_has_the_tabs_the_roadmap_names(themes, full_hub):
    window = ShellWindow(full_hub, themes=themes)
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert titles == list(TAB_TITLES[:4])
    # Custom is PR3; the title constant already reserves its place so the
    # order cannot drift when it lands.
    assert TAB_TITLES[4] == "Custom"


def test_every_existing_widget_lives_in_a_hub_fed_tab(themes, full_hub):
    """Chunk C's first done-when clause, as an assertion.

    "Every existing widget lives in a tab fed by the hub." The four
    panels Phase 2 built -- waterfall, spectrum line, quieting, readout
    -- plus the frequency readout this PR adds. The readout is checked
    separately below, since it needs a rotor.
    """
    window = ShellWindow(full_hub, themes=themes)
    for widget_type in (WaterfallWidget, SpectrumLineWidget, QuietingWidget, FrequencyWidget):
        assert window.findChildren(widget_type), f"{widget_type.__name__} is not in the shell"


def test_the_radio_tab_claims_exactly_two_named_spectrum_feeds(themes, full_hub):
    ShellWindow(full_hub, themes=themes)
    assert full_hub.claimed == ("spectrum-line", "waterfall")


# ----------------------------------------------------------------------
# A second instance: the standing rule, checked rather than asserted
# ----------------------------------------------------------------------


def test_a_second_radio_tab_gets_its_own_feeds_and_every_frame(themes):
    """The Custom tab's whole premise, one PR before the Custom tab.

    The standing Phase 3 rule is that **every widget must work as a
    second instance in the Custom tab, or the design is wrong.** PR3
    builds those instances from a config file; this builds them from a
    second RadioTab, which exercises the identical hub path. If a
    duplicate silently shared the first tab's subscription, this is
    where it shows up -- as one of the four feeds seeing fewer frames
    than the others.
    """
    stream = make_stream(blocks=6)
    hub = FeedHub(spectrum=stream, radio=FakeRadio())
    first = RadioTab(hub, themes=themes)
    second = RadioTab(hub, themes=themes)

    assert hub.claimed == (
        "spectrum-line",
        "waterfall",
        "spectrum-line-2",
        "waterfall-2",
    )

    stream.start()
    thread = stream._thread
    assert thread is not None
    thread.join(5.0)
    assert not thread.is_alive()

    stats = {sub.name: sub for sub in stream._subscribers}
    for name in hub.claimed:
        assert stats[name].stats.frames_offered == 6, name
        assert stats[name].stats.frames_dropped == 0, name

    # Both tabs are stopped by hand: they were built without a window,
    # so nothing else will do it, and a live QTimer outliving its test
    # polls a stream the next test has already finished with.
    for tab in (first, second):
        for timer in tab.findChildren(QTimer):
            timer.stop()


# ----------------------------------------------------------------------
# Absent feeds say so, in words
# ----------------------------------------------------------------------


def test_a_shell_with_nothing_attached_still_opens(themes):
    """A sky-free evening is a shell with every tab in placeholder.

    Not a crash and not an empty window: the roadmap's own asymmetry is
    that a rotor fault costs the antenna pointing and nothing else, and
    the shell inherits it.
    """
    window = ShellWindow(FeedHub(), themes=themes)
    assert window.tabs.count() == 4
    assert window.findChildren(Placeholder)


def test_an_absent_feed_is_named_rather_than_left_blank(themes):
    """ "Off" and "broken" must never look the same.

    Written down after a healthy headless run reported 453 dropped
    blocks and read as catastrophic loss. An empty spectrum panel and a
    dead SDR look identical; only one of them is a fault.
    """
    window = ShellWindow(FeedHub(), themes=themes)
    text = " ".join(p.text() for p in window.findChildren(Placeholder))
    assert "No SDR attached" in text
    assert "No rotor connected" in text


def test_a_spectrum_only_shell_draws_spectrum_and_names_the_missing_rest(themes):
    window = ShellWindow(FeedHub(spectrum=make_stream()), themes=themes)
    assert window.findChildren(WaterfallWidget)
    assert not window.findChildren(QuietingWidget)
    text = " ".join(p.text() for p in window.findChildren(Placeholder))
    assert "no live levels" in text


# ----------------------------------------------------------------------
# Theme switching restyles the shell -- and un-styles it again
# ----------------------------------------------------------------------


@pytest.mark.parametrize("slug", sorted(SHIPPED_THEME_SLUGS))
def test_every_shipped_theme_applies_to_a_live_shell(themes, full_hub, slug):
    """Chunk C's marquee done-when, for the shell rather than a panel.

    A theme that raised only once the shell existed would be found at
    the bench, mid-pass, in the dark.
    """
    window = ShellWindow(full_hub, themes=themes)
    themes.apply(slug)
    assert themes.current.slug == slug
    window.close()


def test_lcars_shows_accent_bars_and_leaving_it_hides_them(themes, full_hub):
    """A chrome effect that could only be turned on is a one-way door.

    The exact failure PR1 met with fonts: ``apply`` had no ``unapply``,
    so leaving LCARS left Antonio on screen. Every branch in
    ``_apply_chrome`` therefore has to restore as well as set, and this
    is the assertion that says so.
    """
    window = ShellWindow(full_hub, themes=themes)
    cards = window.findChildren(Card)
    assert cards

    themes.apply("deep-space")
    assert not any(card._bar.isVisible() for card in cards)

    themes.apply("lcars")
    window.show()
    assert all(card._bar.isVisibleTo(card) for card in cards)

    themes.apply("deep-space")
    assert not any(card._bar.isVisibleTo(card) for card in cards)
    window.close()


def test_uppercase_headings_are_applied_and_undone(themes, full_hub):
    """``text-transform`` is CSS that Qt does not implement.

    PR1 emits it in the stylesheet for LCARS and CRT, where it is
    silently ignored -- which is exactly why ``ChromeStructure`` reports
    ``uppercase_headings`` for the shell to act on. If this ever passes
    without the shell doing the work, the stylesheet has grown a feature
    Qt does not have.
    """
    window = ShellWindow(full_hub, themes=themes)

    themes.apply("deep-space")
    assert window.tabs.tabText(0) == "Radio"

    themes.apply("wopr")
    assert window.tabs.tabText(0) == "RADIO"

    themes.apply("deep-space")
    assert window.tabs.tabText(0) == "Radio"
    window.close()


def test_wopr_adds_scanlines_and_leaving_it_removes_them(themes, full_hub):
    window = ShellWindow(full_hub, themes=themes)
    window.show()

    themes.apply("wopr")
    assert window._scanlines.isVisibleTo(window)

    themes.apply("deep-space")
    assert not window._scanlines.isVisibleTo(window)
    window.close()


def test_the_crt_glow_is_removed_when_the_theme_leaves(themes, full_hub):
    """Graphics effects accumulate if only ever added.

    ``setGraphicsEffect(None)`` deletes the previous one, so nothing
    piles up across eight switches -- but only if the code path that
    removes it exists at all.
    """
    window = ShellWindow(full_hub, themes=themes)
    readouts = [
        label for label in window.findChildren(QLabel) if label.property("role") == "readout"
    ]
    assert readouts, "no readout labels to glow"

    themes.apply("wopr")
    assert all(label.graphicsEffect() is not None for label in readouts)

    themes.apply("deep-space")
    assert all(label.graphicsEffect() is None for label in readouts)
    window.close()


# ----------------------------------------------------------------------
# The top bar
# ----------------------------------------------------------------------


def test_the_picker_lists_every_theme_and_applying_one_changes_the_theme(themes):
    bar = TopBar(themes=themes)
    assert bar.picker.count() == len(themes.slugs)

    index = bar.picker.findData("mars")
    assert index >= 0
    bar.picker.setCurrentIndex(index)
    assert themes.current.slug == "mars"
    bar.stop()


def test_the_picker_follows_ctrl_t_rather_than_disagreeing_with_it(themes, full_hub):
    """A picker showing the wrong theme is a readout that lies.

    Both controls survive from PR1's design -- the dropdown is what an
    operator uses, the shortcut is what finds a widget that fails to
    repaint -- so they have to stay in step.
    """
    window = ShellWindow(full_hub, themes=themes)
    themes.apply("deep-space")

    applied = window.cycle_theme(1)
    assert window.top_bar.picker.currentData() == applied.slug
    window.close()


def test_the_top_bar_names_the_target_only_when_there_is_one(themes):
    bar = TopBar(themes=themes)
    assert not any("target" in label.text() for label in bar.findChildren(QLabel))
    bar.stop()

    named = TopBar(themes=themes, target_name="RS-44")
    assert any("RS-44" in label.text() for label in named.findChildren(QLabel))
    named.stop()


def test_the_top_bar_shows_both_clocks(themes):
    """Local and UTC, always both.

    Session 22 cost an evening to a time-zone error, and Windows reports
    its zone as "Eastern Standard Time" in August as well as January --
    so the offset is not something to work out by eye at the bench.
    """
    bar = TopBar(themes=themes)
    assert bar.local_clock.text()
    assert "UTC" in bar.utc_clock.text()
    bar.stop()


# ----------------------------------------------------------------------
# Teardown
# ----------------------------------------------------------------------


def test_closing_stops_every_polling_panel(themes, full_hub):
    """The stop-everything walk, checked rather than trusted.

    A panel whose timer outlives its session keeps polling a dead object
    -- which is how a clean-looking shutdown becomes a traceback nobody
    can place. The walk exists because PR3's Custom tab builds widgets
    nobody here will know about; this asserts the walk actually reaches
    them.
    """
    window = ShellWindow(full_hub, themes=themes)
    assert running_timers(window) > 0
    window.close()
    assert running_timers(window) == 0


def test_closing_does_not_stop_the_hardware(themes):
    """Whoever built the stream owns its lifetime. Still true here.

    Closing a window mid-pass must not abandon the antenna mid-slew or
    tear down a running reader thread -- the policy every layer below
    the shell already follows.
    """
    stream = SpectrumStream(
        endless_blocks(),
        SpectrumConfig(fft_size=FFT_SIZE, sample_rate_hz=SAMPLE_RATE),
        frame_rate_hz=1_000.0,
    )
    hub = FeedHub(spectrum=stream)
    window = ShellWindow(hub, themes=themes)
    stream.start()
    window.close()
    assert stream.is_running, "the shell stopped a stream it does not own"
    stream.stop()
