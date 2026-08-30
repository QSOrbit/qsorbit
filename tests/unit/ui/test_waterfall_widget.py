"""Tests for the waterfall panel's theming.

The one that matters is
:func:`test_two_waterfalls_on_one_manager_both_restyle`. Two of Chunk C's
done-when clauses meet in it: *switching theme at runtime restyles every
widget -- waterfall colormap included -- without a restart*, and *a
Custom tab showing a duplicate of a built-in widget works*. Neither
widget in that test is wired to the theme by anything; each subscribed
itself at construction, which is the property that makes a
config-file-defined tab possible at all.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

# A submodule, not the package: `import PySide6` succeeds on a machine
# with no Qt system libraries, and only `PySide6.QtWidgets` fails --
# with an ImportError for libEGL.so.1 rather than anything mentioning
# Qt. Guarding the package alone let CI die at collection.
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402

from qsorbit.core.dsp.spectrum import SpectrumConfig  # noqa: E402
from qsorbit.core.dsp.spectrum_stream import SpectrumFrame  # noqa: E402
from qsorbit.ui.theme import DEFAULT_THEMES_DIR, discover_themes  # noqa: E402
from qsorbit.ui.theme_manager import ThemeManager  # noqa: E402
from qsorbit.ui.waterfall_widget import WaterfallWidget  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def manager(qapp):
    return ThemeManager(discover_themes((DEFAULT_THEMES_DIR,)))


@pytest.fixture
def config():
    return SpectrumConfig(sample_rate_hz=2_048_000.0, center_freq_hz=435_605_000.0, fft_size=256)


class FakeSource:
    """A frame source that hands over exactly what it was given.

    Its own object rather than a mock: the widget drains it on a
    ``QTimer`` and the tests below drive that draining by hand, so what
    matters is that ``latest()`` empties, the way a real subscription
    does.
    """

    def __init__(self, config: SpectrumConfig) -> None:
        self._config = config
        self._pending: list[SpectrumFrame] = []

    @property
    def config(self) -> SpectrumConfig:
        return self._config

    def offer(self, power_db: np.ndarray) -> None:
        self._pending.append(SpectrumFrame(power_db=power_db, time=datetime.now(UTC)))

    def latest(self) -> list[SpectrumFrame]:
        frames, self._pending = self._pending, []
        return frames


def carrier(config: SpectrumConfig) -> np.ndarray:
    """A quiet band with one loud bin in the middle."""
    power_db = np.full(config.fft_size, -85.0, dtype=np.float32)
    power_db[config.fft_size // 2] = -30.0
    return power_db


def rendered(widget: WaterfallWidget) -> np.ndarray:
    """The widget's current rendered rows, as one array."""
    return np.stack(tuple(widget._rendered_rows))


# ----------------------------------------------------------------------


def test_a_waterfall_starts_in_the_managers_current_theme(manager, config):
    manager.apply("deep-space")
    widget = WaterfallWidget(FakeSource(config), themes=manager)
    assert widget._colormap is manager.current.waterfall
    widget.stop()


def test_switching_theme_restyles_the_panel_without_a_restart(manager, config):
    manager.apply("deep-space")
    source = FakeSource(config)
    widget = WaterfallWidget(source, themes=manager)

    source.offer(carrier(config))
    widget._on_timer()
    before = rendered(widget).copy()

    manager.apply("mars")

    after = rendered(widget)
    assert widget._colormap is manager.theme("mars").waterfall
    assert not np.array_equal(before, after), "the panel did not repaint in the new theme"
    widget.stop()


def test_history_survives_a_theme_switch(manager, config):
    """A theme change mid-pass must not cost the pass being watched.

    The rows are re-rendered from the raw frames rather than discarded,
    which is also the only way to *see* that a switch worked -- a panel
    that blanked and refilled would look identical to one that had
    simply stopped.
    """
    manager.apply("deep-space")
    source = FakeSource(config)
    widget = WaterfallWidget(source, themes=manager)

    for _ in range(5):
        source.offer(carrier(config))
        widget._on_timer()
    raw_before = [frame.copy() for frame in widget._raw_frames]

    manager.apply("night-ops")

    assert len(widget._raw_frames) == len(raw_before)
    for kept, original in zip(widget._raw_frames, raw_before, strict=True):
        assert np.array_equal(kept, original)
    widget.stop()


def test_the_carrier_stays_the_brightest_thing_after_a_switch(manager, config):
    """Restyling must not lose the signal, only recolour it."""
    manager.apply("deep-space")
    source = FakeSource(config)
    widget = WaterfallWidget(source, themes=manager)
    source.offer(carrier(config))
    widget._on_timer()

    for slug in ("mars", "luna", "wopr", "night-ops"):
        manager.apply(slug)
        row = rendered(widget)[0].astype(np.float64)
        luminance = row.sum(axis=1)
        peak = int(np.argmax(luminance))
        assert luminance[peak] > np.median(luminance) + 40.0, f"carrier lost under {slug}"
    widget.stop()


def test_two_waterfalls_on_one_manager_both_restyle(manager, config):
    """The Custom tab, in miniature.

    Two independent instances, each with its own feed, neither wired to
    the theme by any container -- exactly the shape a config-file-defined
    tab produces. Both must follow a theme change, and the second one
    must not be a special case of any kind.
    """
    manager.apply("deep-space")
    radio_source, custom_source = FakeSource(config), FakeSource(config)
    radio = WaterfallWidget(radio_source, themes=manager)
    custom = WaterfallWidget(custom_source, themes=manager, history_rows=64)

    for source, widget in ((radio_source, radio), (custom_source, custom)):
        source.offer(carrier(config))
        widget._on_timer()

    before = {"radio": rendered(radio).copy(), "custom": rendered(custom).copy()}

    manager.apply("earth")

    earth = manager.theme("earth").waterfall
    assert radio._colormap is earth
    assert custom._colormap is earth
    assert not np.array_equal(before["radio"], rendered(radio))
    assert not np.array_equal(before["custom"], rendered(custom))
    radio.stop()
    custom.stop()


def test_two_waterfalls_do_not_share_rendered_state(manager, config):
    """Second-instance safety: one panel's rows are not the other's.

    Chunk A proved the two panels no longer steal each other's *frames*.
    This is the display-side counterpart -- with different history
    depths and different feeds, neither deck may alias the other.
    """
    manager.apply("deep-space")
    first_source, second_source = FakeSource(config), FakeSource(config)
    first = WaterfallWidget(first_source, themes=manager, history_rows=128)
    second = WaterfallWidget(second_source, themes=manager, history_rows=64)

    first_source.offer(carrier(config))
    first._on_timer()

    assert first._rendered_rows is not second._rendered_rows
    assert first._raw_frames is not second._raw_frames
    assert len(first._rendered_rows) == 128
    assert len(second._rendered_rows) == 64
    # Only the fed panel changed.
    assert not np.array_equal(rendered(first)[0], rendered(second)[0])
    first.stop()
    second.stop()


def test_a_waterfall_needs_a_theme_to_exist(manager, config):
    """No default colormap, because a default would be a colour chosen
    inside a widget -- the one thing the standing rule forbids."""
    with pytest.raises(TypeError):
        WaterfallWidget(FakeSource(config))  # type: ignore[call-arg]


# ----------------------------------------------------------------------
# Paint accounting
# ----------------------------------------------------------------------


def test_a_panel_that_has_never_painted_says_so(manager, config):
    """Zero is a state, not a measurement.

    "never painted" and "painted, cost nothing" are different facts, and
    this project's rule is that off and broken must not look the same.
    """
    widget = WaterfallWidget(FakeSource(config), themes=manager)
    stats = widget.paint_stats
    assert stats.paints == 0
    assert stats.mean_ms == 0.0
    assert "never painted" in stats.describe()
    widget.stop()


def test_painting_accumulates_both_halves_separately(manager, config):
    """The split is the whole point of the measurement.

    Building the image and drawing it have different fixes and could not
    be told apart from outside the widget -- the 28x maximize regression
    was attributed to one of them by inference, which is exactly what
    this exists to replace with a number.
    """
    widget = WaterfallWidget(FakeSource(config), themes=manager)
    widget.resize(640, 400)
    widget.show()
    for _ in range(5):
        widget.grab()

    stats = widget.paint_stats
    assert stats.paints == 5
    assert stats.build_s > 0.0
    assert stats.blit_s > 0.0
    # The two halves plus the trimmings are the whole, never more.
    assert stats.build_s + stats.blit_s <= stats.total_s
    assert stats.worst_s > 0.0
    assert stats.mean_ms > 0.0
    widget.stop()


def test_the_reported_size_is_the_size_that_produced_the_cost(manager, config):
    """A paint time with no size beside it cannot be compared to another run."""
    widget = WaterfallWidget(FakeSource(config), themes=manager)
    widget.resize(800, 500)
    widget.show()
    widget.grab()

    stats = widget.paint_stats
    assert (stats.width, stats.height) == (800, 500)
    assert "800x500" in stats.describe()
    widget.stop()


def test_a_faulted_panel_does_not_count_its_error_message_as_a_repaint(manager, config):
    """It drew a line of text, not a spectrogram.

    Averaging those in would make a dead panel look cheap, which is the
    direction that hides a problem rather than revealing one.
    """
    widget = WaterfallWidget(FakeSource(config), themes=manager)
    widget.resize(640, 400)
    widget.show()
    widget._error = "stopped: the stream went away"
    for _ in range(3):
        widget.grab()

    assert widget.paint_stats.paints == 0
    widget.stop()


def test_the_panel_repaints_at_the_rate_it_was_asked_for_whatever_its_size(manager, config):
    """A gravestone, and it is worth reading before writing a new one.

    This panel briefly throttled its own repaint rate by area, under a
    constant called ``PIXEL_BUDGET_PER_SECOND``, on the theory that
    pixel throughput was what cost the receive path its samples. Six
    tests pinned that arithmetic down and every one of them passed. The
    theory was still wrong: on hardware, cutting the throughput of a
    1178x443 panel from 10.4 to 4.0 Mpix/s moved USB loss from 0.667% to
    0.607%, against a within-configuration spread of 0.07 -- no effect,
    measured three times.

    So the rate is the caller's, at every size, and this asserts it at
    the two extremes rather than trusting that nobody re-adds a budget
    on the same reasoning. **A test suite that only records what a
    module does cannot warn the next person off what it already tried.**
    """
    widget = WaterfallWidget(FakeSource(config), themes=manager, poll_interval_ms=50)
    widget.resize(396, 148)
    assert widget._timer.interval() == 50
    widget.resize(3000, 1600)
    assert widget._timer.interval() == 50
    widget.stop()


def test_a_zero_poll_interval_is_refused_at_construction(manager, config):
    """Because ``describe`` divides by it, and would raise at shutdown.

    A crash in the run report fires *after* a pass is over, when the
    numbers it was going to print are the only record of what happened
    -- so this is caught where the bad value enters rather than where it
    is finally used. The sibling checks on ``history_rows`` and
    ``render_width`` were already here; this one was missing, and became
    load-bearing the moment the interval became a denominator.

    Not hypothetical: PR3's Custom tab builds panels from a config file,
    which is exactly where a hand-typed zero arrives.
    """
    with pytest.raises(ValueError, match="poll_interval_ms"):
        WaterfallWidget(FakeSource(config), themes=manager, poll_interval_ms=0)


def test_the_rate_and_the_size_are_reported_separately(manager, config):
    """Not multiplied into a throughput figure, which is the refuted axis.

    ``describe`` used to print "N Hz budgeted (M Mpix/s)". The Mpix/s
    was the number that turned out not to predict anything, and printing
    it as the headline invited the next reader to reach the same wrong
    conclusion from the same run report.
    """
    widget = WaterfallWidget(FakeSource(config), themes=manager, poll_interval_ms=50)
    widget.resize(1162, 412)
    widget.show()
    widget.grab()

    described = widget.paint_stats.describe()
    assert "1162x412" in described
    assert "Mpix/s" not in described
    assert widget.paint_stats.interval_ms == widget._timer.interval() == 50
    widget.stop()
