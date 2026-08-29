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
