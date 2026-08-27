"""Tests for :class:`~qsorbit.ui.zoom_controller.ZoomController`.

Unlike this project's other Qt shells (``waterfall_widget.py``,
``readout_widget.py``, ``quieting_widget.py``), this one is not
bench-only. It is a plain ``QObject`` with a ``Signal`` - no
``QWidget``, no ``paintEvent``, no rendering backend - and confirmed by
hand in this session that both construction and ``Signal.connect()``/
``.emit()`` work with no ``QApplication`` instantiated at all: Qt's
signal/slot machinery for a direct (same-thread) connection needs no
event loop, only ``QApplication``/``QWidget`` construction needs the
windowing libraries this sandbox is missing (``libEGL.so.1``). So these
tests exercise the real class, including the real ``Signal``, rather
than standing in a structural double the way ``test_cli.py``'s
``FakeQuittableApp`` does for ``_quit_on_sigint``.

Since every decision this class makes is delegated to the pure
``next_zoom_for_*`` functions in ``spectrum_zoom.py`` (already covered
exhaustively in ``test_spectrum_zoom.py``), what is worth testing here
is the orchestration itself: that each method calls through to the
right pure function with the right arguments, that the starting state
is the full band unlocked, and - the one behaviour with no equivalent in
the pure layer - that ``changed`` fires exactly once per real change and
never for a no-op.
"""

from __future__ import annotations

from qsorbit.ui.zoom_controller import ZoomController

BAND_START_HZ = 99_000_000.0
BAND_STOP_HZ = 101_000_000.0


class ChangeCounter:
    """A tiny real Qt slot, so 'did changed fire' is an actual signal delivery."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> None:
        self.count += 1


def a_controller() -> tuple[ZoomController, ChangeCounter]:
    controller = ZoomController(BAND_START_HZ, BAND_STOP_HZ)
    counter = ChangeCounter()
    controller.changed.connect(counter)
    return controller, counter


def test_starts_on_the_whole_band_unlocked():
    controller, _counter = a_controller()
    assert controller.zoom.start_hz == BAND_START_HZ
    assert controller.zoom.stop_hz == BAND_STOP_HZ
    assert controller.locked is False
    assert controller.band_start_hz == BAND_START_HZ
    assert controller.band_stop_hz == BAND_STOP_HZ


def test_set_locked_flips_the_flag_and_emits_once():
    controller, counter = a_controller()
    controller.set_locked(True)
    assert controller.locked is True
    assert counter.count == 1


def test_set_locked_to_the_same_value_does_not_emit():
    controller, counter = a_controller()
    controller.set_locked(False)  # already False
    assert counter.count == 0


def test_set_span_hz_changes_the_span_and_emits():
    controller, counter = a_controller()
    controller.set_span_hz(200_000.0)
    assert controller.zoom.span_hz == 200_000.0
    assert controller.zoom.center_hz == 100_000_000.0
    assert counter.count == 1


def test_set_span_hz_to_the_current_value_does_not_emit():
    controller, counter = a_controller()
    controller.set_span_hz(controller.zoom.span_hz)
    assert counter.count == 0


def test_pan_to_moves_the_center_when_unlocked():
    # Narrowed first: at the default full-band span there is no room to
    # pan at all (the window already fills the whole band, so clamp_zoom
    # pins the center regardless of what is asked for) - that is
    # clamp_zoom doing its job, not something to trip over here.
    controller, counter = a_controller()
    controller.set_span_hz(200_000.0)
    counter.count = 0
    controller.pan_to(100_500_000.0)
    assert controller.zoom.center_hz == 100_500_000.0
    assert counter.count == 1


def test_pan_to_does_nothing_while_locked():
    controller, counter = a_controller()
    controller.set_span_hz(200_000.0)
    controller.set_locked(True)
    counter.count = 0  # ignore the span/lock's own emits
    controller.pan_to(100_500_000.0)
    assert controller.zoom.center_hz == 100_000_000.0
    assert counter.count == 0


def test_zoom_by_narrows_the_span_and_emits():
    controller, counter = a_controller()
    controller.zoom_by(0.5)
    assert controller.zoom.span_hz == 1_000_000.0
    assert counter.count == 1


def test_zoom_by_honours_an_anchor_when_unlocked():
    controller, _counter = a_controller()
    controller.zoom_by(0.5, anchor_hz=BAND_START_HZ)
    assert controller.zoom.center_hz != 100_000_000.0


def test_zoom_by_ignores_the_anchor_while_locked():
    controller, _counter = a_controller()
    controller.set_locked(True)
    controller.zoom_by(0.5, anchor_hz=BAND_START_HZ)
    assert controller.zoom.center_hz == 100_000_000.0


def test_follow_recenters_only_while_locked():
    controller, counter = a_controller()
    controller.set_span_hz(200_000.0)
    counter.count = 0
    controller.follow(100_300_000.0)  # unlocked: no-op
    assert controller.zoom.center_hz == 100_000_000.0
    assert counter.count == 0

    controller.set_locked(True)
    counter.count = 0  # ignore the lock's own emit
    controller.follow(100_300_000.0)
    assert controller.zoom.center_hz == 100_300_000.0
    assert counter.count == 1


def test_locking_mid_session_snaps_into_place_on_the_next_follow_call():
    """set_locked itself never moves the window - the very next follow()
    call is what carries the snap, exactly as the class's own docstring
    promises."""
    controller, counter = a_controller()
    controller.set_span_hz(200_000.0)
    controller.set_locked(True)
    assert controller.zoom.center_hz == 100_000_000.0  # unmoved by set_locked alone
    counter.count = 0
    controller.follow(100_700_000.0)
    assert controller.zoom.center_hz == 100_700_000.0
    assert counter.count == 1


def test_a_no_op_gesture_after_a_real_change_still_only_emits_once():
    controller, counter = a_controller()
    controller.set_span_hz(200_000.0)
    controller.set_span_hz(200_000.0)  # repeat, now a no-op
    assert counter.count == 1


# ----------------------------------------------------------------------
# tracked_frequency_source polling
# ----------------------------------------------------------------------


class FakeTrackedFrequencySource:
    """A minimal double satisfying TrackedFrequencySource structurally."""

    def __init__(self, live_tracked_frequency_hz: float | None = None) -> None:
        self.live_tracked_frequency_hz = live_tracked_frequency_hz


def test_stop_is_a_harmless_no_op_with_no_tracked_source():
    controller, _counter = a_controller()  # a_controller() gives no source
    controller.stop()  # must not raise


def test_polling_a_source_reporting_none_is_a_no_op():
    source = FakeTrackedFrequencySource(live_tracked_frequency_hz=None)
    controller = ZoomController(BAND_START_HZ, BAND_STOP_HZ, tracked_frequency_source=source)
    counter = ChangeCounter()
    controller.changed.connect(counter)
    controller.set_span_hz(200_000.0)
    counter.count = 0

    controller.set_locked(True)
    counter.count = 0
    controller._poll_tracked_frequency()
    assert controller.zoom.center_hz == 100_000_000.0
    assert counter.count == 0


def test_polling_a_source_with_a_reading_follows_it_while_locked():
    source = FakeTrackedFrequencySource(live_tracked_frequency_hz=100_400_000.0)
    controller = ZoomController(BAND_START_HZ, BAND_STOP_HZ, tracked_frequency_source=source)
    controller.set_span_hz(200_000.0)
    controller.set_locked(True)
    counter = ChangeCounter()
    controller.changed.connect(counter)

    controller._poll_tracked_frequency()

    assert controller.zoom.center_hz == 100_400_000.0
    assert counter.count == 1


def test_polling_a_source_does_not_move_the_zoom_while_unlocked():
    source = FakeTrackedFrequencySource(live_tracked_frequency_hz=100_400_000.0)
    controller = ZoomController(BAND_START_HZ, BAND_STOP_HZ, tracked_frequency_source=source)
    controller.set_span_hz(200_000.0)

    controller._poll_tracked_frequency()

    assert controller.zoom.center_hz == 100_000_000.0


def test_stop_stops_the_follow_timer_without_raising():
    source = FakeTrackedFrequencySource(live_tracked_frequency_hz=100_400_000.0)
    controller = ZoomController(BAND_START_HZ, BAND_STOP_HZ, tracked_frequency_source=source)
    controller.stop()
    # Stopping does not itself change the zoom or lock state.
    assert controller.locked is False
