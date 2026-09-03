"""Unit tests for the tracking-profile toggle widget.

The wording lives in ``test_profile_formatting.py``; this file is the Qt
half and is gated on a Qt import like every widget test here.

The behaviour worth pinning is that **the control never claims a state
the rotor is not in**. A press queues a switch; the checked button
follows what is in force, not what was clicked; and a refusal is a
normal outcome that leaves the toggle showing the truth.
"""

from __future__ import annotations

import pytest

from qsorbit.core.pointing import ProfileSwitchError
from qsorbit.core.tracking_profile import TrackingProfile

STOCK = TrackingProfile(name="stock", deadband_deg=2.5, interval_s=1.0)
TRACKING = TrackingProfile(
    name="tracking",
    deadband_deg=0.25,
    interval_s=0.5,
    arrival_window_deg=1.0,
    azimuth_kp=8.0,
    azimuth_ki=0.0,
    azimuth_kd=0.5,
    elevation_kp=10.0,
    elevation_ki=0.0,
    elevation_kd=0.3,
)

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from qsorbit.ui.profile_widget import ProfileWidget  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeSource:
    """A profile source that records requests, without a rotor or a port."""

    def __init__(self, active=STOCK, *, stalled=False, refuse=False):
        self.active_profile = active
        self.pending_profile = None
        self.profile_refusal = None
        self.is_stalled = stalled
        self.requested: list[TrackingProfile] = []
        self._refuse = refuse

    def request_profile(self, profile: TrackingProfile) -> None:
        self.requested.append(profile)
        if self._refuse:
            self.profile_refusal = "refused for the test's own reasons"
            raise ProfileSwitchError(self.profile_refusal)
        self.pending_profile = profile


class TestWidget:
    def test_one_button_per_profile(self, app):
        source = FakeSource()
        widget = ProfileWidget(source, (STOCK, TRACKING))
        assert set(widget._buttons) == {"stock", "tracking"}  # noqa: SLF001

    def test_the_active_profile_is_the_checked_one(self, app):
        widget = ProfileWidget(FakeSource(), (STOCK, TRACKING))
        assert widget._buttons["stock"].isChecked()  # noqa: SLF001
        assert not widget._buttons["tracking"].isChecked()  # noqa: SLF001

    def test_the_active_button_is_disabled(self, app):
        # Pressing the profile you are already on would queue a
        # pointless gain push over the serial port.
        widget = ProfileWidget(FakeSource(), (STOCK, TRACKING))
        assert not widget._buttons["stock"].isEnabled()  # noqa: SLF001
        assert widget._buttons["tracking"].isEnabled()  # noqa: SLF001

    def test_pressing_requests_that_profile(self, app):
        source = FakeSource()
        widget = ProfileWidget(source, (STOCK, TRACKING))
        widget._buttons["tracking"].click()  # noqa: SLF001
        assert source.requested == [TRACKING]

    def test_a_press_does_not_move_the_checked_state(self, app):
        # The switch is queued; nothing has reached the rotor. A control
        # that jumped to "tracking" here would be claiming a state the
        # hardware is not in -- the same silent lie as a frozen readout.
        source = FakeSource()
        widget = ProfileWidget(source, (STOCK, TRACKING))
        widget._buttons["tracking"].click()  # noqa: SLF001
        assert widget._buttons["stock"].isChecked()  # noqa: SLF001
        assert not widget._buttons["tracking"].isChecked()  # noqa: SLF001

    def test_a_pending_switch_shows_in_the_status(self, app):
        source = FakeSource()
        widget = ProfileWidget(source, (STOCK, TRACKING))
        widget._buttons["tracking"].click()  # noqa: SLF001
        assert "Switching to tracking" in widget.status_text

    def test_everything_is_disabled_while_a_switch_is_pending(self, app):
        source = FakeSource()
        widget = ProfileWidget(source, (STOCK, TRACKING))
        widget._buttons["tracking"].click()  # noqa: SLF001
        assert not widget._buttons["tracking"].isEnabled()  # noqa: SLF001

    def test_a_refusal_is_caught_and_shown(self, app):
        # ProfileSwitchError is the loop declining a moment, not a
        # fault. Letting it unwind through Qt would print a traceback
        # and change nothing on screen.
        source = FakeSource(refuse=True)
        widget = ProfileWidget(source, (STOCK, TRACKING))
        widget._buttons["tracking"].click()  # noqa: SLF001
        assert "refused for the test's own reasons" in widget.status_text

    def test_buttons_are_disabled_while_stalled(self, app):
        widget = ProfileWidget(FakeSource(stalled=True), (STOCK, TRACKING))
        assert not widget._buttons["tracking"].isEnabled()  # noqa: SLF001

    def test_the_widget_follows_a_switch_it_did_not_make(self, app):
        # The loop applies the switch on its own thread; the widget
        # finds out by polling, like every other panel here.
        source = FakeSource()
        widget = ProfileWidget(source, (STOCK, TRACKING))
        source.active_profile = TRACKING
        source.pending_profile = None
        widget.refresh()
        assert widget._buttons["tracking"].isChecked()  # noqa: SLF001

    def test_stop_stops_polling(self, app):
        widget = ProfileWidget(FakeSource(), (STOCK, TRACKING))
        widget.stop()
        assert not widget._timer.isActive()  # noqa: SLF001
