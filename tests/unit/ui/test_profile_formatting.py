"""Unit tests for the tracking-profile toggle's wording. No Qt.

Its own file rather than a section of ``test_profile_widget.py``,
because ``pytest.importorskip`` skips a whole module during collection
however far down it sits -- so wording tests sharing a file with widget
tests would silently stop running anywhere Qt cannot be imported. Same
split as ``test_picker_formatting.py`` against ``test_picker_widget.py``.
"""

from __future__ import annotations

from qsorbit.core.tracking_profile import TrackingProfile
from qsorbit.ui.profile_formatting import profile_status_text

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


class TestStatusText:
    def test_a_settled_profile_reports_its_cadence(self):
        text = profile_status_text(STOCK, None, None, stalled=False)
        assert "stock" in text
        assert "2.5 deg deadband at 1 s" in text

    def test_a_profile_with_gains_says_so(self):
        assert "gains pushed" in profile_status_text(TRACKING, None, None, stalled=False)

    def test_a_profile_without_gains_says_controller_defaults(self):
        # Not "no gains": the controller is running the firmware's own
        # values, which is a real configuration rather than an absence.
        assert "controller defaults" in profile_status_text(STOCK, None, None, stalled=False)

    def test_a_pending_switch_says_it_has_not_happened_yet(self):
        text = profile_status_text(STOCK, TRACKING, None, stalled=False)
        assert "Switching to tracking" in text
        assert "next tick" in text

    def test_a_refusal_outranks_everything(self):
        # The most recent thing that happened, and the only one the
        # operator has to act on.
        text = profile_status_text(STOCK, TRACKING, "elevation is stalled", stalled=True)
        assert text == "elevation is stalled"

    def test_a_stall_with_no_refusal_still_explains_the_disabled_buttons(self):
        # Otherwise the toggle is greyed out for no stated reason.
        text = profile_status_text(STOCK, None, None, stalled=True)
        assert "stalled" in text

    def test_an_unlabelled_loop_says_there_is_nothing_to_switch(self):
        text = profile_status_text(None, None, None, stalled=False)
        assert "nothing to switch between" in text

    def test_every_wording_is_printable(self):
        for args in (
            (STOCK, None, None),
            (TRACKING, None, None),
            (STOCK, TRACKING, None),
            (None, None, None),
        ):
            profile_status_text(*args, stalled=False).encode("ascii")
