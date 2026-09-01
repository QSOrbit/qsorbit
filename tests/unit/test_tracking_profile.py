"""Unit tests for named tracking profiles and the cadence arithmetic.

The arithmetic tests are written against **numbers Session 32 measured
on hardware**, not against numbers this module produced. The commanded
step of a cadence was measured off the controller's own command
timestamps before any of this code existed to predict it, so a test that
reproduces 3.0 deg for the shipped configuration is checking a model
against reality rather than against itself.
"""

import math

import pytest

from qsorbit.core.tracking_profile import (
    DEFAULT_INTERVAL_S,
    DEFAULT_PROFILE_NAME,
    KNIFE_EDGE_MARGIN,
    NOMINAL_TRACKING_RATE_DEG_S,
    CadenceError,
    TrackingProfile,
    check_cadence,
    commanded_step_deg,
    knife_edge_ratio,
    ticks_per_command,
)

# ---------------------------------------------------------------------------
# The measured cases
# ---------------------------------------------------------------------------


class TestAgainstMeasurement:
    """Every number here came off hardware in Session 32."""

    def test_the_shipped_cadence_commands_three_degrees_not_two_point_five(self):
        # "The minimum commanded step is rate x tick, whatever the
        # deadband is." At 1 deg/s a 2.5 deg deadband cannot be crossed
        # until the third tick, so the real step is 3.0 -- measured
        # directly off the command timestamps.
        assert commanded_step_deg(2.5, 1.0) == pytest.approx(3.0)
        assert ticks_per_command(2.5, 1.0) == 3

    def test_the_validated_set_commands_half_a_degree(self):
        # Session 32's validated set is described as "0.5 deg steps
        # (deadband 0.25, interval 0.5)". The arithmetic has to agree
        # with the description, or the description was wrong.
        assert commanded_step_deg(0.25, 0.5) == pytest.approx(0.5)
        assert ticks_per_command(0.25, 0.5) == 1

    def test_the_measured_knife_edge_is_refused(self):
        # A run configured for 1.0 deg steps produced 1.15, 2.00, 1.97,
        # 1.95 -- the step silently doubled.
        with pytest.raises(CadenceError, match="knife edge"):
            check_cadence(1.0, 1.0)

    def test_the_validated_set_survives_its_own_guard(self):
        # A guard that refused the configuration it exists to ship would
        # be a guard nobody could use.
        check_cadence(0.25, 0.5)

    def test_the_shipped_cadence_survives_its_own_guard(self):
        # 2.5 / 1.0 = 2.5, half a tick from either neighbour. Every
        # station config written before this module has these values, so
        # this is also the no-upgrade-breakage test.
        check_cadence(2.5, 1.0)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


class TestCheckCadence:
    @pytest.mark.parametrize("multiple", [1, 2, 3, 4])
    def test_every_whole_multiple_is_the_knife_edge(self, multiple):
        # The hazard is not special to one tick: it recurs at every
        # whole multiple, because the rounding decision is unstable
        # wherever it lands on an integer.
        with pytest.raises(CadenceError):
            check_cadence(float(multiple) * 0.5, 0.5)

    @pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75, 1.5, 2.5, 3.5])
    def test_a_ratio_clear_of_a_whole_tick_is_allowed(self, ratio):
        check_cadence(ratio, 1.0)

    def test_a_ratio_below_one_is_safe_by_construction(self):
        # The loop commands on every tick, so there is no rounding
        # decision to be unstable. Includes a zero deadband.
        check_cadence(0.0, 1.0)
        check_cadence(0.1, 1.0)

    def test_the_margin_is_what_decides(self):
        just_inside = 1.0 - (KNIFE_EDGE_MARGIN / 2.0)
        just_outside = 1.0 - (KNIFE_EDGE_MARGIN * 2.0)
        with pytest.raises(CadenceError):
            check_cadence(just_inside, 1.0)
        check_cadence(just_outside, 1.0)

    def test_the_guard_can_actually_fail(self):
        # Canary. Session 27 shipped a check that "proved" no widget
        # hardcoded a colour and could not have failed, because it
        # matched the wrong literal. A guard nobody has watched refuse
        # something is not a guard.
        raised = False
        try:
            check_cadence(1.0, 1.0)
        except CadenceError:
            raised = True
        assert raised

    def test_the_message_says_what_went_wrong_and_what_to_do(self):
        with pytest.raises(CadenceError) as exc:
            check_cadence(1.0, 1.0)
        message = str(exc.value)
        assert "knife edge" in message
        assert "silently doubles" in message
        assert "Halving" in message or "halving" in message

    def test_the_rate_is_a_parameter_not_a_constant(self):
        # The same cadence is a knife edge at one target rate and fine
        # at another, because rate sweeps through a pass. The nominal
        # rate is a reporting choice, not a claim about the sky.
        check_cadence(1.0, 1.0, rate_deg_s=0.4)
        with pytest.raises(CadenceError):
            check_cadence(1.0, 1.0, rate_deg_s=1.0)

    def test_a_non_positive_step_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            ticks_per_command(1.0, 0.0)
        with pytest.raises(ValueError, match="must be positive"):
            knife_edge_ratio(1.0, 1.0, rate_deg_s=0.0)


# ---------------------------------------------------------------------------
# The value object
# ---------------------------------------------------------------------------


class TestTrackingProfile:
    def test_carries_the_validated_set(self):
        profile = TrackingProfile(
            name="tracking", deadband_deg=0.25, interval_s=0.5, arrival_window_deg=1.0
        )
        assert profile.commanded_step_deg == pytest.approx(0.5)
        assert profile.window_against(2.5) == 1.0

    def test_arrival_window_falls_back_to_the_capability_record(self):
        # None is the honest identity value: "this profile does not
        # change where the rotor settles", not a placeholder.
        profile = TrackingProfile(name="stock", deadband_deg=2.5, interval_s=1.0)
        assert profile.arrival_window_deg is None
        assert profile.window_against(2.5) == 2.5
        assert profile.window_against(3.0) == 3.0

    def test_a_blank_name_is_refused(self):
        with pytest.raises(ValueError, match="needs a name"):
            TrackingProfile(name="   ", deadband_deg=2.5, interval_s=1.0)

    def test_a_negative_deadband_is_refused(self):
        with pytest.raises(ValueError, match="deadband_deg"):
            TrackingProfile(name="x", deadband_deg=-0.1, interval_s=1.0)

    def test_a_zero_deadband_is_allowed(self):
        # Command on every tick. A legitimate configuration, and the
        # limit the validated set is walking toward.
        profile = TrackingProfile(name="x", deadband_deg=0.0, interval_s=0.5)
        assert profile.commanded_step_deg == pytest.approx(0.5)

    @pytest.mark.parametrize("interval", [0.0, -1.0])
    def test_a_non_positive_interval_is_refused(self, interval):
        with pytest.raises(ValueError, match="interval_s"):
            TrackingProfile(name="x", deadband_deg=2.5, interval_s=interval)

    @pytest.mark.parametrize("window", [0.0, -1.0])
    def test_a_non_positive_arrival_window_is_refused(self, window):
        with pytest.raises(ValueError, match="arrival_window_deg"):
            TrackingProfile(name="x", deadband_deg=2.5, interval_s=1.0, arrival_window_deg=window)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_values_are_refused(self, bad):
        with pytest.raises(ValueError, match="finite"):
            TrackingProfile(name="x", deadband_deg=bad, interval_s=1.0)
        with pytest.raises(ValueError, match="finite"):
            TrackingProfile(name="x", deadband_deg=2.5, interval_s=bad)

    def test_the_error_names_the_profile(self):
        # A station with several profiles needs to know which one is
        # wrong, not just that something is.
        with pytest.raises(ValueError, match="'sprint'"):
            TrackingProfile(name="sprint", deadband_deg=-1.0, interval_s=1.0)

    def test_a_knife_edge_profile_cannot_be_constructed(self):
        with pytest.raises(CadenceError, match="knife edge"):
            TrackingProfile(name="x", deadband_deg=1.0, interval_s=1.0)

    def test_the_step_is_derived_not_stored(self):
        # Storing it would let it drift from the values it comes from.
        profile = TrackingProfile(name="x", deadband_deg=2.5, interval_s=1.0)
        assert "commanded_step_deg" not in profile.__dict__

    def test_profiles_compare_by_value(self):
        a = TrackingProfile(name="x", deadband_deg=2.5, interval_s=1.0)
        b = TrackingProfile(name="x", deadband_deg=2.5, interval_s=1.0)
        assert a == b


class TestConstants:
    def test_the_nominal_rate_is_the_near_zenith_peak(self):
        assert NOMINAL_TRACKING_RATE_DEG_S == 1.0

    def test_the_margin_leaves_room_for_the_measured_jitter(self):
        # Session 32's failure was a 2% shortfall (0.98 of a configured
        # 1.00). The margin has to be wider than that to catch it.
        assert KNIFE_EDGE_MARGIN > 0.02

    def test_the_default_interval_is_the_one_second_tick(self):
        assert DEFAULT_INTERVAL_S == 1.0

    def test_the_default_profile_name(self):
        assert DEFAULT_PROFILE_NAME == "stock"

    def test_pointing_binds_to_this_modules_interval(self):
        # Two constants that must agree and are written down twice are
        # two constants that will eventually disagree.
        from qsorbit.core.pointing import DEFAULT_TICK_INTERVAL_S

        assert DEFAULT_TICK_INTERVAL_S is DEFAULT_INTERVAL_S
