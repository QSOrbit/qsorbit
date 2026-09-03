"""Unit tests for the gain clamp: what integral gain a rotor may run.

The arithmetic under test is::

    windup at detection = 0.5 x Ki x free_play^2 / rate

which composes two separately-derived facts. Detection takes
``free_play / rate`` seconds, because the stall detector cannot open its
gate until the commanded setpoint has advanced past the slop (Session
34, bench-confirmed at 3.0 s against a 1 deg/s target). Windup over that
time is ``0.5 x Ki x rate x t^2``, which is PID_v1's ``outputSum += ki *
error`` integrated against a linearly growing error.

**The rate cancels once and inverts**, so slow targets are the dangerous
case. Several tests below pin that specifically, because the intuition
runs the other way and a future refactor that "simplifies" the rate out
would look reasonable.

Every hardware number here is measured: azimuth free play 2.95 deg and
breakaway ~17 PWM, elevation 2.55 deg and ~21-26 PWM, on Phil's SatNOGS
rotator with stock firmware v2.2.1.
"""

import math

import pytest

from qsorbit.core.rotor import AzimuthWrap, RotorCapabilities
from qsorbit.core.tracking_profile import (
    DESIGN_RATE_DEG_S,
    NOMINAL_TRACKING_RATE_DEG_S,
    GainClampError,
    GainPolicyError,
    TrackingProfile,
    UnmeasuredMechanicsError,
    check_axis_gain,
    max_safe_free_play_deg,
    max_safe_ki,
    windup_at_detection_pwm,
)

MEASURED = {
    "azimuth_free_play_deg": 2.95,
    "azimuth_breakaway_pwm": 17.0,
    "elevation_free_play_deg": 2.55,
    "elevation_breakaway_pwm": 21.0,
}


def capabilities(**overrides) -> RotorCapabilities:
    """A rotor with Phil's measured travel, mechanics overridable."""
    fields = {
        "azimuth_min_deg": 0.0,
        "azimuth_max_deg": 360.0,
        "elevation_min_deg": 0.0,
        "elevation_max_deg": 180.0,
        "azimuth_wrap": AzimuthWrap.EXTRA_ROTATION,
        "acceptance_window_deg": 2.5,
        "rs485_turnaround_s": 0.15,
        **MEASURED,
    }
    fields.update(overrides)
    return RotorCapabilities(**fields)


def profile(**overrides) -> TrackingProfile:
    """Session 32's validated cadence, gains overridable."""
    fields = {
        "name": "tracking",
        "deadband_deg": 0.25,
        "interval_s": 0.5,
        "arrival_window_deg": 1.0,
        "azimuth_kp": 8.0,
        "azimuth_ki": 1.0,
        "azimuth_kd": 0.5,
        "elevation_kp": 10.0,
        "elevation_ki": 1.0,
        "elevation_kd": 0.3,
    }
    fields.update(overrides)
    return TrackingProfile(**fields)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


class TestWindupArithmetic:
    def test_the_shape_is_ki_times_free_play_squared_over_rate(self):
        assert windup_at_detection_pwm(1.0, 3.0, rate_deg_s=0.25) == pytest.approx(18.0)
        assert windup_at_detection_pwm(1.0, 2.95, rate_deg_s=0.25) == pytest.approx(17.405)

    def test_slow_targets_are_the_dangerous_case(self):
        # The instinct is the opposite, so this is pinned deliberately:
        # halving the rate DOUBLES the exposure.
        fast = windup_at_detection_pwm(1.0, 3.0, rate_deg_s=1.0)
        half = windup_at_detection_pwm(1.0, 3.0, rate_deg_s=0.5)
        quarter = windup_at_detection_pwm(1.0, 3.0, rate_deg_s=0.25)
        assert fast == pytest.approx(4.5)
        assert half == pytest.approx(2 * fast)
        assert quarter == pytest.approx(4 * fast)

    def test_free_play_matters_quadratically(self):
        # Halving the free play quarters the exposure. This is why the
        # wind measurement is worth a morning.
        whole = windup_at_detection_pwm(1.0, 3.0)
        halved = windup_at_detection_pwm(1.0, 1.5)
        assert halved == pytest.approx(whole / 4.0)

    def test_windup_is_linear_in_ki(self):
        assert windup_at_detection_pwm(2.0, 3.0) == pytest.approx(
            2 * windup_at_detection_pwm(1.0, 3.0)
        )

    def test_zero_ki_cannot_wind_up(self):
        assert windup_at_detection_pwm(0.0, 99.0) == 0.0

    def test_a_non_positive_rate_is_refused(self):
        with pytest.raises(ValueError, match="rate_deg_s must be positive"):
            windup_at_detection_pwm(1.0, 3.0, rate_deg_s=0.0)

    def test_the_design_rate_is_slower_than_the_nominal_one(self):
        # They answer different questions and must not be collapsed:
        # nominal asks "what step will this cadence command at the
        # hardest the rotor works", design asks "how slowly could the
        # target be moving when something jams".
        assert DESIGN_RATE_DEG_S < NOMINAL_TRACKING_RATE_DEG_S


# ---------------------------------------------------------------------------
# The per-axis check
# ---------------------------------------------------------------------------


class TestCheckAxisGain:
    def test_the_validated_set_is_refused_on_this_rotators_azimuth(self):
        # The finding that decided PR2c: Session 32's Ki 1.0 misses
        # azimuth's breakaway by 2%, at the measured 2.95 deg.
        with pytest.raises(GainClampError) as caught:
            check_axis_gain(1.0, 2.95, 17.0, axis="azimuth", profile_name="tracking")
        message = str(caught.value)
        assert "17.4" in message
        assert "azimuth" in message

    def test_the_same_set_passes_on_this_rotators_elevation(self):
        # Elevation has more free play headroom because its breakaway is
        # higher -- the boom's weight has to be lifted. This asymmetry is
        # why the clamp is per-axis rather than pooled.
        check_axis_gain(1.0, 2.55, 21.0, axis="elevation", profile_name="tracking")

    def test_zero_ki_passes_whatever_the_mechanics(self):
        check_axis_gain(0.0, 99.0, 0.001, axis="azimuth", profile_name="stock")

    def test_exactly_at_breakaway_passes(self):
        # The hazard is driving the motor, and at breakaway it has not
        # driven yet. A strict inequality here would refuse a
        # configuration that is precisely safe.
        check_axis_gain(1.0, 2.0, 8.0, axis="azimuth", profile_name="edge")

    def test_a_hair_over_breakaway_is_refused(self):
        with pytest.raises(GainClampError):
            check_axis_gain(1.0001, 2.0, 8.0, axis="azimuth", profile_name="edge")

    def test_the_message_names_both_ways_out(self):
        with pytest.raises(GainClampError) as caught:
            check_axis_gain(1.0, 2.95, 17.0, axis="azimuth", profile_name="tracking")
        message = str(caught.value)
        # A safe Ki...
        assert "0.97" in message
        # ...and the free play that would clear the gain as asked.
        assert "2.91" in message

    @pytest.mark.parametrize(
        ("ki", "free_play", "breakaway"),
        [(1.0, 2.95, 17.0), (1.0, 2.55, 12.0), (3.0, 4.0, 20.0), (0.75, 6.0, 30.0)],
    )
    def test_both_suggested_remedies_actually_pass(self, ki, free_play, breakaway):
        # A remedy nobody checked is a remedy that can be wrong, and the
        # first version of this code was: both figures were formatted to
        # two decimals with round-to-nearest, so 0.9767 printed as "0.98"
        # and an operator who followed the advice was refused a second
        # time. They are floored now, and this test is what says so.
        with pytest.raises(GainClampError) as caught:
            check_axis_gain(ki, free_play, breakaway, axis="azimuth", profile_name="t")
        message = str(caught.value)
        safe_ki = float(message.split("Ki to ")[1].split(" or less")[0])
        safe_free_play = float(message.split("at or under ")[1].split(" deg")[0])
        # Each remedy, applied on its own, must clear the check.
        check_axis_gain(safe_ki, free_play, breakaway, axis="azimuth", profile_name="t")
        check_axis_gain(ki, safe_free_play, breakaway, axis="azimuth", profile_name="t")

    def test_the_refusal_message_is_printable_on_a_windows_console(self):
        # Session 34 shipped an em dash into a console message and it
        # rendered as a stray glyph on Phil's terminal. This message is
        # the one an operator reads while something is wrong, so it is
        # the worst possible place for mojibake.
        with pytest.raises(GainClampError) as caught:
            check_axis_gain(1.0, 2.95, 17.0, axis="azimuth", profile_name="tracking")
        str(caught.value).encode("ascii")

    def test_the_unmeasured_message_is_printable_too(self):
        bare = capabilities(
            azimuth_free_play_deg=None,
            azimuth_breakaway_pwm=None,
            elevation_free_play_deg=None,
            elevation_breakaway_pwm=None,
        )
        with pytest.raises(UnmeasuredMechanicsError) as caught:
            profile().check_against(bare)
        str(caught.value).encode("ascii")

    def test_a_faster_design_rate_would_let_the_set_through(self):
        # Not an endorsement -- a record of what the 0.25 choice costs,
        # so that a future change to DESIGN_RATE_DEG_S is a visible one.
        check_axis_gain(1.0, 2.95, 17.0, axis="azimuth", profile_name="t", rate_deg_s=0.5)

    def test_the_error_is_a_value_error(self):
        # So load_station_config re-raises it as a ConfigError carrying
        # the file name, the same way every other rejection is reported.
        assert issubclass(GainClampError, GainPolicyError)
        assert issubclass(GainPolicyError, ValueError)


# ---------------------------------------------------------------------------
# Profile against capabilities
# ---------------------------------------------------------------------------


class TestProfileCheckAgainst:
    def test_a_profile_with_no_gains_passes_on_an_unmeasured_rotor(self):
        # This is what lets `stock` work on anybody's hardware on day one.
        bare = capabilities(
            azimuth_free_play_deg=None,
            azimuth_breakaway_pwm=None,
            elevation_free_play_deg=None,
            elevation_breakaway_pwm=None,
        )
        TrackingProfile(name="stock", deadband_deg=2.5, interval_s=1.0).check_against(bare)

    def test_zero_integral_gain_passes_on_an_unmeasured_rotor(self):
        # Gains that cannot accumulate need no mechanics to check against.
        bare = capabilities(
            azimuth_free_play_deg=None,
            azimuth_breakaway_pwm=None,
            elevation_free_play_deg=None,
            elevation_breakaway_pwm=None,
        )
        profile(azimuth_ki=0.0, elevation_ki=0.0).check_against(bare)

    def test_integral_gain_on_an_unmeasured_rotor_is_refused(self):
        bare = capabilities(
            azimuth_free_play_deg=None,
            azimuth_breakaway_pwm=None,
            elevation_free_play_deg=None,
            elevation_breakaway_pwm=None,
        )
        with pytest.raises(UnmeasuredMechanicsError) as caught:
            profile().check_against(bare)
        assert "no free_play_deg or breakaway_pwm" in str(caught.value)

    def test_unmeasured_is_a_different_error_from_unsafe(self):
        # Different problems, different fixes. An operator told the wrong
        # one changes the wrong number.
        assert not issubclass(UnmeasuredMechanicsError, GainClampError)
        assert not issubclass(GainClampError, UnmeasuredMechanicsError)

    def test_the_validated_set_is_refused_on_this_station(self):
        with pytest.raises(GainClampError):
            profile().check_against(capabilities())

    def test_azimuth_is_what_refuses_it_not_elevation(self):
        # If this ever flips, the mechanics have been swapped somewhere.
        with pytest.raises(GainClampError) as caught:
            profile().check_against(capabilities())
        assert "azimuth" in str(caught.value)

    def test_dropping_only_azimuth_ki_lets_it_through(self):
        profile(azimuth_ki=0.0).check_against(capabilities())

    def test_a_measured_free_play_under_the_threshold_clears_it(self):
        # The wind measurement's whole point.
        profile().check_against(capabilities(azimuth_free_play_deg=1.5))

    def test_elevation_alone_can_refuse_a_profile(self):
        # Per-axis judging, in the direction that would be masked if the
        # axes were pooled on azimuth's larger free play.
        with pytest.raises(GainClampError) as caught:
            profile(azimuth_ki=0.0).check_against(capabilities(elevation_breakaway_pwm=1.0))
        assert "elevation" in str(caught.value)


# ---------------------------------------------------------------------------
# The gains half of the profile value object
# ---------------------------------------------------------------------------


class TestProfileGains:
    def test_a_profile_without_gains_pushes_nothing(self):
        # None is "write nothing", not "write zeros". A controller that
        # has not been written to since power-on is definitionally
        # running the firmware's own values, which is what makes `stock`
        # a real baseline rather than a guess at one.
        assert TrackingProfile(name="stock", deadband_deg=2.5, interval_s=1.0).gains is None

    def test_all_six_registers_are_returned_in_register_order(self):
        from qsorbit.core.rotor import GainRegister

        gains = profile().gains
        assert gains is not None
        assert list(gains) == list(GainRegister)
        assert gains[GainRegister.AZIMUTH_KP] == 8.0
        assert gains[GainRegister.ELEVATION_KD] == 0.3

    @pytest.mark.parametrize(
        "missing",
        ["azimuth_kp", "azimuth_ki", "azimuth_kd", "elevation_kp", "elevation_ki", "elevation_kd"],
    )
    def test_a_partial_gain_set_is_refused(self, missing):
        # Whatever is left out keeps the firmware's compiled default, so
        # a partial set runs a mixture nobody chose -- the same hazard
        # push_gains() reports every register for.
        with pytest.raises(ValueError, match="all six or none"):
            profile(**{missing: None})

    def test_a_negative_gain_is_refused(self):
        with pytest.raises(ValueError, match="must not be negative"):
            profile(azimuth_ki=-1.0)

    def test_a_non_finite_gain_is_refused(self):
        with pytest.raises(ValueError, match="must be a finite number"):
            profile(azimuth_kp=math.inf)

    def test_zero_gains_are_a_legitimate_declaration(self):
        # Ki 0 is stock's actual value, and declaring it explicitly is
        # different from declaring nothing: it writes the register.
        declared = profile(azimuth_ki=0.0, elevation_ki=0.0)
        assert declared.gains is not None

    def test_gains_do_not_disturb_the_cadence_half(self):
        assert profile().commanded_step_deg == pytest.approx(0.5)


class TestSafeValueHelpers:
    """The two "here is what would work" figures, and their rounding.

    Both round **down**. Both call sites got this wrong independently
    with round-to-nearest -- the refusal message and ``qsorbit status``
    -- which is why the arithmetic lives in one place now.
    """

    def test_max_safe_ki_rounds_down(self):
        # Exact value is 0.9767; to nearest it would be 0.98, which the
        # clamp refuses.
        assert max_safe_ki(2.95, 17.0) == 0.97

    def test_max_safe_free_play_rounds_down(self):
        # Exact value is 2.9155; to nearest it would be 2.92.
        assert max_safe_free_play_deg(1.0, 17.0) == 2.91

    @pytest.mark.parametrize(
        ("free_play", "breakaway"),
        [(2.95, 17.0), (2.55, 21.0), (1.0, 17.0), (4.0, 20.0), (0.5, 5.0)],
    )
    def test_the_reported_ki_always_passes_the_check(self, free_play, breakaway):
        check_axis_gain(
            max_safe_ki(free_play, breakaway),
            free_play,
            breakaway,
            axis="azimuth",
            profile_name="t",
        )

    @pytest.mark.parametrize(
        ("ki", "breakaway"),
        [(1.0, 17.0), (1.0, 21.0), (0.5, 17.0), (3.0, 20.0), (2.0, 5.0)],
    )
    def test_the_reported_free_play_always_passes_the_check(self, ki, breakaway):
        check_axis_gain(
            ki,
            max_safe_free_play_deg(ki, breakaway),
            breakaway,
            axis="azimuth",
            profile_name="t",
        )

    def test_a_hair_above_the_reported_ki_is_refused(self):
        # Confirms the figure is a real boundary rather than a
        # comfortable understatement.
        with pytest.raises(GainClampError):
            check_axis_gain(0.99, 2.95, 17.0, axis="azimuth", profile_name="t")

    def test_the_helpers_agree_with_each_other(self):
        # Inverses of the same equation, so a change to one that misses
        # the other should fail here.
        ki = max_safe_ki(2.95, 17.0)
        assert max_safe_free_play_deg(ki, 17.0) >= 2.95
