"""Unit tests for pass prediction.

Uses the same TLE test_satellite.py already uses for its own
propagation tests -- SGP4 correctness against real orbits isn't what
this file is testing; the geometry of AOS/TCA/LOS detection,
bisection, and horizon-mask filtering is.
"""

from datetime import UTC, datetime, timedelta

import pytest

from qsorbit.core.horizon import HorizonMask, HorizonPoint
from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.pass_prediction import predict_passes
from qsorbit.core.tracker.satellite import Satellite
from qsorbit.core.tracker.target import Target

TLE = """1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"""

OBSERVER = ObserverLocation(latitude=40.0, longitude=-83.0, altitude_m=250.0)
START = datetime(2026, 8, 28, 0, 0, 0, tzinfo=UTC)
END = START + timedelta(hours=48)


@pytest.fixture
def satellite():
    return Satellite.from_tle(TLE)


class TestPredictPasses:
    def test_finds_multiple_passes_in_a_two_day_window(self, satellite):
        passes = predict_passes(satellite, OBSERVER, START, END)

        assert len(passes) > 5

    def test_passes_are_chronological_and_non_overlapping(self, satellite):
        passes = predict_passes(satellite, OBSERVER, START, END)

        for earlier, later in zip(passes, passes[1:], strict=False):
            assert earlier.los.time < later.aos.time

    def test_aos_and_los_bracket_tca(self, satellite):
        passes = predict_passes(satellite, OBSERVER, START, END)

        assert passes
        for one_pass in passes:
            assert one_pass.aos.time < one_pass.tca.time < one_pass.los.time

    def test_aos_and_los_elevation_is_near_the_threshold(self, satellite):
        passes = predict_passes(satellite, OBSERVER, START, END, min_elevation_deg=0.0)

        assert passes
        for one_pass in passes:
            assert abs(one_pass.aos.sky_position.elevation) < 0.5
            assert abs(one_pass.los.sky_position.elevation) < 0.5

    def test_tca_elevation_is_the_maximum_of_the_pass(self, satellite):
        passes = predict_passes(satellite, OBSERVER, START, END, track_step_s=15.0)

        assert passes
        for one_pass in passes:
            assert one_pass.max_elevation_deg == one_pass.tca.sky_position.elevation
            assert all(
                event.sky_position.elevation <= one_pass.max_elevation_deg + 1e-6
                for event in one_pass.az_track
            )

    def test_az_track_starts_at_aos_and_ends_at_los(self, satellite):
        passes = predict_passes(satellite, OBSERVER, START, END)

        assert passes
        one_pass = passes[0]
        assert one_pass.az_track[0].time == one_pass.aos.time
        assert one_pass.az_track[-1].time == one_pass.los.time

    def test_raising_the_flat_threshold_shortens_or_removes_passes(self, satellite):
        low = predict_passes(satellite, OBSERVER, START, END, min_elevation_deg=0.0)
        high = predict_passes(satellite, OBSERVER, START, END, min_elevation_deg=60.0)

        assert len(high) <= len(low)
        for one_pass in high:
            assert one_pass.max_elevation_deg >= 60.0 - 1e-6

    def test_a_mask_blocking_the_whole_sky_finds_nothing(self, satellite):
        blocking_everywhere = HorizonMask(points=(HorizonPoint(0.0, 90.0),))

        passes = predict_passes(satellite, OBSERVER, START, END, horizon_mask=blocking_everywhere)

        assert passes == []

    def test_horizon_mask_overrides_flat_min_elevation(self, satellite):
        # A mask present at all replaces min_elevation_deg entirely --
        # an all-zero mask should behave like the 0.0 default even
        # though a high min_elevation_deg is also passed.
        open_sky = HorizonMask(points=(HorizonPoint(0.0, 0.0),))

        with_mask = predict_passes(
            satellite, OBSERVER, START, END, min_elevation_deg=80.0, horizon_mask=open_sky
        )
        flat_zero = predict_passes(satellite, OBSERVER, START, END, min_elevation_deg=0.0)

        assert len(with_mask) == len(flat_zero)

    def test_a_pass_already_underway_at_start_is_not_reported(self, satellite):
        # Find a real AOS, then start the search a bit after it -- the
        # pass that was already above the horizon should not appear.
        full_window = predict_passes(satellite, OBSERVER, START, END)
        assert full_window
        first_pass = full_window[0]
        mid_pass_start = first_pass.aos.time + (first_pass.los.time - first_pass.aos.time) / 2

        passes = predict_passes(satellite, OBSERVER, mid_pass_start, END)

        assert all(one_pass.aos.time != first_pass.aos.time for one_pass in passes)
        assert not any(one_pass.aos.time < mid_pass_start for one_pass in passes)

    def test_naive_start_is_rejected(self, satellite):
        with pytest.raises(ValueError, match="timezone-aware"):
            predict_passes(satellite, OBSERVER, datetime(2026, 8, 28), END)  # noqa: DTZ001

    def test_end_before_start_is_rejected(self, satellite):
        with pytest.raises(ValueError, match="after"):
            predict_passes(satellite, OBSERVER, END, START)

    def test_non_positive_step_s_is_rejected(self, satellite):
        with pytest.raises(ValueError, match="step_s"):
            predict_passes(satellite, OBSERVER, START, END, step_s=0.0)

    def test_non_positive_track_step_s_is_rejected(self, satellite):
        with pytest.raises(ValueError, match="track_step_s"):
            predict_passes(satellite, OBSERVER, START, END, track_step_s=-1.0)


class TestIllumination:
    def test_illuminated_is_none_by_default(self, satellite):
        passes = predict_passes(satellite, OBSERVER, START, END)

        assert passes
        assert all(one_pass.illuminated is None for one_pass in passes)

    def test_illuminated_is_a_bool_when_requested(self, satellite):
        passes = predict_passes(satellite, OBSERVER, START, END, include_illumination=True)

        assert passes
        assert all(isinstance(one_pass.illuminated, bool) for one_pass in passes)

    def test_a_target_without_state_at_raises_on_illumination(self):
        class BareTarget:
            name = "bare"

            def topocentric_state(self, observer, time):
                return Satellite.from_tle(TLE).topocentric_state(observer, time)

        with pytest.raises(TypeError, match="state_at"):
            predict_passes(BareTarget(), OBSERVER, START, END, include_illumination=True)

    def test_bare_target_satisfies_the_target_protocol(self):
        # Confirms the test double above is actually testing the
        # "no state_at" case, not just an unrelated duck-typing miss.
        class BareTarget:
            name = "bare"

            def topocentric_state(self, observer, time):
                return Satellite.from_tle(TLE).topocentric_state(observer, time)

        assert isinstance(BareTarget(), Target)
