"""Unit tests for ground_track().

Reuses the "TEME EXAMPLE" TLE test_satellite.py and test_picker.py
already trust, matching those modules' own reasoning: proving what
this function *does with* a satellite's propagated positions doesn't
need a second real orbit, only a satellite that propagates cleanly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from qsorbit.core.tracker import Satellite
from qsorbit.core.tracker.ground_track import ground_track

_TEME_EXAMPLE_TLE = """\
TEME EXAMPLE
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
"""

_NOW = datetime(2000, 6, 29, 12, 0, 0, tzinfo=UTC)


class TestGroundTrack:
    def test_default_span_and_step_produce_37_points(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)

        track = ground_track(sat, _NOW)

        assert len(track) == 37

    def test_a_smaller_span_and_step_produce_exactly_the_expected_count(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)

        track = ground_track(sat, _NOW, span_minutes=10.0, step_minutes=5.0)

        # -10, -5, 0, 5, 10
        assert len(track) == 5

    def test_first_and_last_points_match_the_spans_own_boundary_instants(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)

        track = ground_track(sat, _NOW, span_minutes=30.0, step_minutes=10.0)

        first_expected = sat.subpoint_at(_NOW - timedelta(minutes=30.0))
        last_expected = sat.subpoint_at(_NOW + timedelta(minutes=30.0))
        assert track[0] == first_expected
        assert track[-1] == last_expected

    def test_points_are_in_chronological_order(self):
        # Not literally checking timestamps (the return value doesn't
        # carry one), but checking that each point matches propagating
        # to progressively later instants -- the same guarantee stated
        # a different way.
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)

        track = ground_track(sat, _NOW, span_minutes=20.0, step_minutes=10.0)
        expected = tuple(
            sat.subpoint_at(_NOW + timedelta(minutes=offset))
            for offset in (-20.0, -10.0, 0.0, 10.0, 20.0)
        )

        assert track == expected

    def test_step_that_does_not_evenly_divide_span_still_reaches_both_ends(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)

        track = ground_track(sat, _NOW, span_minutes=10.0, step_minutes=3.0)

        assert track[0] == sat.subpoint_at(_NOW - timedelta(minutes=10.0))
        assert track[-1] == sat.subpoint_at(_NOW + timedelta(minutes=10.0))

    def test_naive_datetime_is_rejected(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        naive = datetime(2000, 6, 29, 12, 0, 0)

        with pytest.raises(ValueError, match="timezone-aware"):
            ground_track(sat, naive)

    @pytest.mark.parametrize("bad", [0.0, -5.0])
    def test_non_positive_span_minutes_is_an_error(self, bad):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)

        with pytest.raises(ValueError, match="span_minutes"):
            ground_track(sat, _NOW, span_minutes=bad)

    @pytest.mark.parametrize("bad", [0.0, -5.0])
    def test_non_positive_step_minutes_is_an_error(self, bad):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)

        with pytest.raises(ValueError, match="step_minutes"):
            ground_track(sat, _NOW, step_minutes=bad)
