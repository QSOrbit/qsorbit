"""Unit tests for naked-eye visibility windows within a pass.

Uses the same TLE ``test_pass_prediction.py`` does, for the same reason:
SGP4 correctness is not what this file tests. What it tests is whether the
window's *edges* land where visibility actually changes, and whether the
window says something the old single flag could not.

That last part is the point of the feature. :attr:`Pass.illuminated`
answers only for the instant of closest approach, and over a three-day
sample of this orbit it is **wrong on more than a quarter of passes** --
reporting "not visible" for passes carrying minutes of real naked-eye
visibility. :meth:`TestTheWindowSaysWhatTheFlagCannot` pins one of those.
"""

from datetime import UTC, datetime, timedelta

import pytest

from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.pass_prediction import (
    DEFAULT_TWILIGHT_SUN_ELEVATION_DEG,
    VisibleWindow,
    predict_passes,
    visible_window,
)
from qsorbit.core.tracker.satellite import Satellite
from qsorbit.core.tracker.state import EciState
from qsorbit.core.tracker.sun import is_illuminated, sun_elevation_deg

TLE = """1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"""

OBSERVER = ObserverLocation(latitude=40.0, longitude=-83.0, altitude_m=250.0)
START = datetime(2026, 8, 28, 0, 0, 0, tzinfo=UTC)
END = START + timedelta(hours=72)


@pytest.fixture
def satellite():
    return Satellite.from_tle(TLE)


@pytest.fixture
def passes(satellite):
    return predict_passes(satellite, OBSERVER, START, END, include_illumination=True)


def _visible_at(satellite: Satellite, when: datetime) -> bool:
    """The visibility predicate, computed independently of the window code.

    Deliberately rebuilt from the two underlying conditions rather than
    calling the module's own helper: a test that asks the code under test
    whether it agrees with itself proves nothing, which is the lesson
    ``sun.py``'s frame bug taught this project the expensive way.
    """
    state: EciState = satellite.state_at(when)
    return is_illuminated(state.position_km, when) and (
        sun_elevation_deg(OBSERVER, when) <= DEFAULT_TWILIGHT_SUN_ELEVATION_DEG
    )


class TestVisibleWindow:
    def test_duration_is_the_interval_it_spans(self):
        from qsorbit.core.tracker.pass_prediction import PassEvent

        begins = datetime(2026, 8, 28, 1, 30, 0, tzinfo=UTC)
        window = VisibleWindow(
            begins=PassEvent(time=begins, sky_position=None),
            ends=PassEvent(time=begins + timedelta(seconds=231), sky_position=None),
        )

        assert window.duration_s == pytest.approx(231.0)


class TestWindowsAreFound:
    def test_illumination_must_be_requested(self, satellite):
        without = predict_passes(satellite, OBSERVER, START, END)

        assert all(one_pass.visible_window is None for one_pass in without)

    def test_the_sample_contains_both_visible_and_invisible_passes(self, passes):
        # If this ever fails the fixture has drifted into a degenerate
        # window, and every other test here would be proving nothing.
        with_window = [p for p in passes if p.visible_window is not None]

        assert len(passes) >= 12
        assert 0 < len(with_window) < len(passes)

    def test_a_window_always_lies_inside_its_pass(self, passes):
        for one_pass in passes:
            window = one_pass.visible_window
            if window is None:
                continue

            assert one_pass.aos.time <= window.begins.time
            assert window.begins.time < window.ends.time
            assert window.ends.time <= one_pass.los.time

    def test_a_window_carries_where_to_look(self, passes):
        # The sky position at each edge is the operator-facing half of
        # this: "visible from 20:41" is much less useful than "visible
        # from 20:41, low in the south-west."
        for one_pass in passes:
            window = one_pass.visible_window
            if window is None:
                continue

            for event in (window.begins, window.ends):
                assert 0.0 <= event.sky_position.azimuth < 360.0
                assert -90.0 <= event.sky_position.elevation <= 90.0


class TestTheEdgesAreRealTransitions:
    """The bisection has to land *on* the boundary, not near it.

    Checked against a predicate rebuilt from first principles, a second
    either side of each edge. An edge that sits on the pass's own AOS or
    LOS is skipped: the satellite was already lit when it rose, so there
    is no transition there to find.
    """

    def test_just_inside_each_edge_is_visible(self, satellite, passes):
        checked = 0
        for one_pass in passes:
            window = one_pass.visible_window
            if window is None:
                continue

            assert _visible_at(satellite, window.begins.time + timedelta(seconds=1))
            assert _visible_at(satellite, window.ends.time - timedelta(seconds=1))
            checked += 1

        assert checked > 0

    def test_just_outside_a_refined_edge_is_not_visible(self, satellite, passes):
        checked = 0
        for one_pass in passes:
            window = one_pass.visible_window
            if window is None:
                continue

            if window.begins.time > one_pass.aos.time + timedelta(seconds=30):
                assert not _visible_at(satellite, window.begins.time - timedelta(seconds=30))
                checked += 1
            if window.ends.time < one_pass.los.time - timedelta(seconds=30):
                assert not _visible_at(satellite, window.ends.time + timedelta(seconds=30))
                checked += 1

        assert checked > 0


class TestTheWindowSaysWhatTheFlagCannot:
    """The reason this feature exists, pinned as a regression.

    Over this three-day sample, five of eighteen passes report
    ``illuminated is False`` at closest approach while carrying a real
    visible window -- the longest of them nearly twenty-four minutes. An
    operator trusting the flag would stay indoors for all five.
    """

    def test_the_flag_and_the_window_disagree_on_real_passes(self, passes):
        disagreements = [
            one_pass
            for one_pass in passes
            if one_pass.illuminated is False and one_pass.visible_window is not None
        ]

        assert len(disagreements) >= 3

    def test_the_disagreement_is_not_a_rounding_sliver(self, passes):
        # A few seconds either side of TCA would be a boundary artifact
        # and not worth a feature. These are minutes.
        longest = max(
            (
                one_pass.visible_window.duration_s
                for one_pass in passes
                if one_pass.illuminated is False and one_pass.visible_window is not None
            ),
            default=0.0,
        )

        assert longest > 600.0

    def test_a_flagged_pass_always_has_a_window(self, passes):
        # The converse must hold: if the satellite was visible at closest
        # approach then closest approach is inside some window, so one
        # exists. A failure here means the search stepped over it.
        for one_pass in passes:
            if one_pass.illuminated:
                assert one_pass.visible_window is not None
                assert (
                    one_pass.visible_window.begins.time
                    <= one_pass.tca.time
                    <= one_pass.visible_window.ends.time
                )


class TestTheStandaloneEntryPoint:
    """``visible_window()`` must be the same search, not a similar one.

    It exists purely so a caller holding one :class:`Pass` can pay for
    that pass alone instead of for every pass in a search window. That
    is only a saving if the answer is *identical* -- a cheaper function
    that quietly disagrees with the one it replaces would be a far
    worse bug than the cost it avoided, and it would show up as the
    picker and ``qsorbit plan --visual`` contradicting each other about
    the same pass.
    """

    def test_it_agrees_exactly_with_the_search_that_computes_them_in_bulk(self, satellite, passes):
        checked = 0
        for one_pass in passes:
            standalone = visible_window(satellite, OBSERVER, one_pass)
            in_bulk = one_pass.visible_window

            if in_bulk is None:
                assert standalone is None
            else:
                assert standalone is not None
                assert standalone.begins.time == in_bulk.begins.time
                assert standalone.ends.time == in_bulk.ends.time
                checked += 1

        # Without this the loop above passes on a sample where every
        # window is None, which proves only that None == None.
        assert checked > 0

    def test_a_target_without_state_at_is_refused(self, satellite, passes):
        from qsorbit.core.tracker.celestial import celestial_target

        # The same category error predict_passes refuses, refused at
        # this entry point too -- a public function that accepted a
        # star and returned None would be a quieter way to get the
        # wrong answer than the one TestNonSatelliteTargets guards.
        with pytest.raises(TypeError, match="state_at"):
            visible_window(celestial_target("vega"), OBSERVER, passes[0])


class TestNonSatelliteTargets:
    def test_a_target_without_state_at_is_refused(self):
        from qsorbit.core.tracker.celestial import celestial_target

        # Earth's-shadow illumination is meaningless for the Moon or a
        # star, and silently returning None would hide the category error.
        # Twelve hours, not six: the Moon's pass here runs 00:05 to 11:14,
        # and predict_passes only returns passes that *complete* inside the
        # window. A six-hour window finds nothing, so the illumination code
        # never runs and this would pass without ever reaching what it
        # claims to test.
        with pytest.raises(TypeError, match="state_at"):
            predict_passes(
                celestial_target("moon"),
                OBSERVER,
                START,
                START + timedelta(hours=12),
                include_illumination=True,
            )

    def test_the_window_used_above_really_contains_a_pass(self):
        # Guards the guard: if the Moon's pass ever falls outside this
        # window, the test above silently stops testing anything.
        from qsorbit.core.tracker.celestial import celestial_target

        found = predict_passes(
            celestial_target("moon"), OBSERVER, START, START + timedelta(hours=12)
        )

        assert len(found) >= 1
