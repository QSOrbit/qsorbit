"""Unit tests for ObserverLocation and topocentric satellite state.

The elevation scenarios here are geometric certainties (directly below
something means straight up; a rise/set moment means the horizon by
definition) rather than facts needing an external published source, so
each test constructs its own known-good scenario using skyfield's own
trusted geometry helpers (``wgs84.subpoint_of``, ``find_events``)
instead of hand-deriving the horizon geometry or picking arbitrary
numbers.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from skyfield.api import load, wgs84

from qsorbit.core.tracker import ObserverLocation, PropagationError, Satellite

# Same TLE used in Chunk C's tests (Vallado AIAA 2006-6753 Appendix C
# example) — reused here so this chunk's tests build on a time window
# already known to be a valid propagation window for this satellite.
_TEME_EXAMPLE_TLE = """\
TEME EXAMPLE
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
"""

# GOCE's last-ever element set before re-entry, from skyfield's
# "Detecting Propagation Errors" documentation — reused from Chunk C.
_GOCE_DECAY_TLE = """\
GOCE
1 34602U 09013A   13314.96046236  .14220718  20669-5  50412-4 0   930
2 34602 096.5717 344.5256 0009826 296.2811 064.0942 16.58673376272979
"""


class TestObserverLocationValidation:
    def test_typical_location(self):
        obs = ObserverLocation(latitude=40.8939, longitude=-83.8917, elevation_m=280.0)
        assert obs.latitude == 40.8939
        assert obs.longitude == -83.8917
        assert obs.elevation_m == 280.0

    def test_elevation_defaults_to_sea_level(self):
        obs = ObserverLocation(latitude=0.0, longitude=0.0)
        assert obs.elevation_m == 0.0

    def test_latitude_bounds_inclusive(self):
        assert ObserverLocation(latitude=90.0, longitude=0.0).latitude == 90.0
        assert ObserverLocation(latitude=-90.0, longitude=0.0).latitude == -90.0

    def test_longitude_bounds_inclusive(self):
        assert ObserverLocation(latitude=0.0, longitude=180.0).longitude == 180.0
        assert ObserverLocation(latitude=0.0, longitude=-180.0).longitude == -180.0

    def test_latitude_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="latitude"):
            ObserverLocation(latitude=90.1, longitude=0.0)
        with pytest.raises(ValueError, match="latitude"):
            ObserverLocation(latitude=-90.1, longitude=0.0)

    def test_longitude_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="longitude"):
            ObserverLocation(latitude=0.0, longitude=180.1)
        with pytest.raises(ValueError, match="longitude"):
            ObserverLocation(latitude=0.0, longitude=-180.1)


def _height_km(sat: Satellite, t) -> float:
    """The satellite's height above the ellipsoid at ``t`` — a loose sanity bound.

    Not a precision check (that's the job of Chunk C's tests); just
    confirms range_km lands in the right ballpark for "directly
    overhead" rather than, say, being Earth's full radius off due to a
    units or frame mixup. For an observer exactly at the subpoint, range
    and height above the ellipsoid are the same quantity by definition.
    """
    return wgs84.height_of(sat.skyfield_satellite.at(t)).km


class TestTopocentricStateOverhead:
    def test_elevation_near_90_when_observer_is_at_the_subpoint(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        ts = load.timescale(builtin=True)
        epoch = sat.skyfield_satellite.epoch
        t = ts.tt_jd(epoch.whole + 3.0, epoch.tt_fraction)

        subpoint = wgs84.subpoint_of(sat.skyfield_satellite.at(t))
        observer = ObserverLocation(
            latitude=subpoint.latitude.degrees,
            longitude=subpoint.longitude.degrees,
        )

        state = sat.topocentric_state(observer, t.utc_datetime())

        assert state.position.elevation == pytest.approx(90.0, abs=0.1)
        assert state.range_km == pytest.approx(_height_km(sat, t), abs=5.0)


class TestTopocentricStateHorizon:
    def test_elevation_near_0_at_a_rise_or_set_event(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        ts = load.timescale(builtin=True)
        epoch = sat.skyfield_satellite.epoch
        observer = ObserverLocation(latitude=40.0, longitude=-83.0)

        t0 = ts.tt_jd(epoch.whole + 3.0, epoch.tt_fraction)
        t1 = ts.tt_jd(epoch.whole + 4.0, epoch.tt_fraction)
        times, events = sat.skyfield_satellite.find_events(
            observer.skyfield_position, t0, t1, altitude_degrees=0.0
        )
        rise_or_set_times = [ti for ti, event in zip(times, events, strict=True) if event != 1]
        assert rise_or_set_times, "expected at least one rise/set event in this window"

        state = sat.topocentric_state(observer, rise_or_set_times[0].utc_datetime())

        assert state.position.elevation == pytest.approx(0.0, abs=1.0)


class TestTopocentricStatePropagationError:
    def test_raises_when_orbit_is_invalid(self):
        sat = Satellite.from_tle(_GOCE_DECAY_TLE)
        observer = ObserverLocation(latitude=0.0, longitude=0.0)
        with pytest.raises(PropagationError):
            sat.topocentric_state(observer, datetime(2013, 11, 9, tzinfo=UTC))
