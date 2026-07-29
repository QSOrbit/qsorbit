"""Unit tests for TLE loading and satellite position/velocity propagation.

The propagation-correctness tests here check against externally
published reference values rather than positions this code computed
itself in this session — per the project convention for tracker tests.
Two external sources are used:

- The "TEME EXAMPLE" test case (TLE catalog number 00005) from David
  Vallado, Paul Crawford, Richard Hujsak, and T.S. Kelso, "Revisiting
  Spacetrack Report #3", Revision 2, AIAA 2006-6753 — the standard
  reference used to validate SGP4 implementations. The exact reference
  vectors and tolerances below are taken from skyfield's own test suite
  (``test_appendix_c_satellite`` in ``skyfield/tests/test_earth_satellites.py``),
  which is itself sourced from that paper.
- The GOCE decay example from skyfield's official documentation
  ("Detecting Propagation Errors",
  rhodesmill.org/skyfield/earth-satellites.html), used to exercise the
  PropagationError path with a real, documented case instead of a
  synthetic one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from skyfield.api import load

from qsorbit.core.tracker import PropagationError, Satellite, TleError

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# "TEME EXAMPLE" satellite (NORAD catalog #00005), from Appendix C of
# Revisiting Spacetrack Report #3 (AIAA 2006-6753, Rev 2).
_TEME_EXAMPLE_TLE = """\
TEME EXAMPLE
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
"""

# Reference TEME position/velocity at epoch + 3.0 days, and the
# tolerances used to check against them — taken directly from
# skyfield's own verification test (test_appendix_c_satellite).
_REFERENCE_R_TEME_KM = (-9060.47373569, 4658.70952502, 813.68673153)
_REFERENCE_V_TEME_KM_S = (-2.232832783, -4.110453490, -3.157345433)
_POSITION_EPSILON_KM = 1e-4
_VELOCITY_EPSILON_KM_S = 5e-8

# GOCE, in its last-ever element set before re-entry — from skyfield's
# "Detecting Propagation Errors" documentation. Elements are only valid
# from just before noon on a Saturday to just past noon on the
# following Tuesday; asking for a position before that window is
# documented to fail with this exact message.
_GOCE_DECAY_TLE = """\
GOCE
1 34602U 09013A   13314.96046236  .14220718  20669-5  50412-4 0   930
2 34602 096.5717 344.5256 0009826 296.2811 064.0942 16.58673376272979
"""
_GOCE_BEFORE_WINDOW = datetime(2013, 11, 9, tzinfo=UTC)
_GOCE_EXPECTED_MESSAGE = "mean eccentricity is outside the range 0.0 to 1.0"


# ---------------------------------------------------------------------------
# TLE parsing
# ---------------------------------------------------------------------------


class TestFromTle:
    def test_three_line_form_captures_name(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        assert sat.name == "TEME EXAMPLE"

    def test_two_line_form_defaults_name_to_unknown(self):
        lines = _TEME_EXAMPLE_TLE.strip().splitlines()
        two_line_text = "\n".join(lines[1:])
        sat = Satellite.from_tle(two_line_text)
        assert sat.name == "UNKNOWN"

    def test_tolerates_surrounding_whitespace_and_blank_lines(self):
        padded = f"\n\n  {_TEME_EXAMPLE_TLE}  \n\n"
        sat = Satellite.from_tle(padded)
        assert sat.name == "TEME EXAMPLE"

    def test_wrong_line_count_raises_tle_error(self):
        with pytest.raises(TleError):
            Satellite.from_tle("just one line")

    def test_garbled_lines_raise_tle_error(self):
        with pytest.raises(TleError):
            Satellite.from_tle("BOGUS\nnot a tle line\nalso not a tle line")


class TestFromFile:
    def test_reads_tle_from_file(self, tmp_path):
        tle_file = tmp_path / "teme_example.tle"
        tle_file.write_text(_TEME_EXAMPLE_TLE)
        sat = Satellite.from_file(tle_file)
        assert sat.name == "TEME EXAMPLE"

    def test_accepts_string_path(self, tmp_path):
        tle_file = tmp_path / "teme_example.tle"
        tle_file.write_text(_TEME_EXAMPLE_TLE)
        sat = Satellite.from_file(str(tle_file))
        assert sat.name == "TEME EXAMPLE"


class TestEpoch:
    def test_epoch_matches_tle_epoch_date(self):
        # The TLE's epoch field, "00179.78495062", is year 2000, day of
        # year 179 (a calendar fact, not orbit physics) — day 179 of
        # 2000 (a leap year) is June 27th.
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        assert sat.epoch.year == 2000
        assert sat.epoch.month == 6
        assert sat.epoch.day == 27
        assert sat.epoch.tzinfo is not None


# ---------------------------------------------------------------------------
# Propagation correctness — checked against external truth
# ---------------------------------------------------------------------------


class TestPropagationAgainstReferenceVectors:
    """Validates the underlying TLE parsing + SGP4 propagation.

    This checks the satellite's raw TEME state (skyfield's
    ``_position_and_velocity_TEME_km``) rather than going through
    :meth:`Satellite.state_at`, because the published reference vectors
    are in the TEME frame, while ``state_at`` reports GCRS (see its
    docstring). Comparing in TEME is what lets this test check against
    an external, independently-published source instead of a value
    this code computed itself.
    """

    def test_teme_position_matches_reference(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        ts = load.timescale(builtin=True)
        epoch = sat.skyfield_satellite.epoch
        t = ts.tt_jd(epoch.whole + 3.0, epoch.tt_fraction)

        r_teme, _v_teme, error = sat.skyfield_satellite._position_and_velocity_TEME_km(t)

        assert error is None
        assert tuple(r_teme) == pytest.approx(_REFERENCE_R_TEME_KM, abs=_POSITION_EPSILON_KM)

    def test_teme_velocity_matches_reference(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        ts = load.timescale(builtin=True)
        epoch = sat.skyfield_satellite.epoch
        t = ts.tt_jd(epoch.whole + 3.0, epoch.tt_fraction)

        _r_teme, v_teme, error = sat.skyfield_satellite._position_and_velocity_TEME_km(t)

        assert error is None
        assert tuple(v_teme) == pytest.approx(_REFERENCE_V_TEME_KM_S, abs=_VELOCITY_EPSILON_KM_S)


# ---------------------------------------------------------------------------
# state_at() — public API behavior
# ---------------------------------------------------------------------------


class TestStateAt:
    def test_matches_skyfield_directly_at_the_same_instant(self):
        """Wiring check: state_at() should be a thin, correct pass-through.

        The physics is already validated above against external truth;
        this checks that the wrapper doesn't introduce its own bugs
        (swapped axes, wrong units, position/velocity mixed up) when
        translating skyfield's output into an EciState. Tolerances are
        looser than the reference-vector tests above because this also
        absorbs one round-trip through a Python datetime, which is only
        microsecond-precise.
        """
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        ts = load.timescale(builtin=True)
        epoch = sat.skyfield_satellite.epoch
        t = ts.tt_jd(epoch.whole + 3.0, epoch.tt_fraction)

        direct = sat.skyfield_satellite.at(t)
        state = sat.state_at(t.utc_datetime())

        assert state.position_km == pytest.approx(tuple(direct.xyz.km), abs=1e-4)
        assert state.velocity_km_s == pytest.approx(tuple(direct.velocity.km_per_s), abs=1e-6)

    def test_time_is_normalized_to_utc(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        five_hours = timezone(timedelta(hours=5))
        local_time = sat.epoch.astimezone(five_hours) + timedelta(days=3)

        state = sat.state_at(local_time)

        assert state.time == local_time.astimezone(UTC)
        assert state.time.tzinfo == UTC

    def test_naive_datetime_rejected(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        naive = datetime(2000, 6, 30, 18, 50, 20)
        with pytest.raises(ValueError, match="timezone-aware"):
            sat.state_at(naive)


# ---------------------------------------------------------------------------
# Propagation errors
# ---------------------------------------------------------------------------


class TestPropagationError:
    def test_raises_with_sgp4s_message_when_orbit_is_invalid(self):
        sat = Satellite.from_tle(_GOCE_DECAY_TLE)
        with pytest.raises(PropagationError, match=_GOCE_EXPECTED_MESSAGE):
            sat.state_at(_GOCE_BEFORE_WINDOW)
