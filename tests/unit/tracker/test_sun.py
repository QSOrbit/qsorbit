"""Unit tests for the closed-form Sun position and illumination geometry."""

import csv
import math
from datetime import UTC, datetime
from pathlib import Path

import pytest
from skyfield.framelib import true_equator_and_equinox_of_date

from qsorbit.core.tracker._shared import ts
from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.sun import (
    AU_KM,
    EARTH_RADIUS_KM,
    is_illuminated,
    sun_elevation_deg,
    sun_gcrs_km,
)

#: DE421 reference positions, committed rather than computed, so these tests
#: need no 16 MB ephemeris and never skip. Regenerate with
#: ``tools/generate_sun_reference.py``.
REFERENCE_CSV = Path(__file__).parents[2] / "fixtures" / "celestial" / "sun_gcrs.csv"

#: How far the closed-form Sun may sit from DE421, in arcseconds.
#:
#: The measured worst case across 2000-2050 is 38.9, which is the Almanac
#: formula's own roughly 36-arcsecond accuracy plus the nutation that the
#: of-date-to-GCRS rotation approximates away. 60 leaves half again as much
#: headroom without going slack: the frame bug this guards against is 654
#: arcseconds at the fixture's *second* epoch and grows from there, so there
#: is an order of magnitude between "passes" and "the bug is back."
SUN_REFERENCE_TOLERANCE_ARCSEC = 60.0

#: Precession of the equinoxes, in arcseconds per year. Used only to predict
#: how badly an *unrotated* vector should disagree with the reference.
PRECESSION_ARCSEC_PER_YEAR = 50.3


def _unit_and_distance(vector):
    distance = math.sqrt(sum(c * c for c in vector))
    return tuple(c / distance for c in vector), distance


def _separation_arcsec(first, second):
    """The angle between two vectors, in arcseconds."""
    first_unit, _ = _unit_and_distance(first)
    second_unit, _ = _unit_and_distance(second)
    cosine = sum(a * b for a, b in zip(first_unit, second_unit, strict=True))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 3600.0


def _reference_rows():
    """Every committed DE421 reference position, as ``(time, vector)``."""
    with REFERENCE_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield (
                datetime.fromisoformat(row["utc"]),
                (float(row["x_km"]), float(row["y_km"]), float(row["z_km"])),
            )


class TestSunGcrsKm:
    def test_naive_datetime_is_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            sun_gcrs_km(datetime(2026, 8, 28, 12, 0, 0))  # noqa: DTZ001

    def test_distance_is_close_to_one_au(self):
        # Earth's orbit is nearly circular - the Sun's apparent distance
        # never strays far from 1 AU. 3% is generous against the real
        # ~1.67% eccentricity-driven range, and catches a gross unit or
        # formula error without being a photometric-precision assertion.
        _, distance_km = _unit_and_distance(
            sun_gcrs_km(datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC))
        )

        assert distance_km == pytest.approx(AU_KM, rel=0.03)

    def test_moves_over_a_quarter_year(self):
        # The Sun's apparent ecliptic longitude advances roughly
        # 360/365.25 degrees per day - over ~91 days it should have
        # moved close to a quarter of the way around, i.e. the unit
        # vector should now point in a very different direction.
        start = sun_gcrs_km(datetime(2026, 3, 20, 12, 0, 0, tzinfo=UTC))
        later = sun_gcrs_km(datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC))
        start_unit, _ = _unit_and_distance(start)
        later_unit, _ = _unit_and_distance(later)

        cos_angle = sum(a * b for a, b in zip(start_unit, later_unit, strict=True))
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))

        assert angle_deg == pytest.approx(90.0, abs=5.0)


class TestSunGcrsKmAgainstDe421:
    """The absolute check the rest of this file cannot make.

    Every other test here asks a frame-independent question -- how far
    away, how far it moved -- or builds its expected answer out of
    :func:`sun_gcrs_km` itself. That is exactly how this function shipped
    for two chunks returning mean-equator-of-date coordinates under a GCRS
    name: nothing ever compared its *direction* against an outside
    authority, so nothing could notice the frame was wrong. These tests do,
    against JPL's DE421 -- whose answers are committed as a fixture so the
    ephemeris itself never has to be.
    """

    def test_the_reference_fixture_is_present_and_populated(self):
        # A missing fixture must fail rather than quietly collect zero
        # cases: an empty parametrisation is a green suite that proved
        # nothing, which is the failure mode this whole class exists to
        # close.
        rows = list(_reference_rows())

        assert len(rows) >= 20

    def test_agrees_with_de421_across_fifty_years(self):
        worst_arcsec = 0.0
        worst_time = None
        for when, reference_km in _reference_rows():
            separation = _separation_arcsec(sun_gcrs_km(when), reference_km)
            if separation > worst_arcsec:
                worst_arcsec, worst_time = separation, when

        assert worst_arcsec < SUN_REFERENCE_TOLERANCE_ARCSEC, (
            f"worst disagreement {worst_arcsec:.1f} arcsec at {worst_time}"
        )

    def test_the_error_does_not_grow_with_time(self):
        # The signature of the frame bug was not its size but its *slope*:
        # zero at J2000 and growing by precession every year after. A
        # tolerance alone could be passed by a formula that is merely
        # drifting slowly, so this asserts the shape as well -- the last
        # decade sampled must be no worse than the first.
        by_year: dict[int, float] = {}
        for when, reference_km in _reference_rows():
            separation = _separation_arcsec(sun_gcrs_km(when), reference_km)
            by_year[when.year] = max(by_year.get(when.year, 0.0), separation)

        years = sorted(by_year)
        assert by_year[years[-1]] < SUN_REFERENCE_TOLERANCE_ARCSEC
        assert by_year[years[-1]] < 10.0 * max(by_year[years[0]], 1.0)

    def test_the_frame_rotation_is_load_bearing(self):
        # The canary. Without it a reader has no way to tell whether the
        # comparison above is tight enough to catch anything. Here the
        # rotation is deliberately undone -- recovering exactly what the
        # Almanac formula natively produces, and what this function used to
        # return -- and the disagreement is checked against the precession
        # that theory says should appear. Both halves matter: it proves the
        # test can go red, and that it goes red for the predicted reason.
        for when, reference_km in _reference_rows():
            years_since_j2000 = when.year - 2000
            if years_since_j2000 < 10:
                continue

            rotation = true_equator_and_equinox_of_date.rotation_at(ts.from_datetime(when))
            gcrs_km = sun_gcrs_km(when)
            of_date_km = tuple(
                float(sum(rotation[row][axis] * gcrs_km[axis] for axis in range(3)))
                for row in range(3)
            )

            separation = _separation_arcsec(of_date_km, reference_km)
            expected = PRECESSION_ARCSEC_PER_YEAR * years_since_j2000

            assert separation > SUN_REFERENCE_TOLERANCE_ARCSEC
            assert separation == pytest.approx(expected, rel=0.15)


class TestIsIlluminated:
    """Built from the geometry directly, rather than a real ephemeris,
    so each case's expected answer follows from is_illuminated's own
    contract rather than from an independent source of truth agreeing
    with it -- see the module-level shadow-test cases in the project's
    own bench verification (Session 22) for that independent check.

    That self-consistency is why these cases could not see the frame bug
    fixed in Chunk F: the Sun vector appeared on both sides of every
    assertion and cancelled. The last case in this class is the exception
    and exists for exactly that reason."""

    def _sun_unit(self, time):
        unit, _ = _unit_and_distance(sun_gcrs_km(time))
        return unit

    def test_satellite_on_the_sunward_side_is_illuminated(self):
        time = datetime(2026, 8, 28, 17, 0, 0, tzinfo=UTC)
        sun_unit = self._sun_unit(time)
        radius_km = EARTH_RADIUS_KM + 700.0
        position = tuple(c * radius_km for c in sun_unit)

        assert is_illuminated(position, time) is True

    def test_satellite_directly_behind_earth_is_in_shadow(self):
        time = datetime(2026, 8, 28, 17, 0, 0, tzinfo=UTC)
        sun_unit = self._sun_unit(time)
        radius_km = EARTH_RADIUS_KM + 700.0
        position = tuple(-c * radius_km for c in sun_unit)

        assert is_illuminated(position, time) is False

    def test_satellite_off_to_the_side_is_illuminated(self):
        time = datetime(2026, 8, 28, 17, 0, 0, tzinfo=UTC)
        sun_unit = self._sun_unit(time)
        # Any vector perpendicular to sun_unit.
        perp = (-sun_unit[1], sun_unit[0], 0.0)
        perp_unit, _ = _unit_and_distance(perp)
        radius_km = EARTH_RADIUS_KM + 700.0
        position = tuple(c * radius_km for c in perp_unit)

        assert is_illuminated(position, time) is True

    def test_shadow_has_a_finite_width(self):
        # Behind Earth, but offset sideways by more than Earth's radius
        # -- out of the cylinder, so illuminated even though it is on
        # the anti-solar side.
        time = datetime(2026, 8, 28, 17, 0, 0, tzinfo=UTC)
        sun_unit = self._sun_unit(time)
        perp = (-sun_unit[1], sun_unit[0], 0.0)
        perp_unit, _ = _unit_and_distance(perp)
        behind = tuple(-c * (EARTH_RADIUS_KM + 100.0) for c in sun_unit)
        far_side = tuple(
            a + b * (EARTH_RADIUS_KM + 1000.0) for a, b in zip(behind, perp_unit, strict=True)
        )
        near_side = tuple(
            a + b * (EARTH_RADIUS_KM - 1000.0) for a, b in zip(behind, perp_unit, strict=True)
        )

        assert is_illuminated(far_side, time) is True
        assert is_illuminated(near_side, time) is False

    def test_shadow_edge_agrees_with_an_independent_sun_direction(self):
        # The one case in this class whose Sun comes from outside. Every
        # other builds its satellite position out of sun_gcrs_km, so the
        # frame cancels on both sides -- which is precisely why a wrong
        # frame survived here for two chunks. This takes the Sun direction
        # from DE421 and places satellites a deliberately small distance
        # either side of the shadow's edge.
        #
        # 40 km is chosen, not arbitrary. It sits between the two numbers
        # that matter: the frame error this guards against tilts the shadow
        # axis enough to move the edge by about 86 km at this radius, while
        # the residual error of the corrected formula moves it about 1.3 km.
        # So the margin is 30x the noise and half the size of the fault.
        margin_km = 40.0
        radius_km = EARTH_RADIUS_KM + 700.0

        for when, reference_km in _reference_rows():
            sun_unit, _ = _unit_and_distance(reference_km)
            perpendicular, _ = _unit_and_distance((-sun_unit[1], sun_unit[0], 0.0))
            behind = tuple(-c * radius_km for c in sun_unit)

            def offset_by(distance_km, behind=behind, perpendicular=perpendicular):
                return tuple(
                    a + b * distance_km for a, b in zip(behind, perpendicular, strict=True)
                )

            assert is_illuminated(offset_by(EARTH_RADIUS_KM - margin_km), when) is False
            assert is_illuminated(offset_by(EARTH_RADIUS_KM + margin_km), when) is True


class TestSunElevationDeg:
    def test_naive_datetime_is_rejected(self):
        observer = ObserverLocation(latitude=40.0, longitude=-83.0)

        with pytest.raises(ValueError, match="timezone-aware"):
            sun_elevation_deg(observer, datetime(2026, 8, 28, 17, 0, 0))  # noqa: DTZ001

    def test_local_noon_is_higher_than_local_midnight(self):
        observer = ObserverLocation(latitude=40.0, longitude=-83.0, altitude_m=250.0)
        # -83 degrees longitude is roughly UTC-5.5h solar time; use
        # UTC instants a half-day apart so one is near local noon and
        # the other near local midnight, independent of DST.
        near_noon = datetime(2026, 8, 28, 17, 30, 0, tzinfo=UTC)
        near_midnight = datetime(2026, 8, 29, 5, 30, 0, tzinfo=UTC)

        noon_elevation = sun_elevation_deg(observer, near_noon)
        midnight_elevation = sun_elevation_deg(observer, near_midnight)

        assert noon_elevation > 30.0
        assert midnight_elevation < -20.0
        assert noon_elevation > midnight_elevation

    def test_result_is_within_the_physically_possible_range(self):
        observer = ObserverLocation(latitude=40.0, longitude=-83.0, altitude_m=250.0)

        elevation = sun_elevation_deg(observer, datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC))

        assert -90.0 <= elevation <= 90.0
