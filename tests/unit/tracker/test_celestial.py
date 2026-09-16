"""Unit tests for celestial targets: the Sun and catalogued stars.

The important tests here compare against **outside authorities** rather than
against QSOrbit's own arithmetic, because the failure this module is most
exposed to is a systematic one -- a wrong frame, a missing secular term -- and
a self-consistent test cannot see those. ``sun.py`` shipped a wrong reference
frame for two chunks behind a suite that asked only frame-independent
questions; see ``TestSunGcrsKmAgainstDe421`` in ``test_sun.py``.

Two authorities are used, and they prove different things:

* **DE421**, for the Sun and the Moon. Wholly independent of anything in
  this project.
* **PyEphem**, for the stars, because DE421 contains none. This validates the
  *transformation* -- precession, nutation, Earth rotation, proper motion --
  but not the catalogue's absolute correctness, since PyEphem is also where
  the catalogue came from. :class:`TestCatalogueProvenance` closes that gap
  separately, against coordinates sourced elsewhere.
"""

import csv
import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from qsorbit.core.tracker.celestial import (
    STAR_DISTANCE_KM,
    MoonTarget,
    StarTarget,
    SunTarget,
    celestial_target,
    celestial_target_names,
)
from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.star_catalog import STARS, Star
from qsorbit.core.tracker.target import Target

#: Reference az/el positions, committed so these tests never skip. Regenerate
#: with ``tools/generate_celestial_oracle.py``.
ORACLE_CSV = Path(__file__).parents[2] / "fixtures" / "celestial" / "oracle.csv"

#: How far a computed sky position may sit from the reference, in arcseconds.
#:
#: Measured worst cases are 24.9 for the Sun (against DE421) and 40.2 for the
#: stars (against PyEphem, the residual being the annual aberration this
#: module deliberately does not model). 60 leaves headroom over both without
#: going slack -- dropping proper motion takes the stars to 103, and dropping
#: precession would take everything past 1300, so the faults worth catching
#: are well clear of the bound.
ORACLE_TOLERANCE_ARCSEC = 60.0

#: How far the shipped catalogue may sit from the bench script's own
#: coordinates, in arcseconds. Measured worst is 4.49 (Caph); 10 is twice
#: that, and any real transcription error would be arcminutes or degrees.
BENCH_TOLERANCE_ARCSEC = 10.0

#: The fourteen stars ``rotor-track.py`` used at the bench, as ``(ra_hours,
#: dec_degrees)`` at J2000, copied from that script. Their value here is that
#: they came from a *different* source than the shipped catalogue, which is
#: derived from PyEphem -- so agreement is evidence about the coordinates
#: themselves rather than about one library agreeing with itself.
BENCH_STARS = {
    "polaris": (2.530303, 89.264111),
    "schedar": (0.675110, 56.537331),
    "caph": (0.152806, 59.149781),
    "alderamin": (21.309661, 62.585571),
    "alpheratz": (0.139794, 29.090431),
    "markab": (23.079348, 15.205267),
    "enif": (21.736432, 9.875010),
    "mirfak": (3.405363, 49.861179),
    "deneb": (20.690531, 45.280339),
    "altair": (19.846389, 8.868322),
    "vega": (18.615649, 38.783689),
    "capella": (5.278155, 45.997991),
    "aldebaran": (4.598677, 16.509302),
    "arcturus": (14.261030, 19.182410),
}


def _unit(elevation_deg, azimuth_deg):
    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)
    return (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )


def _separation_arcsec(elevation_a, azimuth_a, elevation_b, azimuth_b):
    """The angle between two sky directions, in arcseconds."""
    first = _unit(elevation_a, azimuth_a)
    second = _unit(elevation_b, azimuth_b)
    cosine = sum(a * b for a, b in zip(first, second, strict=True))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 3600.0


def _oracle_rows():
    """Every committed reference position."""
    with ORACLE_CSV.open(encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _observer_for(row):
    return ObserverLocation(
        latitude=float(row["latitude_deg"]),
        longitude=float(row["longitude_deg"]),
        altitude_m=float(row["altitude_m"]),
    )


class TestProtocolConformance:
    """``Target`` is structural, so conformance is worth asserting rather
    than assuming -- a renamed method would otherwise fail much later, in
    the pointing loop."""

    def test_sun_target_satisfies_the_protocol(self):
        assert isinstance(SunTarget(), Target)

    def test_star_target_satisfies_the_protocol(self):
        assert isinstance(StarTarget(STARS["vega"]), Target)

    def test_moon_target_satisfies_the_protocol(self):
        assert isinstance(MoonTarget(), Target)

    def test_targets_report_a_display_name(self):
        assert SunTarget().name == "Sun"
        assert StarTarget(STARS["kaus-australis"]).name == "Kaus Australis"


class TestAgainstTheOracle:
    def test_the_oracle_fixture_is_present_and_populated(self):
        # A missing fixture must fail rather than silently iterate zero
        # rows, which would be a green suite that checked nothing.
        rows = list(_oracle_rows())

        assert len(rows) >= 200
        assert {row["source"] for row in rows} == {"de421", "pyephem"}

    def test_every_reference_position_is_matched(self):
        worst_arcsec = 0.0
        worst_row = None
        for row in _oracle_rows():
            state = celestial_target(row["target"]).topocentric_state(
                _observer_for(row), datetime.fromisoformat(row["utc"])
            )
            separation = _separation_arcsec(
                state.sky_position.elevation,
                state.sky_position.azimuth,
                float(row["elevation_deg"]),
                float(row["azimuth_deg"]),
            )
            if separation > worst_arcsec:
                worst_arcsec, worst_row = separation, row

        assert worst_arcsec < ORACLE_TOLERANCE_ARCSEC, (
            f"worst {worst_arcsec:.1f} arcsec on {worst_row['target']} "
            f"at {worst_row['utc']} from {worst_row['observer']}"
        )

    def test_proper_motion_is_load_bearing(self):
        # The canary. Proper motion is two numbers per star and easy to
        # think redundant -- it moves nothing more than tens of arcseconds,
        # far inside what any rotator can point to. What makes it worth
        # carrying is that the error *grows*, so this drops it and checks
        # the resulting disagreement against the displacement the
        # catalogue's own rates predict. It proves both that the comparison
        # above can go red, and that it goes red for the stated reason.
        #
        # Arcturus at the fixture's most distant epoch is the strongest
        # case: it has the largest proper motion of any bright star.
        row = max(
            (r for r in _oracle_rows() if r["target"] == "arcturus"),
            key=lambda r: r["utc"],
        )
        catalogued = STARS["arcturus"]
        frozen = Star(
            name=catalogued.name,
            ra_hours=catalogued.ra_hours,
            dec_degrees=catalogued.dec_degrees,
            pm_ra_hours_per_year=0.0,
            pm_dec_degrees_per_year=0.0,
            magnitude=catalogued.magnitude,
        )

        when = datetime.fromisoformat(row["utc"])
        observer = _observer_for(row)
        # Compare the shipped target against the frozen one, NOT against the
        # reference: a frozen star disagrees with the reference whether or
        # not the production path applies proper motion, so that comparison
        # would pass with the feature removed. This one cannot -- if the
        # rates stop being applied the two positions become identical.
        applied = StarTarget(catalogued).topocentric_state(observer, when)
        frozen_state = StarTarget(frozen).topocentric_state(observer, when)
        separation = _separation_arcsec(
            applied.sky_position.elevation,
            applied.sky_position.azimuth,
            frozen_state.sky_position.elevation,
            frozen_state.sky_position.azimuth,
        )

        years = when.year - 2000
        expected = years * math.hypot(
            catalogued.pm_ra_hours_per_year
            * 15.0
            * 3600.0
            * math.cos(math.radians(catalogued.dec_degrees)),
            catalogued.pm_dec_degrees_per_year * 3600.0,
        )

        assert separation > ORACLE_TOLERANCE_ARCSEC
        assert separation == pytest.approx(expected, rel=0.25)


class TestCatalogueProvenance:
    """The only check on whether the catalogue's coordinates are *right*.

    Everything else compares against PyEphem, which is where the catalogue
    came from -- agreement there says the transformation works, not that
    the numbers are correct. These fourteen were sourced independently for
    the bench script, so they are a second opinion rather than an echo.
    """

    def test_the_bench_stars_are_all_in_the_catalogue(self):
        missing = sorted(set(BENCH_STARS) - set(STARS))

        assert missing == []

    def test_the_bench_stars_agree_with_the_catalogue(self):
        worst_arcsec = 0.0
        worst_name = None
        for key, (ra_hours, dec_degrees) in BENCH_STARS.items():
            catalogued = STARS[key]
            # Right ascension converts to an on-sky angle by cos(dec);
            # near the pole a large coordinate difference is a small one
            # on the sky, and Polaris would otherwise dominate this.
            delta_ra_arcsec = (
                (catalogued.ra_hours - ra_hours)
                * 15.0
                * 3600.0
                * math.cos(math.radians(dec_degrees))
            )
            delta_dec_arcsec = (catalogued.dec_degrees - dec_degrees) * 3600.0
            separation = math.hypot(delta_ra_arcsec, delta_dec_arcsec)
            if separation > worst_arcsec:
                worst_arcsec, worst_name = separation, key

        assert worst_arcsec < BENCH_TOLERANCE_ARCSEC, (
            f"worst {worst_arcsec:.2f} arcsec on {worst_name}"
        )


class TestStarCatalogue:
    def test_every_key_is_command_line_safe(self):
        # Keys are typed by a user, so a space or capital would need
        # quoting and would not match what celestial_target() normalises to.
        assert all(key == key.lower() and " " not in key for key in STARS)

    def test_keys_are_derived_from_names(self):
        for key, star in STARS.items():
            assert key == star.name.lower().replace(" ", "-")

    def test_coordinates_are_in_range(self):
        for star in STARS.values():
            assert 0.0 <= star.ra_hours < 24.0
            assert -90.0 <= star.dec_degrees <= 90.0

    def test_magnitudes_are_naked_eye(self):
        # Every catalogued star is meant to be visible without optics.
        # A value outside this range means a wrong column was read.
        for star in STARS.values():
            assert -2.0 <= star.magnitude <= 4.0

    def test_the_catalogue_spans_both_hemispheres(self):
        # Deliberate: the catalogue describes the sky rather than one
        # station's view of it, so it carries stars that never rise from
        # mid-northern latitudes. If this ever fails, someone has filtered
        # the catalogue to where they happen to live.
        assert any(star.dec_degrees < -48.5 for star in STARS.values())
        assert any(star.dec_degrees > 80.0 for star in STARS.values())


class TestCelestialTargetLookup:
    def test_the_solar_system_bodies_are_available_by_name(self):
        assert celestial_target("sun").name == "Sun"
        assert celestial_target("moon").name == "Moon"

    def test_lookup_ignores_case_and_accepts_spaces(self):
        assert celestial_target("KAUS AUSTRALIS").name == "Kaus Australis"
        assert celestial_target("  vega  ").name == "Vega"

    def test_an_unknown_name_is_refused_and_says_what_exists(self):
        with pytest.raises(ValueError, match="No celestial target named"):
            celestial_target("vegga")

    def test_the_refusal_lists_the_alternatives(self):
        # A user who mistyped a star name has no other way to discover
        # the options, so the message has to carry them.
        with pytest.raises(ValueError) as caught:
            celestial_target("not-a-star")

        assert "vega" in str(caught.value)
        assert "sun" in str(caught.value)

    def test_every_advertised_name_resolves(self):
        for name in celestial_target_names():
            assert isinstance(celestial_target(name), Target)

    def test_names_include_every_target(self):
        # Pins the complete set rather than a sample. Adding the Moon in PR2
        # broke exactly this test and nothing else, which is what a
        # completeness assertion is for.
        assert set(celestial_target_names()) == {*STARS, "sun", "moon"}


class TestSkyProperties:
    """Cheap invariants that need no reference data, and would catch a
    gross error before the oracle comparison has to explain a subtle one."""

    OBSERVER = ObserverLocation(latitude=41.5, longitude=-72.4, altitude_m=100.0)

    def test_naive_datetimes_are_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            SunTarget().topocentric_state(self.OBSERVER, datetime(2026, 9, 16, 3, 0, 0))  # noqa: DTZ001

        with pytest.raises(ValueError, match="timezone-aware"):
            celestial_target("vega").topocentric_state(
                self.OBSERVER,
                datetime(2026, 9, 16, 3, 0, 0),  # noqa: DTZ001
            )

    def test_polaris_sits_near_true_north_at_the_observers_latitude(self):
        # Polaris is about 0.66 degrees from the pole in the late 2020s, so
        # it circles north at an elevation close to the latitude. True all
        # year, which is what makes it a usable alignment reference -- and
        # what makes this a real check rather than a single lucky instant.
        for month in (1, 4, 7, 10):
            state = celestial_target("polaris").topocentric_state(
                self.OBSERVER, datetime(2026, month, 15, 6, 0, 0, tzinfo=UTC)
            )
            azimuth = state.sky_position.azimuth
            from_north = min(azimuth, 360.0 - azimuth)

            assert from_north < 1.0
            assert state.sky_position.elevation == pytest.approx(41.5, abs=1.0)

    def test_positions_are_always_representable(self):
        when = datetime(2026, 9, 16, 3, 0, 0, tzinfo=UTC)
        for name in celestial_target_names():
            state = celestial_target(name).topocentric_state(self.OBSERVER, when)

            assert 0.0 <= state.sky_position.azimuth < 360.0
            assert -90.0 <= state.sky_position.elevation <= 90.0

    def test_a_star_reports_the_stand_in_distance_rather_than_a_real_one(self):
        # Stars are modelled as directions, not places. Asserting the range
        # is the placeholder keeps anyone from reading it as astronomy.
        state = celestial_target("vega").topocentric_state(
            self.OBSERVER, datetime(2026, 9, 16, 3, 0, 0, tzinfo=UTC)
        )

        assert state.range_km == pytest.approx(STAR_DISTANCE_KM, rel=1e-6)

    def test_the_moons_range_is_topocentric_rather_than_geocentric(self):
        # The Moon is close enough that where you stand matters: over a day
        # the observer is carried from one side of the Earth to the other,
        # so the reported range must swing by something close to an Earth
        # diameter. A geocentric range would barely move.
        ranges = [
            MoonTarget()
            .topocentric_state(self.OBSERVER, datetime(2026, 9, 16, hour, 0, 0, tzinfo=UTC))
            .range_km
            for hour in range(0, 24, 2)
        ]

        assert max(ranges) - min(ranges) > 8000.0

    def test_the_sun_is_about_one_au_away(self):
        state = SunTarget().topocentric_state(
            self.OBSERVER, datetime(2026, 9, 16, 3, 0, 0, tzinfo=UTC)
        )

        assert state.range_km == pytest.approx(149_597_870.7, rel=0.03)

    def test_range_rate_reflects_the_observer_being_carried_by_the_earth(self):
        # A star cannot approach or recede, so any range rate reported for
        # one is the observer's own motion -- Earth's surface speed at this
        # latitude is about 0.35 km/s, and the line-of-sight component of
        # it cannot exceed that. A rate outside this band would mean the
        # finite difference is measuring something else.
        when = datetime(2026, 9, 16, 3, 0, 0, tzinfo=UTC)
        for name in ("vega", "sirius", "polaris"):
            state = celestial_target(name).topocentric_state(self.OBSERVER, when)

            assert abs(state.range_rate_km_s) < 0.36
