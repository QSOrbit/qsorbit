"""Regenerate the celestial oracle fixture that ``test_celestial.py`` checks against.

QSOrbit computes celestial positions from analytic series and ships no
ephemeris (see :mod:`qsorbit.core.tracker.celestial` for why). That decision is
only defensible if the series are *checked* against the professional answer, so
this script produces the reference values offline and commits the numbers. The
16 MB ephemeris stays out of the repository; its answers do not.

**Two sources, because no single one covers everything.**

* **The Sun and the Moon** come from JPL's DE421 ephemeris through skyfield.
  Genuinely independent of anything QSOrbit computes.
* **Stars** come from PyEphem. DE421 contains no stars, so there is nothing
  else to ask. Note carefully what this can and cannot prove: PyEphem is also
  where ``star_catalog.py``'s coordinates came from, so this validates the
  *transformation* -- precession, nutation, Earth rotation, proper motion --
  and **not** the catalogue's absolute correctness. What covers that gap is a
  separate check: the fourteen stars inherited from ``rotor-track.py`` were
  sourced elsewhere and agree with PyEphem to about one part in 100,000. See
  ``tests/fixtures/celestial/README.md``.

**Reference positions are apparent and airless.** ``apparent()`` is used
rather than ``astrometric()``, so the reference includes annual aberration --
a term QSOrbit deliberately does not model. Comparing against the flattering
number would hide a real 20-arcsecond gap; comparing against the true one
makes it visible and forces the tolerance to account for it honestly.
Refraction is excluded on both sides (``pressure = 0``), because it is a
property of the atmosphere on the night rather than of the computation, and
because a rotor is pointed geometrically.

Usage::

    uv run python tools/generate_celestial_oracle.py \\
        --ephemeris /path/to/de421.bsp \\
        --out tests/fixtures/celestial/oracle.csv

``de421.bsp`` is not in this repository and is not needed to *run* the tests --
only to regenerate this fixture. Skyfield will fetch it for you
(``python -c "from skyfield.api import load; load('de421.bsp')"``), or it can
be downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from pathlib import Path

import ephem
from skyfield.api import load, wgs84

#: Observers the fixture samples. Chosen to break things that a single
#: mid-northern station would not: a southern-hemisphere site (azimuth
#: conventions and the sign of everything), a site inside the Arctic circle
#: (where a target can stay up all day), the equator (where the geodetic and
#: geocentric verticals coincide, so a bug in that correction hides), and the
#: prime meridian (where a longitude sign error is invisible). The first entry
#: is this project's own station, near the latitude where the geodetic
#: correction is largest.
OBSERVERS = [
    ("home", 41.5, -72.4, 100.0),
    ("equator", 0.0, 0.0, 0.0),
    ("arctic", 68.4, 20.5, 350.0),
    ("southern", -33.87, 151.21, 50.0),
    ("greenwich", 51.48, 0.0, 47.0),
]

#: Instants the fixture samples, in UTC. Spread across the seasons (so the
#: Sun is sampled near both solstices and an equinox), across the day, and
#: across twenty years -- the last one matters because proper motion and
#: precession both grow with time, and a fixture that only ever asks about
#: today cannot tell a missing secular term from a correct one.
EPOCHS = [
    dt.datetime(2026, 1, 15, 7, 30, 0, tzinfo=dt.UTC),
    dt.datetime(2026, 3, 20, 16, 5, 0, tzinfo=dt.UTC),
    dt.datetime(2026, 6, 21, 12, 0, 0, tzinfo=dt.UTC),
    dt.datetime(2026, 9, 16, 3, 0, 0, tzinfo=dt.UTC),
    dt.datetime(2036, 11, 2, 21, 45, 0, tzinfo=dt.UTC),
    dt.datetime(2045, 7, 4, 6, 30, 0, tzinfo=dt.UTC),
]

#: Which observers stars are sampled from. Every observer would triple the
#: fixture for little added coverage -- the transformation under test does
#: not vary by body -- so stars take one northern and one southern site,
#: which between them exercise both hemispheres' azimuth behaviour.
STAR_OBSERVERS = ["home", "southern"]

#: Which epochs stars are sampled at: one contemporary, one twenty years out.
#: The second is what makes a missing proper-motion term fail rather than pass.
STAR_EPOCHS = [EPOCHS[3], EPOCHS[5]]

FIELDS = [
    "target",
    "observer",
    "latitude_deg",
    "longitude_deg",
    "altitude_m",
    "utc",
    "azimuth_deg",
    "elevation_deg",
    "source",
]


def solar_system_rows(ephemeris) -> list[dict[str, object]]:
    """Reference Sun and Moon positions, from DE421 via skyfield."""
    timescale = load.timescale(builtin=True)
    earth = ephemeris["earth"]
    bodies = {"sun": ephemeris["sun"], "moon": ephemeris["moon"]}

    rows = []
    for name, body in bodies.items():
        for label, latitude, longitude, altitude_m in OBSERVERS:
            here = wgs84.latlon(latitude, longitude, elevation_m=altitude_m)
            for when in EPOCHS:
                t = timescale.from_datetime(when)
                altitude, azimuth, _ = (earth + here).at(t).observe(body).apparent().altaz()
                rows.append(
                    {
                        "target": name,
                        "observer": label,
                        "latitude_deg": latitude,
                        "longitude_deg": longitude,
                        "altitude_m": altitude_m,
                        "utc": when.isoformat(),
                        "azimuth_deg": f"{azimuth.degrees % 360.0:.6f}",
                        "elevation_deg": f"{altitude.degrees:.6f}",
                        "source": "de421",
                    }
                )
    return rows


def star_rows(names: list[str]) -> list[dict[str, object]]:
    """Reference star positions, from PyEphem."""
    by_label = {label: (lat, lon, alt) for label, lat, lon, alt in OBSERVERS}

    rows = []
    for label in STAR_OBSERVERS:
        latitude, longitude, altitude_m = by_label[label]
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        observer.elevation = altitude_m
        # Airless, for the reason in this module's docstring.
        observer.pressure = 0
        for when in STAR_EPOCHS:
            observer.date = when.strftime("%Y/%m/%d %H:%M:%S")
            for name in names:
                body = ephem.star(name)
                body.compute(observer)
                rows.append(
                    {
                        "target": name.lower().replace(" ", "-"),
                        "observer": label,
                        "latitude_deg": latitude,
                        "longitude_deg": longitude,
                        "altitude_m": altitude_m,
                        "utc": when.isoformat(),
                        "azimuth_deg": f"{math.degrees(float(body.az)) % 360.0:.6f}",
                        "elevation_deg": f"{math.degrees(float(body.alt)):.6f}",
                        "source": "pyephem",
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ephemeris",
        type=Path,
        required=True,
        help="Path to de421.bsp (or any DE kernel skyfield can read).",
    )
    parser.add_argument("--out", type=Path, required=True, help="CSV fixture to write.")
    parser.add_argument(
        "--stars",
        type=Path,
        default=None,
        help=(
            "Optional file of star names, one per line. Defaults to every star in "
            "qsorbit.core.tracker.star_catalog."
        ),
    )
    args = parser.parse_args(argv)

    if args.stars is not None:
        names = [line.strip() for line in args.stars.read_text().splitlines() if line.strip()]
    else:
        from qsorbit.core.tracker.star_catalog import STARS

        names = [star.name for star in STARS.values()]

    ephemeris = load(str(args.ephemeris))
    rows = solar_system_rows(ephemeris) + star_rows(names)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} reference positions to {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
