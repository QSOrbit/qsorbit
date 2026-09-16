"""Regenerate the DE421 reference the closed-form Sun position is checked against.

:func:`qsorbit.core.tracker.sun.sun_gcrs_km` computes the Sun from an analytic
series rather than an ephemeris, for the reasons that module's docstring gives.
That is only defensible if the series is checked against the professional
answer, so this script produces the reference offline and the *numbers* are
committed. The 16 MB ephemeris stays out of the repository.

The reference is **astrometric geocentric** -- light-time corrected, but with
no aberration and no refraction. That matches what the closed-form series
produces, so the comparison measures the series rather than a term neither side
models.

**Why the epochs span fifty years.** The bug this fixture exists to prevent was
a frame error: the series is referred to the mean equinox *of date* and was
being returned as though it were GCRS. Those two frames coincide at J2000 and
drift apart by precession, roughly 50.3 arcseconds a year. A fixture sampling
only recent dates would be nearly as blind to that as the tests it replaces --
so the rows reach from 2000, where the error is zero, to 2050, where it is
0.7 degrees. The early rows are what make the late rows meaningful: they prove
the comparison is not simply loose.

Usage::

    uv run python tools/generate_sun_reference.py \\
        --ephemeris /path/to/de421.bsp \\
        --out tests/fixtures/celestial/sun_gcrs.csv

``de421.bsp`` is not in this repository and is not needed to *run* the tests --
only to regenerate this fixture. Skyfield will fetch it
(``python -c "from skyfield.api import load; load('de421.bsp')"``), or take it
from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

from skyfield.api import load

#: Years sampled, reaching far enough either side of J2000 that a missing
#: precession rotation cannot hide in the tolerance.
YEARS = [2000, 2013, 2026, 2038, 2050]

#: Dates within each year, spread across the seasons so the Sun is sampled at
#: a range of ecliptic longitudes rather than one.
DATES = [(1, 15, 6, 0), (4, 5, 11, 30), (7, 21, 18, 15), (10, 30, 23, 45)]

FIELDS = ["utc", "x_km", "y_km", "z_km"]


def reference_rows(ephemeris) -> list[dict[str, str]]:
    """The Sun's astrometric geocentric position, in GCRS kilometers."""
    timescale = load.timescale(builtin=True)
    earth, sun = ephemeris["earth"], ephemeris["sun"]

    rows = []
    for year in YEARS:
        for month, day, hour, minute in DATES:
            when = dt.datetime(year, month, day, hour, minute, tzinfo=dt.UTC)
            position = earth.at(timescale.from_datetime(when)).observe(sun)
            x, y, z = position.position.km
            rows.append(
                {
                    "utc": when.isoformat(),
                    "x_km": f"{x:.3f}",
                    "y_km": f"{y:.3f}",
                    "z_km": f"{z:.3f}",
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
    args = parser.parse_args(argv)

    rows = reference_rows(load(str(args.ephemeris)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} reference Sun positions to {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
