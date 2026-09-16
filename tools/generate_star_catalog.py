"""Regenerate ``qsorbit.core.tracker.star_catalog`` from PyEphem's star database.

QSOrbit ships no star catalogue of its own and does not want to: typing sixty
stars' coordinates by hand is exactly the kind of transcription that produces a
wrong number nothing can catch, because a hand-typed catalogue is its own
authority. This script derives them instead, from PyEphem's built-in database
(a Yale Bright Star Catalog subset), which is offline, versioned, and entirely
independent of whoever runs this.

**PyEphem is a development dependency only.** Nothing shipped imports it --
this script runs at the desk, its output is committed, and a user installing
QSOrbit never sees it.

Usage::

    uv run python tools/generate_star_catalog.py \
        --out src/qsorbit/core/tracker/star_catalog.py

Proper motion is not read from a field, because PyEphem does not expose one.
It is *measured* from the catalogue's own behaviour: the star's astrometric
position is computed at J2000 and again a century later with the epoch pinned
to J2000, so precession cancels and what is left is the star's own motion
across the sky. Dividing by 100 gives the annual rate. That is a derivation
rather than a lookup, so it is worth saying plainly -- and worth checking, as
``tests/unit/tracker/test_celestial.py`` does against an independent
computation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path

import ephem

#: The stars the catalogue ships with.
#:
#: The first fourteen are the set ``rotor-track.py`` used at the bench for
#: alignment calibration -- kept as a block, and in their original order, so
#: the provenance stays visible. The rest are the conventionally bright and
#: well-known stars, including several far enough south that they never rise
#: from mid-northern latitudes; the catalogue describes the sky rather than
#: one station's view of it, and whether a star is reachable is a question
#: for the observer's latitude to answer, not for this list to pre-decide.
WANTED = [
    # The bench calibration set, from rotor-track.py.
    "Polaris",
    "Schedar",
    "Caph",
    "Alderamin",
    "Alpheratz",
    "Markab",
    "Enif",
    "Mirfak",
    "Deneb",
    "Altair",
    "Vega",
    "Capella",
    "Aldebaran",
    "Arcturus",
    # Bright and well known.
    "Sirius",
    "Canopus",
    "Rigil Kentaurus",
    "Procyon",
    "Betelgeuse",
    "Achernar",
    "Agena",
    "Acrux",
    "Antares",
    "Spica",
    "Pollux",
    "Fomalhaut",
    "Mimosa",
    "Regulus",
    "Adhara",
    "Castor",
    "Gacrux",
    "Shaula",
    "Bellatrix",
    "Elnath",
    "Miaplacidus",
    "Alnilam",
    "Alnair",
    "Alnitak",
    "Alioth",
    "Dubhe",
    "Mintaka",
    "Wezen",
    "Kaus Australis",
    "Avior",
    "Alkaid",
    "Nunki",
    "Menkalinan",
    "Atria",
    "Peacock",
    "Suhail",
    "Mirzam",
    "Alphard",
    "Hamal",
    "Algol",
    "Denebola",
    "Rasalhague",
    "Mizar",
    "Sadr",
    "Albireo",
    "Diphda",
]

#: How far ahead the proper-motion baseline reaches, in years. Long enough
#: that a slow star's motion is well above the catalogue's own rounding, short
#: enough that treating the motion as linear stays honest.
PM_BASELINE_YEARS = 100.0

HEADER = '''"""Bright stars, as sky positions QSOrbit can point at.

**This file is generated. Do not edit it by hand.** Regenerate with::

    uv run python tools/generate_star_catalog.py \\
        --out src/qsorbit/core/tracker/star_catalog.py

Positions are ICRS/J2000 -- the frame skyfield's own ``altaz()`` expects, so
they are handed over unrotated and skyfield applies precession, nutation and
Earth rotation itself. See :mod:`qsorbit.core.tracker.celestial` for why that
matters and what it saves.

Proper motion is carried because without it this catalogue would quietly rot.
Arcturus moves about 2.3 arcseconds a year, so a J2000 position is already
about 60 arcseconds wrong today and drifts further every year this project
runs. Sixty arcseconds is far inside what any rotator can point to, but it is
an error that *grows*, and the fix is two numbers per star. Applying it takes
the catalogue's worst-case agreement with an independent implementation from
103 to 40 arcseconds, and -- more to the point -- stops it getting worse.

The rates are in the same units as the coordinates they correct (hours per
year for right ascension, degrees per year for declination) so that applying
them is an addition and nothing else. Note that the right-ascension rate is a
*coordinate* rate, not an angular rate on the sky: near the pole a large
change in right ascension is a small change in where you point, which is why
Polaris's rate looks enormous next to Arcturus's and is not.

Magnitudes are apparent visual, and are here so a caller can say how hard a
target is to see -- not for any computation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Star:
    """One catalogued star: where it is, how it moves, how bright it looks.

    ``Star`` is a value object: immutable and comparable by value. It is
    plain data about the sky -- turning one into something pointable is
    :class:`~qsorbit.core.tracker.celestial.StarTarget`'s job.

    Args:
        name: The star's common name, capitalized for display.
        ra_hours: Right ascension at J2000, in hours, ``0 <= ra < 24``.
        dec_degrees: Declination at J2000, in degrees, ``-90 <= dec <= 90``.
        pm_ra_hours_per_year: Annual change in ``ra_hours``. A coordinate
            rate, not an on-sky rate -- see this module's docstring.
        pm_dec_degrees_per_year: Annual change in ``dec_degrees``.
        magnitude: Apparent visual magnitude. Lower is brighter; Sirius is
            about -1.4 and the naked-eye limit under a dark sky is about 6.
    """

    name: str
    ra_hours: float
    dec_degrees: float
    pm_ra_hours_per_year: float
    pm_dec_degrees_per_year: float
    magnitude: float


'''


def catalogue_key(name: str) -> str:
    """Turn a display name into the key a caller types.

    Lowercase, spaces to hyphens: ``"Kaus Australis"`` becomes
    ``"kaus-australis"``. Chosen so a key is safe on a command line and
    needs no quoting.
    """
    return name.lower().replace(" ", "-")


def measure(name: str) -> tuple[float, float, float, float, float]:
    """Read one star's J2000 position and proper motion out of PyEphem.

    Returns:
        ``(ra_hours, dec_degrees, pm_ra_hours_per_year,
        pm_dec_degrees_per_year, magnitude)``.
    """
    observer = ephem.Observer()
    observer.lat = "0"
    observer.lon = "0"
    observer.elevation = 0
    observer.pressure = 0
    # Pinning the epoch to J2000 is what makes the century-apart difference
    # below a measurement of proper motion rather than of precession.
    observer.epoch = ephem.J2000

    samples = []
    for when in (dt.datetime(2000, 1, 1, 12), dt.datetime(2100, 1, 1, 12)):
        observer.date = when
        star = ephem.star(name)
        star.compute(observer)
        samples.append(
            (
                star.a_ra * 12.0 / math.pi,
                star.a_dec * 180.0 / math.pi,
                float(star.mag),
            )
        )

    (ra0, dec0, magnitude), (ra1, dec1, _) = samples
    # Right ascension wraps at 24 h; a star sitting near the wrap would
    # otherwise show a century of "motion" almost a full turn wide.
    delta_ra = (ra1 - ra0 + 12.0) % 24.0 - 12.0
    return (
        ra0,
        dec0,
        delta_ra / PM_BASELINE_YEARS,
        (dec1 - dec0) / PM_BASELINE_YEARS,
        magnitude,
    )


def render(rows: list[tuple[str, tuple[float, float, float, float, float]]]) -> str:
    """Render the generated module source."""
    lines = [HEADER, "#: Every catalogued star, keyed by the name a caller types.\n"]
    lines.append("STARS: dict[str, Star] = {\n")
    for name, (ra, dec, pm_ra, pm_dec, mag) in rows:
        lines.append(f'    "{catalogue_key(name)}": Star(\n')
        lines.append(f'        name="{name}",\n')
        lines.append(f"        ra_hours={ra:.9f},\n")
        lines.append(f"        dec_degrees={dec:.9f},\n")
        lines.append(f"        pm_ra_hours_per_year={pm_ra:.12f},\n")
        lines.append(f"        pm_dec_degrees_per_year={pm_dec:.12f},\n")
        lines.append(f"        magnitude={mag:.2f},\n")
        lines.append("    ),\n")
    lines.append("}\n")
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the generated module to.",
    )
    args = parser.parse_args(argv)

    rows = []
    for name in WANTED:
        try:
            rows.append((name, measure(name)))
        except KeyError:
            # Worth failing loudly rather than silently shipping 59 stars:
            # PyEphem's database is versioned and a name can disappear.
            print(f"PyEphem does not know a star named {name!r}.")
            return 1

    args.out.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} stars to {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
