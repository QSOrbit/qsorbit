"""Bright stars, as sky positions QSOrbit can point at.

**This file is generated. Do not edit it by hand.** Regenerate with::

    uv run python tools/generate_star_catalog.py \
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


#: Every catalogued star, keyed by the name a caller types.
STARS: dict[str, Star] = {
    "polaris": Star(
        name="Polaris",
        ra_hours=2.530301000,
        dec_degrees=89.264109490,
        pm_ra_hours_per_year=0.000063743020,
        pm_dec_degrees_per_year=-0.000003260257,
        magnitude=1.97,
    ),
    "schedar": Star(
        name="Schedar",
        ra_hours=0.675122370,
        dec_degrees=56.537331070,
        pm_ra_hours_per_year=0.000001690895,
        pm_dec_degrees_per_year=-0.000008933771,
        magnitude=2.24,
    ),
    "caph": Star(
        name="Caph",
        ra_hours=0.152968080,
        dec_degrees=59.149779500,
        pm_ra_hours_per_year=0.000018896166,
        pm_dec_degrees_per_year=-0.000050103543,
        magnitude=2.28,
    ),
    "alderamin": Star(
        name="Alderamin",
        ra_hours=21.309658760,
        dec_degrees=62.585572560,
        pm_ra_hours_per_year=0.000006027896,
        pm_dec_degrees_per_year=0.000013404823,
        magnitude=2.45,
    ),
    "alpheratz": Star(
        name="Alpheratz",
        ra_hours=0.139794050,
        dec_degrees=29.090431970,
        pm_ra_hours_per_year=0.000002874549,
        pm_dec_degrees_per_year=-0.000045252039,
        magnitude=2.07,
    ),
    "markab": Star(
        name="Markab",
        ra_hours=23.079348270,
        dec_degrees=15.205264410,
        pm_ra_hours_per_year=0.000001172222,
        pm_dec_degrees_per_year=-0.000011819126,
        magnitude=2.49,
    ),
    "enif": Star(
        name="Enif",
        ra_hours=21.736432810,
        dec_degrees=9.875011260,
        pm_ra_hours_per_year=0.000000564139,
        pm_dec_degrees_per_year=0.000000383233,
        magnitude=2.38,
    ),
    "mirfak": Star(
        name="Mirfak",
        ra_hours=3.405380650,
        dec_degrees=49.861179580,
        pm_ra_hours_per_year=0.000000692423,
        pm_dec_degrees_per_year=-0.000007223108,
        magnitude=1.79,
    ),
    "deneb": Star(
        name="Deneb",
        ra_hours=20.690531870,
        dec_degrees=45.280338000,
        pm_ra_hours_per_year=0.000000041046,
        pm_dec_degrees_per_year=0.000000430443,
        magnitude=1.25,
    ),
    "altair": Star(
        name="Altair",
        ra_hours=19.846388640,
        dec_degrees=8.868322030,
        pm_ra_hours_per_year=0.000010058758,
        pm_dec_degrees_per_year=0.000107066400,
        magnitude=0.76,
    ),
    "vega": Star(
        name="Vega",
        ra_hours=18.615649030,
        dec_degrees=38.783691850,
        pm_ra_hours_per_year=0.000004774266,
        pm_dec_degrees_per_year=0.000079829089,
        magnitude=0.03,
    ),
    "capella": Star(
        name="Capella",
        ra_hours=5.278155280,
        dec_degrees=45.997991060,
        pm_ra_hours_per_year=0.000002012646,
        pm_dec_degrees_per_year=-0.000118616155,
        magnitude=0.08,
    ),
    "aldebaran": Star(
        name="Aldebaran",
        ra_hours=4.598677400,
        dec_degrees=16.509301380,
        pm_ra_hours_per_year=0.000001212266,
        pm_dec_degrees_per_year=-0.000052586226,
        magnitude=0.87,
    ),
    "arcturus": Star(
        name="Arcturus",
        ra_hours=14.261020010,
        dec_degrees=19.182410380,
        pm_ra_hours_per_year=-0.000021433837,
        pm_dec_degrees_per_year=-0.000555243460,
        magnitude=-0.05,
    ),
    "sirius": Star(
        name="Sirius",
        ra_hours=6.752476970,
        dec_degrees=-16.716115690,
        pm_ra_hours_per_year=-0.000010554673,
        pm_dec_degrees_per_year=-0.000339655471,
        magnitude=-1.44,
    ),
    "canopus": Star(
        name="Canopus",
        ra_hours=6.399197180,
        dec_degrees=-52.695660450,
        pm_ra_hours_per_year=0.000000610658,
        pm_dec_degrees_per_year=0.000006573278,
        magnitude=-0.62,
    ),
    "rigil-kentaurus": Star(
        name="Rigil Kentaurus",
        ra_hours=14.660137790,
        dec_degrees=-60.833975880,
        pm_ra_hours_per_year=-0.000139731034,
        pm_dec_degrees_per_year=0.000133809394,
        magnitude=-0.01,
    ),
    "procyon": Star(
        name="Procyon",
        ra_hours=7.655032830,
        dec_degrees=5.224993140,
        pm_ra_hours_per_year=-0.000013321695,
        pm_dec_degrees_per_year=-0.000287308094,
        magnitude=0.40,
    ),
    "betelgeuse": Star(
        name="Betelgeuse",
        ra_hours=5.919529240,
        dec_degrees=7.407062740,
        pm_ra_hours_per_year=0.000000510236,
        pm_dec_degrees_per_year=0.000003015877,
        magnitude=0.45,
    ),
    "achernar": Star(
        name="Achernar",
        ra_hours=1.628568490,
        dec_degrees=-57.236757440,
        pm_ra_hours_per_year=0.000003011210,
        pm_dec_degrees_per_year=-0.000011130419,
        magnitude=0.45,
    ),
    "agena": Star(
        name="Agena",
        ra_hours=14.063723470,
        dec_degrees=-60.373039320,
        pm_ra_hours_per_year=-0.000001271818,
        pm_dec_degrees_per_year=-0.000006959288,
        magnitude=0.61,
    ),
    "acrux": Star(
        name="Acrux",
        ra_hours=12.443304390,
        dec_degrees=-63.099091680,
        pm_ra_hours_per_year=-0.000001447298,
        pm_dec_degrees_per_year=-0.000004090595,
        magnitude=0.77,
    ),
    "antares": Star(
        name="Antares",
        ra_hours=16.490128030,
        dec_degrees=-26.432002500,
        pm_ra_hours_per_year=-0.000000210058,
        pm_dec_degrees_per_year=-0.000006445534,
        magnitude=1.06,
    ),
    "spica": Star(
        name="Spica",
        ra_hours=13.419883130,
        dec_degrees=-11.161322030,
        pm_ra_hours_per_year=-0.000000802000,
        pm_dec_degrees_per_year=-0.000008811581,
        magnitude=0.98,
    ),
    "pollux": Star(
        name="Pollux",
        ra_hours=7.755263970,
        dec_degrees=28.026198650,
        pm_ra_hours_per_year=-0.000013122677,
        pm_dec_degrees_per_year=-0.000012760546,
        magnitude=1.16,
    ),
    "fomalhaut": Star(
        name="Fomalhaut",
        ra_hours=22.960846260,
        dec_degrees=-29.622236010,
        pm_ra_hours_per_year=0.000007011444,
        pm_dec_degrees_per_year=-0.000045604721,
        magnitude=1.17,
    ),
    "mimosa": Star(
        name="Mimosa",
        ra_hours=12.795350870,
        dec_degrees=-59.688763640,
        pm_ra_hours_per_year=-0.000001769576,
        pm_dec_degrees_per_year=-0.000003560179,
        magnitude=1.25,
    ),
    "regulus": Star(
        name="Regulus",
        ra_hours=10.139530740,
        dec_degrees=11.967207090,
        pm_ra_hours_per_year=-0.000004719889,
        pm_dec_degrees_per_year=0.000001363532,
        magnitude=1.36,
    ),
    "adhara": Star(
        name="Adhara",
        ra_hours=6.977096790,
        dec_degrees=-28.972083740,
        pm_ra_hours_per_year=0.000000055656,
        pm_dec_degrees_per_year=0.000000635945,
        magnitude=1.50,
    ),
    "castor": Star(
        name="Castor",
        ra_hours=7.576628550,
        dec_degrees=31.888276310,
        pm_ra_hours_per_year=-0.000004498900,
        pm_dec_degrees_per_year=-0.000041150335,
        magnitude=1.58,
    ),
    "gacrux": Star(
        name="Gacrux",
        ra_hours=12.519433140,
        dec_degrees=-57.113211750,
        pm_ra_hours_per_year=0.000000952652,
        pm_dec_degrees_per_year=-0.000073405775,
        magnitude=1.59,
    ),
    "shaula": Star(
        name="Shaula",
        ra_hours=17.560144440,
        dec_degrees=-37.103821150,
        pm_ra_hours_per_year=-0.000000206599,
        pm_dec_degrees_per_year=-0.000008317266,
        magnitude=1.62,
    ),
    "bellatrix": Star(
        name="Bellatrix",
        ra_hours=5.418850850,
        dec_degrees=6.349702230,
        pm_ra_hours_per_year=-0.000000162995,
        pm_dec_degrees_per_year=-0.000003687923,
        magnitude=1.64,
    ),
    "elnath": Star(
        name="Elnath",
        ra_hours=5.438198160,
        dec_degrees=28.607450000,
        pm_ra_hours_per_year=0.000000490931,
        pm_dec_degrees_per_year=-0.000048381773,
        magnitude=1.65,
    ),
    "miaplacidus": Star(
        name="Miaplacidus",
        ra_hours=9.219993180,
        dec_degrees=-69.717207760,
        pm_ra_hours_per_year=-0.000008420112,
        pm_dec_degrees_per_year=0.000030244857,
        magnitude=1.67,
    ),
    "alnilam": Star(
        name="Alnilam",
        ra_hours=5.603559290,
        dec_degrees=-1.201919830,
        pm_ra_hours_per_year=0.000000027591,
        pm_dec_degrees_per_year=-0.000000294367,
        magnitude=1.69,
    ),
    "alnair": Star(
        name="Alnair",
        ra_hours=22.137218190,
        dec_degrees=-46.960975390,
        pm_ra_hours_per_year=0.000003461329,
        pm_dec_degrees_per_year=-0.000041075355,
        magnitude=1.73,
    ),
    "alnitak": Star(
        name="Alnitak",
        ra_hours=5.679313090,
        dec_degrees=-1.942572240,
        pm_ra_hours_per_year=0.000000073912,
        pm_dec_degrees_per_year=0.000000705371,
        magnitude=1.74,
    ),
    "alioth": Star(
        name="Alioth",
        ra_hours=12.900485950,
        dec_degrees=55.959821230,
        pm_ra_hours_per_year=0.000003695629,
        pm_dec_degrees_per_year=-0.000002496568,
        magnitude=1.76,
    ),
    "dubhe": Star(
        name="Dubhe",
        ra_hours=11.062130190,
        dec_degrees=61.751033240,
        pm_ra_hours_per_year=-0.000005337746,
        pm_dec_degrees_per_year=-0.000009789103,
        magnitude=1.81,
    ),
    "mintaka": Star(
        name="Mintaka",
        ra_hours=5.533444640,
        dec_degrees=-0.299092040,
        pm_ra_hours_per_year=0.000000030918,
        pm_dec_degrees_per_year=0.000000155515,
        magnitude=2.25,
    ),
    "wezen": Star(
        name="Wezen",
        ra_hours=7.139856740,
        dec_degrees=-26.393199670,
        pm_ra_hours_per_year=-0.000000056837,
        pm_dec_degrees_per_year=0.000000924758,
        magnitude=1.83,
    ),
    "kaus-australis": Star(
        name="Kaus Australis",
        ra_hours=18.402866200,
        dec_degrees=-34.384616110,
        pm_ra_hours_per_year=-0.000000888595,
        pm_dec_degrees_per_year=-0.000034449310,
        magnitude=1.79,
    ),
    "avior": Star(
        name="Avior",
        ra_hours=8.375232110,
        dec_degrees=-59.509483070,
        pm_ra_hours_per_year=-0.000000924597,
        pm_dec_degrees_per_year=0.000006309458,
        magnitude=1.86,
    ),
    "alkaid": Star(
        name="Alkaid",
        ra_hours=13.792343790,
        dec_degrees=49.313265120,
        pm_ra_hours_per_year=-0.000003442757,
        pm_dec_degrees_per_year=-0.000004321090,
        magnitude=1.85,
    ),
    "nunki": Star(
        name="Nunki",
        ra_hours=18.921090480,
        dec_degrees=-26.296722250,
        pm_ra_hours_per_year=0.000000286426,
        pm_dec_degrees_per_year=-0.000014621170,
        magnitude=2.05,
    ),
    "menkalinan": Star(
        name="Menkalinan",
        ra_hours=5.992145250,
        dec_degrees=44.947432770,
        pm_ra_hours_per_year=-0.000001475589,
        pm_dec_degrees_per_year=-0.000000244380,
        magnitude=1.90,
    ),
    "atria": Star(
        name="Atria",
        ra_hours=16.811081910,
        dec_degrees=-69.027715050,
        pm_ra_hours_per_year=0.000000923314,
        pm_dec_degrees_per_year=-0.000009142050,
        magnitude=1.91,
    ),
    "peacock": Star(
        name="Peacock",
        ra_hours=20.427460510,
        dec_degrees=-56.735090090,
        pm_ra_hours_per_year=0.000000260233,
        pm_dec_degrees_per_year=-0.000023924291,
        magnitude=1.94,
    ),
    "suhail": Star(
        name="Suhail",
        ra_hours=9.133266240,
        dec_degrees=-43.432589350,
        pm_ra_hours_per_year=-0.000000591727,
        pm_dec_degrees_per_year=0.000003965628,
        magnitude=2.23,
    ),
    "mirzam": Star(
        name="Mirzam",
        ra_hours=6.378329240,
        dec_degrees=-17.955917720,
        pm_ra_hours_per_year=-0.000000067142,
        pm_dec_degrees_per_year=-0.000000130521,
        magnitude=1.98,
    ),
    "alphard": Star(
        name="Alphard",
        ra_hours=9.459789800,
        dec_degrees=-8.658602530,
        pm_ra_hours_per_year=-0.000000271356,
        pm_dec_degrees_per_year=0.000009233693,
        magnitude=1.99,
    ),
    "hamal": Star(
        name="Hamal",
        ra_hours=2.119557530,
        dec_degrees=23.462423100,
        pm_ra_hours_per_year=0.000003849373,
        pm_dec_degrees_per_year=-0.000040481063,
        magnitude=2.01,
    ),
    "algol": Star(
        name="Algol",
        ra_hours=3.136147650,
        dec_degrees=40.955647660,
        pm_ra_hours_per_year=0.000000058589,
        pm_dec_degrees_per_year=-0.000000399895,
        magnitude=2.09,
    ),
    "denebola": Star(
        name="Denebola",
        ra_hours=11.817660430,
        dec_degrees=14.572060380,
        pm_ra_hours_per_year=-0.000009545760,
        pm_dec_degrees_per_year=-0.000031597280,
        magnitude=2.14,
    ),
    "rasalhague": Star(
        name="Rasalhague",
        ra_hours=17.582241830,
        dec_degrees=12.560034810,
        pm_ra_hours_per_year=0.000002087952,
        pm_dec_degrees_per_year=-0.000061819919,
        magnitude=2.08,
    ),
    "mizar": Star(
        name="Mizar",
        ra_hours=13.398761920,
        dec_degrees=54.925361830,
        pm_ra_hours_per_year=0.000003905750,
        pm_dec_degrees_per_year=-0.000006112288,
        magnitude=2.23,
    ),
    "sadr": Star(
        name="Sadr",
        ra_hours=20.370472750,
        dec_degrees=40.256679240,
        pm_ra_hours_per_year=0.000000058950,
        pm_dec_degrees_per_year=-0.000000258266,
        magnitude=2.23,
    ),
    "albireo": Star(
        name="Albireo",
        ra_hours=19.512022390,
        dec_degrees=27.959681120,
        pm_ra_hours_per_year=-0.000000148608,
        pm_dec_degrees_per_year=-0.000001563479,
        magnitude=3.05,
    ),
    "diphda": Star(
        name="Diphda",
        ra_hours=0.726491960,
        dec_degrees=-17.986604570,
        pm_ra_hours_per_year=0.000004531245,
        pm_dec_degrees_per_year=0.000009083732,
        magnitude=2.04,
    ),
}
