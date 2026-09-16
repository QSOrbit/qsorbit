# Celestial reference fixtures

Reference positions from an authority outside this project, used to check
QSOrbit's closed-form celestial arithmetic.

**Unlike `tests/fixtures/iq/`, these files *are* committed.** They are
kilobytes of text rather than megabytes of binary, so committing them costs
nothing and buys something real: the tests that use them run on every clone and
in CI, rather than skipping when a fixture is absent. A skipped suite is not a
passing suite.

## Files

| File | Rows | Source | What it pins down |
|------|------|--------|-------------------|
| `sun_gcrs.csv` | 20 | JPL DE421 via skyfield | The Sun's geocentric position vector, in GCRS kilometers, sampled 2000–2050 |
| `oracle.csv` | 270 | DE421 (Sun) and PyEphem (stars) | Topocentric azimuth and elevation, from five observers across both hemispheres |

## Why an ephemeris is the checker and not a dependency

QSOrbit computes the Sun from an analytic series and ships no ephemeris file —
see `qsorbit/core/tracker/sun.py` for that argument. The series is small, does
not expire, and is far more accurate than any rotator can use. But "far more
accurate" is a claim, and a claim about arithmetic deserves an outside
authority.

So DE421 is used once, at the desk, by `tools/generate_sun_reference.py`, and
**its answers are committed while the 16 MB file is not**. The correctness is
bought; the dependency is not.

## What each fixture can and cannot prove

Worth stating plainly, because the natural assumption — "it's checked against
JPL, so it's right" — is true of some of this and not all of it.

**`sun_gcrs.csv` is a genuinely independent check.** Nothing in QSOrbit
contributed to these numbers. If the closed-form series is wrong in any way
that moves the Sun's direction, this catches it.

**`oracle.csv` is two different claims in one file, and the `source` column
says which is which.**

*Sun rows (`de421`)* are independent in the same way `sun_gcrs.csv` is, but
one layer further out: they check the whole topocentric path — the observer's
geodetic vertical, sidereal rotation, parallax — rather than just the
geocentric vector.

*Star rows (`pyephem`)* are **not** an independent check on the coordinates.
PyEphem is where `star_catalog.py` came from, so these rows can only prove
that the *transformation* is right — precession, nutation, Earth rotation,
proper motion — and would happily agree with a catalogue full of wrong stars.

What closes that gap is a separate test, not this fixture:
`TestCatalogueProvenance` in `tests/unit/tracker/test_celestial.py` checks the
fourteen stars inherited from the bench script `rotor-track.py`, whose
coordinates were sourced elsewhere entirely, against the shipped catalogue.
They agree to **4.49 arcseconds worst** (Caph), most to well under one. That
is the second opinion; everything else here is one library agreeing with
itself.

**The observers are chosen to break things**, not to be representative: a
southern-hemisphere site (azimuth conventions and the sign of everything), one
inside the Arctic circle, the equator (where the geodetic and geocentric
verticals coincide, so a bug in that correction hides), and the prime meridian
(where a longitude sign error is invisible).

**The epochs are the point, not just the coverage.** They run from 2000 to
2050 because the fault this fixture was created in response to was a *frame*
error — the Almanac's series is referred to the mean equinox of date and was
being returned as though it were GCRS. Those frames coincide exactly at J2000
and separate by precession, about 50.3 arcseconds a year. A fixture sampling
only recent dates would have been nearly as blind to that as the tests it
replaced. The 2000 rows are what make the 2050 rows meaningful: they show the
comparison is tight rather than merely loose.

## Accuracy actually measured

Closed-form Sun against DE421, worst angular separation per sampled year:

| Year | Worst |
|------|-------|
| 2000 | 8.5″ |
| 2013 | 38.9″ |
| 2026 | 30.1″ |
| 2038 | 18.9″ |
| 2050 | 34.7″ |

Flat, not growing — which is the signature of the precession term being
correctly removed. The residual is the Almanac formula's own ~36″ accuracy plus
the nutation the of-date-to-GCRS rotation approximates away. The test tolerance
is 60″, roughly half again as much headroom; the fault it guards against is
654″ at the fixture's *second* epoch, so there is an order of magnitude between
"passes" and "the bug is back."

## Regenerating

```
uv run python tools/generate_sun_reference.py \
    --ephemeris /path/to/de421.bsp \
    --out tests/fixtures/celestial/sun_gcrs.csv

uv run python tools/generate_celestial_oracle.py \
    --ephemeris /path/to/de421.bsp \
    --out tests/fixtures/celestial/oracle.csv
```

The second also needs PyEphem, which is a development dependency (`uv sync`
installs it). Nothing shipped imports it.

`de421.bsp` is not in this repository and is not needed to *run* the tests.
Skyfield will fetch it:

```
python -c "from skyfield.api import load; load('de421.bsp')"
```

or take it from
<https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/>.

DE421 is a product of NASA's Jet Propulsion Laboratory, distributed by the
Navigation and Ancillary Information Facility. It carries no license and is
not redistributed here; only values derived from it are, with thanks.

Note that regenerating is not a routine step. These are reference values for a
sky that does not change — if a regenerated fixture differs from the committed
one by more than rounding, that is a finding about the ephemeris version or the
generator, and it deserves reading rather than committing.
