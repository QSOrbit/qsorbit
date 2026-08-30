"""The target picker's data layer: which curated satellites are worth pointing at.

Pure data over what Chunks B and D-PR1 already built -- pass
prediction, the horizon mask, and the curated profile catalogue with
its tier-1 alive status. No Qt here, same reasoning
:mod:`qsorbit.ui.feed_hub` gives for staying import-free of PySide6:
the filtering logic is worth testing without a display, and this
module is what makes that possible.

**The fifth filter axis lives on the entry, not the profile.**
"Ever-visible-from-this-latitude" (Chunk D PR2b) is a fact about an
orbit and this station's latitude -- see
:mod:`qsorbit.core.orbit_geometry` -- not about a
:class:`~qsorbit.core.profiles.profile.SatelliteProfile` itself, so it
does not fit :func:`passes_filters`'s ``profile``-only signature.
:func:`entry_passes_filters` wraps it around the untouched
:func:`passes_filters` instead of reworking that already-shipped
function.

**Filtering reads permissively, display reads canonically.** A profile
can carry more than one transmitter (RS-44's CW beacon and SSB
transponder, say), and the two questions "would this satellite pass
the band/modulation filter" and "what should this row's table cells
say" have different right answers. Filtering asks *any* transmitter
matches (Session 21's reasoning for :meth:`~qsorbit.core.profiles.
profile.SatelliteProfile.best_reliability` extended to the other two
axes: the operator cares whether there is anything to hear, not
whether every transmitter agrees). Display picks *one* canonical
transmitter -- :func:`primary_transmitter`, the one matching the
profile's own :meth:`~qsorbit.core.profiles.profile.SatelliteProfile.
best_reliability` -- because a table row has one downlink cell, not a
list.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from qsorbit.core.horizon import HorizonMask
from qsorbit.core.orbit_geometry import is_ever_visible_from_latitude
from qsorbit.core.profiles import (
    Mode,
    ProfileCatalog,
    ReliabilityClass,
    SatelliteProfile,
    Transmitter,
)
from qsorbit.core.tracker import ObserverLocation, Pass, Satellite, TrackerError, predict_passes

#: How far ahead the picker looks for a next pass, by default. A
#: separate constant from __main__.py's DEFAULT_PLAN_HOURS -- core/
#: does not import from the CLI entry point -- even though the two
#: currently agree.
DEFAULT_LOOKAHEAD_HOURS = 24.0

#: Amateur 2 m band edges, in Hz. Generous enough to cover the whole
#: allocation without needing per-country band-plan detail this filter
#: has no use for.
_TWO_METERS_HZ = (144_000_000.0, 148_000_000.0)

#: Amateur 70 cm band edges, in Hz.
_SEVENTY_CM_HZ = (420_000_000.0, 450_000_000.0)


class Band(Enum):
    """A coarse amateur-band bucket, for the picker's band filter."""

    TWO_METERS = "2m"
    SEVENTY_CM = "70cm"
    OTHER = "other"


def classify_band(downlink_hz: float) -> Band:
    """Which :class:`Band` a downlink frequency falls in.

    Args:
        downlink_hz: A transmitter's downlink frequency, in hertz.

    Returns:
        :data:`Band.TWO_METERS` or :data:`Band.SEVENTY_CM` if
        ``downlink_hz`` falls in that allocation, else
        :data:`Band.OTHER` -- every satellite in the curated starter
        set falls in one of the first two today, but nothing here
        assumes that stays true.
    """
    if _TWO_METERS_HZ[0] <= downlink_hz <= _TWO_METERS_HZ[1]:
        return Band.TWO_METERS
    if _SEVENTY_CM_HZ[0] <= downlink_hz <= _SEVENTY_CM_HZ[1]:
        return Band.SEVENTY_CM
    return Band.OTHER


class ModeGroup(Enum):
    """A coarse modulation bucket, for the picker's modulation filter.

    Matches the mockup's three filter chips exactly:
    :data:`Mode.FM` is its own group, :data:`Mode.SSB` and
    :data:`Mode.CW` group together, and everything that needs a
    decoder rather than an ear or an SSB rig -- AFSK, BPSK, SSTV --
    groups as :data:`DIGITAL`.
    """

    FM = "fm"
    SSB_CW = "ssb_cw"
    DIGITAL = "digital"


_MODE_GROUPS: dict[Mode, ModeGroup] = {
    Mode.FM: ModeGroup.FM,
    Mode.SSB: ModeGroup.SSB_CW,
    Mode.CW: ModeGroup.SSB_CW,
    Mode.AFSK1200: ModeGroup.DIGITAL,
    Mode.BPSK: ModeGroup.DIGITAL,
    Mode.SSTV: ModeGroup.DIGITAL,
}


def mode_group(mode: Mode) -> ModeGroup:
    """Which :class:`ModeGroup` a :class:`~qsorbit.core.profiles.profile.Mode` belongs to."""
    return _MODE_GROUPS[mode]


def primary_transmitter(profile: SatelliteProfile) -> Transmitter | None:
    """The one transmitter a table row's downlink/mode cells should show.

    The first transmitter matching :meth:`~qsorbit.core.profiles.
    profile.SatelliteProfile.best_reliability` -- see the module
    docstring's "filtering reads permissively, display reads
    canonically".

    Returns:
        The canonical transmitter, or ``None`` if ``profile`` has none.
    """
    best = profile.best_reliability()
    if best is None:
        return None
    for transmitter in profile.transmitters:
        if transmitter.reliability is best:
            return transmitter
    raise AssertionError(  # pragma: no cover - best_reliability only returns a present class
        f"best_reliability() returned {best!r}, not present on any transmitter"
    )


@dataclass(frozen=True)
class PickerFilters:
    """The picker's active filter state -- five axes, all "no restriction" by default.

    Every set (or bool) field means "if active, keep only entries
    matching it; if not, this axis restricts nothing" -- so the
    default, all-empty ``PickerFilters()`` passes everything, and the
    UI layer decides its own default chip states rather than this type
    prescribing them.

    Args:
        needs_transmitter: If ``True``, exclude profiles with no
            transmitters at all -- entries that exist only to carry a
            curated alive-status fact (see :mod:`qsorbit.core.profiles.
            profile`'s module docstring).
        bands: Keep a profile if *any* transmitter falls in one of
            these bands. Empty means no band restriction.
        mode_groups: Keep a profile if *any* transmitter's mode groups
            into one of these. Empty means no modulation restriction.
        reliability_classes: Keep a profile if its
            :meth:`~qsorbit.core.profiles.profile.SatelliteProfile.
            best_reliability` is one of these. Empty means no
            reliability restriction. A profile with no transmitters at
            all (``best_reliability() is None``) never matches a
            non-empty set here -- pair with ``needs_transmitter=True``
            deliberately if that's not wanted.
        require_visible_from_latitude: If ``True``, exclude entries
            whose :attr:`PickerEntry.visible_from_latitude` is
            ``False`` -- an orbit that geometrically never rises from
            this station's latitude, per :mod:`qsorbit.core.
            orbit_geometry`. Checked by :func:`entry_passes_filters`,
            not :func:`passes_filters` -- it needs the entry, not just
            the profile.
    """

    needs_transmitter: bool = False
    bands: frozenset[Band] = frozenset()
    mode_groups: frozenset[ModeGroup] = frozenset()
    reliability_classes: frozenset[ReliabilityClass] = frozenset()
    require_visible_from_latitude: bool = False


def passes_filters(profile: SatelliteProfile, filters: PickerFilters) -> bool:
    """Whether ``profile`` survives every active axis of ``filters``."""
    if filters.needs_transmitter and not profile.transmitters:
        return False
    if filters.bands and not any(
        classify_band(t.downlink_hz) in filters.bands for t in profile.transmitters
    ):
        return False
    if filters.mode_groups and not any(
        mode_group(t.mode) in filters.mode_groups for t in profile.transmitters
    ):
        return False
    if filters.reliability_classes:
        best = profile.best_reliability()
        if best is None or best not in filters.reliability_classes:
            return False
    return True


def entry_passes_filters(entry: PickerEntry, filters: PickerFilters) -> bool:
    """Whether ``entry`` survives every active axis of ``filters``, including the latitude axis.

    Combines :func:`passes_filters` on ``entry.profile`` with the one
    filter axis :func:`passes_filters` cannot check itself --
    ``require_visible_from_latitude`` needs ``entry.
    visible_from_latitude``, a fact about the matched satellite's
    orbit and this station, not about the profile alone.
    """
    if filters.require_visible_from_latitude and not entry.visible_from_latitude:
        return False
    return passes_filters(entry.profile, filters)


@dataclass(frozen=True)
class PickerEntry:
    """One picker row's worth of data: a curated profile and its next pass.

    ``PickerEntry`` is a value object: immutable and comparable by
    value.

    Args:
        profile: The curated profile.
        next_pass: The earliest predicted :class:`~qsorbit.core.
            tracker.pass_prediction.Pass` within the search window, or
            ``None`` if this satellite has none in that window -- still
            a real entry (it has a curated profile and a matching TLE),
            just with nothing to show in the time columns.
        visible_from_latitude: Whether this satellite's orbit can ever
            put it above this station's flat horizon at all, per
            :func:`~qsorbit.core.orbit_geometry.is_ever_visible_from_latitude`
            -- a permanent fact about the orbit and this station's
            latitude, not about the current search window. ``False``
            means ``next_pass`` will stay ``None`` forever from here,
            not just today.
    """

    profile: SatelliteProfile
    next_pass: Pass | None
    visible_from_latitude: bool


def build_picker_entries(
    catalog: ProfileCatalog,
    tle_dir: str | Path,
    observer: ObserverLocation,
    horizon: HorizonMask,
    now: datetime,
    *,
    hours: float = DEFAULT_LOOKAHEAD_HOURS,
) -> tuple[PickerEntry, ...]:
    """Build one :class:`PickerEntry` per TLE in ``tle_dir`` with a matching curated profile.

    Mirrors ``qsorbit plan``'s own matching (:func:`qsorbit.__main__.
    _command_plan`): a TLE that fails to parse, or has no curated
    profile by NORAD id, is silently skipped -- the catalogue is a
    curated subset by design, so an unmatched TLE is the expected case.
    Unlike the CLI, this returns one entry per matched satellite even
    when it has no pass in the window, rather than only printing the
    ones that do -- the picker's table has a row for "nothing coming
    up" to draw, the CLI's plain-text report does not.

    Args:
        catalog: The curated profile catalogue to match against.
        tle_dir: Directory of ``*.tle`` files.
        observer: This station's location.
        horizon: This station's own horizon mask.
        now: Search window start.
        hours: Search window length. Defaults to
            :data:`DEFAULT_LOOKAHEAD_HOURS`.

    Returns:
        Entries sorted with an upcoming pass first (earliest AOS
        first), then entries with none, alphabetically by name.
    """
    end = now + timedelta(hours=hours)
    entries: list[PickerEntry] = []
    for tle_path in sorted(Path(tle_dir).glob("*.tle")):
        try:
            satellite = Satellite.from_file(tle_path)
        except TrackerError:
            continue

        profile = catalog.by_norad_id(satellite.norad_id)
        if profile is None:
            continue

        passes = predict_passes(satellite, observer, now, end, horizon_mask=horizon)
        next_pass = min(passes, key=lambda one_pass: one_pass.aos.time) if passes else None
        visible_from_latitude = is_ever_visible_from_latitude(
            satellite.inclination_deg, satellite.mean_altitude_km, observer.latitude
        )
        entries.append(
            PickerEntry(
                profile=profile,
                next_pass=next_pass,
                visible_from_latitude=visible_from_latitude,
            )
        )

    entries.sort(key=_sort_key)
    return tuple(entries)


def _sort_key(entry: PickerEntry) -> tuple[int, datetime | str]:
    if entry.next_pass is not None:
        return (0, entry.next_pass.aos.time)
    return (1, entry.profile.name)
