"""Turning the target picker's data into the words and roles a Plan-tab table shows.

Same split as :mod:`qsorbit.ui.frequency_formatting`,
:mod:`qsorbit.ui.quieting_formatting`, and :mod:`qsorbit.ui.
readout_formatting`: this module imports nothing from PySide6, only
plain values and :mod:`qsorbit.core` types, so the decisions about what
a row *says* can be tested without a window to check. :mod:`qsorbit.ui.
picker_widget` is the thin Qt remainder -- own the table and the filter
chips, put this module's strings and roles in cells and labels.

**The tier column doubles as the dead-satellite explanation.** The
mockup gives a not-alive satellite a row that says ``"dead 2025-06"``
in the same cell a live satellite's reliability letter would occupy,
rather than a separate column most rows would leave blank -- see
:func:`picker_row_text`. Everything else about that row (pass time, max
elevation, downlink, mode) still renders normally when it's known; a
curated-dead satellite is excluded by the picker's own filtering logic
long before formatting ever sees it (:func:`~qsorbit.core.picker.
passes_filters` has no alive-status axis today -- see that module's
docstring for why -- so a dead satellite can still reach this table,
and this is what keeps its row honest about that).

**Colour never appears here, only a ``role`` string.** Same reasoning
:mod:`qsorbit.ui.theme_qss` enforces everywhere else: ``"ok"``,
``"warn"``, and ``"dim"`` are theme tokens a QSS selector resolves to an
actual colour, not colours themselves -- see :func:`alive_status_role`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, tzinfo
from typing import Final

from qsorbit.core.picker import Band, PickerEntry, primary_transmitter
from qsorbit.core.profiles import AliveStatus, CatalogManifest, ReliabilityClass

#: Shown wherever a row has nothing to say for a cell -- no pass in the
#: search window, no transmitter to read a downlink or mode from, or no
#: reliability class to letter. Matches the em-dash convention
#: :data:`qsorbit.ui.frequency_formatting.NO_READING` already
#: established: a placeholder, never a number that looks measured but
#: isn't.
NO_DATA_LABEL: Final = "-"

#: The mockup's tier letters, in :data:`~qsorbit.core.profiles.profile.
#: _RELIABILITY_ORDER`'s own order -- A is the most favorable
#: (unconditional), C the least (dependent on another operator).
_RELIABILITY_LETTERS: dict[ReliabilityClass, str] = {
    ReliabilityClass.UNCONDITIONAL: "A",
    ReliabilityClass.SCHEDULED: "B",
    ReliabilityClass.DEPENDENT: "C",
}

#: The status dot's role per the mockup's own legend: "green = curated
#: alive, amber = intermittent, grey = dead". :class:`~qsorbit.core.
#: profiles.profile.AliveStatus` only has three values and none of them
#: is spelled "intermittent" -- UNKNOWN is what that legend means:
#: a status nobody has been able to confirm one way or the other reads
#: as caution, not as confidence.
_ALIVE_ROLES: dict[AliveStatus, str] = {
    AliveStatus.ACTIVE: "ok",
    AliveStatus.UNKNOWN: "warn",
    AliveStatus.INACTIVE: "dim",
}

#: Band labels with the space the mockup's own chips use ("2 m", not
#: qsorbit.core.picker.Band's terser filter-chip value "2m").
_BAND_LABELS: dict[Band, str] = {
    Band.TWO_METERS: "2 m",
    Band.SEVENTY_CM: "70 cm",
    Band.OTHER: "other",
}


def reliability_letter(reliability: ReliabilityClass) -> str:
    """The tier column's letter for a reliability class -- see :data:`_RELIABILITY_LETTERS`."""
    return _RELIABILITY_LETTERS[reliability]


def alive_status_role(status: AliveStatus) -> str:
    """Which QSS ``role`` the picker's status dot wears for a curated alive status."""
    return _ALIVE_ROLES[status]


def format_band(band: Band) -> str:
    """A band filter's display label, e.g. ``"2 m"``."""
    return _BAND_LABELS[band]


@dataclass(frozen=True)
class PickerRowText:
    """Everything one picker table row needs to display, already decided.

    A plain value object so the widget's paint step is just "put these
    strings and this role in this row" -- all the judgment about how to
    describe an entry lives in :func:`picker_row_text`, not in widget
    code.

    Args:
        name: The satellite's display name.
        status_role: Which ``role`` the leading status dot wears -- see
            :func:`alive_status_role`.
        pass_text: ``"HH:MM → HH:MM"`` (AOS to LOS) in local time, or
            :data:`NO_DATA_LABEL` if this satellite has no pass in the
            picker's search window.
        max_elevation_text: The pass's peak elevation, e.g. ``"62°"``,
            or :data:`NO_DATA_LABEL`.
        downlink_text: The primary transmitter's downlink frequency in
            MHz, or :data:`NO_DATA_LABEL` if it has none -- see
            :func:`~qsorbit.core.picker.primary_transmitter`.
        mode_text: The primary transmitter's mode, or
            :data:`NO_DATA_LABEL`.
        tier_text: The reliability letter (:func:`reliability_letter`),
            or, for a satellite curated as not alive, ``"dead
            {as_of:%Y-%m}"`` -- see the module docstring.
    """

    name: str
    status_role: str
    pass_text: str
    max_elevation_text: str
    downlink_text: str
    mode_text: str
    tier_text: str


def picker_row_text(entry: PickerEntry, *, local_zone: tzinfo | None = None) -> PickerRowText:
    """Build one table row's display strings and role from a picker entry.

    Args:
        entry: The curated profile and its next pass, from
            :func:`~qsorbit.core.picker.build_picker_entries`.
        local_zone: The zone the pass time is shown in. Defaults to
            ``None``, which -- exactly as with :meth:`datetime.
            astimezone` itself -- means "ask the OS for the system's
            configured zone", matching :func:`~qsorbit.ui.
            readout_formatting.format_time`'s identical parameter and
            for the same reason: tests should pass a fixed zone rather
            than depend on the machine they happen to run on.
    """
    profile = entry.profile
    transmitter = primary_transmitter(profile)
    best_reliability = profile.best_reliability()

    if profile.alive.status is AliveStatus.INACTIVE:
        tier_text = f"dead {profile.alive.as_of:%Y-%m}"
    elif best_reliability is not None:
        tier_text = reliability_letter(best_reliability)
    else:
        tier_text = NO_DATA_LABEL

    if entry.next_pass is not None:
        aos_local = entry.next_pass.aos.time.astimezone(local_zone)
        los_local = entry.next_pass.los.time.astimezone(local_zone)
        pass_text = f"{aos_local:%H:%M} → {los_local:%H:%M}"
        max_elevation_text = f"{entry.next_pass.max_elevation_deg:.0f}°"
    else:
        pass_text = NO_DATA_LABEL
        max_elevation_text = NO_DATA_LABEL

    if transmitter is not None:
        downlink_text = f"{transmitter.downlink_hz / 1_000_000.0:.3f}"
        mode_text = transmitter.mode.name
    else:
        downlink_text = NO_DATA_LABEL
        mode_text = NO_DATA_LABEL

    return PickerRowText(
        name=profile.name,
        status_role=alive_status_role(profile.alive.status),
        pass_text=pass_text,
        max_elevation_text=max_elevation_text,
        downlink_text=downlink_text,
        mode_text=mode_text,
        tier_text=tier_text,
    )


def catalogue_staleness_text(manifest: CatalogManifest | None, today: date) -> str | None:
    """The Plan tab's staleness line, mirroring ``qsorbit plan``'s own wording.

    TLE-fetch staleness (the mockup's "TLEs fetched 6 h ago") is
    deliberately absent -- Chunk D PR1 shipped only the catalogue
    manifest and left TLE-staleness display out of scope entirely, a
    choice this function does not revisit.

    Args:
        manifest: The catalogue's optional manifest -- see
            :func:`~qsorbit.core.profiles.load_catalog_manifest`.
        today: Today's date.

    Returns:
        ``"catalogue: shipped {date} ({n} d)"``, or ``None`` if there is
        no manifest to report on -- callers hide the whole staleness
        line in that case, exactly as ``qsorbit plan`` skips its own
        print.
    """
    if manifest is None:
        return None
    age_days = (today - manifest.shipped).days
    return f"catalogue: shipped {manifest.shipped.isoformat()} ({age_days} d)"
