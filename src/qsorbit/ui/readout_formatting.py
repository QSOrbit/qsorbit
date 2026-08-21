"""Pure text formatting for the readout window.

Kept apart from :mod:`qsorbit.ui.readout_window` on purpose: this module
imports nothing from PySide6, only :mod:`qsorbit.core` types, so it can
be imported and its output tested without Qt installed at all. Every
label the window shows is built by a function in here; the window's own
job is only to own the timer and put these strings in widgets.

The formatting mirrors the honesty already established in
:mod:`qsorbit.__main__`: sky position and rotor axis position are
labelled as the distinct things they are, and nothing here implies a
calibration that doesn't exist yet. It duplicates a couple of small
helpers from the CLI (``_format_position``-style axis formatting, the
range-rate wording) rather than importing them — ``__main__`` is an
entry point, not something ``ui`` should reach into, and the duplication
is a few lines of pure string formatting, not the seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from qsorbit.core.geometry import AzEl
from qsorbit.core.pointing import TickOutcome, TrackSample, rotor_to_sky
from qsorbit.core.rotor import Position

#: Shown at all times beneath the readout, matching UNCALIBRATED_NOTE in
#: qsorbit.__main__ — the same honesty applies here: rotor_to_sky() and
#: sky_to_rotor() are both still identity conversions.
UNCALIBRATED_NOTE = (
    "No alignment calibration is applied. Sky position is astronomical truth; "
    "rotor axis position is a raw hardware reading, not a compass bearing - "
    "see rotor_to_sky() for what it means as a sky direction."
)


@dataclass(frozen=True)
class ReadoutText:
    """Every string one TrackSample's tick contributes to the readout window.

    A plain value object so the window's paint step is just "put these
    strings in these labels" — all the judgment about how to describe a
    sample lives in the functions that build this, not in the widget code.

    Args:
        target_name: What is being tracked.
        time: When this sample was computed, formatted.
        sky_position: Where the target actually is.
        rotor_axis: The rotor's raw axis reading.
        rotor_as_sky: The same reading, converted through
            :func:`~qsorbit.core.pointing.rotor_to_sky` so it can be
            compared directly against ``sky_position`` - the raw
            reading alone isn't, once cable wrap or a fresh homing
            settle puts it outside a compass bearing's normal range.
        range_and_rate: Distance and closing/receding rate, in words.
        outcome: What the tick did about the rotor.
    """

    target_name: str
    time: str
    sky_position: str
    rotor_axis: str
    rotor_as_sky: str
    range_and_rate: str
    outcome: str


def format_time(instant: datetime) -> str:
    """Format an instant for display, always in UTC regardless of its zone."""
    return instant.astimezone(UTC).strftime("%H:%M:%S UTC")


def format_azel(sky: AzEl) -> str:
    """Format a sky direction, e.g. ``"AZ 180.0  EL 45.0"``."""
    return f"AZ {sky.azimuth:.1f}  EL {sky.elevation:.1f}"


def format_axis_position(position: Position) -> str:
    """Format a rotor axis reading, labelled distinctly from a sky direction.

    Args:
        position: The rotor's raw reading, e.g.
            :attr:`~qsorbit.core.pointing.TrackSample.rotor_position`.
    """
    return f"AZ {position.azimuth:.1f}  EL {position.elevation:.1f}  (axis reading)"


def format_rotor_as_sky(position: Position) -> str:
    """Format a rotor axis reading as the sky direction it would mean.

    Runs the reading through
    :func:`~qsorbit.core.pointing.rotor_to_sky` first, so a reading like
    ``AZ 380.3`` (past a full turn) or ``AZ -1.5`` (a fresh homing
    settle) shows up as the same compass bearing a person would get by
    doing that math themselves, directly comparable to
    :func:`format_azel`'s output. Labelled ``(uncalibrated)`` rather
    than ``(axis reading)``: this is a derived sky direction, not a
    hardware measurement, and it carries no alignment correction - see
    :func:`~qsorbit.core.pointing.rotor_to_sky`'s own docs.
    """
    sky = rotor_to_sky(position)
    return f"AZ {sky.azimuth:.1f}  EL {sky.elevation:.1f}  (uncalibrated)"


def format_range(range_km: float, range_rate_km_s: float) -> str:
    """Format range and range rate, spelling out the rate's sign in words.

    The sign alone is easy to misread, exactly as in
    :mod:`qsorbit.__main__`'s own ``_range_description``.
    """
    if range_rate_km_s > 0.0:
        rate = f"receding at {range_rate_km_s:.3f} km/s"
    elif range_rate_km_s < 0.0:
        rate = f"approaching at {abs(range_rate_km_s):.3f} km/s"
    else:
        rate = "range steady"
    return f"{range_km:.0f} km, {rate}"


def format_outcome(outcome: TickOutcome) -> str:
    """Format a tick's outcome as a short, readable phrase."""
    return outcome.value.replace("_", " ")


def readout_text(sample: TrackSample, *, target_name: str) -> ReadoutText:
    """Build every display string for one tick's sample.

    Args:
        sample: The tick's :class:`~qsorbit.core.pointing.TrackSample`.
        target_name: The name of what's being tracked
            (:attr:`~qsorbit.core.tracker.Target.name`). Not carried on
            ``TrackSample`` itself — the loop tracks one target for its
            whole life, and the sample is the per-tick seam, so the name
            is threaded through here instead.

    Returns:
        Every field the window's labels bind to.
    """
    return ReadoutText(
        target_name=target_name,
        time=format_time(sample.time),
        sky_position=format_azel(sample.sky_position),
        rotor_axis=format_axis_position(sample.rotor_position),
        rotor_as_sky=format_rotor_as_sky(sample.rotor_position),
        range_and_rate=format_range(sample.range_km, sample.range_rate_km_s),
        outcome=format_outcome(sample.outcome),
    )
