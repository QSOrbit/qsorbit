"""Wording for the tracking-profile toggle. No Qt.

Split out for the same reason :mod:`qsorbit.ui.readout_formatting` and
:mod:`qsorbit.ui.quieting_formatting` are: the sentences are the part
worth arguing about, and checking them should not need a display, a Qt
import, or a running window.
"""

from __future__ import annotations

from qsorbit.core.tracking_profile import TrackingProfile


def profile_status_text(
    active: TrackingProfile | None,
    pending: TrackingProfile | None,
    refusal: str | None,
    *,
    stalled: bool,
) -> str:
    """One line describing what the toggle is doing.

    Precedence is refusal, then pending, then settled. A refusal is the
    most recent thing that happened and the only one the operator has to
    act on, so it outranks a queued switch that is no longer going to
    land.

    Args:
        active: The profile in force, or ``None`` for a loop running a
            cadence no profile names.
        pending: A profile queued for the next tick, or ``None``.
        refusal: Why the last switch was refused, or ``None``.
        stalled: Whether an axis is currently stalled, which is why the
            buttons are disabled.
    """
    if refusal is not None:
        return refusal
    if pending is not None:
        return (
            f"Switching to {pending.name}. Queued for the next tick, because the "
            "gain write happens on the tracking loop's own thread."
        )
    if stalled:
        return "An axis is stalled. Switching is refused until it follows again."
    if active is None:
        return "Running a cadence no profile names, so there is nothing to switch between."
    return (
        f"{active.name}: {active.deadband_deg:g} deg deadband at {active.interval_s:g} s, "
        f"{'gains pushed' if active.gains is not None else 'controller defaults'}."
    )
