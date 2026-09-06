"""Pure formatting for the live squelch-quieting indicator.

Same split as :mod:`qsorbit.ui.readout_formatting`: this module imports
nothing from PySide6, only plain values, so it can be tested without Qt
installed at all. :mod:`qsorbit.ui.quieting_widget` is the thin Qt
remainder — own a timer, put this module's numbers in a label and a bar.

Session 22's request was concrete: "it can be hard to hear when the
quieting happens - it would be useful if there was a visual cue when one
is detected." What that becomes here is a number (the live dB reading)
and a bar (the same number, scaled to something a glance can read) -
:attr:`~qsorbit.core.receive.ReceiveSession.live_quieting_db` and
:attr:`~qsorbit.core.receive.ReceiveSession.live_squelch_open` are the
two properties this module turns into both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: The bar's dB range. Taken from qsorbit.core.dsp.squelch's own measured
#: numbers, not chosen fresh here: a real empty channel floors out around
#: -2.3 dB (the arithmetic floor) to -2.0 dB (live bench, see
#: quieting_db()'s docstring), and that same module's docstring uses
#: "20 dB quieting" as the amateur-radio idiom for a solid signal. So the
#: bar spans a little below the measured empty floor to that idiom's
#: ceiling, rather than to MAX_QUIETING_DB (60.0 dB) - a bound only a
#: literally silent block ever reaches, which would leave every real
#: signal this project has actually measured bunched in the first third
#: of the bar.
DEFAULT_BAR_FLOOR_DB: Final = -5.0
DEFAULT_BAR_CEILING_DB: Final = 20.0

#: Shown when no squelch is in use at all - distinct from a squelch that
#: is simply closed, matching the honesty ``readout_formatting`` already
#: applies to "uncalibrated" vs "aligned".
NO_SQUELCH_LABEL = "no squelch in use"

#: Shown when a squelch is attached but has not measured a block yet -
#: the gap between "session started" and "first block demodulated" is
#: real (see ReceiveSession.live_squelch_open's own docs: is_open starts
#: False before any measurement), and a live panel should say so rather
#: than print a number that looks measured but isn't.
AWAITING_FIRST_MEASUREMENT_LABEL = "awaiting first measurement"

#: Marks the branch whose audio is actually reaching the speaker, on a
#: station with more than one. Only ever set on one panel at a time --
#: :meth:`~qsorbit.core.receive.ReceiveSession.listen_to` enforces that
#: at the source, so a display showing it twice would be reporting a
#: fault rather than a layout choice.
HEARD_LABEL = "heard"


@dataclass(frozen=True)
class QuietingText:
    """Everything one live poll of the squelch contributes to the panel.

    Args:
        quieting_label: The dB reading in words, or why there isn't one
            yet - see :data:`NO_SQUELCH_LABEL` and
            :data:`AWAITING_FIRST_MEASUREMENT_LABEL`.
        gate_label: ``"open"``, ``"closed"``, or ``"-"`` when there is no
            squelch running at all to have a state.
        bar_fraction: How full to draw the bar, ``0.0`` to ``1.0``.
            ``0.0`` whenever there is nothing to show yet, never negative
            or over ``1.0`` even when a reading sits outside the bar's
            configured range.
        ear_label: :data:`HEARD_LABEL` when this branch's audio is the
            audio, and ``""`` otherwise. Empty on a single-branch
            station too, where marking the only panel "heard" would be
            noise -- there is nothing it could be distinguished from.
    """

    quieting_label: str
    gate_label: str
    bar_fraction: float
    ear_label: str = ""


def quieting_text(
    quieting_db: float | None,
    is_open: bool | None,
    *,
    listening: bool | None = None,
    floor_db: float = DEFAULT_BAR_FLOOR_DB,
    ceiling_db: float = DEFAULT_BAR_CEILING_DB,
) -> QuietingText:
    """Build the panel's text and bar fill from one live poll.

    Args:
        quieting_db: :attr:`~qsorbit.core.receive.ReceiveSession.live_quieting_db`
            - ``None`` when there is no squelch, or when there is one but
            it has not measured a block yet.
        is_open: :attr:`~qsorbit.core.receive.ReceiveSession.live_squelch_open`
            - ``None`` only when there is no squelch at all; a squelch
            with no measurement yet still reports ``False`` (its start
            state), which is what distinguishes "no squelch" from "not
            measured yet" below.
        listening: Whether this branch's audio is reaching the speaker,
            or ``None`` on a station with only one branch. **Three
            states rather than two, and the third is the point**: on a
            single-branch station there is no choice being made, so
            marking the only panel would imply one. ``False`` and
            ``None`` both render nothing, which is fine because the
            context differs -- one panel against two.
        floor_db: The bar's empty end, in dB. Defaults to
            :data:`DEFAULT_BAR_FLOOR_DB`.
        ceiling_db: The bar's full end, in dB. Defaults to
            :data:`DEFAULT_BAR_CEILING_DB`.

    Returns:
        The panel's text and bar fill for this poll.

    The ear marker is independent of everything else here: a branch can
    hold the speaker while its gate is shut and its reading is absent,
    which is exactly what a combiner does for the first block of a run
    and again whenever every branch has gone quiet. So it is applied to
    all three returns below rather than only to the measured one.
    """
    ear_label = HEARD_LABEL if listening else ""
    if is_open is None:
        # No squelch at all: ReceiveSession.live_squelch_open only
        # returns None in that one case (see its own docstring).
        return QuietingText(
            quieting_label=NO_SQUELCH_LABEL,
            gate_label="-",
            bar_fraction=0.0,
            ear_label=ear_label,
        )

    if quieting_db is None:
        # A squelch is attached but has not evaluated a block yet.
        gate_label = "open" if is_open else "closed"
        return QuietingText(
            quieting_label=AWAITING_FIRST_MEASUREMENT_LABEL,
            gate_label=gate_label,
            bar_fraction=0.0,
            ear_label=ear_label,
        )

    span = ceiling_db - floor_db
    fraction = (quieting_db - floor_db) / span if span > 0.0 else 0.0
    fraction = max(0.0, min(1.0, fraction))
    return QuietingText(
        quieting_label=f"{quieting_db:.1f} dB quieting",
        gate_label="open" if is_open else "closed",
        bar_fraction=fraction,
        ear_label=ear_label,
    )
