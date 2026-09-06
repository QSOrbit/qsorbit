"""Choosing which branch gets the speaker, and when that changes.

**Non-coherent combining, and the "non-coherent" is the design.** Two
dongles have two crystals and two PLLs; their audio is two independent
demodulations of two independently-tuned radios, phase-unrelated. Adding
them would not reinforce the signal, so this picks one at a time instead
— selection diversity — and the whole problem reduces to *when to
switch*.

**Why a margin, and why it is not the ~2 dB from Session 22.** That
number is a *fade depth*: how far a signal dropped. What a switching
margin has to clear is a *noise floor on the metric*: how far
``quieting_A - quieting_B`` wanders when nothing about the signal has
changed. Two different quantities that happen to have similar
magnitudes, which is the most dangerous kind of confusion available —
so the margin here was measured rather than inherited.

**Where 3.0 dB comes from** (2026-09-06, four 180 s runs on NOAA
162.550, 2,812 paired samples each, antennas swapped between dongles
between runs):

* The difference's standard deviation ran 0.39, 0.39, 0.51 and 0.64 dB
  across the four runs, the variation driven by the propagation path
  moving rather than by anything in the receiver.
* Peak excursions were 3.45σ to 4.11σ, against ~3.5σ expected for 2,812
  samples — so the distribution holds no rogue transient, and extremes
  scale with σ in the ordinary way.
* A ten-minute pass at ~15.6 blocks per second is ~9,400 samples, where
  the expected maximum is about 3.8σ. At the **worst** observed σ that
  is 2.4 dB, rounded up to 3.0.

Sizing against the worst σ rather than the average, and extrapolating to
a pass longer than anything measured, are both deliberate: **the costs
are asymmetric.** A margin that is too small chatters, and every switch
is an audible discontinuity. A margin that is too large only declines to
chase differences under 3 dB — and a real polarization fade is far
deeper than that, so it still switches when switching matters.

**A margin of zero is allowed and is not a mistake.** It means "always
take the best reading", i.e. no hysteresis at all, which is the control
that demonstrates why hysteresis is needed. The acceptance pass can run
it deliberately.

**Staleness is not the selector's problem, and that is on purpose.** A
branch whose radio has died keeps its last measurement forever, and a
stale number is indistinguishable from a live one by value alone. So a
reading of ``None`` means "no opinion" and the caller decides what makes
a reading stale — see :meth:`~qsorbit.core.receive.Branch.fresh_quieting_db`.
Keeping that judgement out of here is what lets :meth:`BranchSelector.choose`
stay a pure function of its arguments, testable without a clock, a
thread, or a radio.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

#: The switching margin, in dB of quieting. Derived in the module
#: docstring from measurement rather than assumed; see there before
#: changing it.
DEFAULT_MARGIN_DB: Final = 3.0


@dataclass(frozen=True)
class SelectorStats:
    """What the combiner did over a run.

    Args:
        margin_db: The margin in force, echoed so a run's report is
            self-contained — the switch count means nothing without it.
        evaluations: How many times a decision was made. One per
            demodulated block per branch, so roughly twice the block
            rate on a two-branch station.
        switches: How many of those decisions changed the branch. **This
            is the number to read first**: with a correctly sized margin
            it should be small and driven by real fades, and a switch
            count in the hundreds means the margin is being crossed by
            noise.
        evaluations_by_branch: How many decisions each branch was the
            chosen one for. Reported per branch rather than as a
            fraction, because "held the ear for none of the run" and
            "was never a candidate" are different facts and a
            percentage hides the second.
    """

    margin_db: float
    evaluations: int
    switches: int
    evaluations_by_branch: tuple[tuple[str, int], ...]

    def describe(self) -> str:
        """Summarise the combiner, for the end-of-run report."""
        if not self.evaluations:
            return f"combiner: on at {self.margin_db:.1f} dB margin, but never ran."
        share = ", ".join(
            f"{label} {count / self.evaluations:.1%}" for label, count in self.evaluations_by_branch
        )
        return (
            f"combiner: {self.switches:,} switch(es) in {self.evaluations:,} decision(s) "
            f"at a {self.margin_db:.1f} dB margin\n"
            f"  time on each branch: {share}"
        )


class BranchSelector:
    """Chooses which branch holds the speaker, with hysteresis.

    Args:
        margin_db: How far a challenger must exceed the current branch
            before the ear moves. Zero means no hysteresis — see the
            module docstring.

    Raises:
        ValueError: If ``margin_db`` is negative.
    """

    def __init__(self, *, margin_db: float = DEFAULT_MARGIN_DB) -> None:
        if margin_db < 0.0:
            raise ValueError(
                f"margin_db must not be negative, got {margin_db}. Zero is allowed "
                "and means no hysteresis at all."
            )
        self._margin_db = margin_db
        self._lock = threading.Lock()
        self._evaluations = 0
        self._switches = 0
        self._by_branch: dict[str, int] = {}

    @property
    def margin_db(self) -> float:
        """The margin in force."""
        return self._margin_db

    def choose(self, current: str, readings: Mapping[str, float | None]) -> str:
        """Return the branch that should hold the speaker.

        Args:
            current: The branch holding it now.
            readings: Each branch's quieting measurement, or ``None``
                for a branch with no usable reading — no squelch, no
                block yet, or a radio that has stopped. ``None`` is "no
                opinion", never "zero".

        Returns:
            The label to listen to, which is ``current`` unless a
            challenger cleared the margin.

        Three rules, and the second is the one that is easy to leave out:

        1. A challenger takes the ear only by exceeding the current
           branch **by the margin**. Equal readings never switch.
        2. **If the current branch has no usable reading, the ear moves
           immediately, margin or not.** The margin exists to stop
           switching on noise; applying it here would keep the speaker
           on a radio that has stopped, which is the one case where not
           switching is the failure.
        3. If nothing has a usable reading, keep the current branch.
           There is nothing better to do, and switching blind would
           only add a discontinuity.
        """
        live = {label: value for label, value in readings.items() if value is not None}
        chosen = current
        if live:
            best = max(live, key=lambda label: live[label])
            if current not in live:
                # Rule 2: the ear is on a branch that has stopped.
                chosen = best
            elif live[best] > live[current] + self._margin_db:
                chosen = best
        with self._lock:
            self._evaluations += 1
            self._by_branch[chosen] = self._by_branch.get(chosen, 0) + 1
            if chosen != current:
                self._switches += 1
        return chosen

    @property
    def stats(self) -> SelectorStats:
        """What this selector has done so far."""
        with self._lock:
            return SelectorStats(
                margin_db=self._margin_db,
                evaluations=self._evaluations,
                switches=self._switches,
                evaluations_by_branch=tuple(self._by_branch.items()),
            )
