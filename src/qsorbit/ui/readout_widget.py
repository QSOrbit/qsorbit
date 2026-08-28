"""The rotor-vs-sky readout — QSOrbit's first PySide6 code, as a widget.

Originally ``ReadoutWindow`` (Chunk B). Chunk F turned it into a plain
:class:`QWidget` so it can be placed in whatever container the moment
calls for: today a :class:`~qsorbit.ui.instrument_window.InstrumentWindow`
beside a waterfall, later a tab, a dock, or a panel on a custom tab. The
convention it now follows, adopted in Session 19, is that **a UI element
receives its feed and knows nothing about what contains it** — which is
what makes the eventual shell a container job rather than a rewrite.

Its purpose is unchanged: make
:class:`~qsorbit.core.pointing.TrackingLoop` watchable while it runs
against a real rotor, showing sky target and rotor axis position as the
distinct things :class:`~qsorbit.core.pointing.TrackSample` already keeps
them as.

**This widget follows; it no longer drives.** Until Chunk A PR2 each
timer timeout called :meth:`~qsorbit.core.pointing.TrackingLoop.tick`
directly, on the GUI thread, on the reasoning that a tick "never blocks
or sleeps". That reasoning was wrong about the hardware: a tick writes a
command, sleeps out the RS-485 turnaround, and blocks on a serial read,
so it froze the interface for a sixth of a second every second and for
the port's whole timeout whenever a reply went missing. The loop is now
ticked by :class:`~qsorbit.core.receive.LoopRangeRate` on the receive
session's tracking thread, and this widget reads
:attr:`~qsorbit.core.pointing.TrackingLoop.latest_sample` and paints it
-- the same pull-on-a-timer shape the waterfall next door already uses,
now that the two turn out not to be dissimilar after all.

Every label's text comes from :mod:`qsorbit.ui.readout_formatting`, which
is plain Python with no Qt import. This module is the thin remainder:
own the timer, own the widgets, put the formatted strings in them.

Boundary rule: this module imports :mod:`qsorbit.core`; nothing in
``core`` imports ``qsorbit.ui``.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from qsorbit.core.pointing import TrackingLoop
from qsorbit.ui.readout_formatting import alignment_note, readout_text

#: How often the widget polls the loop, in milliseconds.
#:
#: A 1 Hz refresh is plenty for a human reading a label — this is a
#: display cadence, unrelated to
#: :data:`~qsorbit.core.pointing.DEFAULT_TICK_INTERVAL_S`, which paces
#: :meth:`~qsorbit.core.pointing.TrackingLoop.run` instead. It no longer
#: sets the tick cadence: the widget follows, so polling faster than the
#: ticker only repaints the same sample twice.
DEFAULT_POLL_INTERVAL_MS = 1000


def _no_fault() -> BaseException | None:
    """The default ``fault`` source: nothing is watching the ticker.

    Used when the readout is driven by something that does its own error
    reporting -- a bench script, a test. Returning ``None`` forever is
    honest here rather than optimistic: this widget genuinely has no
    information about a ticker it was never told about.
    """
    return None


#: Row labels, in display order.
_FIELDS = ("Time", "Sky target", "Rotor axis", "Rotor axis (sky dir.)", "Range", "Last tick")


class ReadoutWidget(QWidget):
    """Shows sky position and rotor axis position as distinct things, live.

    Three rotor-adjacent rows, deliberately not collapsed into one:
    "Sky target" is astronomical truth; "Rotor axis" is the raw
    mechanical reading, useful for noticing things like cable wrap;
    "Rotor axis (sky dir.)" runs that same reading through
    :func:`~qsorbit.core.pointing.rotor_to_sky` so it is directly
    comparable to "Sky target" without doing mod-360 arithmetic by eye.

    The widget **follows** ``loop``: each timer timeout reads
    :attr:`~qsorbit.core.pointing.TrackingLoop.latest_sample` and
    repaints from it, without touching the rotor. Whoever ticks the loop
    is somebody else's job, and in a ``receive --window`` run it is the
    session's tracking thread.

    Because the tick happens elsewhere, a tick that raises no longer
    arrives here as an exception. It arrives through ``fault``, and the
    widget stops polling and shows it in place of the last-tick line --
    the same outcome as before, reached differently. Without that, a
    dead rotor would leave its last plausible-looking numbers frozen on
    screen for the rest of a pass.

    Args:
        loop: The tracking loop to display. Something else ticks it;
            this widget only reads what it produced. The widget neither
            builds it nor owns the rotor's connection — whoever
            constructed the loop is responsible for both, exactly as
            :class:`~qsorbit.core.pointing.TrackingLoop`'s own docs
            describe. Stopping the widget stops polling; it does not
            stop the rotor, matching the loop's own policy.
        fault: Asked on every timeout whether the thing ticking the
            loop has died, and handed whatever killed it. Defaults to a
            source that never reports one, for a caller that ticks the
            loop itself and reports its own errors.
        poll_interval_ms: How often to repaint, in milliseconds.
            Defaults to :data:`DEFAULT_POLL_INTERVAL_MS`.
    """

    def __init__(
        self,
        loop: TrackingLoop,
        *,
        fault: Callable[[], BaseException | None] = _no_fault,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._loop = loop
        self._fault = fault
        self._target_name = loop.target.name
        # Read once, not per tick: the loop's own offset does not
        # change mid-run, and the note label is built once here too.
        self._alignment_offset = loop.alignment_offset

        layout = QGridLayout(self)
        self._value_labels = {}
        for row, caption in enumerate(_FIELDS):
            layout.addWidget(QLabel(f"{caption}:"), row, 0)
            value_label = QLabel("-")
            layout.addWidget(value_label, row, 1)
            self._value_labels[caption] = value_label

        note_label = QLabel(alignment_note(self._alignment_offset))
        note_label.setWordWrap(True)
        layout.addWidget(note_label, len(_FIELDS), 0, 1, 2)

        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()

    @property
    def target_name(self) -> str:
        """What is being tracked. A host window uses this for its title."""
        return self._target_name

    def stop(self) -> None:
        """Stop polling. Does not stop the rotor - see the class docstring."""
        self._timer.stop()

    def _on_timer(self) -> None:
        fault = self._fault()
        if fault is not None:
            self._timer.stop()
            self._value_labels["Last tick"].setText(f"stopped: {fault}")
            return

        sample = self._loop.latest_sample
        if sample is None:
            # Nothing has been ticked yet. Normal in the moment between
            # the window appearing and the first sample landing, and not
            # a reason to repaint or to blank what is already shown.
            return

        text = readout_text(
            sample, target_name=self._target_name, alignment_offset=self._alignment_offset
        )
        self._value_labels["Time"].setText(text.time)
        self._value_labels["Sky target"].setText(text.sky_position)
        self._value_labels["Rotor axis"].setText(text.rotor_axis)
        self._value_labels["Rotor axis (sky dir.)"].setText(text.rotor_as_sky)
        self._value_labels["Range"].setText(text.range_and_rate)
        self._value_labels["Last tick"].setText(text.outcome)
