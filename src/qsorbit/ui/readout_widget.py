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

**No threading, still.** :meth:`~qsorbit.core.pointing.TrackingLoop.tick`
never blocks or sleeps, so a ``QTimer`` on the GUI thread can call it
directly on every timeout. The streaming feed next door needs a worker
thread and a bounded buffer; this one genuinely does not, and pretending
otherwise would add a thread to make two dissimilar things look alike.

Every label's text comes from :mod:`qsorbit.ui.readout_formatting`, which
is plain Python with no Qt import. This module is the thin remainder:
own the timer, own the widgets, put the formatted strings in them.

Boundary rule: this module imports :mod:`qsorbit.core`; nothing in
``core`` imports ``qsorbit.ui``.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from qsorbit.core.pointing import TrackingLoop
from qsorbit.ui.readout_formatting import alignment_note, readout_text

#: How often the widget polls the loop, in milliseconds.
#:
#: A 1 Hz refresh is plenty for a human reading a label — this is a
#: display cadence, unrelated to
#: :data:`~qsorbit.core.pointing.DEFAULT_TICK_INTERVAL_S`, which paces
#: :meth:`~qsorbit.core.pointing.TrackingLoop.run` instead. Here the
#: widget drives the loop itself, one tick per timer timeout, so this
#: value sets the tick cadence too.
DEFAULT_POLL_INTERVAL_MS = 1000

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

    The widget drives ``loop`` itself: each timer timeout calls
    :meth:`~qsorbit.core.pointing.TrackingLoop.tick` directly and repaints
    from the sample it returns. A tick that raises — a
    :class:`~qsorbit.core.pointing.TravelGuardError`, a
    :class:`~qsorbit.core.rotor.PositionLimitError`, a serial fault, a
    propagation error, anything — stops the timer and shows the error in
    place of the last-tick line, rather than letting the exception cross
    into Qt's event loop or leaving stale numbers on screen while a real
    fault sits underneath.

    Args:
        loop: The tracking loop to drive and display. The widget neither
            builds it nor owns the rotor's connection — whoever
            constructed the loop is responsible for both, exactly as
            :class:`~qsorbit.core.pointing.TrackingLoop`'s own docs
            describe. Stopping the widget stops polling; it does not
            stop the rotor, matching the loop's own policy.
        poll_interval_ms: How often to tick, in milliseconds. Defaults
            to :data:`DEFAULT_POLL_INTERVAL_MS`.
    """

    def __init__(
        self,
        loop: TrackingLoop,
        *,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._loop = loop
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
        try:
            sample = self._loop.tick()
        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            self._timer.stop()
            self._value_labels["Last tick"].setText(f"stopped: {exc}")
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
