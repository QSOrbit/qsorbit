"""The rotor-vs-sky readout window — QSOrbit's first PySide6 code.

A deliberately plain lab instrument (see the Phase 2 brief), not the
real UI shell: one window, a handful of labels, and a timer. Its purpose
is to make :class:`~qsorbit.core.pointing.TrackingLoop` watchable while
it runs against a real rotor, showing sky target and rotor axis
position as the distinct things :class:`~qsorbit.core.pointing.TrackSample`
already keeps them as.

**No threading.** :meth:`~qsorbit.core.pointing.TrackingLoop.tick` never
blocks or sleeps — see its own docstring — so a ``QTimer`` on the GUI
thread can call it directly on every timeout. That is only possible
because the loop's design (Chunk A) split ``tick()`` out from ``run()``
specifically so a caller with its own clock, like Qt's event loop, could
drive it without a background thread. The harder question — how a
*streaming* core feed meets Qt without blocking the event loop — is
deliberately deferred to the waterfall panel (Chunk F), which will have
real background work (spectrum frames) to hand off. This window has
none: reading the rotor and computing one sample is the entire cost of
a tick, already proven fast enough at 1 Hz on the bench in Chunk A.

Every label's text comes from :mod:`qsorbit.ui.readout_formatting`,
which is plain Python with no Qt import — kept that way so the display
logic can be read, and tested, without pulling PySide6 in at all. This
module is the thin remainder: own the timer, own the widgets, and put
the formatted strings in them.

Boundary rule: this module imports :mod:`qsorbit.core`; nothing in
``core`` imports ``qsorbit.ui``. This is the first module in the repo
where that rule is actually load-bearing rather than aspirational.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QGridLayout, QLabel, QMainWindow, QWidget

from qsorbit.core.pointing import TrackingLoop
from qsorbit.ui.readout_formatting import UNCALIBRATED_NOTE, readout_text

#: How often the window polls the loop, in milliseconds.
#:
#: A 1 Hz refresh is plenty for a human reading a label — this is a
#: display cadence, unrelated to
#: :data:`~qsorbit.core.pointing.DEFAULT_TICK_INTERVAL_S`, which paces
#: :meth:`~qsorbit.core.pointing.TrackingLoop.run` instead. Here the
#: window drives the loop itself, one tick per timer timeout, so this
#: value sets the tick cadence too.
DEFAULT_POLL_INTERVAL_MS = 1000

#: Row labels, in display order.
_FIELDS = ("Time", "Sky target", "Rotor axis", "Rotor axis (sky dir.)", "Range", "Last tick")


class ReadoutWindow(QMainWindow):
    """Shows sky position and rotor axis position as distinct things, live.

    Three rotor-adjacent rows, deliberately not collapsed into one:
    "Sky target" is astronomical truth; "Rotor axis" is the raw
    mechanical reading, useful for noticing things like cable wrap;
    "Rotor axis (sky dir.)" runs that same reading through
    :func:`~qsorbit.core.pointing.rotor_to_sky` so it is directly
    comparable to "Sky target" without doing mod-360 arithmetic by eye.

    The window drives ``loop`` itself: each timer timeout calls
    :meth:`~qsorbit.core.pointing.TrackingLoop.tick` directly and repaints
    from the sample it returns. A tick that raises — a
    :class:`~qsorbit.core.pointing.TravelGuardError`, a
    :class:`~qsorbit.core.rotor.PositionLimitError`, a serial fault, a
    propagation error, anything — stops the timer and shows the error in
    place of the last-tick line, rather than letting the exception cross
    into Qt's event loop or leaving stale numbers on screen while a real
    fault sits underneath.

    Args:
        loop: The tracking loop to drive and display. The window neither
            builds it nor owns the rotor's connection — whoever
            constructed the loop is responsible for both, exactly as
            :class:`~qsorbit.core.pointing.TrackingLoop`'s own docs
            describe. Closing the window stops polling; it does not
            stop the rotor, matching the loop's own policy.
        poll_interval_ms: How often to tick, in milliseconds. Defaults
            to :data:`DEFAULT_POLL_INTERVAL_MS`.
    """

    def __init__(
        self,
        loop: TrackingLoop,
        *,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
    ) -> None:
        super().__init__()
        self._loop = loop
        self._target_name = loop.target.name

        self.setWindowTitle(f"QSOrbit - tracking {self._target_name}")

        central = QWidget(self)
        layout = QGridLayout(central)
        self._value_labels = {}
        for row, caption in enumerate(_FIELDS):
            layout.addWidget(QLabel(f"{caption}:"), row, 0)
            value_label = QLabel("-")
            layout.addWidget(value_label, row, 1)
            self._value_labels[caption] = value_label

        note_label = QLabel(UNCALIBRATED_NOTE)
        note_label.setWordWrap(True)
        layout.addWidget(note_label, len(_FIELDS), 0, 1, 2)
        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop polling on close. Does not stop the rotor - see the class docstring."""
        self._timer.stop()
        super().closeEvent(event)

    def _on_timer(self) -> None:
        try:
            sample = self._loop.tick()
        except Exception as exc:
            self._timer.stop()
            self._value_labels["Last tick"].setText(f"stopped: {exc}")
            return

        text = readout_text(sample, target_name=self._target_name)
        self._value_labels["Time"].setText(text.time)
        self._value_labels["Sky target"].setText(text.sky_position)
        self._value_labels["Rotor axis"].setText(text.rotor_axis)
        self._value_labels["Rotor axis (sky dir.)"].setText(text.rotor_as_sky)
        self._value_labels["Range"].setText(text.range_and_rate)
        self._value_labels["Last tick"].setText(text.outcome)
