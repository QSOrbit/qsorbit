"""The live quieting panel — a number and a bar, polling a receive session.

Same division of labour as :mod:`qsorbit.ui.readout_widget` and
:mod:`qsorbit.ui.waterfall_widget`: everything worth arguing about lives
in :mod:`qsorbit.ui.quieting_formatting`, which imports no Qt. This
module owns a timer, a label, and a progress bar standing in for "the
bar" Session 22 asked for.

**This widget polls; nothing pushes to it**, for the same reason
:class:`~qsorbit.ui.waterfall_widget.WaterfallWidget` polls its source
rather than being handed frames: the demodulating thread runs on its own
schedule, and a signal-per-block would be posting events onto the GUI
queue from a thread that has no idea whether the GUI is keeping up.
Polling a plain property is also the cheapest read available -
:attr:`~qsorbit.core.receive.ReceiveSession.live_quieting_db` and
:attr:`~qsorbit.core.receive.ReceiveSession.live_squelch_open` are
documented as safe for exactly this: a single attribute read, tolerant
of being one block stale.

**The widget knows nothing about what contains it**, the Session 19
convention every panel in this package follows: it takes its source as a
constructor argument and never reaches for a window or a session's other
state.
"""

from __future__ import annotations

from typing import Final, Protocol

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from qsorbit.ui.quieting_formatting import quieting_text

#: How often the widget polls its source, in milliseconds.
#:
#: Faster than :data:`~qsorbit.ui.readout_widget.DEFAULT_POLL_INTERVAL_MS`
#: on purpose - Session 22 asked for a visual cue "when [quieting] is
#: detected", and a 1 Hz panel would visibly lag a squelch that can open
#: and close inside a single second. 200 ms is a plain property read, not
#: a demodulation, so polling five times a second costs nothing worth
#: naming.
DEFAULT_POLL_INTERVAL_MS: Final = 200

#: The bar's resolution. Arbitrary beyond "smooth enough to look like a
#: bar rather than a stepped meter" - the underlying value is a float
#: fraction from :func:`~qsorbit.ui.quieting_formatting.quieting_text`.
_BAR_STEPS: Final = 1000

#: Height of the bar, in pixels. Slim on purpose: it is a level, and a
#: full-height progress bar reads as a task making progress towards
#: finishing, which is not what a squelch does.
_BAR_HEIGHT: Final = 10


class QuietingSource(Protocol):
    """Anything with a live quieting reading, declared structurally.

    Satisfied by :class:`~qsorbit.core.receive.ReceiveSession` without
    subclassing anything, the same reasoning
    :class:`~qsorbit.ui.waterfall_widget.FrameSource` already gives -
    and it lets a test double stand in without building a real session's
    stream, audio device, and rotor.
    """

    @property
    def live_quieting_db(self) -> float | None:
        """See :attr:`ReceiveSession.live_quieting_db`."""
        ...

    @property
    def live_squelch_open(self) -> bool | None:
        """See :attr:`ReceiveSession.live_squelch_open`."""
        ...


def _listening_state(source: object) -> bool | None:
    """Whether ``source`` is a branch holding the speaker, or ``None``.

    Read with :func:`getattr` rather than by widening
    :class:`QuietingSource`, and the reason is the widget rule. A
    session-wide feed has no ear to hold -- it *is* the ear -- so
    requiring the attribute would force every source to answer a
    question only some of them have. ``None`` is "no choice is being
    made here", which is the honest answer for a single-branch station
    and what stops the only panel on screen being labelled as the
    chosen one.
    """
    listening = getattr(source, "is_listening", None)
    return listening if isinstance(listening, bool) else None


class QuietingWidget(QWidget):
    """A live "how quiet is the channel" readout: a number and a bar.

    Args:
        source: Where the live reading comes from. The widget neither
            builds it nor owns its lifetime - whoever constructed it is
            responsible for both, exactly as
            :class:`~qsorbit.ui.readout_widget.ReadoutWidget` treats its
            loop.
        poll_interval_ms: How often to poll, in milliseconds. Defaults
            to :data:`DEFAULT_POLL_INTERVAL_MS`.

    Reads ``source.live_quieting_db`` and ``source.live_squelch_open``
    on every timeout and hands them straight to
    :func:`~qsorbit.ui.quieting_formatting.quieting_text`, which is the
    only place that decides what the number, the bar, and the "no
    squelch"/"awaiting first measurement" wording actually say.
    """

    def __init__(
        self,
        source: QuietingSource,
        *,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source

        # Bar above, labels below, rather than one long horizontal row.
        # **Changed in Chunk C PR2, and by measurement rather than
        # taste.** The row shape was fine as the only thing in a
        # full-width instrument window; in the shell's 300 px side
        # column it rendered as "Quieting: -6.2 dB quietin" with the
        # text cut off mid-word and the bar squeezed to a stub. That is
        # the standing widget rule failing in the direction nobody
        # checks -- a panel has to work in whatever container it is put
        # in, and a second instance in a narrow Custom-tab cell is
        # exactly what PR3 will do to it. Stacking also happens to be
        # the mockup's own layout: a full-width bar, then the gate state
        # on the left and the figure on the right.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._bar = QProgressBar()
        self._bar.setRange(0, _BAR_STEPS)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(_BAR_HEIGHT)
        layout.addWidget(self._bar)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._gate_label = QLabel("-")
        self._gate_label.setProperty("role", "dim")
        # Between the gate state and the figure, so it reads as a
        # property of this panel rather than of the number. Empty on a
        # single-branch station, where it takes no space at all -- the
        # layout below it is unchanged for every station that had one
        # meter before this existed.
        self._ear_label = QLabel("")
        self._ear_label.setProperty("role", "dim")
        self._value_label = QLabel("-")
        self._value_label.setProperty("role", "value")
        row.addWidget(self._gate_label)
        row.addStretch(1)
        row.addWidget(self._ear_label)
        row.addWidget(self._value_label)
        layout.addLayout(row)

        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()

    def stop(self) -> None:
        """Stop polling. Does not stop the receive session."""
        self._timer.stop()

    def _on_timer(self) -> None:
        text = quieting_text(
            self._source.live_quieting_db,
            self._source.live_squelch_open,
            # Read every poll, not captured at construction: with a
            # combiner running the ear moves mid-pass, and a marker
            # fixed at build time would be wrong for most of the run.
            listening=_listening_state(self._source),
        )
        self._value_label.setText(text.quieting_label)
        self._gate_label.setText(text.gate_label)
        self._ear_label.setText(text.ear_label)
        self._bar.setValue(round(text.bar_fraction * _BAR_STEPS))
