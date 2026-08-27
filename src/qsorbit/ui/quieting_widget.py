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
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

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

        layout = QHBoxLayout(self)
        layout.addWidget(QLabel("Quieting:"))
        self._value_label = QLabel("-")
        layout.addWidget(self._value_label)
        self._bar = QProgressBar()
        self._bar.setRange(0, _BAR_STEPS)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()

    def stop(self) -> None:
        """Stop polling. Does not stop the receive session."""
        self._timer.stop()

    def _on_timer(self) -> None:
        text = quieting_text(self._source.live_quieting_db, self._source.live_squelch_open)
        self._value_label.setText(f"{text.quieting_label} ({text.gate_label})")
        self._bar.setValue(round(text.bar_fraction * _BAR_STEPS))
