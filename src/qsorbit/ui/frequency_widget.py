"""A live readout of where the tracked downlink actually is.

The Radio tab's most prominent number, and a **second consumer of a
level feed** -- which is why it is in this PR rather than waiting for a
later one. The feed hub's central claim is that rotor position, quieting
and tracked frequency are levels rather than streams, and that any
number of consumers can therefore share one feed without stealing from
each other. A claim like that wants more than one consumer in the tree
to mean anything: this widget reads the same
:class:`~qsorbit.ui.feed_hub.TrackedFrequencyFeed` that a
:class:`~qsorbit.ui.zoom_controller.ZoomController` is already polling
to drive its frequency lock, and neither knows the other exists.

Every string it shows is decided by
:func:`~qsorbit.ui.frequency_formatting.frequency_text`, which has no Qt
in it. This file moves those strings into labels and does nothing else.
"""

from __future__ import annotations

from typing import Final, Protocol

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from qsorbit.ui.frequency_formatting import frequency_text

#: How often to repaint. The Doppler tracker recomputes per block --
#: about 16 times a second at 2.048 Msps -- and a readout that changed
#: that fast would be unreadable. Four times a second is fast enough
#: that the digits look live and slow enough that they can be read.
DEFAULT_POLL_INTERVAL_MS: Final = 250


class TrackedFrequencySource(Protocol):
    """Anything that can report the tracked downlink's live frequency.

    Declared structurally rather than importing the feed, matching every
    other widget in this package: a test double satisfies it with one
    property and no session behind it.
    """

    @property
    def live_tracked_frequency_hz(self) -> float | None:
        """See :attr:`~qsorbit.core.receive.ReceiveSession.live_tracked_frequency_hz`."""
        ...


class FrequencyWidget(QWidget):
    """The tracked downlink, big, with its Doppler correction under it.

    Args:
        source: Where the live frequency comes from. The widget neither
            builds it nor owns its lifetime, exactly as every other
            panel here treats its feed.
        nominal_hz: The transmitter's rest frequency, so the shift can
            be shown. ``None`` when the caller does not know it, in
            which case the frequency is displayed and no shift is
            claimed -- see
            :func:`~qsorbit.ui.frequency_formatting.frequency_text`.
        poll_interval_ms: How often to repaint.
    """

    def __init__(
        self,
        source: TrackedFrequencySource,
        *,
        nominal_hz: float | None = None,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._nominal_hz = nominal_hz

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self._megahertz = QLabel("-", self)
        self._megahertz.setProperty("role", "readout")
        self._hertz = QLabel("", self)
        self._hertz.setProperty("role", "value")
        # Bottom-aligned so the small half sits on the big half's
        # baseline rather than floating in the middle of its line
        # height, which is what makes "435.605" and ".000 MHz" read as
        # one number instead of two.
        row.addWidget(self._megahertz, 0, Qt.AlignmentFlag.AlignBottom)
        row.addWidget(self._hertz, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        layout.addLayout(row)

        self._doppler = QLabel("", self)
        self._doppler.setProperty("role", "dim")
        layout.addWidget(self._doppler)

        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()
        self._on_timer()

    def stop(self) -> None:
        """Stop polling. Does not stop the receive session."""
        self._timer.stop()

    def _on_timer(self) -> None:
        text = frequency_text(self._source.live_tracked_frequency_hz, self._nominal_hz)
        self._megahertz.setText(text.megahertz)
        self._hertz.setText(text.hertz)
        self._doppler.setText(text.doppler)
        if self._doppler.property("role") != text.role:
            # Re-polished rather than re-styled: the role property is
            # what the application stylesheet selects on, and Qt only
            # re-evaluates those selectors when a widget is told to.
            # Skipping the no-op case keeps a four-times-a-second timer
            # from asking for a restyle it does not need.
            self._doppler.setProperty("role", text.role)
            style = self._doppler.style()
            style.unpolish(self._doppler)
            style.polish(self._doppler)
