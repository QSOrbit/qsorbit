"""The Rotor tab's tracking-profile toggle.

Chunk H's whole point in one control: switch the rotor between stock
gains and the bench-validated set **during a pass**, so the two can be
compared without the between-run variability Session 32 spent an
afternoon fighting. Back-to-back runs would reintroduce exactly the
confound the interleaved sweep was designed to remove.

**A switch is not instant, and this widget says so rather than
pretending.** Changing profile means up to six serial writes, a settle,
and six reads back -- one to two seconds on the port the tracking loop
already owns. So a press *queues* the switch
(:meth:`~qsorbit.core.pointing.TrackingLoop.request_profile`) and the
loop applies it at the top of its next tick. Between those two moments
the button shows "switching", because a control that reports done
before anything has reached the hardware is the same silent lie this
project keeps finding in frozen readouts.

**The toggle refuses while an axis is stalled**, and shows why. A
stalled axis is holding error that the frozen setpoint is containing;
new gains would act on all of it at once, which is the runaway the
stall guard exists to prevent.

**This widget knows nothing about what contains it**, the Session 19
convention every panel here follows: it takes a source and a list of
profiles, and never reaches for a window, a rotor, or a serial port. It
also hardcodes no colour -- the pressed state is
``QPushButton:checked`` in the theme's own stylesheet.
"""

from __future__ import annotations

from typing import Final, Protocol

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from qsorbit.core.pointing import ProfileSwitchError
from qsorbit.core.tracking_profile import TrackingProfile
from qsorbit.ui.profile_formatting import profile_status_text

#: How often the widget re-reads its source, in milliseconds.
#:
#: Matched to the readout rather than the quieting panel: a profile
#: switch resolves within one tracking tick, and the fastest cadence
#: this project ships is 500 ms.
DEFAULT_POLL_INTERVAL_MS: Final = 250


class ProfileSource(Protocol):
    """Anything that can be asked to change tracking profile.

    Satisfied by :class:`~qsorbit.core.pointing.TrackingLoop` without
    subclassing, the same structural convention the other panels use.
    """

    @property
    def active_profile(self) -> TrackingProfile | None: ...

    @property
    def pending_profile(self) -> TrackingProfile | None: ...

    @property
    def profile_refusal(self) -> str | None: ...

    @property
    def is_stalled(self) -> bool: ...

    def request_profile(self, profile: TrackingProfile) -> None: ...


class ProfileWidget(QWidget):
    """A button per profile, plus a line saying what is happening.

    Args:
        source: What to read and ask. Ticked by somebody else.
        profiles: Every profile this station declares, in config order.
        poll_interval_ms: How often to re-read ``source``.
        parent: Qt parent.
    """

    def __init__(
        self,
        source: ProfileSource,
        profiles: tuple[TrackingProfile, ...],
        *,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._profiles = profiles

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        for index, profile in enumerate(profiles):
            button = QPushButton(profile.name)
            button.setCheckable(True)
            button.clicked.connect(
                # Bound per profile rather than read back off the sender,
                # so the handler cannot be wrong about which button it is.
                lambda _checked, chosen=profile: self._request(chosen)
            )
            self._group.addButton(button, index)
            self._buttons[profile.name] = button
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)

        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    @property
    def status_text(self) -> str:
        """What the status line currently reads. For tests and for tabs."""
        return self._status.text()

    def _request(self, profile: TrackingProfile) -> None:
        """Ask for a switch, and let a refusal be a normal outcome.

        :class:`~qsorbit.core.pointing.ProfileSwitchError` is caught
        rather than propagated: it is the loop declining a *moment*, not
        a fault, and it already carries the sentence the operator needs.
        Letting it unwind through Qt's dispatch would print a traceback
        and change nothing on screen.
        """
        try:
            self._source.request_profile(profile)
        except ProfileSwitchError:
            pass
        self.refresh()

    def refresh(self) -> None:
        """Re-read the source and redraw. Cheap: plain property reads."""
        active = self._source.active_profile
        pending = self._source.pending_profile
        stalled = self._source.is_stalled

        # The checked button follows what is *in force*, never what was
        # clicked. A press that is queued or refused must not leave the
        # control claiming a state the rotor is not in.
        shown = active.name if active is not None else None
        for name, button in self._buttons.items():
            button.setChecked(name == shown)
            button.setEnabled(not stalled and pending is None and name != shown)

        self._status.setText(
            profile_status_text(
                active,
                pending,
                self._source.profile_refusal,
                stalled=stalled,
            )
        )

    def stop(self) -> None:
        """Stop polling. Does not touch the loop."""
        self._timer.stop()
