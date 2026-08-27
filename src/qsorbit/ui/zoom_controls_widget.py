"""Numeric zoom controls — a span field and a lock checkbox.

Phil's own scoping choice for the waterfall zoom item was **both**
numeric and mouse-driven interaction — precise, discoverable controls
alongside the scroll/drag feel :class:`~qsorbit.ui.waterfall_widget.WaterfallWidget`
and :class:`~qsorbit.ui.spectrum_line_widget.SpectrumLineWidget` already
give the mouse. This is the numeric half: a span spinbox and a
lock-to-tracked-frequency checkbox, both driving and driven by the same
:class:`~qsorbit.ui.zoom_controller.ZoomController` the mouse gestures
use, so the two ways of controlling the view never disagree about what
the view currently is.
"""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QWidget

from qsorbit.ui.spectrum_zoom import MIN_ZOOM_SPAN_HZ
from qsorbit.ui.zoom_controller import ZoomController

#: The spinbox works in kHz rather than raw Hz, matching how this
#: project's own CLI already talks about tuning (``--offset`` in
#: ``qsorbit receive --help`` is kHz too) and how an amateur operator
#: thinks about a channel's width.
_SPAN_STEP_KHZ = 1.0
_SPAN_DECIMALS = 1


class ZoomControlsWidget(QWidget):
    """A span spinbox and a lock-to-tracked-frequency checkbox over one controller.

    Args:
        controller: The shared zoom state this widget both drives (the
            spinbox and checkbox act on it directly) and mirrors (its own
            fields update whenever a mouse gesture on either spectrum
            panel changes the same controller).
        parent: Standard Qt parent.

    Without the mirroring, a user who zoomed with the mouse wheel would
    see the spinbox still showing the span from before — worse than
    merely stale, since typing into it would then jump the view rather
    than nudge it from where it visibly is. Both fields block their own
    signals while being written from the controller, so reflecting its
    state back does not immediately fire another change straight back at
    it.
    """

    def __init__(self, controller: ZoomController, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Span:", self))

        band_width_khz = (controller.band_stop_hz - controller.band_start_hz) / 1000.0
        self._span_spinbox = QDoubleSpinBox(self)
        self._span_spinbox.setDecimals(_SPAN_DECIMALS)
        self._span_spinbox.setSingleStep(_SPAN_STEP_KHZ)
        self._span_spinbox.setSuffix(" kHz")
        self._span_spinbox.setRange(MIN_ZOOM_SPAN_HZ / 1000.0, band_width_khz)
        self._span_spinbox.setValue(controller.zoom.span_hz / 1000.0)
        self._span_spinbox.valueChanged.connect(self._on_span_spinbox_changed)
        layout.addWidget(self._span_spinbox)

        self._lock_checkbox = QCheckBox("Lock to tracked frequency", self)
        self._lock_checkbox.setChecked(controller.locked)
        self._lock_checkbox.toggled.connect(controller.set_locked)
        layout.addWidget(self._lock_checkbox)

        layout.addStretch(1)

        controller.changed.connect(self._on_controller_changed)

    def stop(self) -> None:
        """No timer of its own — present only so
        :meth:`~qsorbit.ui.instrument_window.InstrumentWindow.closeEvent`
        can call ``.stop()`` uniformly across every panel it holds."""

    def _on_span_spinbox_changed(self, value_khz: float) -> None:
        self._controller.set_span_hz(value_khz * 1000.0)

    def _on_controller_changed(self) -> None:
        span_khz = self._controller.zoom.span_hz / 1000.0
        if abs(self._span_spinbox.value() - span_khz) > 1e-6:
            self._span_spinbox.blockSignals(True)
            self._span_spinbox.setValue(span_khz)
            self._span_spinbox.blockSignals(False)

        if self._lock_checkbox.isChecked() != self._controller.locked:
            self._lock_checkbox.blockSignals(True)
            self._lock_checkbox.setChecked(self._controller.locked)
            self._lock_checkbox.blockSignals(False)
