"""Shared frequency-axis painting for the waterfall and the spectrum-line panel.

Both panels draw the exact same axis strip - same ticks, same labels,
same "MHz" corner note - directly beneath their own plot, and Phil's own
request for the two panels was that they share a frequency axis so a
peak on one lines up with the same frequency on the other. Kept as one
function rather than letting each panel's ``paintEvent`` grow its own
near-identical copy, which is exactly the kind of drift that would let
the two panels' axes quietly disagree with each other.

Not part of :mod:`qsorbit.ui.waterfall_render` - that module's whole
point is importing no Qt at all, which is what makes it fully unit
tested. This one function does need a live ``QPainter``, so it lives
here instead rather than compromising that module's own invariant.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter

from qsorbit.ui.waterfall_render import frequency_ticks, tick_position

#: Roughly how many pixels each frequency label needs to itself. Used to
#: scale the tick count with the panel, so a narrow window thins its
#: labels out instead of overprinting them into a smear.
_PIXELS_PER_LABEL = 90


def paint_frequency_axis(
    painter: QPainter,
    plot_rect: QRect,
    start_hz: float,
    stop_hz: float,
    axis_height_px: int,
    color: QColor,
) -> None:
    """Draw frequency ticks and labels in a strip directly below ``plot_rect``.

    Args:
        painter: An already-active painter on the panel being drawn.
        plot_rect: The plot area the axis sits beneath - ticks are
            positioned against its width, exactly as
            :func:`~qsorbit.ui.waterfall_render.tick_position` maps a
            frequency onto it.
        start_hz: Frequency at the plot's left edge.
        stop_hz: Frequency at the plot's right edge.
        axis_height_px: Vertical space reserved for the strip.
        color: Pen color for the ticks, labels and corner note - the
            caller's own palette color, so the axis matches whatever
            theme the panel is drawn in rather than a color fixed here.
    """
    painter.setPen(color)
    top = plot_rect.bottom() + 1
    max_ticks = max(2, min(9, plot_rect.width() // _PIXELS_PER_LABEL))

    for frequency_hz, label in frequency_ticks(start_hz, stop_hz, max_ticks):
        x = int(round(tick_position(frequency_hz, start_hz, stop_hz, plot_rect.width())))
        painter.drawLine(x, top, x, top + 4)
        painter.drawText(
            QRect(x - 45, top + 5, 90, axis_height_px - 5),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            label,
        )

    painter.drawText(
        QRect(plot_rect.right() - 40, top + 5, 38, axis_height_px - 5),
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        "MHz",
    )
