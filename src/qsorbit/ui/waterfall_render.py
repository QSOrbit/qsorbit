"""Turning spectrum frames into pixels. No Qt import anywhere in here.

Split out from :mod:`qsorbit.ui.waterfall_widget` for exactly the reason
:mod:`qsorbit.ui.readout_formatting` is split out from the readout
widget: every decision worth arguing about lives in plain functions that
can be read and tested without PySide6 installed, and the widget is the
thin remainder that owns a timer and blits an image. That split is
load-bearing rather than aspirational — it only works because
``ui/__init__.py`` stays empty, so importing this module does not
transitively drag Qt in through the package.

Two choices in here are worth more than their line count.

**Downsampling bins to pixels takes the maximum, not the mean.** A
waterfall almost always has more FFT bins than horizontal pixels — 2048
bins into an 800-pixel panel is typical — so something has to combine
them. Averaging is the obvious answer and it is the wrong one for this
instrument. A satellite downlink is *narrow*: a carrier can occupy one or
two bins out of 2048, and averaging it with its 30 dB-quieter neighbours
drags it most of the way back down to the noise floor. Max-hold keeps it.
The distinction is the same one Session 14's bring-up settled for a
different reason — this panel exists to answer *is the signal present*,
not *what is the average power here* — and averaging answers the second
question while quietly destroying the evidence for the first.

**The dB scale is fixed, not auto-ranged.** Auto-scaling looks better on
a screenshot and is actively harmful here: it moves the mapping in step
with the signal, so a carrier appearing, fading, or dropping out shows up
as no change at all. A fixed window means brightness *means* something
across time, which is the entire point of watching a pass. The defaults
come from a real measurement rather than taste — see
:class:`WaterfallScale`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Colour ramp control points, dark to bright, as ``(position, r, g, b)``
#: with position in ``[0, 1]``. Black through blue and cyan into yellow
#: and white: a conventional waterfall ramp, chosen because brightness
#: rises monotonically across the whole range, so "brighter is stronger"
#: holds everywhere and no band of the scale looks like a feature that
#: is not there. Interpolated linearly into a 256-entry table once, at
#: import, rather than per row.
_RAMP: tuple[tuple[float, int, int, int], ...] = (
    (0.00, 0, 0, 0),
    (0.25, 0, 0, 128),
    (0.50, 0, 160, 190),
    (0.75, 240, 220, 60),
    (1.00, 255, 255, 255),
)


def _build_colormap() -> np.ndarray:
    """Interpolate :data:`_RAMP` into a 256x3 uint8 lookup table."""
    positions = np.array([point[0] for point in _RAMP], dtype=np.float64)
    table = np.empty((256, 3), dtype=np.uint8)
    x = np.linspace(0.0, 1.0, 256)
    for channel in range(3):
        values = np.array([point[channel + 1] for point in _RAMP], dtype=np.float64)
        table[:, channel] = np.interp(x, positions, values).astype(np.uint8)
    return table


#: The colour table, built once at import.
COLORMAP: np.ndarray = _build_colormap()


@dataclass(frozen=True)
class WaterfallScale:
    """The dB window mapped onto the colour ramp.

    Follows the frozen-dataclass-plus-``__post_init__``-validation
    template every config object in this project uses, for the same
    reason: a nonsensical scale should fail at construction rather than
    produce a uniformly black panel that looks like a dead radio.

    Args:
        floor_db: Power at or below this maps to the darkest colour.
        ceiling_db: Power at or above this maps to the brightest.

    The defaults are measured rather than guessed. Running the real
    ``wbfm-99.9.iq`` capture (RTL-SDR Blog V4, 32.8 dB manual gain)
    through :func:`~qsorbit.core.dsp.spectrum.power_spectrum_db` puts the
    noise floor near −67 dB and the broadcast carrier near −40 dB, with
    the receiver's own DC spike around −57 dB. A −90 to −20 window
    therefore leaves the noise floor dark, the carrier bright, and
    headroom on both sides for a stronger or weaker signal without
    reaching for the controls. **These are defaults, not constants** —
    gain, antenna and band all move the whole picture, so a caller
    watching a quiet satellite downlink should expect to lower both.

    Raises:
        ValueError: If either bound is not finite, or ``ceiling_db`` is
            not above ``floor_db``.
    """

    floor_db: float = -90.0
    ceiling_db: float = -20.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.floor_db):
            raise ValueError(f"floor_db must be finite, got {self.floor_db!r}.")
        if not math.isfinite(self.ceiling_db):
            raise ValueError(f"ceiling_db must be finite, got {self.ceiling_db!r}.")
        if self.ceiling_db <= self.floor_db:
            raise ValueError(
                f"ceiling_db must be above floor_db, got floor={self.floor_db!r} "
                f"ceiling={self.ceiling_db!r}."
            )

    @property
    def span_db(self) -> float:
        """How many dB the ramp covers."""
        return self.ceiling_db - self.floor_db


def bins_to_pixels(power_db: np.ndarray, width: int) -> np.ndarray:
    """Resample one frame's bins onto ``width`` pixels, keeping peaks.

    Args:
        power_db: One frame's power values, any length.
        width: Pixels to produce.

    Returns:
        ``width`` values, taken as the **maximum** over each pixel's span
        of bins when there are more bins than pixels, and by
        nearest-neighbour repetition when there are fewer.

        Max rather than mean is the whole point of this function; see the
        module docstring. A single-bin carrier surviving a 2048-to-800
        reduction is the behaviour, not a side effect, and there is a
        test that fails if someone swaps in an average.

    Raises:
        ValueError: If ``width`` is not positive, or ``power_db`` is not
            a non-empty one-dimensional array.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width!r}.")
    if power_db.ndim != 1 or power_db.shape[0] == 0:
        raise ValueError(f"power_db must be a non-empty 1-D array, got shape {power_db.shape}.")

    bins = power_db.shape[0]
    if width >= bins:
        # Fewer bins than pixels: repeat, do not interpolate. Interpolation
        # would invent intermediate values between two real measurements
        # and make a narrow carrier look like a broad one.
        return power_db[(np.arange(width) * bins) // width]

    # Strictly increasing because bins/width > 1, which is what reduceat
    # needs to give one output per span.
    edges = (np.arange(width) * bins) // width
    return np.maximum.reduceat(power_db, edges)


def db_to_index(power_db: np.ndarray, scale: WaterfallScale) -> np.ndarray:
    """Map dB values onto ``0..255`` colour-table indices, clamping outside."""
    normalised = (power_db - scale.floor_db) / scale.span_db
    return (np.clip(normalised, 0.0, 1.0) * 255.0).astype(np.uint8)


def colorize(indices: np.ndarray) -> np.ndarray:
    """Look ``0..255`` indices up in :data:`COLORMAP`, returning RGB."""
    return COLORMAP[indices]


def render_row(power_db: np.ndarray, width: int, scale: WaterfallScale) -> np.ndarray:
    """Turn one spectrum frame into one row of RGB pixels.

    Args:
        power_db: The frame's power values, in dB.
        width: Pixels wide.
        scale: The dB window to map onto the ramp.

    Returns:
        A ``(width, 3)`` uint8 RGB array, ready to be written into an
        image buffer.

    The order matters and is deliberate: resample **first**, in the dB
    domain, and only then map to colour. Maximum is monotonic, so taking
    it before the scaling gives the same answer as after and does the
    clamping and lookup on ``width`` values instead of on every bin —
    which, at 2048 bins against 800 pixels several times a second, is
    most of the work avoided.
    """
    return colorize(db_to_index(bins_to_pixels(power_db, width), scale))


def _label_decimals(step_hz: float) -> int:
    """Smallest decimal count that writes ``step_hz`` in MHz without rounding.

    Derived rather than fixed because the step is chosen from the span:
    a 0.5 MHz step needs one decimal and a 0.25 MHz step needs two, and
    hard-coding either produces labels that repeat ("99.2, 99.2") or
    trail meaningless zeros.
    """
    step_mhz = step_hz / 1e6
    for decimals in range(7):
        if abs(round(step_mhz, decimals) - step_mhz) < 1e-12:
            return decimals
    return 6


def frequency_ticks(start_hz: float, stop_hz: float, max_ticks: int = 6) -> list[tuple[float, str]]:
    """Choose round frequencies to label across a span, with their labels.

    Args:
        start_hz: Frequency at the left edge.
        stop_hz: Frequency at the right edge.
        max_ticks: Upper bound on how many labels to produce.

    Returns:
        ``(frequency_hz, label)`` pairs in increasing order, at round
        1/2/2.5/5-times-a-power-of-ten steps — the same family of steps
        any instrument uses, because they are the ones a person can do
        arithmetic between without thinking. Labels are in MHz, carrying
        exactly as many decimals as the chosen step needs.

    Raises:
        ValueError: If the span is not positive or ``max_ticks`` is below 2.
    """
    if not stop_hz > start_hz:
        raise ValueError(f"stop_hz must be above start_hz, got {start_hz!r}..{stop_hz!r}.")
    if max_ticks < 2:
        raise ValueError(f"max_ticks must be at least 2, got {max_ticks!r}.")

    span = stop_hz - start_hz
    magnitude = 10.0 ** math.floor(math.log10(span / max_ticks))
    step = magnitude
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiplier * magnitude
        if span / step <= max_ticks:
            break

    decimals = _label_decimals(step)
    ticks: list[tuple[float, str]] = []
    index = math.ceil(start_hz / step)
    while True:
        frequency = index * step
        if frequency > stop_hz + step * 1e-9:
            break
        ticks.append((frequency, f"{frequency / 1e6:.{decimals}f}"))
        index += 1
    return ticks


def tick_position(frequency_hz: float, start_hz: float, stop_hz: float, width: int) -> float:
    """Where a frequency falls, in pixels from the left edge of ``width``.

    Linear because every step between bins and pixels is: the FFT's bins
    are evenly spaced in frequency, :func:`bins_to_pixels` maps equal
    spans of bins onto each pixel, and the widget scales the finished
    image uniformly. If any of those three stops being true, this does
    too.
    """
    return (frequency_hz - start_hz) / (stop_hz - start_hz) * width


def blank_row(width: int) -> np.ndarray:
    """One row of "nothing received", for pre-filling a waterfall's history.

    Exists so a display can start with a full-height buffer rather than
    growing one. A history that grows from empty has to be stretched to
    fill its panel, which means the **time** scale moves for as long as
    it is filling — and a moving time axis makes a Doppler slope appear
    to change angle when nothing about the signal changed. That is the
    same objection this module already raises against auto-ranging the
    dB scale, applied to the other axis.

    Deliberately identical to what :func:`render_row` produces for a
    silent frame, so pre-filled rows are indistinguishable from genuine
    silence instead of being a slightly different black that reads as a
    seam. There is a test that fails if the two ever drift apart.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width!r}.")
    return np.repeat(COLORMAP[0][np.newaxis, :], width, axis=0)
