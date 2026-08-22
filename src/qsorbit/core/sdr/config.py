"""Device configuration for an RTL-SDR: what to tune, how fast, how loud.

The dividing line this module holds to is **what can be checked without
a device present**. Those checks live here, in the value object, and
raise :class:`ValueError` — the same contract
:class:`~qsorbit.core.rotor.Position` has. Anything that needs the
hardware to answer (which gain steps this tuner actually offers, whether
the R828D will go to 23 MHz) is left to the device, and comes back as a
:class:`~qsorbit.core.sdr.exceptions.DeviceError`. There is deliberately
no finer-grained "this setting is out of range" exception: librtlsdr
answers every failure with a bare negative integer, and nothing in it
distinguishes an out-of-range frequency from any other fault, so a
separate class would be promising a distinction we cannot actually
make.

Note what is deliberately *not* here: the device index and the driver
directory. Those answer "which radio, and how do we load its library",
not "how is it configured" — they belong with the connection, the way a
port name belongs to :class:`~qsorbit.core.rotor.SerialPort` rather than
to a rotor's capabilities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

#: Sample rates the RTL2832U will accept, as ``(low, high)`` pairs in Hz,
#: inclusive. This is not a policy of ours — the chip has a genuine hole
#: between the two windows and the driver rejects anything inside it.
#: Knowable without a device, so it is checked here.
SAMPLE_RATE_WINDOWS_HZ: Final[tuple[tuple[int, int], ...]] = (
    (225_001, 300_000),
    (900_001, 3_200_000),
)

#: Above this rate most USB host controllers cannot keep up and samples
#: are dropped somewhere between the dongle and the application — often
#: without any error surfacing. Not an error (2.56 Msps works on some
#: machines), so it is exposed as :attr:`SdrConfig.may_drop_samples`
#: rather than enforced. 2.048 Msps is what the bring-up captures used.
RELIABLE_MAX_SAMPLE_RATE_HZ: Final[int] = 2_400_000

#: Bound on the crystal correction, in parts per million. A real dongle
#: is out by single or low double digits; anything past this is a typo,
#: not a calibration.
MAX_PPM: Final[int] = 1_000


class AutoGain(Enum):
    """Sentinel for "let the tuner pick its own gain".

    A single-member enum rather than ``None``, because ``None`` in a
    configuration object reads as *unset* — and the difference between
    "unset" and "deliberately automatic" is exactly the difference this
    field must not blur.

    Auto gain is supported but is **not** the default, and the reason is
    on the record: during bring-up on 2026-08-22 the V4 in auto mode
    reported 0.0 dB and returned a flat, empty capture. Nothing raised;
    the samples simply contained nothing. Manual gain is what every
    working ``rtl_*`` capture does in practice.
    """

    AUTO = "auto"


#: Convenience alias so call sites read ``gain_db=AUTO_GAIN``.
AUTO_GAIN: Final[AutoGain] = AutoGain.AUTO


@dataclass(frozen=True)
class SdrConfig:
    """How one RTL-SDR should be tuned and configured.

    Every field is required except the two that have an honest default:
    a zero ppm correction is a real, meaningful value, and the RTL2832's
    digital AGC is off unless asked for.

    Gain in particular has **no default**. A default gain is precisely
    how a capture comes back empty without anyone noticing, so the
    caller is made to say what it wants.

    Args:
        center_hz: Centre frequency to tune to, in Hz. Whether this
            tuner can reach it is a device question, checked at
            configure time — this only rejects numbers no radio could
            use.
        sample_rate_hz: Sample rate in Hz. Must fall inside one of
            :data:`SAMPLE_RATE_WINDOWS_HZ`.
        gain_db: Tuner gain in dB, or :data:`AUTO_GAIN`. A manual value
            is snapped to the nearest step this tuner actually offers —
            see :func:`nearest_gain_step` — because the gain control is
            a table of discrete steps, not a continuous knob.
        ppm: Crystal frequency correction in parts per million. Usually
            a station constant; see ``[sdr] ppm`` in the station config.
        enable_agc: Whether to enable the RTL2832's *digital* AGC. This
            is a separate control from tuner gain and the two are easily
            confused; enabling this does not set the tuner's gain.

    Raises:
        ValueError: If a value is one no RTL-SDR could accept.
    """

    center_hz: float
    sample_rate_hz: float
    gain_db: float | AutoGain
    ppm: int = 0
    enable_agc: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.center_hz) or self.center_hz <= 0.0:
            raise ValueError(
                f"center_hz must be a positive, finite frequency, got {self.center_hz!r}."
            )

        if not math.isfinite(self.sample_rate_hz):
            raise ValueError(f"sample_rate_hz must be finite, got {self.sample_rate_hz!r}.")
        if not any(low <= self.sample_rate_hz <= high for low, high in SAMPLE_RATE_WINDOWS_HZ):
            windows = " or ".join(f"{low:,}-{high:,} Hz" for low, high in SAMPLE_RATE_WINDOWS_HZ)
            raise ValueError(
                f"sample_rate_hz must be {windows}, got {self.sample_rate_hz:,.0f}. "
                "The RTL2832U has a real gap between those two windows; a rate "
                "inside it is rejected by the driver, not merely discouraged."
            )

        if not isinstance(self.gain_db, AutoGain):
            if isinstance(self.gain_db, bool) or not isinstance(self.gain_db, (int, float)):
                raise ValueError(
                    f"gain_db must be a number in dB or AUTO_GAIN, got {self.gain_db!r}."
                )
            if not math.isfinite(self.gain_db) or self.gain_db < 0.0:
                raise ValueError(
                    f"gain_db must be a finite, non-negative number of dB, got {self.gain_db!r}."
                )

        if isinstance(self.ppm, bool) or not isinstance(self.ppm, int):
            raise ValueError(f"ppm must be an integer, got {self.ppm!r}.")
        if abs(self.ppm) > MAX_PPM:
            raise ValueError(
                f"ppm must be within +/-{MAX_PPM}, got {self.ppm}. A real dongle is "
                "out by single or low double digits."
            )

    @property
    def uses_auto_gain(self) -> bool:
        """``True`` if the tuner has been asked to choose its own gain."""
        return isinstance(self.gain_db, AutoGain)

    @property
    def may_drop_samples(self) -> bool:
        """``True`` if this rate is above what USB reliably sustains.

        Not a fault and not enforced — some machines manage 2.56 or even
        3.2 Msps. It is exposed so a caller that later finds gaps in a
        stream has somewhere to look first.
        """
        return self.sample_rate_hz > RELIABLE_MAX_SAMPLE_RATE_HZ

    # There is deliberately no offset_from() here, though there was in
    # PR1. "Where does a station land in this capture" has to be
    # measured from the centre the tuner *actually* reached, not the one
    # it was asked for, so the method belongs on
    # :class:`~qsorbit.core.sdr.device.AppliedSettings` and lives there.
    # Keeping a copy here that answered the same question from the
    # requested centre would be offering a subtly wrong number under a
    # right-looking name.


def nearest_gain_step(requested_db: float, supported_db: Sequence[float]) -> float:
    """Return the supported gain step closest to ``requested_db``.

    The RTL-SDR's tuner gain is a table of discrete steps, not a
    continuous control — the V4's R828D reports 29 of them, from 0.0 to
    49.6 dB. The table is a property of the *tuner*, so it is read from
    the device rather than hardcoded here; a different tuner reports a
    different table, and a table that does not match the hardware is a
    useful sign that the wrong library got loaded.

    Args:
        requested_db: The gain the caller asked for, in dB.
        supported_db: The steps this tuner reports, in dB, in any order.

    Returns:
        The closest supported step. Exact ties resolve **downward**, on
        the principle that being slightly quiet costs signal-to-noise
        while being slightly loud can overload the ADC and smear the
        whole spectrum.

    Raises:
        ValueError: If ``supported_db`` is empty — which is itself
            diagnostic, since a device that reports no gain steps is not
            one we can drive.
    """
    if not supported_db:
        raise ValueError(
            "The device reported no supported gain steps, so no gain can be set. "
            "This usually means the library loaded but the tuner was not identified."
        )
    return min(supported_db, key=lambda step: (abs(step - requested_db), step))
