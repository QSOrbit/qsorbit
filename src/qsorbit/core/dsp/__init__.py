"""DSP module: turning raw IQ into a spectrum, and (soon) into audio.

Chunk E first light: IQ format conversion, power-spectrum framing for a
waterfall consumer, and integer decimation. WBFM demodulation and audio
output follow in this same module, in a later PR (see the Phase 2 brief).
"""

from qsorbit.core.dsp.decimate import MAX_SINGLE_STAGE_FACTOR, decimate
from qsorbit.core.dsp.iq import IQ_SCALE, IQ_ZERO_OFFSET, unpack_uint8_iq
from qsorbit.core.dsp.spectrum import (
    DEFAULT_FLOOR_DB,
    SpectrumConfig,
    frame_iq,
    frequency_axis_hz,
    power_spectrum_db,
)

__all__ = [
    "DEFAULT_FLOOR_DB",
    "IQ_SCALE",
    "IQ_ZERO_OFFSET",
    "MAX_SINGLE_STAGE_FACTOR",
    "SpectrumConfig",
    "decimate",
    "frame_iq",
    "frequency_axis_hz",
    "power_spectrum_db",
    "unpack_uint8_iq",
]
