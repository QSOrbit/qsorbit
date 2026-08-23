"""DSP module: turning raw IQ into a spectrum, and into audible WBFM.

Chunk E, complete: IQ format conversion, power-spectrum framing for a
waterfall consumer, integer decimation, WBFM demodulation, and audio
output. See the Phase 2 brief.
"""

from qsorbit.core.dsp.audio import (
    DEFAULT_QUEUE_BLOCKS,
    AudioError,
    AudioOutput,
    AudioStats,
)
from qsorbit.core.dsp.decimate import MAX_SINGLE_STAGE_FACTOR, decimate
from qsorbit.core.dsp.demod import (
    AUDIO_CLIP_RANGE,
    DEFAULT_AUDIO_RATE_HZ,
    DEFAULT_DEEMPHASIS_US,
    DEFAULT_DEVIATION_HZ,
    WbfmConfig,
    demodulate_wbfm,
    shift_to_baseband,
)
from qsorbit.core.dsp.iq import IQ_SCALE, IQ_ZERO_OFFSET, unpack_uint8_iq
from qsorbit.core.dsp.spectrum import (
    DEFAULT_FLOOR_DB,
    SpectrumConfig,
    frame_iq,
    frequency_axis_hz,
    power_spectrum_db,
)

__all__ = [
    "AUDIO_CLIP_RANGE",
    "DEFAULT_AUDIO_RATE_HZ",
    "DEFAULT_DEEMPHASIS_US",
    "DEFAULT_DEVIATION_HZ",
    "DEFAULT_FLOOR_DB",
    "DEFAULT_QUEUE_BLOCKS",
    "IQ_SCALE",
    "IQ_ZERO_OFFSET",
    "MAX_SINGLE_STAGE_FACTOR",
    "AudioError",
    "AudioOutput",
    "AudioStats",
    "SpectrumConfig",
    "WbfmConfig",
    "decimate",
    "demodulate_wbfm",
    "frame_iq",
    "frequency_axis_hz",
    "power_spectrum_db",
    "shift_to_baseband",
    "unpack_uint8_iq",
]
