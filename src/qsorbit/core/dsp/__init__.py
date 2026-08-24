"""DSP module: turning raw IQ into a spectrum, and into audible FM.

Chunk E, complete: IQ format conversion, power-spectrum framing for a
waterfall consumer, integer decimation, WBFM demodulation, and audio
output. Chunk F adds the streaming half of the spectrum path — frames
produced on a worker thread at a rate a display can use, with no Qt
import anywhere in here. Chunk G adds narrowband FM — the mode the
satellite downlinks actually use — and the optional noise squelch that
makes a narrowband listen bearable. See the Phase 2 brief.
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
    DEFAULT_NBFM_DEEMPHASIS_US,
    DEFAULT_NBFM_DEVIATION_HZ,
    DEFAULT_NBFM_IF_RATE_HZ,
    MIN_IF_SAMPLES,
    NbfmConfig,
    WbfmConfig,
    demodulate_nbfm,
    demodulate_wbfm,
    discriminate,
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
from qsorbit.core.dsp.spectrum_stream import (
    DEFAULT_FRAME_RATE_HZ,
    DEFAULT_QUEUE_FRAMES,
    SpectrumFrame,
    SpectrumStream,
    SpectrumStreamStats,
    hop_for_frame_rate,
)
from qsorbit.core.dsp.squelch import (
    DEFAULT_CLOSE_BELOW_DB,
    DEFAULT_NOISE_BAND_LOW_HZ,
    DEFAULT_OPEN_ABOVE_DB,
    MAX_QUIETING_DB,
    NoiseSquelch,
    SquelchStats,
    quieting_db,
)

__all__ = [
    "AUDIO_CLIP_RANGE",
    "DEFAULT_AUDIO_RATE_HZ",
    "DEFAULT_CLOSE_BELOW_DB",
    "DEFAULT_DEEMPHASIS_US",
    "DEFAULT_DEVIATION_HZ",
    "DEFAULT_FLOOR_DB",
    "DEFAULT_FRAME_RATE_HZ",
    "DEFAULT_NBFM_DEEMPHASIS_US",
    "DEFAULT_NBFM_DEVIATION_HZ",
    "DEFAULT_NBFM_IF_RATE_HZ",
    "DEFAULT_NOISE_BAND_LOW_HZ",
    "DEFAULT_OPEN_ABOVE_DB",
    "DEFAULT_QUEUE_BLOCKS",
    "DEFAULT_QUEUE_FRAMES",
    "IQ_SCALE",
    "IQ_ZERO_OFFSET",
    "MAX_QUIETING_DB",
    "MAX_SINGLE_STAGE_FACTOR",
    "MIN_IF_SAMPLES",
    "AudioError",
    "AudioOutput",
    "AudioStats",
    "NbfmConfig",
    "NoiseSquelch",
    "SpectrumConfig",
    "SpectrumFrame",
    "SpectrumStream",
    "SpectrumStreamStats",
    "SquelchStats",
    "WbfmConfig",
    "decimate",
    "demodulate_nbfm",
    "demodulate_wbfm",
    "discriminate",
    "frame_iq",
    "frequency_axis_hz",
    "hop_for_frame_rate",
    "power_spectrum_db",
    "quieting_db",
    "shift_to_baseband",
    "unpack_uint8_iq",
]
