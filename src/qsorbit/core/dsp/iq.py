"""IQ sample format conversion.

Every capture in ``tests/fixtures/iq/`` and everything the SDR device layer
streams is raw 8-bit unsigned interleaved I/Q: ``I, Q, I, Q, ...``, offset
binary, where 127.5 is zero — exactly what the RTL-SDR delivers and exactly
what ``rtl_sdr.exe`` writes. See ``tests/fixtures/iq/README.md``.

This module is the single place that format becomes complex baseband
samples. It is deliberately not folded into :mod:`qsorbit.core.sdr` and
deliberately not baked into the fixtures — the README says why: baking the
conversion into the fixture would remove it from test coverage. Everything
downstream in :mod:`qsorbit.core.dsp` works in complex64, never in raw
bytes.
"""

from __future__ import annotations

import numpy as np

#: Offset-binary zero point. 127.5, not 127 or 128, because the format is
#: unsigned 8-bit with no exact centre sample - splitting the difference is
#: what makes the conversion unbiased rather than favouring one polarity.
IQ_ZERO_OFFSET: float = 127.5

#: Scale that maps the offset-centred byte range onto exactly [-1, 1].
#: Dividing by 127.5 rather than 128 or 127 is what makes both extremes
#: land exactly on the boundary: byte 0 gives (0 - 127.5) / 127.5 = -1.0,
#: and byte 255 gives (255 - 127.5) / 127.5 = +1.0. Either of the other
#: two "obvious" choices would leave one end of the range clipped short.
IQ_SCALE: float = 1.0 / IQ_ZERO_OFFSET


def unpack_uint8_iq(raw: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Convert raw offset-binary uint8 interleaved I/Q to complex64 samples.

    Args:
        raw: An even-length sequence of bytes, ``I, Q, I, Q, ...``. A
            :class:`numpy.ndarray` is accepted directly if it is already
            ``uint8`` and one-dimensional; anything else is interpreted as
            a raw byte buffer via :func:`numpy.frombuffer`.

    Returns:
        Complex64 samples, one per I/Q pair, each component scaled to
        roughly ``[-1.0, 1.0]``.

    Raises:
        ValueError: If the input has an odd length. An odd length means a
            truncated I/Q pair — a bug in whatever produced it, not
            something this function can recover from by guessing which
            half of the last pair is missing.
    """
    if isinstance(raw, np.ndarray):
        if raw.ndim != 1:
            raise ValueError(f"raw must be one-dimensional, got shape {raw.shape!r}.")
        samples = raw if raw.dtype == np.uint8 else raw.astype(np.uint8)
    else:
        samples = np.frombuffer(raw, dtype=np.uint8)

    if samples.size % 2 != 0:
        raise ValueError(
            f"IQ byte stream has odd length {samples.size}; that means a truncated "
            "I/Q pair, not a valid capture."
        )

    centered = (samples.astype(np.float32) - IQ_ZERO_OFFSET) * IQ_SCALE
    i_samples = centered[0::2]
    q_samples = centered[1::2]
    return (i_samples + 1j * q_samples).astype(np.complex64)
