"""Integer decimation of complex IQ samples.

A shared utility rather than something private to demodulation: spectrum
framing can use it to reduce a wide capture to a manageable rate before an
FFT, and Chunk G's NBFM channel filter will need the same operation again
for its narrower channel. One implementation, one place to have gotten the
anti-aliasing right.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import decimate as _scipy_decimate

#: scipy's own documentation for :func:`scipy.signal.decimate` recommends
#: calling it multiple times for downsampling factors higher than this,
#: rather than passing one large factor — its default IIR filter design
#: becomes numerically unreliable past this point. This is scipy's
#: constraint, not a policy of ours, so decimate() below chains
#: automatically instead of asking every caller to remember it.
MAX_SINGLE_STAGE_FACTOR: int = 13


def decimate(iq: np.ndarray, factor: int) -> np.ndarray:
    """Decimate ``iq`` by an integer ``factor``, anti-alias filtered.

    Large factors are split into a chain of stages no larger than
    :data:`MAX_SINGLE_STAGE_FACTOR`, per scipy's own advice for
    :func:`scipy.signal.decimate` — a factor of, say, 40 runs as two
    stages of 4 and 10 rather than one stage of 40. If ``factor`` itself
    is prime and larger than :data:`MAX_SINGLE_STAGE_FACTOR` (e.g. 17),
    it cannot be split further and runs as a single oversized stage;
    that is inherent to the requested factor, not something chaining can
    fix.

    Args:
        iq: Complex samples to decimate.
        factor: How much to reduce the sample rate by. Must be a positive
            integer. ``factor == 1`` returns an unmodified copy.

    Returns:
        The decimated signal, as complex64, roughly ``len(iq) / factor``
        samples long (see :func:`scipy.signal.decimate` for the exact
        edge-sample behaviour).

    Raises:
        ValueError: If ``factor`` is not a positive integer.
    """
    if isinstance(factor, bool) or not isinstance(factor, int):
        raise ValueError(f"factor must be an int, got {factor!r}.")
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor!r}.")
    if factor == 1:
        return iq.copy()

    result = iq
    for stage in _stage_factors(factor):
        # zero_phase=True avoids the group delay an ordinary IIR filter
        # would introduce, which matters here because a phase shift in
        # decimation becomes a timing error in whatever range-rate or
        # Doppler correction consumes the result downstream.
        result = _scipy_decimate(result, stage, zero_phase=True)
    return result.astype(np.complex64)


def _stage_factors(factor: int, max_stage: int = MAX_SINGLE_STAGE_FACTOR) -> list[int]:
    """Split ``factor`` into a chain of integer stages, each ``<= max_stage``.

    Factors ``factor`` into primes and greedily groups them so each stage's
    product stays at or under ``max_stage``. A prime factor larger than
    ``max_stage`` cannot be grouped with anything and becomes its own
    (oversized) stage.
    """
    primes = _prime_factors(factor)
    stages: list[int] = []
    current = 1
    for prime in sorted(primes, reverse=True):
        if current * prime <= max_stage:
            current *= prime
        else:
            if current > 1:
                stages.append(current)
            current = prime
    stages.append(current)
    return stages


def _prime_factors(n: int) -> list[int]:
    """Return the prime factorization of ``n`` (with multiplicity)."""
    factors = []
    divisor = 2
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    if n > 1:
        factors.append(n)
    return factors
