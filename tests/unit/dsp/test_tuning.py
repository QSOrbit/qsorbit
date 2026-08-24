"""Tests for Doppler-corrected tuning.

Two halves. The first exercises :class:`DopplerTracker`'s extrapolation,
staleness handling and accounting directly. The second is the chunk's
**synthetic-Doppler round trip**: generate a narrowband FM signal, sweep
it along a range-rate profile taken from real orbital geometry, correct
it, and assert the modulating tone comes back. That is what tests the
live-correction path without needing a satellite.

**The round-trip tests are built to fail on a sign flip**, which is the
whole reason the named wrappers exist. A test that only checked "the tone
is recovered to within N Hz" would pass with the correction applied
backwards at low range rates, because the error is symmetric about zero —
so several of these assert that the *wrong* sign is measurably worse,
not merely that the right one is good.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from qsorbit.core.dsp import (
    DEFAULT_MAX_EXTRAPOLATION_S,
    DopplerError,
    DopplerStats,
    DopplerTracker,
    NbfmConfig,
    demodulate_nbfm,
)

EPOCH = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)
TRANSMIT_HZ = 435_000_000.0
CENTER_HZ = 434_750_000.0  # tuned 250 kHz low, this project's standing convention

# Round numbers chosen so the expected frequencies are hand-checkable.
EARTH_R_KM, MU_KM3_S2, C_KM_S = 6371.0, 398600.4418, 299792.458


def at(seconds: float) -> datetime:
    return EPOCH + timedelta(seconds=seconds)


def orbital_range_rate(t_s: np.ndarray, alt_km: float = 420.0) -> np.ndarray:
    """Range rate for an overhead circular pass, analytically differentiated.

    Real geometry rather than an invented ramp, so the profile has the
    curvature a linear extrapolator actually has to cope with. ``t_s`` is
    seconds from closest approach, where the Doppler rate is highest —
    the hardest part of a pass, chosen deliberately.
    """
    a = EARTH_R_KM + alt_km
    w = np.sqrt(MU_KM3_S2 / a) / a
    r = np.sqrt(EARTH_R_KM**2 + a**2 - 2 * EARTH_R_KM * a * np.cos(w * t_s))
    return (EARTH_R_KM * a * w * np.sin(w * t_s)) / r


class TestDopplerTrackerConstruction:
    def test_a_reasonable_tracker_is_accepted(self):
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)

        assert tracker.transmit_hz == TRANSMIT_HZ
        assert tracker.center_hz == CENTER_HZ
        assert tracker.has_samples is False

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_an_impossible_transmit_frequency(self, bad):
        with pytest.raises(ValueError, match="transmit_hz"):
            DopplerTracker(bad, CENTER_HZ)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_a_non_finite_centre(self, bad):
        with pytest.raises(ValueError, match="center_hz"):
            DopplerTracker(TRANSMIT_HZ, bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_an_impossible_extrapolation_limit(self, bad):
        with pytest.raises(ValueError, match="max_extrapolation_s"):
            DopplerTracker(TRANSMIT_HZ, CENTER_HZ, max_extrapolation_s=bad)


class TestDopplerTrackerUpdates:
    def test_asking_before_any_sample_raises_rather_than_guessing(self):
        # A programming error, not a runtime condition: demodulation
        # started before the loop produced its first tick.
        with pytest.raises(DopplerError, match="No range rate"):
            DopplerTracker(TRANSMIT_HZ, CENTER_HZ).offset_at(at(0.0))

    def test_a_naive_datetime_is_refused(self):
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)

        with pytest.raises(ValueError, match="timezone-aware"):
            tracker.update(datetime(2026, 8, 24, 12, 0, 0), 0.0)  # noqa: DTZ001

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_range_rate_is_refused(self, bad):
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)

        with pytest.raises(ValueError, match="range_rate_km_s"):
            tracker.update(at(0.0), bad)

    def test_time_running_backwards_is_refused(self):
        # Not fussiness: a negative span inverts the extrapolation slope,
        # which corrects the wrong way -- a sign flip by another route.
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)
        tracker.update(at(10.0), 1.0)

        with pytest.raises(ValueError, match="backwards"):
            tracker.update(at(9.0), 1.0)

    def test_one_sample_holds_that_value_with_no_slope(self):
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)
        tracker.update(at(0.0), 0.0)

        # Zero range rate means no shift, so the offset is exactly the
        # nominal frequency's distance from the tuned centre.
        assert tracker.offset_at(at(0.5)) == pytest.approx(TRANSMIT_HZ - CENTER_HZ)
        assert tracker.offset_at(at(5.0)) == pytest.approx(TRANSMIT_HZ - CENTER_HZ)

    def test_two_samples_extrapolate_linearly(self):
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)
        tracker.update(at(0.0), 0.0)
        tracker.update(at(1.0), 1.0)

        # The received frequency moved by one "km/s worth" per second, so
        # half a second past the newest sample it should have moved half
        # that again beyond it.
        f0 = tracker.frequency_at(at(1.0))
        f1 = tracker.frequency_at(at(1.5))
        step = TRANSMIT_HZ * (1.0 / C_KM_S)

        assert f1 - f0 == pytest.approx(-step * 0.5, rel=1e-6)

    def test_only_the_two_newest_samples_are_kept(self):
        # An hour-long pass at 1 Hz would otherwise accumulate thousands
        # of samples for a two-point fit.
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)
        for i in range(50):
            tracker.update(at(float(i)), float(i) * 0.1)

        assert tracker.stats.updates == 50
        assert len(tracker._samples) == 2  # noqa: SLF001 - the bound is the point

    def test_two_samples_at_the_same_instant_fall_back_rather_than_dividing_by_zero(self):
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)
        tracker.update(at(1.0), 2.0)
        tracker.update(at(1.0), 2.0)

        assert np.isfinite(tracker.offset_at(at(1.5)))


class TestStaleness:
    def test_extrapolation_is_capped_and_counted(self):
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ, max_extrapolation_s=2.0)
        tracker.update(at(0.0), 0.0)
        tracker.update(at(1.0), 1.0)

        capped = tracker.frequency_at(at(1.0 + 2.0))
        way_past = tracker.frequency_at(at(1.0 + 60.0))

        assert way_past == pytest.approx(capped)
        assert tracker.stats.stale_queries == 1  # frequency_at counts, offset_at wraps it

    def test_a_fresh_query_is_not_stale(self):
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)
        tracker.update(at(0.0), 0.0)
        tracker.update(at(1.0), 1.0)
        tracker.offset_at(at(1.5))

        assert tracker.stats.stale_queries == 0
        assert tracker.is_stale_at(at(1.5)) is False

    def test_is_stale_at_flags_a_stalled_loop_without_waiting_for_the_stats(self):
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)
        tracker.update(at(0.0), 0.0)

        assert tracker.is_stale_at(at(DEFAULT_MAX_EXTRAPOLATION_S + 1.0)) is True

    def test_a_tracker_with_no_samples_reads_as_stale(self):
        assert DopplerTracker(TRANSMIT_HZ, CENTER_HZ).is_stale_at(at(0.0)) is True


class TestDopplerAccounting:
    def test_a_fresh_tracker_reports_nothing(self):
        stats = DopplerTracker(TRANSMIT_HZ, CENTER_HZ).stats

        assert isinstance(stats, DopplerStats)
        assert stats.updates == 0
        assert stats.last_offset_hz is None
        assert "never queried" in stats.describe()

    def test_the_offset_range_brackets_the_pass(self):
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)
        for i, rr in enumerate((-7.0, 0.0, 7.0)):
            tracker.update(at(float(i)), rr)
            tracker.offset_at(at(float(i)))

        stats = tracker.stats
        assert stats.min_offset_hz < stats.max_offset_hz
        # Approaching gives a higher frequency than receding, so the span
        # should be roughly twice the one-way shift at 7 km/s.
        one_way = TRANSMIT_HZ * 7.0 / C_KM_S
        assert stats.max_offset_hz - stats.min_offset_hz == pytest.approx(2 * one_way, rel=0.2)

    def test_describe_calls_out_a_stalled_loop_in_words(self):
        # Silence about a stale correction is the failure mode this
        # counter exists for, so the report has to say it, not imply it.
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ, max_extrapolation_s=1.0)
        tracker.update(at(0.0), 0.0)
        tracker.update(at(1.0), 1.0)
        tracker.offset_at(at(30.0))

        assert "STALE" in tracker.stats.describe()

    def test_stale_queries_can_never_exceed_queries(self):
        # An earlier version counted queries in offset_at() and stale
        # queries in frequency_at(), which offset_at() calls -- so a
        # caller mixing the two got "4 of 1 query(ies) ran on a STALE
        # range rate". Nonsense output in the one report that exists to
        # be trusted when something has gone wrong.
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ, max_extrapolation_s=1.0)
        tracker.update(at(0.0), 0.0)
        tracker.update(at(1.0), 1.0)

        for _ in range(3):
            tracker.frequency_at(at(30.0))
        tracker.offset_at(at(30.0))

        stats = tracker.stats
        assert stats.queries == 4
        assert stats.stale_queries <= stats.queries

    def test_each_public_query_is_counted_exactly_once(self):
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)
        tracker.update(at(0.0), 0.0)

        tracker.frequency_at(at(0.1))
        tracker.offset_at(at(0.2))

        assert tracker.stats.queries == 2

    def test_describe_says_so_when_everything_was_fresh(self):
        tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)
        tracker.update(at(0.0), 0.0)
        tracker.update(at(1.0), 1.0)
        tracker.offset_at(at(1.5))

        assert "all queries used a fresh range rate" in tracker.stats.describe()


# ----------------------------------------------------------------------
# The synthetic-Doppler round trip
# ----------------------------------------------------------------------

RATE_HZ = 256_000.0
BLOCK = 16_384  # 64 ms, the same cadence a real 256 KiB IQ block has
IF_RATE_HZ = 32_000.0
TONE_HZ = 1_000.0
DEVIATION_HZ = 5_000.0

#: Long enough to span several tracking-loop ticks, so the extrapolation
#: has a slope to work from rather than a single held sample. 48 blocks of
#: 64 ms is about 3.1 s, which at ~15.6 blocks/s is three ticks.
NBLOCKS = 48

#: Where in the pass the window sits, in seconds after closest approach.
#: Deliberately **not** centred on TCA: the range-rate profile is
#: antisymmetric there, so a window straddling it has a mean range rate of
#: about zero — and a correction applied with the wrong sign would produce
#: the same near-zero mean error as a correct one. Offsetting the window
#: gives a consistently receding satellite, which is what makes a sign flip
#: show up as a large error rather than cancelling itself out.
WINDOW_START_S = 10.0


def swept_nbfm(range_rate: np.ndarray) -> np.ndarray:
    """An NBFM signal at baseband, swept by the Doppler of ``range_rate``.

    Built the way the receiver will see it: the satellite transmits at
    TRANSMIT_HZ, Doppler moves it, and the tuner sits at CENTER_HZ — so
    what lands in the capture is the difference between the two.
    """
    n = range_rate.shape[0]
    message = np.sin(2 * np.pi * TONE_HZ * np.arange(n) / RATE_HZ)
    fm_phase = 2 * np.pi * DEVIATION_HZ * np.cumsum(message) / RATE_HZ
    received_hz = TRANSMIT_HZ * (1.0 - range_rate / C_KM_S)
    offset_phase = 2 * np.pi * np.cumsum(received_hz - CENTER_HZ) / RATE_HZ
    return np.exp(1j * (fm_phase + offset_phase)).astype(np.complex64)


def demodulate_pass(iq: np.ndarray, range_rate: np.ndarray, *, flip_sign: bool = False):
    """Run a swept signal through the tracker and the demodulator.

    Feeds the tracker one range rate per second, exactly as the tracking
    loop would, and asks it for a fresh offset at each block's midpoint.
    """
    tracker = DopplerTracker(TRANSMIT_HZ, CENTER_HZ)
    base = NbfmConfig(
        sample_rate_hz=RATE_HZ,
        if_rate_hz=IF_RATE_HZ,
        deviation_hz=DEVIATION_HZ,
        de_emphasis_us=None,
    )
    audio = []
    next_tick_s = 0.0
    for b in range(NBLOCKS):
        block_start_s = b * BLOCK / RATE_HZ
        # Feed on a time basis, not a block count: 256 kHz / 16,384 is
        # 15.625 blocks per second, and truncating that to 15 would drift
        # the tick cadence against the block cadence over a long pass.
        if block_start_s >= next_tick_s:
            rr = float(range_rate[b * BLOCK])
            tracker.update(at(block_start_s), -rr if flip_sign else rr)
            next_tick_s += 1.0
        mid_s = (b * BLOCK + BLOCK // 2) / RATE_HZ
        config = NbfmConfig(
            sample_rate_hz=RATE_HZ,
            if_rate_hz=IF_RATE_HZ,
            deviation_hz=DEVIATION_HZ,
            de_emphasis_us=None,
            channel_offset_hz=tracker.offset_at(at(mid_s)),
        )
        assert config.audio_rate_hz == base.audio_rate_hz
        audio.append(demodulate_nbfm(iq[b * BLOCK : (b + 1) * BLOCK], config))
    return np.concatenate(audio), tracker


def mean_frequency_error_hz(audio: np.ndarray) -> float:
    """Residual tuning error, in Hz: the discriminator's DC term times deviation."""
    return float(audio.mean() * DEVIATION_HZ)


class TestSyntheticDopplerRoundTrip:
    """Sweep a signal along a real pass profile, correct it, get the tone back."""

    def setup_method(self):
        n = NBLOCKS * BLOCK
        t_s = WINDOW_START_S + np.arange(n) / RATE_HZ
        self.range_rate = orbital_range_rate(t_s)
        self.iq = swept_nbfm(self.range_rate)

    def test_the_profile_is_a_real_one(self):
        # Guards the test's own premise twice over: a flat profile would
        # make every assertion below pass trivially, and a profile with a
        # near-zero mean would make the sign-flip tests toothless.
        shift_hz = TRANSMIT_HZ * self.range_rate / C_KM_S
        assert abs(shift_hz).max() > 1_000.0
        assert abs(shift_hz.max() - shift_hz.min()) > 300.0
        assert self.range_rate.mean() > 1.0

    def test_the_corrected_signal_recovers_the_modulating_tone(self):
        audio, _ = demodulate_pass(self.iq, self.range_rate)

        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / IF_RATE_HZ)
        peak_hz = freqs[np.argmax(spectrum)]

        assert peak_hz == pytest.approx(TONE_HZ, abs=IF_RATE_HZ / len(audio) * 3)

    def test_the_residual_tuning_error_is_small(self):
        audio, _ = demodulate_pass(self.iq, self.range_rate)

        assert abs(mean_frequency_error_hz(audio)) < 50.0

    def test_flipping_the_range_rate_sign_is_measurably_worse(self):
        # THE test this whole PR exists for. The magnitude of the error is
        # identical either way, so only a comparison catches it.
        right, _ = demodulate_pass(self.iq, self.range_rate)
        wrong, _ = demodulate_pass(self.iq, self.range_rate, flip_sign=True)

        assert abs(mean_frequency_error_hz(wrong)) > 10 * abs(mean_frequency_error_hz(right))

    def test_flipping_the_sign_doubles_the_offset_instead_of_cancelling_it(self):
        """Stronger than "worse": it pins the *direction* of the failure.

        A receding satellite arrives low, so a correct correction shifts
        the signal back **up**. A flipped one shifts it further **down**,
        leaving a residual with the same sign as the uncorrected offset
        and roughly twice its size.

        The bound is deliberately loose on magnitude and strict on sign.
        The naive expectation is exactly 2x the Doppler shift, but the
        measured residual comes back at about 70% of that, because at
        4.3 kHz off-centre the signal is partly cut by the channel filter
        and the discriminator output is at 0.86 of full scale and starting
        to clip. Both are real effects of a badly mistuned receiver, so
        asserting the naive 2x would be asserting something false. Do not
        "tighten" this to an exact multiple.

        (The first version of this test asserted the wrong sign — in the
        PR whose entire purpose is preventing sign errors. Left recorded
        here rather than quietly fixed, because it is the argument for
        :func:`~qsorbit.core.doppler.downlink_receive_frequency` existing
        at all: the direction is genuinely easy to get backwards, even
        while concentrating on it.)
        """
        wrong, _ = demodulate_pass(self.iq, self.range_rate, flip_sign=True)
        wrong_hz = mean_frequency_error_hz(wrong)
        one_shift_hz = TRANSMIT_HZ * float(np.mean(self.range_rate)) / C_KM_S

        assert one_shift_hz > 0.0, "window should have a receding satellite"
        assert wrong_hz < 0.0, "a flipped correction must push the error further down, not up"
        assert abs(wrong_hz) > one_shift_hz

    def test_no_correction_at_all_is_worse_than_correcting(self):
        # The other negative-space direction: proves the correction is
        # doing work, not that the profile happened to be gentle.
        corrected, _ = demodulate_pass(self.iq, self.range_rate)

        config = NbfmConfig(
            sample_rate_hz=RATE_HZ,
            if_rate_hz=IF_RATE_HZ,
            deviation_hz=DEVIATION_HZ,
            de_emphasis_us=None,
            channel_offset_hz=TRANSMIT_HZ - CENTER_HZ,  # nominal, never updated
        )
        uncorrected = np.concatenate(
            [demodulate_nbfm(self.iq[b * BLOCK : (b + 1) * BLOCK], config) for b in range(NBLOCKS)]
        )

        assert abs(mean_frequency_error_hz(uncorrected)) > abs(mean_frequency_error_hz(corrected))

    def test_the_tracker_reports_a_correction_span_matching_the_pass(self):
        _, tracker = demodulate_pass(self.iq, self.range_rate)
        stats = tracker.stats

        expected_span = TRANSMIT_HZ * (self.range_rate.max() - self.range_rate.min()) / C_KM_S
        assert stats.max_offset_hz - stats.min_offset_hz == pytest.approx(expected_span, rel=0.25)
        assert stats.stale_queries == 0
