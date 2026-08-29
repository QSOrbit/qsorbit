"""Tests for the frequency card's wording. No Qt.

Same shape as ``test_readout_formatting`` and
``test_quieting_formatting``: the decisions about what a display *says*
are the part worth testing, and they should not need a window.
"""

from __future__ import annotations

import pytest

from qsorbit.ui.frequency_formatting import (
    AWAITING_LABEL,
    NO_READING,
    frequency_text,
)


def test_before_the_first_sample_nothing_is_claimed():
    """A zero would be a number somebody can believe.

    ``live_tracked_frequency_hz`` is ``None`` until the tracker has been
    fed once, and there is no honest frequency to report before then --
    so the card shows a placeholder and says what it is waiting for,
    rather than a plausible reading that is not a measurement.
    """
    text = frequency_text(None, 435_600_000.0)
    assert text.megahertz == NO_READING
    assert text.hertz == ""
    assert text.doppler == AWAITING_LABEL
    assert text.role == "dim"


def test_the_split_falls_at_the_kilohertz():
    text = frequency_text(435_605_213.0, None)
    assert text.megahertz == "435.605"
    assert text.hertz == ".213 MHz"


def test_the_two_halves_are_split_after_rounding_not_before():
    """Otherwise the display shows a frequency that was never true.

    Formatting each half independently lets the megahertz field round
    up while the hertz field still shows the digits it rounded past --
    a reading a kilohertz wrong in exactly the digit an operator is
    watching move.
    """
    text = frequency_text(435_605_999.6, None)
    assert text.megahertz == "435.606"
    assert text.hertz == ".000 MHz"


def test_a_rising_downlink_gets_an_up_arrow_and_a_signed_figure():
    text = frequency_text(435_604_213.0, 435_600_000.0)
    assert text.doppler.startswith("▲")
    assert "+4,213 Hz" in text.doppler
    assert "corrected" in text.doppler
    assert text.role == "accent"


def test_a_falling_downlink_gets_a_down_arrow():
    text = frequency_text(435_595_787.0, 435_600_000.0)
    assert text.doppler.startswith("▼")
    assert "-4,213 Hz" in text.doppler


def test_zero_doppler_still_reads_as_the_correction_working():
    """A geostationary target is the station's own proving signal.

    Session 27's bench runs used a geostationary TLE precisely so the
    Doppler span would be a measured no-op, and the convention is to
    prove the chain against a signal that cannot be absent before
    pointing at one that can. This line therefore has to read as
    success, not as a tracker that has not started.
    """
    text = frequency_text(435_600_000.2, 435_600_000.0)
    assert text.doppler == "0 Hz doppler - corrected"
    assert text.role == "accent"
    assert text.megahertz == "435.600"


def test_without_a_nominal_frequency_no_shift_is_claimed():
    """A replayed capture, or a bench run on an ISM frequency."""
    text = frequency_text(433_920_000.0, None)
    assert text.megahertz == "433.920"
    assert "no shift is claimed" in text.doppler
    assert text.role == "dim"


@pytest.mark.parametrize(
    ("hz", "megahertz", "hertz"),
    [
        (0.0, "0.000", ".000 MHz"),
        (1_000.0, "0.001", ".000 MHz"),
        (999.0, "0.000", ".999 MHz"),
        (137_100_000.0, "137.100", ".000 MHz"),
        (1_700_000_000.0, "1,700.000", ".000 MHz"),
    ],
)
def test_the_split_holds_across_the_whole_tuning_range(hz, megahertz, hertz):
    """0 Hz to 1.7 GHz -- the RTL-SDR's range, plus both ends of it."""
    text = frequency_text(hz, None)
    assert (text.megahertz, text.hertz) == (megahertz, hertz)
