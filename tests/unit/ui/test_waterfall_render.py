"""Tests for the waterfall's pixel logic.

All headless — this module imports no Qt, which is the point of the
split. Rendering itself (does it look right on screen) is verified by
eyes at the bench, per the Chunk F spec; what is testable here is every
decision that turns numbers into pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsorbit.ui.waterfall_render import (
    WaterfallScale,
    bins_to_pixels,
    blank_row,
    colorize,
    db_to_index,
    frequency_ticks,
    render_row,
    tick_position,
)

# ----------------------------------------------------------------------
# WaterfallScale
# ----------------------------------------------------------------------


def test_default_scale_brackets_a_real_capture():
    """Defaults must contain the measured noise floor and carrier.

    Bounded from both sides, per Session 16: a scale assertion with only
    one bound passes when the window collapses or runs away.
    """
    scale = WaterfallScale()
    measured_noise_floor_db = -66.8
    measured_carrier_db = -40.0
    assert scale.floor_db < measured_noise_floor_db
    assert scale.ceiling_db > measured_carrier_db
    assert scale.span_db == pytest.approx(70.0)


def test_scale_rejects_an_inverted_window():
    with pytest.raises(ValueError, match="above floor_db"):
        WaterfallScale(floor_db=-20.0, ceiling_db=-90.0)


def test_scale_rejects_a_zero_width_window():
    with pytest.raises(ValueError, match="above floor_db"):
        WaterfallScale(floor_db=-50.0, ceiling_db=-50.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_scale_rejects_non_finite_bounds(bad):
    with pytest.raises(ValueError):
        WaterfallScale(floor_db=bad)


# ----------------------------------------------------------------------
# db_to_index
# ----------------------------------------------------------------------


def test_floor_and_ceiling_map_to_the_ends_of_the_ramp():
    scale = WaterfallScale(floor_db=-90.0, ceiling_db=-20.0)
    ends = db_to_index(np.array([-90.0, -20.0]), scale)
    assert ends[0] == 0
    assert ends[1] == 255


def test_values_outside_the_window_clamp_rather_than_wrap():
    """Wrapping would draw a strong signal as black. Worth pinning."""
    scale = WaterfallScale(floor_db=-90.0, ceiling_db=-20.0)
    out = db_to_index(np.array([-200.0, 0.0, 1e6]), scale)
    assert out[0] == 0
    assert out[1] == 255
    assert out[2] == 255


def test_midpoint_lands_in_the_middle():
    scale = WaterfallScale(floor_db=-90.0, ceiling_db=-20.0)
    assert db_to_index(np.array([-55.0]), scale)[0] == pytest.approx(127, abs=1)


def test_index_output_is_uint8():
    scale = WaterfallScale()
    assert db_to_index(np.linspace(-120.0, 0.0, 64), scale).dtype == np.uint8


# ----------------------------------------------------------------------
# bins_to_pixels — the max-hold decision
# ----------------------------------------------------------------------


def test_a_single_bin_carrier_survives_downsampling():
    """The test that fails if someone swaps max-hold for an average.

    This is the behaviour the whole module exists for: a satellite
    downlink can occupy one bin in two thousand, and it has to still be
    visible after those bins are squeezed onto a panel's pixels.
    """
    bins = np.full(2048, -90.0)
    bins[1000] = -30.0
    pixels = bins_to_pixels(bins, 800)
    assert pixels.max() == pytest.approx(-30.0)


def test_averaging_would_have_lost_that_carrier():
    """Shows the counterfactual, so the previous test's point is explicit.

    Not testing our code — testing that the alternative really is worse,
    so nobody 'simplifies' max-hold into a mean believing it equivalent.
    """
    bins = np.full(2048, -90.0)
    bins[1000] = -30.0
    span = 2048 // 800  # bins collapsed into one pixel
    averaged_peak = ((-30.0) + (-90.0) * (span - 1)) / span
    assert averaged_peak < -50.0  # buried, versus max-hold's -30.0
    assert bins_to_pixels(bins, 800).max() > averaged_peak + 20.0


def test_downsampling_produces_exactly_the_requested_width():
    assert bins_to_pixels(np.zeros(2048), 800).shape == (800,)


def test_every_bin_is_covered_by_some_pixel():
    """No bin may fall between spans — a gap is a blind spot in the panel."""
    bins = np.full(1000, -100.0)
    for index in range(1000):
        probe = bins.copy()
        probe[index] = 0.0
        assert bins_to_pixels(probe, 97).max() == pytest.approx(0.0), index


def test_upsampling_repeats_rather_than_interpolates():
    """Interpolation would invent values between real measurements."""
    bins = np.array([-90.0, -30.0])
    pixels = bins_to_pixels(bins, 8)
    assert set(np.unique(pixels)) == {-90.0, -30.0}


def test_width_equal_to_bin_count_is_the_identity():
    bins = np.linspace(-90.0, -20.0, 64)
    assert np.allclose(bins_to_pixels(bins, 64), bins)


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_an_unusable_width(bad):
    with pytest.raises(ValueError, match="width"):
        bins_to_pixels(np.zeros(64), bad)


def test_rejects_an_empty_frame():
    with pytest.raises(ValueError, match="non-empty"):
        bins_to_pixels(np.zeros(0), 8)


# ----------------------------------------------------------------------
# colorize
# ----------------------------------------------------------------------


def test_colorize_shapes_indices_into_rgb(colormap):
    out = colorize(np.array([0, 128, 255], dtype=np.uint8), colormap)
    assert out.shape == (3, 3)
    assert out.dtype == np.uint8


# ----------------------------------------------------------------------
# render_row
# ----------------------------------------------------------------------


def test_render_row_produces_one_rgb_row_of_the_requested_width(colormap):
    row = render_row(np.full(2048, -70.0), 640, WaterfallScale(), colormap)
    assert row.shape == (640, 3)
    assert row.dtype == np.uint8


def test_render_row_draws_a_carrier_brighter_than_its_noise_floor(colormap):
    """End to end, on the numbers a real capture actually produces."""
    bins = np.full(2048, -66.8)
    bins[1024] = -40.0
    row = render_row(bins, 800, WaterfallScale(), colormap)

    luminance = row.astype(np.float64).sum(axis=1)
    carrier_pixel = int(np.argmax(luminance))
    assert luminance[carrier_pixel] > np.median(luminance) + 60.0
    # And it lands where the bin was, not somewhere else.
    assert abs(carrier_pixel - (1024 * 800) // 2048) <= 1


def test_render_row_of_a_silent_frame_is_uniformly_the_ramp_floor(colormap):
    """Silence is the bottom of *this theme's* ramp, which is not always dark.

    This assertion used to read ``row.max() <= 8`` -- "silent means
    nearly black" -- which was true only because the ramp was a module
    constant starting at pure black. Daylight's ramp floor is
    ``#f7f9fc``, so under a theme system the old assertion was testing
    the wrong property and would have failed on a perfectly correct
    light theme. The property that actually holds everywhere is exact
    rather than approximate: every pixel of a silent frame is the
    colormap's own floor colour.
    """
    row = render_row(np.full(1024, -200.0), 400, WaterfallScale(), colormap)
    expected = np.array(colormap.floor_color, dtype=np.uint8)
    assert np.array_equal(np.unique(row.reshape(-1, 3), axis=0), expected[np.newaxis, :])


# ----------------------------------------------------------------------
# Frequency axis
# ----------------------------------------------------------------------


def test_ticks_land_on_round_megahertz_for_a_real_capture_span():
    """The actual bench span: 2.048 MHz centred on 99.65 MHz."""
    start, stop = 99_650_000.0 - 1_024_000.0, 99_650_000.0 + 1_024_000.0
    ticks = frequency_ticks(start, stop)
    labels = [label for _, label in ticks]
    assert "99.5" in labels
    assert "100.0" in labels
    assert all(start <= hz <= stop for hz, _ in ticks)


def test_ticks_stay_within_the_requested_count():
    start, stop = 99_650_000.0 - 1_024_000.0, 99_650_000.0 + 1_024_000.0
    for max_ticks in (3, 4, 6, 8, 12):
        assert len(frequency_ticks(start, stop, max_ticks)) <= max_ticks


def test_ticks_are_ordered_and_evenly_spaced():
    ticks = frequency_ticks(144_000_000.0, 146_000_000.0)
    frequencies = [hz for hz, _ in ticks]
    assert frequencies == sorted(frequencies)
    gaps = np.diff(frequencies)
    assert np.allclose(gaps, gaps[0])


def test_labels_carry_enough_decimals_to_be_distinct():
    """A step finer than the label's precision produces repeated labels.

    The failure this catches is quiet and ugly: an axis reading
    '99.2  99.2  99.5  99.5' looks like a rendering glitch rather than a
    formatting bug, and nothing else would flag it.
    """
    for span_hz in (2_048_000.0, 500_000.0, 200_000.0, 48_000.0, 2_000_000_000.0):
        ticks = frequency_ticks(100_000_000.0, 100_000_000.0 + span_hz)
        labels = [label for _, label in ticks]
        assert len(labels) == len(set(labels)), (span_hz, labels)


def test_labels_do_not_trail_pointless_zeros():
    ticks = frequency_ticks(99_000_000.0, 101_000_000.0)
    assert all(not label.endswith("00") or "." not in label for _, label in ticks)


def test_a_narrow_span_still_produces_usable_ticks():
    """A zoomed-in NBFM channel is only tens of kHz wide."""
    ticks = frequency_ticks(162_540_000.0, 162_560_000.0)
    assert len(ticks) >= 2
    assert len({label for _, label in ticks}) == len(ticks)


def test_tick_position_maps_the_edges_and_centre():
    assert tick_position(100.0, 100.0, 200.0, 800) == pytest.approx(0.0)
    assert tick_position(200.0, 100.0, 200.0, 800) == pytest.approx(800.0)
    assert tick_position(150.0, 100.0, 200.0, 800) == pytest.approx(400.0)


def test_ticks_reject_an_inverted_or_empty_span():
    with pytest.raises(ValueError, match="above start_hz"):
        frequency_ticks(100.0, 100.0)
    with pytest.raises(ValueError, match="above start_hz"):
        frequency_ticks(200.0, 100.0)


def test_ticks_reject_a_useless_tick_budget():
    with pytest.raises(ValueError, match="max_ticks"):
        frequency_ticks(100.0, 200.0, 1)


def test_blank_row_is_indistinguishable_from_genuine_silence(colormap):
    """A pre-filled row must not read as a seam against real quiet data."""
    silent = render_row(np.full(512, -1e6), 320, WaterfallScale(), colormap)
    assert np.array_equal(blank_row(320, colormap), silent)


def test_blank_row_has_the_right_shape_and_dtype(colormap):
    row = blank_row(640, colormap)
    assert row.shape == (640, 3)
    assert row.dtype == np.uint8


def test_blank_row_rejects_an_unusable_width(colormap):
    with pytest.raises(ValueError, match="width"):
        blank_row(0, colormap)
