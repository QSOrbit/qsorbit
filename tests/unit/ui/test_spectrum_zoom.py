"""Tests for the shared pan/zoom arithmetic behind the waterfall and the
new spectrum-line panel.

Headless, like ``test_waterfall_render.py`` - no Qt import here. What's
testable in this module is every decision about which slice of a
captured frame is "on screen": span/center validation, clamping to the
captured band, the actual-vs-requested edges ``visible_slice`` reports,
and where the tuner's DC spike falls relative to the visible window.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsorbit.ui.spectrum_zoom import (
    MIN_ZOOM_SPAN_HZ,
    ZoomSpan,
    clamp_zoom,
    dc_spike_in_view,
    full_band_zoom,
    next_zoom_for_follow,
    next_zoom_for_pan,
    next_zoom_for_scroll,
    next_zoom_for_span,
    rescale_around,
    visible_slice,
    zoom_spanning,
)

# ----------------------------------------------------------------------
# ZoomSpan
# ----------------------------------------------------------------------


def test_start_and_stop_bracket_the_center():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=2_000_000.0)
    assert zoom.start_hz == pytest.approx(99_000_000.0)
    assert zoom.stop_hz == pytest.approx(101_000_000.0)


def test_rejects_a_non_positive_span():
    with pytest.raises(ValueError, match="span_hz"):
        ZoomSpan(center_hz=0.0, span_hz=0.0)
    with pytest.raises(ValueError, match="span_hz"):
        ZoomSpan(center_hz=0.0, span_hz=-1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_rejects_a_non_finite_span(bad):
    with pytest.raises(ValueError, match="span_hz"):
        ZoomSpan(center_hz=0.0, span_hz=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_rejects_a_non_finite_center(bad):
    with pytest.raises(ValueError, match="center_hz"):
        ZoomSpan(center_hz=bad, span_hz=1_000.0)


# ----------------------------------------------------------------------
# full_band_zoom
# ----------------------------------------------------------------------


def test_full_band_zoom_spans_the_whole_axis():
    axis_hz = np.linspace(99_000_000.0, 101_000_000.0, 2048)
    zoom = full_band_zoom(axis_hz)
    assert zoom.start_hz == pytest.approx(axis_hz[0])
    assert zoom.stop_hz == pytest.approx(axis_hz[-1])
    assert zoom.center_hz == pytest.approx(100_000_000.0)


def test_full_band_zoom_on_a_single_bin_axis_is_degenerate_but_does_not_raise():
    """A one-bin axis has zero width - clamp_zoom is what protects against
    that downstream, not full_band_zoom itself. Documenting the boundary
    rather than assuming it can't happen."""
    axis_hz = np.array([100_000_000.0])
    with pytest.raises(ValueError, match="span_hz"):
        full_band_zoom(axis_hz)


# ----------------------------------------------------------------------
# clamp_zoom
# ----------------------------------------------------------------------


def test_clamp_leaves_a_zoom_already_inside_the_band_untouched():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=30_000.0)
    clamped = clamp_zoom(zoom, 99_000_000.0, 101_000_000.0)
    assert clamped == zoom


def test_clamp_pulls_a_span_wider_than_the_band_down_to_the_band_width():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=10_000_000.0)
    clamped = clamp_zoom(zoom, 99_000_000.0, 101_000_000.0)
    assert clamped.span_hz == pytest.approx(2_000_000.0)


def test_clamp_enforces_the_minimum_span():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=1.0)
    clamped = clamp_zoom(zoom, 99_000_000.0, 101_000_000.0)
    assert clamped.span_hz == pytest.approx(MIN_ZOOM_SPAN_HZ)


def test_clamp_uses_a_caller_supplied_minimum_span():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=1.0)
    clamped = clamp_zoom(zoom, 99_000_000.0, 101_000_000.0, min_span_hz=500.0)
    assert clamped.span_hz == pytest.approx(500.0)


def test_clamp_pulls_a_center_beyond_the_low_edge_back_inside():
    zoom = ZoomSpan(center_hz=98_000_000.0, span_hz=200_000.0)
    clamped = clamp_zoom(zoom, 99_000_000.0, 101_000_000.0)
    assert clamped.start_hz == pytest.approx(99_000_000.0)
    assert clamped.span_hz == pytest.approx(200_000.0)


def test_clamp_pulls_a_center_beyond_the_high_edge_back_inside():
    zoom = ZoomSpan(center_hz=102_000_000.0, span_hz=200_000.0)
    clamped = clamp_zoom(zoom, 99_000_000.0, 101_000_000.0)
    assert clamped.stop_hz == pytest.approx(101_000_000.0)
    assert clamped.span_hz == pytest.approx(200_000.0)


def test_clamp_span_wins_over_a_center_that_would_no_longer_fit():
    """A wide-but-valid span combined with an edge-hugging center: span
    is honoured (clamped to the band width first) and center is then
    pulled to whatever still fits that span - not the other way round,
    per the function's own documented ordering."""
    zoom = ZoomSpan(center_hz=101_000_000.0, span_hz=1_500_000.0)
    clamped = clamp_zoom(zoom, 99_000_000.0, 101_000_000.0)
    assert clamped.span_hz == pytest.approx(1_500_000.0)
    assert clamped.stop_hz <= 101_000_000.0 + 1e-6
    assert clamped.start_hz >= 99_000_000.0 - 1e-6


def test_clamp_rejects_an_inverted_or_zero_width_band():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=1_000.0)
    with pytest.raises(ValueError, match="above band_start_hz"):
        clamp_zoom(zoom, 100_000_000.0, 100_000_000.0)
    with pytest.raises(ValueError, match="above band_start_hz"):
        clamp_zoom(zoom, 101_000_000.0, 99_000_000.0)


# ----------------------------------------------------------------------
# visible_slice
# ----------------------------------------------------------------------


def test_visible_slice_keeps_only_bins_inside_the_zoom():
    axis_hz = np.linspace(99_000_000.0, 101_000_000.0, 2001)  # 1 kHz/bin
    power_db = np.full(2001, -90.0)
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=10_000.0)
    sliced, start_hz, stop_hz = visible_slice(power_db, axis_hz, zoom)
    assert sliced.shape[0] < axis_hz.shape[0]
    assert start_hz >= zoom.start_hz - 1_000.0
    assert stop_hz <= zoom.stop_hz + 1_000.0
    assert start_hz <= stop_hz


def test_visible_slice_reports_actual_bin_edges_not_the_requested_ones():
    """Bins land at fixed frequencies; a zoom asking for an edge between
    two bins should get back whichever bin was actually kept."""
    axis_hz = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    power_db = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    zoom = ZoomSpan(center_hz=120.0, span_hz=17.0)  # 111.5 .. 128.5
    sliced, start_hz, stop_hz = visible_slice(power_db, axis_hz, zoom)
    assert start_hz == 120.0
    assert stop_hz == 120.0
    assert list(sliced) == [3.0]


def test_visible_slice_covering_the_whole_band_returns_everything():
    axis_hz = np.linspace(99_000_000.0, 101_000_000.0, 512)
    power_db = np.linspace(-90.0, -20.0, 512)
    zoom = full_band_zoom(axis_hz)
    sliced, start_hz, stop_hz = visible_slice(power_db, axis_hz, zoom)
    assert sliced.shape == power_db.shape
    assert np.array_equal(sliced, power_db)
    assert start_hz == pytest.approx(axis_hz[0])
    assert stop_hz == pytest.approx(axis_hz[-1])


def test_visible_slice_always_returns_at_least_one_bin():
    """A zoom entirely below or above the band must not slice to nothing -
    the caller still needs something to draw."""
    axis_hz = np.linspace(99_000_000.0, 101_000_000.0, 100)
    power_db = np.zeros(100)
    below = ZoomSpan(center_hz=1_000_000.0, span_hz=10_000.0)
    sliced, _, _ = visible_slice(power_db, axis_hz, below)
    assert sliced.shape[0] >= 1

    above = ZoomSpan(center_hz=200_000_000.0, span_hz=10_000.0)
    sliced, _, _ = visible_slice(power_db, axis_hz, above)
    assert sliced.shape[0] >= 1


def test_visible_slice_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        visible_slice(np.zeros(10), np.zeros(11), ZoomSpan(center_hz=0.0, span_hz=1.0))


def test_visible_slice_rejects_an_empty_axis():
    with pytest.raises(ValueError, match="must not be empty"):
        visible_slice(np.zeros(0), np.zeros(0), ZoomSpan(center_hz=0.0, span_hz=1.0))


# ----------------------------------------------------------------------
# dc_spike_in_view
# ----------------------------------------------------------------------


def test_dc_spike_reported_when_inside_the_window():
    assert dc_spike_in_view(100_000_000.0, 99_000_000.0, 101_000_000.0) == pytest.approx(
        100_000_000.0
    )


def test_dc_spike_reported_at_the_exact_edges():
    assert dc_spike_in_view(99_000_000.0, 99_000_000.0, 101_000_000.0) == pytest.approx(
        99_000_000.0
    )
    assert dc_spike_in_view(101_000_000.0, 99_000_000.0, 101_000_000.0) == pytest.approx(
        101_000_000.0
    )


def test_dc_spike_absent_when_outside_the_window():
    """The common case once locked on a tracked downlink: QSOrbit tunes
    with an offset specifically to keep the spike out of view."""
    assert dc_spike_in_view(100_000_000.0, 100_100_000.0, 100_200_000.0) is None
    assert dc_spike_in_view(100_000_000.0, 99_800_000.0, 99_900_000.0) is None


# ----------------------------------------------------------------------
# zoom_spanning
# ----------------------------------------------------------------------


def test_zoom_spanning_covers_exactly_the_given_edges():
    zoom = zoom_spanning(99_000_000.0, 101_000_000.0)
    assert zoom.start_hz == pytest.approx(99_000_000.0)
    assert zoom.stop_hz == pytest.approx(101_000_000.0)
    assert zoom.center_hz == pytest.approx(100_000_000.0)


def test_full_band_zoom_agrees_with_zoom_spanning_on_the_same_edges():
    axis_hz = np.linspace(99_000_000.0, 101_000_000.0, 2048)
    assert full_band_zoom(axis_hz) == zoom_spanning(float(axis_hz[0]), float(axis_hz[-1]))


# ----------------------------------------------------------------------
# rescale_around
# ----------------------------------------------------------------------


def test_rescale_around_the_center_leaves_the_center_fixed():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=2_000_000.0)
    rescaled = rescale_around(zoom, 0.5)
    assert rescaled.center_hz == pytest.approx(100_000_000.0)
    assert rescaled.span_hz == pytest.approx(1_000_000.0)


def test_rescale_around_an_off_center_anchor_keeps_its_fractional_position():
    """The point under the cursor has to stay under the cursor - checked
    as the anchor's fraction of the way from the low edge to the high
    edge, which a correct scroll-to-zoom keeps invariant."""
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=2_000_000.0)  # 99e6..101e6
    anchor_hz = 99_500_000.0  # a quarter of the way in

    def fraction(z):
        return (anchor_hz - z.start_hz) / z.span_hz

    before = fraction(zoom)
    rescaled = rescale_around(zoom, 0.5, anchor_hz)
    after = fraction(rescaled)
    assert before == pytest.approx(0.25)
    assert after == pytest.approx(before)
    assert rescaled.span_hz == pytest.approx(1_000_000.0)


def test_rescale_around_zooming_out_widens_the_span():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=1_000_000.0)
    rescaled = rescale_around(zoom, 2.0)
    assert rescaled.span_hz == pytest.approx(2_000_000.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_rescale_around_rejects_a_non_positive_or_non_finite_factor(bad):
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=1_000_000.0)
    with pytest.raises(ValueError, match="factor"):
        rescale_around(zoom, bad)


# ----------------------------------------------------------------------
# next_zoom_for_span
# ----------------------------------------------------------------------


def test_next_zoom_for_span_keeps_the_center_and_changes_the_span():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=500_000.0)
    result = next_zoom_for_span(zoom, 99_000_000.0, 101_000_000.0, 100_000.0)
    assert result.center_hz == pytest.approx(100_000_000.0)
    assert result.span_hz == pytest.approx(100_000.0)


def test_next_zoom_for_span_is_clamped_to_the_band():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=500_000.0)
    result = next_zoom_for_span(zoom, 99_000_000.0, 101_000_000.0, 50_000_000.0)
    assert result.span_hz == pytest.approx(2_000_000.0)


# ----------------------------------------------------------------------
# next_zoom_for_pan
# ----------------------------------------------------------------------


def test_next_zoom_for_pan_moves_the_center_when_unlocked():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=200_000.0)
    result = next_zoom_for_pan(zoom, False, 99_000_000.0, 101_000_000.0, 100_500_000.0)
    assert result is not None
    assert result.center_hz == pytest.approx(100_500_000.0)
    assert result.span_hz == pytest.approx(200_000.0)


def test_next_zoom_for_pan_is_a_no_op_while_locked():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=200_000.0)
    assert next_zoom_for_pan(zoom, True, 99_000_000.0, 101_000_000.0, 100_500_000.0) is None


# ----------------------------------------------------------------------
# next_zoom_for_scroll
# ----------------------------------------------------------------------


def test_next_zoom_for_scroll_honours_the_anchor_when_unlocked():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=2_000_000.0)
    result = next_zoom_for_scroll(zoom, False, 90_000_000.0, 110_000_000.0, 0.5, 99_500_000.0)
    assert result.center_hz != pytest.approx(100_000_000.0)
    assert result.span_hz == pytest.approx(1_000_000.0)


def test_next_zoom_for_scroll_ignores_the_anchor_while_locked():
    """Locked, the center is the lock's to own - the wheel may only
    change the span, never recenter around a cursor position."""
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=2_000_000.0)
    result = next_zoom_for_scroll(zoom, True, 90_000_000.0, 110_000_000.0, 0.5, 99_500_000.0)
    assert result.center_hz == pytest.approx(100_000_000.0)
    assert result.span_hz == pytest.approx(1_000_000.0)


def test_next_zoom_for_scroll_is_clamped_to_the_band():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=2_000_000.0)
    result = next_zoom_for_scroll(zoom, False, 99_000_000.0, 101_000_000.0, 100.0, None)
    assert result.span_hz == pytest.approx(2_000_000.0)


# ----------------------------------------------------------------------
# next_zoom_for_follow
# ----------------------------------------------------------------------


def test_next_zoom_for_follow_recenters_while_locked():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=200_000.0)
    result = next_zoom_for_follow(zoom, True, 99_000_000.0, 101_000_000.0, 100_300_000.0)
    assert result is not None
    assert result.center_hz == pytest.approx(100_300_000.0)
    assert result.span_hz == pytest.approx(200_000.0)


def test_next_zoom_for_follow_is_a_no_op_while_unlocked():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=200_000.0)
    assert next_zoom_for_follow(zoom, False, 99_000_000.0, 101_000_000.0, 100_300_000.0) is None


def test_next_zoom_for_follow_is_clamped_to_the_band():
    zoom = ZoomSpan(center_hz=100_000_000.0, span_hz=200_000.0)
    result = next_zoom_for_follow(zoom, True, 99_000_000.0, 101_000_000.0, 200_000_000.0)
    assert result is not None
    assert result.stop_hz <= 101_000_000.0 + 1e-6
