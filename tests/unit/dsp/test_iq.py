"""Tests for raw uint8 IQ unpacking."""

from __future__ import annotations

import numpy as np
import pytest

from qsorbit.core.dsp import unpack_uint8_iq


class TestUnpackUint8Iq:
    def test_minimum_byte_maps_to_exactly_minus_one(self):
        result = unpack_uint8_iq(bytes([0, 0]))

        assert result[0].real == pytest.approx(-1.0)
        assert result[0].imag == pytest.approx(-1.0)

    def test_maximum_byte_maps_to_exactly_plus_one(self):
        result = unpack_uint8_iq(bytes([255, 255]))

        assert result[0].real == pytest.approx(1.0)
        assert result[0].imag == pytest.approx(1.0)

    def test_the_zero_point_is_between_127_and_128(self):
        low = unpack_uint8_iq(bytes([127, 127]))
        high = unpack_uint8_iq(bytes([128, 128]))

        assert low[0].real < 0.0
        assert high[0].real > 0.0
        # Symmetric around true zero, not favouring one side.
        assert low[0].real == pytest.approx(-high[0].real)

    def test_interleaving_is_i_then_q(self):
        # I=0 (min), Q=255 (max) for one sample.
        result = unpack_uint8_iq(bytes([0, 255]))

        assert result[0].real == pytest.approx(-1.0)
        assert result[0].imag == pytest.approx(1.0)

    def test_multiple_samples_unpack_in_order(self):
        raw = bytes([0, 0, 255, 255, 128, 128])

        result = unpack_uint8_iq(raw)

        assert len(result) == 3
        assert result[0].real == pytest.approx(-1.0)
        assert result[1].real == pytest.approx(1.0)
        assert result[2].real > 0.0

    def test_output_dtype_is_complex64(self):
        result = unpack_uint8_iq(bytes([0, 0, 255, 255]))

        assert result.dtype == np.complex64

    def test_accepts_a_bytearray(self):
        result = unpack_uint8_iq(bytearray([0, 0]))

        assert result[0].real == pytest.approx(-1.0)

    def test_accepts_a_uint8_ndarray_directly(self):
        raw = np.array([0, 0, 255, 255], dtype=np.uint8)

        result = unpack_uint8_iq(raw)

        assert len(result) == 2
        assert result[0].real == pytest.approx(-1.0)
        assert result[1].real == pytest.approx(1.0)

    def test_rejects_an_odd_length(self):
        with pytest.raises(ValueError, match="odd length"):
            unpack_uint8_iq(bytes([0, 0, 255]))

    def test_rejects_a_multidimensional_array(self):
        raw = np.zeros((2, 2), dtype=np.uint8)

        with pytest.raises(ValueError, match="one-dimensional"):
            unpack_uint8_iq(raw)

    def test_an_empty_input_is_a_valid_zero_length_result(self):
        result = unpack_uint8_iq(b"")

        assert len(result) == 0
