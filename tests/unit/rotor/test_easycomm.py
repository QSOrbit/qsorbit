"""Unit tests for EasyComm II command formatting and response parsing."""

import pytest

from qsorbit.core.rotor import (
    Position,
    ProtocolError,
    format_get_position,
    format_set_position,
    format_stop,
    parse_position,
)


class TestFormatSetPosition:
    def test_typical(self):
        assert format_set_position(Position(180.0, 45.0)) == b"AZ180.0 EL45.0\n"

    def test_one_decimal_precision(self):
        # Values are rounded, not truncated, to one decimal place.
        assert format_set_position(Position(123.456, 7.891)) == b"AZ123.5 EL7.9\n"

    def test_zero_zero(self):
        assert format_set_position(Position(0.0, 0.0)) == b"AZ0.0 EL0.0\n"

    def test_negative_elevation(self):
        assert format_set_position(Position(359.9, -5.0)) == b"AZ359.9 EL-5.0\n"


class TestFormatQueries:
    def test_get_position(self):
        assert format_get_position() == b"AZ EL\n"

    def test_stop(self):
        assert format_stop() == b"SA SE\n"


class TestParsePosition:
    def test_typical(self):
        assert parse_position(b"AZ180.0 EL45.0") == Position(180.0, 45.0)

    def test_crlf_terminator_tolerated(self):
        assert parse_position(b"AZ180.0 EL45.0\r\n") == Position(180.0, 45.0)

    def test_lf_terminator_tolerated(self):
        assert parse_position(b"AZ180.0 EL45.0\n") == Position(180.0, 45.0)

    def test_integer_angles(self):
        assert parse_position(b"AZ180 EL45") == Position(180.0, 45.0)

    def test_negative_elevation(self):
        assert parse_position(b"AZ0.0 EL-5.0") == Position(0.0, -5.0)

    def test_extra_whitespace_tolerated(self):
        assert parse_position(b"  AZ10.0   EL20.0  ") == Position(10.0, 20.0)

    def test_roundtrip(self):
        # Python -> bytes -> Python survives intact.
        original = Position(287.3, 12.8)
        assert parse_position(format_set_position(original)) == original


class TestParsePositionErrors:
    def test_empty_line(self):
        with pytest.raises(ProtocolError, match="Could not parse"):
            parse_position(b"")

    def test_garbage(self):
        with pytest.raises(ProtocolError, match="Could not parse"):
            parse_position(b"ERROR: motor stalled")

    def test_missing_elevation(self):
        with pytest.raises(ProtocolError, match="Could not parse"):
            parse_position(b"AZ180.0")

    def test_swapped_order_rejected(self):
        with pytest.raises(ProtocolError, match="Could not parse"):
            parse_position(b"EL45.0 AZ180.0")

    def test_out_of_range_azimuth_is_protocol_error(self):
        # The rotor reporting 400 degrees is the rotor's fault, not a
        # programming error - so ProtocolError, not ValueError.
        with pytest.raises(ProtocolError, match="out-of-range"):
            parse_position(b"AZ400.0 EL45.0")

    def test_out_of_range_elevation_is_protocol_error(self):
        with pytest.raises(ProtocolError, match="out-of-range"):
            parse_position(b"AZ180.0 EL95.0")
