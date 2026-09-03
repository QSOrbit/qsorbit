"""Unit tests for SatNOGS command formatting and response parsing."""

import pytest

from qsorbit.core.rotor import (
    Command,
    GainRegister,
    Position,
    ProtocolError,
    RotorErrorCode,
    format_get_error,
    format_get_gain,
    format_get_position,
    format_get_version,
    format_set_gain,
    format_set_position,
    format_stop,
    parse_error,
    parse_gain,
    parse_position,
    parse_version,
)


class TestFormatSetPosition:
    def test_typical(self):
        assert bytes(format_set_position(Position(180.0, 45.0))) == b"AZ180.0 EL45.0\n"

    def test_one_decimal_precision(self):
        # Values are rounded, not truncated, to one decimal place.
        assert bytes(format_set_position(Position(123.456, 7.891))) == b"AZ123.5 EL7.9\n"

    def test_zero_zero(self):
        assert bytes(format_set_position(Position(0.0, 0.0))) == b"AZ0.0 EL0.0\n"

    def test_negative_elevation(self):
        assert bytes(format_set_position(Position(359.9, -5.0))) == b"AZ359.9 EL-5.0\n"

    def test_always_sends_both_axes(self):
        # A lone axis is unsafe on v2.2 and earlier: a bare "AZ" slews
        # azimuth to zero, and "AZ10.0" with no EL token dereferences a
        # null pointer. Taking a whole Position is what makes emitting a
        # lone axis structurally impossible - this pins that shape.
        for position in (Position(0.0, 0.0), Position(12.3, -4.5), Position(359.9, 89.9)):
            data = bytes(format_set_position(position))
            assert data.startswith(b"AZ")
            assert b" EL" in data

    def test_expects_no_reply(self):
        # The firmware answers a set-position command with nothing.
        assert format_set_position(Position(10.0, 20.0)).expects_reply is False


class TestFormatQueries:
    def test_get_position(self):
        assert bytes(format_get_position()) == b"AZ EL\n"

    def test_get_position_expects_reply(self):
        assert format_get_position().expects_reply is True

    def test_get_version(self):
        assert bytes(format_get_version()) == b"VE\n"

    def test_get_version_expects_reply(self):
        assert format_get_version().expects_reply is True

    def test_get_error(self):
        assert bytes(format_get_error()) == b"GE\n"

    def test_get_error_expects_reply(self):
        assert format_get_error().expects_reply is True

    def test_stop(self):
        assert bytes(format_stop()) == b"SA SE\n"

    def test_stop_expects_a_reply(self):
        # The one that bites: SA SE answers with a position report. Left
        # unread it becomes the answer to the *next* query, shifting
        # every later read by one message.
        assert format_stop().expects_reply is True


class TestCommand:
    def test_is_a_value_object(self):
        assert Command(b"VE\n", True) == Command(b"VE\n", True)

    def test_differs_on_reply_expectation(self):
        assert Command(b"VE\n", True) != Command(b"VE\n", False)

    def test_bytes_conversion(self):
        assert bytes(Command(b"AZ EL\n", True)) == b"AZ EL\n"


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

    def test_real_homed_rotor_response(self):
        # Captured verbatim from Phil's SatNOGS rotator at 19200 baud,
        # immediately after homing. Regression test: this exact line used
        # to raise ProtocolError because Position rejected negative
        # azimuth, which meant QSOrbit could not read its own rotor.
        assert parse_position(b"AZ-1.5 EL2.0\n") == Position(-1.5, 2.0)

    def test_azimuth_beyond_360_is_not_an_error(self):
        # Multi-turn azimuth axes reach past 360; 380 and 20 are different
        # physical places.
        assert parse_position(b"AZ380.0 EL10.0") == Position(380.0, 10.0)

    def test_extra_whitespace_tolerated(self):
        assert parse_position(b"  AZ10.0   EL20.0  ") == Position(10.0, 20.0)

    def test_roundtrip(self):
        # Python -> bytes -> Python survives intact.
        original = Position(287.3, 12.8)
        assert parse_position(bytes(format_set_position(original))) == original


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

    def test_impossible_azimuth_is_protocol_error(self):
        # A garbled frame producing a number no working rotor could report
        # is the rotor's fault (or the link's), not a programming error -
        # so ProtocolError, not ValueError. Note 400.0 is NOT an error:
        # multi-turn azimuth axes legitimately exceed 360.
        with pytest.raises(ProtocolError, match="out-of-range"):
            parse_position(b"AZ99999.0 EL45.0")

    def test_impossible_elevation_is_protocol_error(self):
        with pytest.raises(ProtocolError, match="out-of-range"):
            parse_position(b"AZ180.0 EL-99999.0")


class TestParseVersion:
    def test_real_rotor_reply(self):
        # The firmware concatenates the version onto the "VE" prefix with
        # no separator.
        assert parse_version(b"VESatNOGS-v2.2.1\n") == "SatNOGS-v2.2.1"

    def test_older_firmware(self):
        assert parse_version(b"VESatNOGS-v2.2\n") == "SatNOGS-v2.2"

    def test_crlf_tolerated(self):
        assert parse_version(b"VESatNOGS-v2.2.1\r\n") == "SatNOGS-v2.2.1"

    def test_unfamiliar_version_string_returned_verbatim(self):
        # Capabilities are declared, never inferred from the version, so
        # an unfamiliar string is data rather than an error.
        assert parse_version(b"VESomeOtherFirmware-9.9\n") == "SomeOtherFirmware-9.9"

    def test_garbage_rejected(self):
        with pytest.raises(ProtocolError, match="firmware version"):
            parse_version(b"AZ180.0 EL45.0\n")

    def test_empty_version_rejected(self):
        with pytest.raises(ProtocolError, match="firmware version"):
            parse_version(b"VE\n")


class TestParseError:
    def test_no_error(self):
        assert parse_error(b"GE1\n") == RotorErrorCode.NO_ERROR

    def test_homing_error(self):
        assert parse_error(b"GE4\n") == RotorErrorCode.HOMING_ERROR

    def test_over_temperature_is_twelve(self):
        # 12 is not a power of two. Masked as bit flags it would read as
        # HOMING_ERROR | MOTOR_ERROR, which is why these values have to
        # be compared for equality.
        assert parse_error(b"GE12\n") == RotorErrorCode.OVER_TEMPERATURE

    def test_all_defined_codes_round_trip(self):
        for member in RotorErrorCode:
            assert parse_error(f"GE{member.value}\n".encode("ascii")) is member

    def test_crlf_tolerated(self):
        assert parse_error(b"GE1\r\n") == RotorErrorCode.NO_ERROR

    def test_unknown_code_rejected(self):
        # An undefined code means firmware QSOrbit hasn't been verified
        # against - surfaced, not silently treated as "fine".
        with pytest.raises(ProtocolError, match="unrecognized error code 20"):
            parse_error(b"GE20\n")

    def test_garbage_rejected(self):
        with pytest.raises(ProtocolError, match="error reply"):
            parse_error(b"AZ180.0 EL45.0\n")

    def test_missing_code_rejected(self):
        with pytest.raises(ProtocolError, match="error reply"):
            parse_error(b"GE\n")


class TestRotorErrorCode:
    def test_values_match_firmware(self):
        assert [member.value for member in RotorErrorCode] == [1, 2, 4, 8, 12, 16]

    def test_homing_error_is_the_latching_one(self):
        # Pinned because it is the code with an operational consequence:
        # nothing sent over serial clears it, only a power cycle.
        assert RotorErrorCode.HOMING_ERROR.value == 4


# ---------------------------------------------------------------------------
# Gain registers (Chunk H PR2b)
# ---------------------------------------------------------------------------


class TestGainCommands:
    """The read and write forms are not symmetric, and that is upstream.

    ``CW`` tests ``buffer[2]`` and ``CR`` tests ``buffer[3]``, so the
    write takes no space and the read requires one. ``CR1`` parses as
    nothing and the controller answers nothing at all — which reads as a
    dead link rather than a malformed command, and cost bench time once.
    """

    def test_a_write_has_no_space(self):
        assert bytes(format_set_gain(GainRegister.AZIMUTH_KP, 8.0)) == b"CW1,8.00\n"

    def test_a_read_has_a_space(self):
        assert bytes(format_get_gain(GainRegister.AZIMUTH_KP)) == b"CR 1\n"

    def test_a_write_expects_no_reply(self):
        # The firmware answers a gain write with nothing. A caller that
        # read a line here would shift every later reply by one.
        assert format_set_gain(GainRegister.AZIMUTH_KI, 1.0).expects_reply is False

    def test_a_read_expects_a_reply(self):
        assert format_get_gain(GainRegister.AZIMUTH_KI).expects_reply is True

    def test_values_carry_two_decimals(self):
        # Matching what the firmware parses and what CR reports back, so
        # a written value and its read-back are comparable without
        # worrying about how either was rounded.
        assert bytes(format_set_gain(GainRegister.ELEVATION_KD, 0.3)) == b"CW6,0.30\n"
        assert bytes(format_set_gain(GainRegister.AZIMUTH_KI, 1.0)) == b"CW2,1.00\n"

    @pytest.mark.parametrize(
        ("register", "index"),
        [
            (GainRegister.AZIMUTH_KP, 1),
            (GainRegister.AZIMUTH_KI, 2),
            (GainRegister.AZIMUTH_KD, 3),
            (GainRegister.ELEVATION_KP, 4),
            (GainRegister.ELEVATION_KI, 5),
            (GainRegister.ELEVATION_KD, 6),
        ],
    )
    def test_the_register_numbering(self, register, index):
        # 7 and 8 are park positions and 9 is the read-only control
        # mode. Getting an index wrong writes one of those.
        assert int(register) == index

    def test_the_dangerous_neighbours_are_not_reachable(self):
        # There is deliberately no enum member for 7, 8 or 9.
        assert max(int(r) for r in GainRegister) == 6


class TestParseGain:
    def test_parses_a_reply(self):
        assert parse_gain(b"2,1.00\n", GainRegister.AZIMUTH_KI) == 1.0

    def test_parses_stock_values(self):
        assert parse_gain(b"1,8.00\n", GainRegister.AZIMUTH_KP) == 8.0
        assert parse_gain(b"6,0.30\n", GainRegister.ELEVATION_KD) == 0.3

    def test_tolerates_surrounding_whitespace(self):
        assert parse_gain(b"  2,1.00  \n", GainRegister.AZIMUTH_KI) == 1.0

    def test_a_reply_for_another_register_is_refused(self):
        # Silently accepting it would attribute a Kd figure to a Ki slot
        # with nothing on screen to show for it, and every later read
        # would be off by the same shift.
        with pytest.raises(ProtocolError, match="answered for register"):
            parse_gain(b"3,0.50\n", GainRegister.AZIMUTH_KI)

    def test_the_mismatch_message_names_both_registers(self):
        with pytest.raises(ProtocolError) as exc:
            parse_gain(b"3,0.50\n", GainRegister.AZIMUTH_KI)
        assert "AZIMUTH_KI" in str(exc.value)
        assert "register 3" in str(exc.value)

    @pytest.mark.parametrize(
        "line", [b"\n", b"CR 2\n", b"2\n", b"2,\n", b"two,1.00\n", b"AZ1.0 EL2.0\n"]
    )
    def test_unparseable_replies_are_refused(self, line):
        with pytest.raises(ProtocolError, match="Could not parse gain reply"):
            parse_gain(line, GainRegister.AZIMUTH_KI)

    def test_a_negative_gain_parses(self):
        # Not a value QSOrbit would write, but the firmware would report
        # one and refusing to parse it would hide the fact.
        assert parse_gain(b"3,-0.50\n", GainRegister.AZIMUTH_KD) == -0.5
