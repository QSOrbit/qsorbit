"""Unit tests for the Rotor facade.

No hardware and no real waiting: the serial port is a fake that returns
scripted replies, and the clock is a fake whose ``sleep`` advances its
own ``monotonic``. A homing-timeout test that genuinely slept for two
minutes would never get run.
"""

import pytest

from qsorbit.core.rotor import (
    Arrival,
    AzimuthWrap,
    GainRegister,
    GainVerificationError,
    HomingError,
    Position,
    PositionLimitError,
    ProtocolError,
    Rotor,
    RotorCapabilities,
    RotorErrorCode,
    SerialConnectionError,
    SerialTimeoutError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Placed in a reply script to mean "the controller said nothing" — what
#: happens while it is homing, and what a dead link looks like too.
SILENCE = None


class FakePort:
    """A stand-in for SerialPort that replays scripted replies.

    Args:
        replies: Reply lines, in the order they will be returned. A
            ``SILENCE`` entry raises SerialTimeoutError instead, as a
            real timed-out read does. Running out of replies also times
            out, so a test that reads more than it scripted fails
            loudly rather than hanging.
    """

    def __init__(self, replies: list[bytes | None] | None = None) -> None:
        self.replies: list[bytes | None] = list(replies or [])
        self.writes: list[bytes] = []
        self.is_open = False
        self.open_count = 0
        self.close_count = 0

    def open(self) -> None:
        self.is_open = True
        self.open_count += 1

    def close(self) -> None:
        self.is_open = False
        self.close_count += 1

    def write(self, data: bytes) -> None:
        if not self.is_open:
            raise SerialConnectionError("Serial port is not open.")
        self.writes.append(data)

    def readline(self) -> bytes:
        if not self.is_open:
            raise SerialConnectionError("Serial port is not open.")
        if not self.replies:
            raise SerialTimeoutError("Fake port ran out of scripted replies.")
        reply = self.replies.pop(0)
        if reply is SILENCE:
            raise SerialTimeoutError("Fake port: silence.")
        return reply


class FakeClock:
    """A clock whose sleep() advances its own monotonic()."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


def capabilities(**overrides) -> RotorCapabilities:
    fields = {
        "azimuth_min_deg": 0.0,
        "azimuth_max_deg": 360.0,
        "elevation_min_deg": 0.0,
        "elevation_max_deg": 180.0,
        "azimuth_wrap": AzimuthWrap.EXTRA_ROTATION,
        "acceptance_window_deg": 2.5,
        "rs485_turnaround_s": 0.15,
        "firmware_version": "SatNOGS-v2.2.1",
    }
    fields.update(overrides)
    return RotorCapabilities(**fields)


def make_rotor(replies=None, *, caps=None, **kwargs) -> tuple[Rotor, FakePort, FakeClock]:
    port = FakePort(replies)
    clock = FakeClock()
    rotor = Rotor(
        port,  # type: ignore[arg-type]  # structural stand-in for SerialPort
        caps or capabilities(),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **kwargs,
    )
    return rotor, port, clock


#: The three replies a healthy connect sequence consumes.
HEALTHY_CONNECT = [b"VESatNOGS-v2.2.1\n", b"GE1\n", b"AZ-1.5 EL2.0\n"]


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------


class TestConnect:
    def test_happy_path_sequence(self):
        rotor, port, _ = make_rotor(HEALTHY_CONNECT)

        status = rotor.connect()

        assert port.writes == [b"VE\n", b"GE\n", b"AZ EL\n"]
        assert status.firmware_version == "SatNOGS-v2.2.1"
        assert status.error is RotorErrorCode.NO_ERROR
        assert status.position == Position(-1.5, 2.0)
        assert status.healthy

    def test_opens_the_port_once(self):
        rotor, port, _ = make_rotor(HEALTHY_CONNECT)
        rotor.connect()
        assert port.open_count == 1

    def test_records_firmware_version(self):
        rotor, _, _ = make_rotor(HEALTHY_CONNECT)
        assert rotor.firmware_version is None
        rotor.connect()
        assert rotor.firmware_version == "SatNOGS-v2.2.1"

    def test_waits_through_homing_silence(self):
        # The controller is deaf while homing: homing() blocks and never
        # services the serial link. Silence is expected, not a failure.
        rotor, port, _ = make_rotor([SILENCE, SILENCE, *HEALTHY_CONNECT])

        status = rotor.connect()

        assert status.firmware_version == "SatNOGS-v2.2.1"
        assert port.writes.count(b"VE\n") == 3

    def test_reports_progress_while_waiting(self):
        # So a CLI can say "still homing (12s)" rather than appear hung.
        waits: list[float] = []
        rotor, _, _ = make_rotor(
            [SILENCE, SILENCE, *HEALTHY_CONNECT],
            on_homing_wait=waits.append,
        )

        rotor.connect()

        assert len(waits) == 2
        assert waits[0] < waits[1]

    def test_tolerates_unparseable_bytes_during_homing(self):
        # Partial or stale bytes can arrive as the controller comes out
        # of homing. That is noise to be retried, not a fatal error.
        rotor, _, _ = make_rotor([b"\x00\xff garbage\n", *HEALTHY_CONNECT])

        assert rotor.connect().firmware_version == "SatNOGS-v2.2.1"

    def test_gives_up_after_the_homing_timeout(self):
        rotor, _, clock = make_rotor([SILENCE] * 50, homing_timeout_s=10.0)

        with pytest.raises(SerialConnectionError, match="No usable reply"):
            rotor.connect()

        assert clock.now >= 10.0

    def test_timeout_message_names_the_likely_causes(self):
        rotor, _, _ = make_rotor([SILENCE] * 50, homing_timeout_s=5.0)

        with pytest.raises(SerialConnectionError, match="baud rate"):
            rotor.connect()

    def test_timeout_message_quotes_unrecognized_bytes(self):
        rotor, _, _ = make_rotor([b"nonsense\n"] * 50, homing_timeout_s=5.0)

        with pytest.raises(SerialConnectionError, match="nonsense"):
            rotor.connect()

    def test_latched_homing_error_raises(self):
        # It cannot be cleared over serial, and it invalidates the zero
        # that every later position is measured from.
        rotor, _, _ = make_rotor([b"VESatNOGS-v2.2.1\n", b"GE4\n"])

        with pytest.raises(HomingError, match="Power-cycle"):
            rotor.connect()

    def test_other_errors_are_reported_not_raised(self):
        # Over-temperature is worth showing an operator, but it does not
        # invalidate the position reference the way a homing failure does.
        rotor, _, _ = make_rotor([b"VESatNOGS-v2.2.1\n", b"GE12\n", b"AZ10.0 EL20.0\n"])

        status = rotor.connect()

        assert status.error is RotorErrorCode.OVER_TEMPERATURE
        assert not status.healthy
        assert not status.homing_error_latched

    def test_unknown_error_code_surfaces(self):
        rotor, _, _ = make_rotor([b"VESatNOGS-v2.2.1\n", b"GE99\n"])

        with pytest.raises(ProtocolError, match="unrecognized error code"):
            rotor.connect()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestQueries:
    def test_read_position(self):
        rotor, port, _ = make_rotor([*HEALTHY_CONNECT, b"AZ123.4 EL56.7\n"])
        rotor.connect()

        assert rotor.read_position() == Position(123.4, 56.7)
        assert port.writes[-1] == b"AZ EL\n"

    def test_read_error(self):
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, b"GE1\n"])
        rotor.connect()

        assert rotor.read_error() is RotorErrorCode.NO_ERROR

    def test_status_refreshes_error_and_position(self):
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, b"GE1\n", b"AZ90.0 EL45.0\n"])
        rotor.connect()

        status = rotor.status()

        assert status.position == Position(90.0, 45.0)
        assert status.firmware_version == "SatNOGS-v2.2.1"

    def test_status_reports_a_latched_homing_error_rather_than_raising(self):
        # connect() raises on this; status() is what a CLI uses to
        # *display* the state, so it must not blow up on it.
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, b"GE4\n", b"AZ0.0 EL0.0\n"])
        rotor.connect()

        status = rotor.status()

        assert status.homing_error_latched
        assert not status.healthy

    def test_status_before_connect_is_refused(self):
        rotor, _, _ = make_rotor([])

        with pytest.raises(SerialConnectionError, match="Not connected"):
            rotor.status()

    def test_queries_wait_for_rs485_turnaround(self):
        # Half-duplex: the transceiver needs a gap before it can answer.
        rotor, _, clock = make_rotor([*HEALTHY_CONNECT, b"AZ0.0 EL0.0\n"])
        rotor.connect()
        clock.sleeps.clear()

        rotor.read_position()

        assert clock.sleeps == [0.15]

    def test_turnaround_comes_from_capabilities(self):
        rotor, _, clock = make_rotor(
            [*HEALTHY_CONNECT, b"AZ0.0 EL0.0\n"], caps=capabilities(rs485_turnaround_s=0.4)
        )
        rotor.connect()
        clock.sleeps.clear()

        rotor.read_position()

        assert clock.sleeps == [0.4]


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------


class TestMoveTo:
    def test_sends_both_axes(self):
        rotor, port, _ = make_rotor(HEALTHY_CONNECT)
        rotor.connect()

        rotor.move_to(Position(180.0, 45.0))

        assert port.writes[-1] == b"AZ180.0 EL45.0\n"

    def test_consumes_no_reply(self):
        # A set-position command draws no response. Reading one anyway
        # would steal the answer to the next query.
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, b"AZ180.0 EL45.0\n"])
        rotor.connect()

        rotor.move_to(Position(180.0, 45.0))

        # The scripted reply is still there, so it answers the next read.
        assert rotor.read_position() == Position(180.0, 45.0)

    def test_waits_for_no_turnaround(self):
        rotor, _, clock = make_rotor(HEALTHY_CONNECT)
        rotor.connect()
        clock.sleeps.clear()

        rotor.move_to(Position(10.0, 20.0))

        assert clock.sleeps == []

    def test_out_of_range_is_refused(self):
        rotor, port, _ = make_rotor(HEALTHY_CONNECT)
        rotor.connect()
        writes_before = len(port.writes)

        with pytest.raises(PositionLimitError, match="Azimuth"):
            rotor.move_to(Position(380.0, 45.0))

        assert len(port.writes) == writes_before

    def test_refusal_happens_before_the_wire(self):
        # The firmware has no limits at any level, so a command that
        # reaches it is a command it attempts. Nothing may be written.
        rotor, port, _ = make_rotor(HEALTHY_CONNECT)
        rotor.connect()

        with pytest.raises(PositionLimitError):
            rotor.move_to(Position(0.0, 400.0))

        assert b"EL400.0" not in b"".join(port.writes)

    def test_move_before_connect_fails(self):
        rotor, _, _ = make_rotor([])

        with pytest.raises(SerialConnectionError):
            rotor.move_to(Position(10.0, 10.0))


class TestStop:
    def test_returns_the_reported_position(self):
        rotor, port, _ = make_rotor([*HEALTHY_CONNECT, b"AZ42.0 EL13.0\n"])
        rotor.connect()

        assert rotor.stop() == Position(42.0, 13.0)
        assert port.writes[-1] == b"SA SE\n"

    def test_consumes_its_reply(self):
        # The regression that motivated the Command type: SA SE answers
        # with a position. Left unread it becomes the answer to the next
        # query, and every later read is off by one message.
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, b"AZ42.0 EL13.0\n", b"AZ99.0 EL11.0\n"])
        rotor.connect()

        rotor.stop()

        assert rotor.read_position() == Position(99.0, 11.0)


# ---------------------------------------------------------------------------
# Arrival
# ---------------------------------------------------------------------------


class TestWaitForArrival:
    def test_arrives_immediately(self):
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, b"AZ180.0 EL45.0\n"])
        rotor.connect()

        result = rotor.wait_for_arrival(Position(180.0, 45.0), timeout_s=10.0)

        assert isinstance(result, Arrival)
        assert result.arrived
        assert result.position == Position(180.0, 45.0)
        # One round trip's worth of RS-485 turnaround, no polling wait.
        assert result.elapsed_s == pytest.approx(0.15)

    def test_arrives_after_a_few_polls(self):
        rotor, _, _ = make_rotor(
            [
                *HEALTHY_CONNECT,
                b"AZ100.0 EL20.0\n",
                b"AZ150.0 EL35.0\n",
                b"AZ178.6 EL43.1\n",
            ]
        )
        rotor.connect()

        result = rotor.wait_for_arrival(Position(180.0, 45.0), timeout_s=30.0)

        assert result.arrived
        assert result.position == Position(178.6, 43.1)

    def test_short_of_target_but_inside_the_window_counts(self):
        # The normal outcome with stock gains: ~1.5 az / ~2.1 el short.
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, b"AZ178.5 EL42.9\n"])
        rotor.connect()

        assert rotor.wait_for_arrival(Position(180.0, 45.0), timeout_s=10.0).arrived

    def test_timeout_reports_rather_than_raises(self):
        # A stalled axis and a slow one look identical from here, and
        # neither is this method's call to make.
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, *([b"AZ0.0 EL0.0\n"] * 50)])
        rotor.connect()

        result = rotor.wait_for_arrival(Position(180.0, 45.0), timeout_s=3.0)

        assert not result.arrived
        assert result.position == Position(0.0, 0.0)
        assert result.elapsed_s >= 3.0

    def test_never_asks_the_rotor_whether_it_arrived(self):
        # GS reports idle for a stalled axis too, so arrival is decided
        # by comparing positions and GS is never sent.
        rotor, port, _ = make_rotor([*HEALTHY_CONNECT, b"AZ180.0 EL45.0\n"])
        rotor.connect()

        rotor.wait_for_arrival(Position(180.0, 45.0), timeout_s=10.0)

        assert b"GS\n" not in port.writes


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_connects_on_enter_and_closes_on_exit(self):
        rotor, port, _ = make_rotor(HEALTHY_CONNECT)

        with rotor as connected:
            assert connected is rotor
            assert port.is_open

        assert port.close_count == 1
        assert not port.is_open

    def test_closes_even_when_the_body_raises(self):
        rotor, port, _ = make_rotor(HEALTHY_CONNECT)

        with pytest.raises(RuntimeError), rotor:
            raise RuntimeError("boom")

        assert port.close_count == 1

    def test_close_is_safe_twice(self):
        rotor, _, _ = make_rotor(HEALTHY_CONNECT)
        rotor.connect()
        rotor.close()
        rotor.close()


# ---------------------------------------------------------------------------
# Gain registers (Chunk H PR2b)
# ---------------------------------------------------------------------------


class TestReadGain:
    def test_reads_a_register(self):
        rotor, port, _ = make_rotor([*HEALTHY_CONNECT, b"2,1.00\n"])
        rotor.connect()
        assert rotor.read_gain(GainRegister.AZIMUTH_KI) == 1.0
        assert port.writes[-1] == b"CR 2\n"

    def test_a_reply_for_another_register_raises(self):
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, b"3,0.50\n"])
        rotor.connect()
        with pytest.raises(ProtocolError, match="answered for register"):
            rotor.read_gain(GainRegister.AZIMUTH_KI)

    def test_silence_times_out(self):
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, SILENCE])
        rotor.connect()
        with pytest.raises(SerialTimeoutError):
            rotor.read_gain(GainRegister.AZIMUTH_KI)


class TestWriteGain:
    def test_writes_the_register(self):
        rotor, port, _ = make_rotor(HEALTHY_CONNECT)
        rotor.connect()
        rotor.write_gain(GainRegister.AZIMUTH_KI, 1.0)
        assert port.writes[-1] == b"CW2,1.00\n"

    def test_reads_nothing_back(self):
        # The firmware answers a write with nothing. Reading here would
        # shift every later reply by one message.
        rotor, port, _ = make_rotor([*HEALTHY_CONNECT, b"2,1.00\n"])
        rotor.connect()
        rotor.write_gain(GainRegister.AZIMUTH_KI, 1.0)
        # The scripted reply is still queued, unconsumed.
        assert port.replies == [b"2,1.00\n"]


class TestReadGains:
    """The read-only counterpart to push_gains."""

    def test_it_reads_every_register_in_order(self):
        rotor, port, _ = make_rotor(
            [
                *HEALTHY_CONNECT,
                b"1,8.00\n",
                b"2,0.97\n",
                b"3,0.50\n",
                b"4,10.00\n",
                b"5,1.00\n",
                b"6,0.30\n",
            ]
        )
        rotor.connect()

        gains = rotor.read_gains()

        assert gains == {
            GainRegister.AZIMUTH_KP: 8.0,
            GainRegister.AZIMUTH_KI: 0.97,
            GainRegister.AZIMUTH_KD: 0.5,
            GainRegister.ELEVATION_KP: 10.0,
            GainRegister.ELEVATION_KI: 1.0,
            GainRegister.ELEVATION_KD: 0.3,
        }
        assert port.writes[-6:] == [
            b"CR 1\n",
            b"CR 2\n",
            b"CR 3\n",
            b"CR 4\n",
            b"CR 5\n",
            b"CR 6\n",
        ]

    def test_it_writes_nothing(self):
        # The whole point of having it separate from push_gains: asking
        # what a controller is running must not change what it is
        # running.
        rotor, port, _ = make_rotor(
            [
                *HEALTHY_CONNECT,
                b"1,8.00\n",
                b"2,0.97\n",
                b"3,0.50\n",
                b"4,10.00\n",
                b"5,1.00\n",
                b"6,0.30\n",
            ]
        )
        rotor.connect()

        rotor.read_gains()

        assert not [w for w in port.writes if w.startswith(b"CW")]


class TestPushGains:
    """Every register is verified, not a sample.

    Gains are RAM-only and re-pushed at every connect, so a write that
    silently fails leaves the rotor on compiled defaults while the
    application believes it is running a tuned set -- and every metric
    measured afterwards is attributed to the wrong configuration.
    """

    TRACKING = {GainRegister.AZIMUTH_KI: 1.0, GainRegister.ELEVATION_KI: 1.0}

    def test_writes_then_reads_every_register(self):
        rotor, port, _ = make_rotor([*HEALTHY_CONNECT, b"2,1.00\n", b"5,1.00\n"])
        rotor.connect()
        result = rotor.push_gains(self.TRACKING)

        assert result == {GainRegister.AZIMUTH_KI: 1.0, GainRegister.ELEVATION_KI: 1.0}
        assert port.writes[-4:] == [b"CW2,1.00\n", b"CR 2\n", b"CW5,1.00\n", b"CR 5\n"]

    def test_never_sends_two_writes_without_a_read_between_them(self):
        """The guard on the mechanism, and it replaces a test that asserted the bug.

        The previous version of this test asserted that every write went
        out before any read -- the batching was tested, so it looked
        deliberate. It silently loses gains: the firmware drains the
        whole serial buffer in one call and only the FIRST ``CW`` in a
        drain applies, because the handler's ``strtok_r`` mutates the
        pointer the next one parses from. A read between writes is a
        round trip, so the host cannot send the second ``CW`` until the
        firmware has answered -- one write per drain, by construction.

        Asserted structurally rather than by counting sleeps: what has
        to be true is the *interleave*, and a duration would go stale
        the moment anyone retuned the turnaround.
        """
        rotor, port, _ = make_rotor([*HEALTHY_CONNECT, b"1,8.00\n", b"2,1.00\n", b"5,1.00\n"])
        rotor.connect()
        rotor.push_gains(
            {
                GainRegister.AZIMUTH_KP: 8.0,
                GainRegister.AZIMUTH_KI: 1.0,
                GainRegister.ELEVATION_KI: 1.0,
            }
        )

        gain_traffic = [w for w in port.writes if w.startswith((b"CW", b"CR"))]
        assert gain_traffic == [
            b"CW1,8.00\n",
            b"CR 1\n",
            b"CW2,1.00\n",
            b"CR 2\n",
            b"CW5,1.00\n",
            b"CR 5\n",
        ]
        # Stated as the rule as well as the sequence, so a future edit
        # that reorders these is told what it broke.
        for earlier, later in zip(gain_traffic, gain_traffic[1:], strict=False):
            assert not (earlier.startswith(b"CW") and later.startswith(b"CW"))

    def test_a_register_that_did_not_take_raises(self):
        # This guards the OUTCOME, and it has a blind spot worth stating
        # rather than discovering twice: it can only see a lost write
        # whose target differs from what the register already held. That
        # is exactly how the burst defect hid behind Kp, which was being
        # pushed at the firmware's own compiled default and so read back
        # correct whether or not the write landed. The interleave test
        # above is what guards the mechanism.
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, b"2,0.00\n", b"5,1.00\n"])
        rotor.connect()
        with pytest.raises(GainVerificationError, match="AZIMUTH_KI asked 1.00 got 0.00"):
            rotor.push_gains(self.TRACKING)

    def test_the_message_names_every_register_that_disagreed(self):
        # One wrong register and six wrong registers are different
        # faults: one is a dropped byte, six is a controller that took
        # nothing at all.
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, b"2,0.00\n", b"5,0.00\n"])
        rotor.connect()
        with pytest.raises(GainVerificationError) as exc:
            rotor.push_gains(self.TRACKING)
        message = str(exc.value)
        assert "2 of 2" in message
        assert "AZIMUTH_KI" in message
        assert "ELEVATION_KI" in message

    def test_rounding_at_the_last_digit_is_not_a_mismatch(self):
        # Both directions carry two decimals, so anything inside half of
        # the last digit is the format, not a disagreement.
        rotor, _, _ = make_rotor([*HEALTHY_CONNECT, b"2,1.00\n", b"5,1.00\n"])
        rotor.connect()
        rotor.push_gains({GainRegister.AZIMUTH_KI: 1.001, GainRegister.ELEVATION_KI: 0.999})

    def test_an_empty_set_writes_nothing_and_does_not_wait(self):
        # The `stock` profile pushes nothing at all, and must not cost a
        # settle delay on every connect to do it.
        rotor, port, clock = make_rotor(HEALTHY_CONNECT)
        rotor.connect()
        writes_before = len(port.writes)
        sleeps_before = len(clock.sleeps)

        assert rotor.push_gains({}) == {}
        assert len(port.writes) == writes_before
        # Not `clock.sleeps == []`: connecting sleeps the RS-485
        # turnaround on every exchange. What must not happen is a *new*
        # sleep for a push that pushed nothing.
        assert len(clock.sleeps) == sleeps_before
