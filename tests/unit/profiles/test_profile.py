"""Unit tests for the profile data model: Transmitter, SatelliteProfile, and friends."""

from datetime import date

import pytest

from qsorbit.core.profiles.profile import (
    AliveRecord,
    AliveStatus,
    Mode,
    ReliabilityClass,
    SatelliteProfile,
    Transmitter,
)


def make_transmitter(**overrides):
    defaults = {
        "downlink_hz": 435605000.0,
        "mode": Mode.CW,
        "reliability": ReliabilityClass.UNCONDITIONAL,
    }
    defaults.update(overrides)
    return Transmitter(**defaults)


def make_alive(**overrides):
    defaults = {
        "status": AliveStatus.ACTIVE,
        "as_of": date(2026, 8, 25),
        "source": "test fixture",
    }
    defaults.update(overrides)
    return AliveRecord(**defaults)


class TestTransmitter:
    def test_accepts_a_minimal_beacon(self):
        transmitter = make_transmitter()

        assert transmitter.downlink_hz == 435605000.0
        assert transmitter.uplink_hz is None
        assert transmitter.baud is None
        assert transmitter.notes == ""

    def test_accepts_a_full_transponder(self):
        transmitter = make_transmitter(
            downlink_hz=435640000.0,
            uplink_hz=145965000.0,
            mode=Mode.SSB,
            reliability=ReliabilityClass.DEPENDENT,
            baud=None,
            notes="linear inverting transponder",
        )

        assert transmitter.uplink_hz == 145965000.0
        assert transmitter.reliability is ReliabilityClass.DEPENDENT

    def test_zero_downlink_is_rejected(self):
        with pytest.raises(ValueError, match="downlink_hz"):
            make_transmitter(downlink_hz=0.0)

    def test_negative_downlink_is_rejected(self):
        with pytest.raises(ValueError, match="downlink_hz"):
            make_transmitter(downlink_hz=-1.0)

    def test_zero_uplink_is_rejected(self):
        with pytest.raises(ValueError, match="uplink_hz"):
            make_transmitter(uplink_hz=0.0)

    def test_zero_baud_is_rejected(self):
        with pytest.raises(ValueError, match="baud"):
            make_transmitter(baud=0.0)

    def test_is_a_value_object(self):
        assert make_transmitter() == make_transmitter()
        assert make_transmitter(downlink_hz=1.0) != make_transmitter(downlink_hz=2.0)


class TestAliveRecord:
    def test_accepts_a_valid_record(self):
        record = make_alive()

        assert record.status is AliveStatus.ACTIVE
        assert record.as_of == date(2026, 8, 25)

    def test_empty_source_is_rejected(self):
        with pytest.raises(ValueError, match="source"):
            make_alive(source="")

    def test_whitespace_only_source_is_rejected(self):
        with pytest.raises(ValueError, match="source"):
            make_alive(source="   ")


class TestSatelliteProfile:
    def test_accepts_a_minimal_profile(self):
        profile = SatelliteProfile(
            norad_id=44909, name="RS-44", transmitters=(make_transmitter(),), alive=make_alive()
        )

        assert profile.norad_id == 44909
        assert profile.also_known_as == ()

    def test_zero_norad_id_is_rejected(self):
        with pytest.raises(ValueError, match="norad_id"):
            SatelliteProfile(norad_id=0, name="X", transmitters=(), alive=make_alive())

    def test_negative_norad_id_is_rejected(self):
        with pytest.raises(ValueError, match="norad_id"):
            SatelliteProfile(norad_id=-5, name="X", transmitters=(), alive=make_alive())

    def test_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="name"):
            SatelliteProfile(norad_id=1, name="", transmitters=(), alive=make_alive())

    def test_profile_with_no_transmitters_is_valid(self):
        # A confirmed-dead satellite kept in the catalogue on purpose,
        # so it's excluded rather than silently absent.
        profile = SatelliteProfile(norad_id=1, name="Dead-Sat", transmitters=(), alive=make_alive())

        assert profile.transmitters == ()


class TestBestReliability:
    def test_none_when_there_are_no_transmitters(self):
        profile = SatelliteProfile(norad_id=1, name="X", transmitters=(), alive=make_alive())

        assert profile.best_reliability() is None

    def test_single_transmitter_reliability(self):
        profile = SatelliteProfile(
            norad_id=1,
            name="X",
            transmitters=(make_transmitter(reliability=ReliabilityClass.DEPENDENT),),
            alive=make_alive(),
        )

        assert profile.best_reliability() is ReliabilityClass.DEPENDENT

    def test_unconditional_beats_dependent(self):
        # RS-44's own real shape: a beacon (unconditional) and a
        # transponder (dependent) on the same bird.
        profile = SatelliteProfile(
            norad_id=44909,
            name="RS-44",
            transmitters=(
                make_transmitter(reliability=ReliabilityClass.UNCONDITIONAL),
                make_transmitter(reliability=ReliabilityClass.DEPENDENT),
            ),
            alive=make_alive(),
        )

        assert profile.best_reliability() is ReliabilityClass.UNCONDITIONAL

    def test_scheduled_beats_dependent_but_not_unconditional(self):
        scheduled_and_dependent = SatelliteProfile(
            norad_id=1,
            name="X",
            transmitters=(
                make_transmitter(reliability=ReliabilityClass.SCHEDULED),
                make_transmitter(reliability=ReliabilityClass.DEPENDENT),
            ),
            alive=make_alive(),
        )
        all_three = SatelliteProfile(
            norad_id=2,
            name="Y",
            transmitters=(
                make_transmitter(reliability=ReliabilityClass.SCHEDULED),
                make_transmitter(reliability=ReliabilityClass.DEPENDENT),
                make_transmitter(reliability=ReliabilityClass.UNCONDITIONAL),
            ),
            alive=make_alive(),
        )

        assert scheduled_and_dependent.best_reliability() is ReliabilityClass.SCHEDULED
        assert all_three.best_reliability() is ReliabilityClass.UNCONDITIONAL
