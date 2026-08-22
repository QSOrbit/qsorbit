"""Hardware integration tests: a real RTL-SDR on a real USB port.

These are the SDR half of the pattern the rotor suite established, with
one difference worth naming. The rotor suite is strictly read-only
because a test that slews an antenna needs someone standing near a power
switch. Tuning a receiver has no such consequence — nothing moves,
nothing radiates — so these tests *do* configure the device and read
samples from it. That is the point: the ctypes signatures in
:mod:`qsorbit.core.sdr.librtlsdr` cannot be verified by any unit test,
because nothing in a unit test marshals across a C boundary. Only real
hardware proves them.

They are deselected by default (see ``addopts`` in ``pyproject.toml``).
Run them at the bench with::

    uv run pytest -m integration

They use the real station config, so they exercise the ``[sdr]`` section
the operator actually runs — including ``driver_dir``, which is the
setting most likely to be wrong on a fresh Windows machine.

**Close SDR#, SDR++, and any stray ``rtl_test`` first.** A device that
is already open cannot be opened again, and that shows up here as a
failure rather than a skip, because it is a real finding about the
state of the machine.

As with the rotor suite: no device means skip, because there is no bug
to report in "the dongle isn't plugged in". A device that answers and
then misbehaves *fails*.
"""

from __future__ import annotations

import pytest

from qsorbit.core.sdr import (
    DeviceNotFoundError,
    DriverError,
    RtlSdr,
    SdrConfig,
    TunerType,
)
from qsorbit.core.station import ConfigError, load_station_config

pytestmark = pytest.mark.integration

#: A capture the device can produce anywhere, with no antenna and no
#: assumptions about what is on the air. Centred in the FM broadcast
#: band only because it is a frequency every RTL-SDR can reach.
PROBE_CONFIG = SdrConfig(
    center_hz=99_650_000,
    sample_rate_hz=2_048_000,
    gain_db=32.8,
)

#: How far the tuner may land from the requested centre before something
#: is wrong. The PLL quantises, so exact is not expected; kilohertz of
#: error at 100 MHz would be. Bring-up measured this device landing
#: exactly on request.
CENTRE_TOLERANCE_HZ = 1_000.0

#: The R828D's gain table, as reported by this V4 during bring-up. Used
#: as a fingerprint rather than a specification: a *different* table
#: means a different library got loaded than the one intended.
R828D_GAIN_STEPS = 29
R828D_MAX_GAIN_DB = 49.6


@pytest.fixture(scope="module")
def config():
    """The operator's real station config, or a skip."""
    try:
        return load_station_config(None)
    except ConfigError as exc:
        pytest.skip(f"No usable station config, so there is no SDR to talk to: {exc}")


@pytest.fixture(scope="module")
def sdr(config):
    """One open device, held for the whole module.

    Opening is not free — it resets the device and re-negotiates USB —
    and nothing here needs a fresh one per test.
    """
    device = RtlSdr(config.sdr.device_index, driver_dir=config.sdr.driver_dir)
    try:
        device.open()
    except DriverError as exc:
        # Could not even load the library. Not a device fault, and not
        # something the suite can work around.
        pytest.skip(f"Could not load librtlsdr: {exc}")
    except DeviceNotFoundError as exc:
        pytest.skip(f"No RTL-SDR attached: {exc}")

    yield device
    device.close()


@pytest.fixture(scope="module")
def configured(sdr, config):
    """The device, tuned and ready, with the station's own ppm applied."""
    return sdr.configure(
        SdrConfig(
            center_hz=PROBE_CONFIG.center_hz,
            sample_rate_hz=PROBE_CONFIG.sample_rate_hz,
            gain_db=PROBE_CONFIG.gain_db,
            ppm=config.sdr.ppm,
        )
    )


class TestIdentity:
    def test_the_device_identifies_itself(self, sdr):
        assert sdr.info is not None
        assert sdr.info.describe()

    def test_it_is_the_blog_v4_this_project_was_built_against(self, sdr):
        if not sdr.info.is_blog_v4:
            pytest.skip(
                f"Attached device is not an RTL-SDR Blog V4 ({sdr.info.describe()}). "
                "The device layer is not V4-only, but the rest of this module's "
                "expectations were measured against one."
            )
        assert sdr.info.tuner is TunerType.R828D


class TestGainTable:
    """The gain table doubles as a fingerprint of the loaded library."""

    def test_the_device_reports_gain_steps(self, sdr):
        assert sdr.supported_gains_db()

    def test_the_steps_are_the_ones_the_r828d_reports(self, sdr):
        if not sdr.info.is_blog_v4:
            pytest.skip("Fingerprint is specific to the V4's R828D.")
        gains = sdr.supported_gains_db()

        assert len(gains) == R828D_GAIN_STEPS, (
            "This tuner reported a different gain table than the V4's R828D. "
            "The most likely explanation is that a different librtlsdr got "
            "loaded than the one driver_dir points at."
        )
        assert max(gains) == pytest.approx(R828D_MAX_GAIN_DB)


class TestConfiguration:
    def test_it_tunes_close_to_where_it_was_told(self, configured):
        assert abs(configured.center_error_hz) < CENTRE_TOLERANCE_HZ

    def test_the_actual_centre_is_read_from_the_device(self, configured):
        # Not echoed back from the request — that would make the check
        # above meaningless.
        assert configured.center_hz > 0

    def test_the_sample_rate_is_close_to_what_was_asked_for(self, configured):
        assert abs(configured.sample_rate_error_hz) < 1_000.0

    def test_manual_gain_is_in_force(self, configured):
        assert configured.manual_gain

    def test_the_gain_is_not_zero(self, configured):
        # The bring-up failure in executable form: a device reporting
        # 0.0 dB captures nothing, and nothing else reports an error.
        assert not configured.reports_zero_gain

    def test_the_gain_landed_on_a_real_step(self, sdr, configured):
        assert configured.gain_db in sdr.supported_gains_db()

    def test_setting_an_unchanged_ppm_is_not_an_error(self, sdr, configured):
        # librtlsdr returns -2 for "already that value". This is the one
        # place that return path meets real hardware.
        sdr.configure(configured.requested)

    def test_it_can_be_retuned(self, sdr, config):
        applied = sdr.configure(
            SdrConfig(
                center_hz=162_300_000,
                sample_rate_hz=2_048_000,
                gain_db=49.6,
                ppm=config.sdr.ppm,
            )
        )

        assert abs(applied.center_error_hz) < CENTRE_TOLERANCE_HZ
        assert applied.gain_db == pytest.approx(49.6)


class TestReading:
    def test_a_read_returns_the_requested_number_of_bytes(self, sdr, configured):
        assert len(sdr.read_raw(65_536)) == 65_536

    def test_repeated_reads_keep_working(self, sdr, configured):
        for _ in range(3):
            assert len(sdr.read_raw(16_384)) == 16_384

    def test_the_samples_are_not_a_constant(self, sdr, configured):
        # A dead capture is a real failure mode and this is the cheapest
        # check that can catch it. Note what it deliberately does NOT
        # claim: a varying capture proves the ADC is running, not that
        # any signal is present. Signal presence is a dynamic-range
        # question and needs an antenna, a known station and a spectrum —
        # which is the capture utility's job, not this one's.
        raw = sdr.read_raw(16_384)

        assert len(set(raw)) > 1, (
            "Every byte of the capture was identical, so the ADC is not "
            "producing data. Check the gain is not 0 dB and that nothing "
            "else has the device open."
        )
