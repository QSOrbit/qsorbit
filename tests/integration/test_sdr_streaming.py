"""Bench measurement: does reading an RTL-SDR synchronously keep up?

This suite exists to answer one question with a number, and the question
was created deliberately. Session 15 chose a reader thread looping
``rtlsdr_read_sync`` over ``rtlsdr_read_async``, knowing the cost: no USB
transfer is in flight between one synchronous read returning and the
next being issued, so the device's FIFO may overflow under sustained
load. Rather than argue about whether that matters, the decision was to
**measure it here**, with the surface small, instead of discovering it
in Chunk F underneath a waterfall.

**There are two tests because there are two different faults**, and a
single "dropped blocks" figure cannot tell them apart:

:class:`TestSyncReadKeepsUp`
    A bare ``read_raw`` loop. No thread, no buffer, no disk — the
    consumer is structurally incapable of being the bottleneck, so
    anything measured is the sync-read gap itself. **This is the design
    question.** If this one shows loss, the fix is
    ``rtlsdr_read_async`` behind the same interface.

:class:`TestThePipelineKeepsUp`
    The whole shipped stack: reader thread, bounded buffer, capture to
    disk. Any loss here beyond the first test's is ours — a slow
    consumer or too shallow a buffer — and rewriting the reader would
    not touch it.

If the first is clean and the second is not, rewriting the reader would
have been wasted work, which is the entire reason they are separate.
**That comparison only means something well above
:data:`MEASUREMENT_NOISE_FLOOR_FRACTION`.** Once both figures are near
zero the gap between them is stopwatch noise and reverses run to run;
two measurements on 2026-08-22 had the pipeline above the bare loop and
then below it.

**How the loss is measured.** The device's sample clock cannot lie about
how many bytes it must have produced, so expected bytes are compared
against bytes that actually arrived, per read. Test mode's incrementing
counter was considered and rejected: an 8-bit counter recovers a gap
only modulo 256, and real losses arrive in USB-transfer-shaped chunks
that are overwhelmingly multiples of 256, which is why ``rtl_test``
hedges its own figure as "Samples per million lost (**minimum**)".

**The numbers survive the run.** A passing pytest swallows stdout, so
every run also writes :data:`REPORT_NAME` next to the IQ fixtures.
Paste that into ``project-notes.md``; do not reconstruct it from
scrollback.

Run at the bench, with the V4 attached and SDR#/SDR++ closed::

    uv run pytest -m integration -k streaming -s
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from qsorbit.core.sdr import (
    DEFAULT_READ_BYTES,
    DeviceNotFoundError,
    DriverError,
    RtlSdr,
    SdrConfig,
    ThroughputMonitor,
    byte_rate_for,
    capture_to_file,
)
from qsorbit.core.station import ConfigError, load_station_config

pytestmark = pytest.mark.integration

#: The rate the whole of Phase 2 is built around, and the one bring-up
#: captured at.
SAMPLE_RATE_HZ = 2_048_000

#: Long enough that a once-a-second hiccup has somewhere to show up, and
#: short enough to run twice at a bench without it becoming a chore.
MEASUREMENT_SECONDS = 30.0

#: Shorter than the bare-loop run: this one writes ~20 MB to disk, and
#: the question it answers is comparative rather than absolute.
PIPELINE_SECONDS = 5.0

#: Bytes per read. **The parameter that governs this whole risk.** The
#: sync design's exposure is the gap *between* reads, so halving this
#: doubles how many gaps occur per second. If a measurement comes back
#: bad, try a larger block before concluding the design is wrong.
BLOCK_BYTES = DEFAULT_READ_BYTES

#: Where a bad number stops being a curiosity.
#:
#: Set from a real baseline rather than invented. The first measurement
#: (2026-08-22, RTL-SDR Blog V4, Windows, Python 3.13) ran at 5% purely
#: as a catastrophe guard, caught a 9.5% failure, and the fix took it to
#: **0.0021%** for the bare loop and **0.0058%** for the full pipeline.
#: One percent therefore leaves well over a hundredfold headroom for a
#: slower machine while being tight enough that the specific bug that
#: was found — a fixed per-read cost inside the read path — cannot come
#: back quietly at a third of its old size.
MAX_ACCEPTABLE_LOSS_FRACTION = 0.01

#: What a healthy run looks like, from that same baseline. Not asserted —
#: reported, so a run that clears the guard above but sits an order of
#: magnitude away from this is still visible rather than silently green.
TARGET_LOSS_FRACTION = 0.001

#: Below roughly this, the figure is the stopwatch rather than the
#: device. One millisecond is 4,096 bytes at 2.048 Msps, and across two
#: runs on 2026-08-22 the bare loop reported 0.62 ms and 1.69 ms of
#: discrepancy over 30 seconds while the pipeline reported 0.29 ms and
#: **minus** 0.12 ms over 5. Three things feed it: the dongle's crystal
#: is not the PC's, and ``byte_rate`` comes from the nominal rate
#: librtlsdr reports rather than the true one (a 56 ppm crystal alone
#: accounts for run 2's figure); the device's FIFO can bank samples
#: before the window opens and hand them over inside it, which is what
#: produces a negative; and one late final read charges the window for
#: samples that are still safely buffered.
#:
#: **Do not read a difference between the two tests at this scale.** An
#: earlier reading of a single pair — pipeline slightly above bare loop,
#: "the cost of a thread and a disk write" — did not survive the second
#: run. The 9.5% signal that found the marshalling bug was 1,700x this
#: floor, which is why *that* comparison meant something.
MEASUREMENT_NOISE_FLOOR_FRACTION = 0.0001

#: Written beside the IQ fixtures, where ``.gitignore`` already excludes
#: ``*.json``, so a measurement never accidentally gets committed.
REPORT_NAME = "streaming-measurement.json"


def report_dir() -> Path:
    """Return ``tests/fixtures/iq/``, creating it if a fresh clone lacks it."""
    directory = Path(__file__).resolve().parents[1] / "fixtures" / "iq"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_report(section: str, payload: dict) -> None:
    """Merge one measurement into the report file, keeping the other.

    Merged rather than overwritten so that running a single test with
    ``-k`` does not silently discard the other half of the comparison —
    which is the half that says whose fault a bad number is.
    """
    path = report_dir() / REPORT_NAME
    existing: dict = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing[section] = payload
    existing["written_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")


@pytest.fixture(scope="module")
def config():
    """The operator's real station config, or a skip."""
    try:
        return load_station_config(None)
    except ConfigError as exc:
        pytest.skip(f"No usable station config, so there is no SDR to talk to: {exc}")


@pytest.fixture(scope="module")
def sdr(config):
    """One open device for the whole module."""
    device = RtlSdr(config.sdr.device_index, driver_dir=config.sdr.driver_dir)
    try:
        device.open()
    except DriverError as exc:
        pytest.skip(f"Could not load librtlsdr: {exc}")
    except DeviceNotFoundError as exc:
        pytest.skip(f"No RTL-SDR attached: {exc}")
    yield device
    device.close()


def a_config(config) -> SdrConfig:
    """A capture configuration using the station's own ppm correction."""
    return SdrConfig(
        center_hz=99_650_000,
        sample_rate_hz=SAMPLE_RATE_HZ,
        gain_db=32.8,
        ppm=config.sdr.ppm,
    )


class TestSyncReadKeepsUp:
    """The design question, isolated from everything else we wrote."""

    @pytest.fixture(scope="class")
    def measurement(self, sdr, config):
        applied = sdr.configure(a_config(config))
        monitor = ThroughputMonitor(byte_rate_for(applied.sample_rate_hz))

        deadline = time.monotonic() + MEASUREMENT_SECONDS
        reads = 0
        while time.monotonic() < deadline:
            block = sdr.read_raw(BLOCK_BYTES)
            monitor.record(len(block))
            reads += 1

        report = monitor.report()
        print(f"\nBare sync-read loop: {report.describe()}")
        write_report(
            "bare_sync_read_loop",
            {
                "block_bytes": BLOCK_BYTES,
                "requested_seconds": MEASUREMENT_SECONDS,
                "actual_sample_rate_hz": applied.sample_rate_hz,
                "reads": reads,
                "accounted_reads": report.reads,
                "bytes_read": report.bytes_read,
                "elapsed_s": report.elapsed_s,
                "lost_bytes": report.lost_bytes,
                "loss_fraction": report.loss_fraction,
                "stalls": report.stalls,
                "worst_stall_s": report.worst_stall_s,
                "worst_ten_deficits_s": sorted(report.deficits_s, reverse=True)[:10],
                "summary": report.describe(),
            },
        )
        return report

    def test_the_loop_actually_ran_for_its_full_duration(self, measurement):
        # Guards the measurement itself. A run that ended early would
        # report a small, meaningless loss and look like good news.
        assert measurement.elapsed_s >= MEASUREMENT_SECONDS * 0.9
        assert measurement.reads > 0

    def test_sync_reads_keep_up_with_the_device(self, measurement):
        mean_deficit_ms = (
            sum(measurement.deficits_s) / len(measurement.deficits_s) * 1000
            if measurement.deficits_s
            else 0.0
        )
        block_ms = BLOCK_BYTES / measurement.byte_rate * 1000

        assert measurement.loss_fraction < MAX_ACCEPTABLE_LOSS_FRACTION, (
            f"Synchronous reads lost {measurement.loss_fraction * 100:.3f}% of the "
            f"stream ({measurement.lost_bytes:,.0f} bytes) at {SAMPLE_RATE_HZ:,} sps "
            f"with {BLOCK_BYTES:,}-byte blocks.\n\n"
            f"Mean deficit {mean_deficit_ms:.2f} ms per read against a "
            f"{block_ms:.1f} ms block; worst {measurement.worst_stall_s * 1000:.1f} ms; "
            f"{measurement.stalls} of {measurement.reads} reads stalled.\n\n"
            "READ THE SHAPE BEFORE CHOOSING A FIX. If nearly every read is "
            "short by nearly the same amount, this is a fixed cost being paid "
            "inside the read path, not the device failing to keep up - and "
            "rtlsdr_read_async would not touch it, because the same work "
            "happens in the callback. That is exactly what happened on "
            "2026-08-22: bytes(buffer[:n]) in the ctypes binding was building "
            "a quarter-million-element Python list per read, costing 6.7 ms "
            "against a 64 ms block, and both this test and the pipeline test "
            "reported an identical 9.5%. Profile one read first.\n\n"
            "If instead the deficits are bursty and heavy-tailed, it really is "
            "the sync-read gap. Try a larger block before rewriting anything: "
            "the loss window sits between reads, so fewer and bigger means "
            "fewer gaps."
        )

    def test_the_deficits_agree_with_the_aggregate(self, measurement):
        # The accounting's own self-check. If these disagree the
        # instrument is broken, and the number above means nothing.
        assert sum(measurement.deficits_s) * measurement.byte_rate == pytest.approx(
            measurement.lost_bytes, abs=1.0
        )

    def test_the_device_did_not_deliver_an_impossible_surplus(self, measurement):
        # The other side of the bound, and it guards the instrument
        # rather than the device. A *small* negative is ordinary - see
        # MEASUREMENT_NOISE_FLOOR_FRACTION - but a large one cannot
        # happen physically: no dongle produces samples faster than its
        # own clock. It would mean byte_rate is wrong, which silently
        # invalidates every figure this suite reports, loss and target
        # alike. Without this, only over-reporting is ever caught.
        assert measurement.loss_fraction > -MAX_ACCEPTABLE_LOSS_FRACTION, (
            f"The run delivered {-measurement.lost_bytes:,.0f} bytes MORE than the "
            f"sample clock could have produced ({measurement.loss_fraction * 100:.3f}%). "
            "Small surpluses are normal - the device banks samples in its FIFO. A "
            "large one means the byte rate this was measured against is wrong, so "
            "check that the sample rate came from AppliedSettings and not from what "
            "was requested."
        )


class TestThePipelineKeepsUp:
    """The shipped stack: reader thread, bounded buffer, disk."""

    @pytest.fixture(scope="class")
    def capture(self, sdr, config, tmp_path_factory):
        target = tmp_path_factory.mktemp("streaming") / "pipeline.iq"
        result = capture_to_file(
            sdr,
            a_config(config),
            target,
            seconds=PIPELINE_SECONDS,
            station_hz=99_900_000,
            block_bytes=BLOCK_BYTES,
        )
        print(f"\nFull pipeline: {result.describe()}")
        write_report(
            "full_pipeline_capture",
            {
                "block_bytes": BLOCK_BYTES,
                "requested_seconds": PIPELINE_SECONDS,
                "bytes_written": result.metadata["bytes"],
                "contiguous": result.is_contiguous,
                "blocks_read": result.stats.blocks_read,
                "blocks_dropped": result.stats.blocks_dropped,
                "reader_stopped_cleanly": result.stats.reader_stopped_cleanly,
                "lost_bytes": result.stats.loss.lost_bytes,
                "loss_fraction": result.stats.loss.loss_fraction,
                "stalls": result.stats.loss.stalls,
                "worst_stall_s": result.stats.loss.worst_stall_s,
                "summary": result.stats.describe(),
            },
        )
        return result

    def test_the_capture_is_the_size_it_should_be(self, capture):
        expected = int(round(PIPELINE_SECONDS * byte_rate_for(capture.applied.sample_rate_hz)))

        assert capture.iq_path.stat().st_size == expected

    def test_nothing_was_dropped_at_our_buffer(self, capture):
        # If this fails while the bare-loop test passes, the fault is
        # ours - a slow consumer or too shallow a buffer - and swapping
        # to rtlsdr_read_async would not help at all.
        assert capture.stats.blocks_dropped == 0, (
            f"{capture.stats.blocks_dropped} block(s) were discarded because the "
            "buffer filled. That is a consumer-side fault, not a device one: "
            "increase queue_blocks or find what stalled the write."
        )
        assert capture.is_contiguous

    def test_the_pipeline_keeps_up_too(self, capture):
        loss = capture.stats.loss

        assert loss.loss_fraction < MAX_ACCEPTABLE_LOSS_FRACTION, (
            f"The full pipeline lost {loss.loss_fraction * 100:.3f}%. Compare "
            "against the bare-loop figure in the report file before concluding "
            "anything: if that one is clean, the fault is in our thread, our "
            "buffer or our consumer, and changing how the device is read would "
            "not touch it.\n\n"
            "Only compare the two when both are well clear of "
            f"{MEASUREMENT_NOISE_FLOOR_FRACTION * 100:.2f}%. Below that the "
            "difference between them is timing noise and means nothing."
        )

    def test_the_pipeline_did_not_deliver_an_impossible_surplus(self, capture):
        # Same instrument guard as the bare loop's. See that test.
        assert capture.stats.loss.loss_fraction > -MAX_ACCEPTABLE_LOSS_FRACTION, (
            f"The capture delivered {-capture.stats.loss.lost_bytes:,.0f} bytes more "
            "than the sample clock could have produced. A small surplus is the "
            "device's FIFO; a large one means byte_rate is wrong."
        )

    def test_the_reader_thread_shut_down_cleanly(self, capture):
        # librtlsdr uses an infinite bulk timeout, so a wedged read
        # cannot be interrupted. A dirty shutdown here means one was.
        assert capture.stats.reader_stopped_cleanly

    def test_the_sidecar_records_the_actual_tuning(self, capture):
        metadata = json.loads(capture.sidecar_path.read_text(encoding="utf-8"))

        assert metadata["actual_center_hz"] == capture.applied.center_hz
        assert metadata["actual_sample_rate_hz"] == capture.applied.sample_rate_hz
        # Off-centre by design: the station must not sit on the DC spike.
        assert abs(metadata["station_offset_hz"]) > 100_000

    def test_the_capture_is_not_a_constant(self, capture):
        # Cheapest possible guard against a dead capture. Says nothing
        # about signal presence, which is a dynamic-range question and
        # needs a spectrum - that arrives with numpy in Chunk E.
        head = capture.iq_path.read_bytes()[:65_536]

        assert len(set(head)) > 1
