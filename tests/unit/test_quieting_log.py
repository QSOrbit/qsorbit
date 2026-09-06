"""Tests for the per-branch quieting log.

The log exists to produce one number: the switching margin a combiner
selects on. So the tests are about the properties that number depends
on -- that both branches measure elapsed time from **one** origin, that
rows are on disk before the run ends, and that a branch which wrote
nothing is visible rather than absorbed into a total.
"""

from __future__ import annotations

import csv
import threading
from datetime import UTC, datetime, timedelta

import pytest

from qsorbit.core.quieting_log import CSV_COLUMNS, QuietingLog

AN_INSTANT = datetime(2026, 9, 6, 18, 30, 0, tzinfo=UTC)


def at(seconds: float) -> datetime:
    return AN_INSTANT + timedelta(seconds=seconds)


def a_log(tmp_path, name="quieting.csv", start=AN_INSTANT):
    return QuietingLog(tmp_path / name, now=lambda: start)


def rows_of(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


class TestHeader:
    def test_it_writes_the_header_on_open(self, tmp_path):
        log = a_log(tmp_path)
        log.open()
        log.close()

        assert rows_of(log.path) == [list(CSV_COLUMNS)]

    def test_opening_replaces_an_existing_file(self, tmp_path):
        # Two runs concatenated is worse than no log: the time column
        # restarts in the middle and nothing says so.
        path = tmp_path / "quieting.csv"
        path.write_text("stale rubbish\n", encoding="utf-8")

        log = QuietingLog(path, now=lambda: AN_INSTANT)
        log.open()
        log.close()

        assert rows_of(path) == [list(CSV_COLUMNS)]


class TestRows:
    def test_it_records_a_branch_measurement(self, tmp_path):
        log = a_log(tmp_path)
        log.open()
        log.record(at(0.5), "A - Arrow V", 17.42, gate_open=True)
        log.close()

        assert rows_of(log.path)[1] == ["0.500", "A - Arrow V", "17.42", "1"]

    def test_a_closed_gate_is_zero_not_blank(self, tmp_path):
        # Blank would be indistinguishable from a column that was never
        # written, which is the same class of confusion as "off" and
        # "broken" looking alike.
        log = a_log(tmp_path)
        log.open()
        log.record(at(0.5), "B", -1.8, gate_open=False)
        log.close()

        assert rows_of(log.path)[1] == ["0.500", "B", "-1.80", "0"]

    def test_rows_are_flushed_as_they_are_written(self, tmp_path):
        # Read back while the log is still open. A run that ends in a
        # fault is the one whose data matters most, and a buffered
        # final write is what loses it.
        log = a_log(tmp_path)
        log.open()
        log.record(at(0.1), "A", 5.0, gate_open=True)

        assert len(rows_of(log.path)) == 2

        log.close()

    def test_recording_before_opening_is_refused(self, tmp_path):
        with pytest.raises(RuntimeError, match="not been opened"):
            a_log(tmp_path).record(at(0.0), "A", 5.0, gate_open=True)


class TestTimeOrigin:
    def test_both_branches_measure_from_one_origin(self, tmp_path):
        # The property the whole measurement rests on. A difference
        # between two series carried on two origins measures the
        # origins, so the log holds exactly one and neither caller
        # supplies elapsed time.
        log = a_log(tmp_path)
        log.open()
        log.record(at(1.0), "A", 10.0, gate_open=True)
        log.record(at(1.0), "B", 4.0, gate_open=False)
        log.close()

        first, second = rows_of(log.path)[1:]
        assert first[0] == second[0] == "1.000"

    def test_the_origin_is_fixed_at_open_not_at_the_first_row(self, tmp_path):
        # Taken at open() so that whichever branch happens to read a
        # block first cannot move the other one's clock.
        log = a_log(tmp_path)
        log.open()
        log.record(at(4.0), "B", 4.0, gate_open=False)
        log.record(at(1.0), "A", 10.0, gate_open=True)
        log.close()

        assert [row[0] for row in rows_of(log.path)[1:]] == ["4.000", "1.000"]

    def test_a_block_older_than_the_log_is_written_negative(self, tmp_path):
        # Not clamped: it means a block predates the log, which is worth
        # seeing rather than hiding behind a zero.
        log = a_log(tmp_path)
        log.open()
        log.record(at(-0.25), "A", 10.0, gate_open=True)
        log.close()

        assert rows_of(log.path)[1][0] == "-0.250"


class TestCounting:
    def test_it_counts_rows_per_branch(self, tmp_path):
        log = a_log(tmp_path)
        log.open()
        log.record(at(0.1), "A", 1.0, gate_open=True)
        log.record(at(0.2), "B", 2.0, gate_open=True)
        log.record(at(0.3), "A", 3.0, gate_open=True)
        log.close()

        assert log.rows == 3
        assert log.rows_per_branch == {"A": 2, "B": 1}

    def test_describe_breaks_the_count_down_by_branch(self, tmp_path):
        # A total alone is what would let one branch write nothing and
        # still look right -- and a branch that wrote nothing is the
        # failure the whole measurement would then be built on.
        log = a_log(tmp_path)
        log.open()
        log.record(at(0.1), "A - Arrow V", 1.0, gate_open=True)
        log.record(at(0.2), "B - Arrow H", 2.0, gate_open=True)
        log.record(at(0.3), "A - Arrow V", 3.0, gate_open=True)
        log.close()

        described = log.describe()
        assert "3 row(s)" in described
        assert "A - Arrow V 2" in described
        assert "B - Arrow H 1" in described

    def test_describe_says_so_when_nothing_was_written(self, tmp_path):
        log = a_log(tmp_path)
        log.open()
        log.close()

        assert "no rows" in log.describe()


class TestConcurrency:
    def test_two_writers_do_not_interleave_inside_a_row(self, tmp_path):
        # Two demodulating threads write to one log. Without the lock a
        # row can be torn, and a torn row is a corrupt sample in the
        # middle of the series rather than an obvious failure.
        log = a_log(tmp_path)
        log.open()
        start = threading.Barrier(2)

        def write(label, count):
            start.wait()
            for index in range(count):
                log.record(at(index * 0.064), label, float(index), gate_open=True)

        threads = [
            threading.Thread(target=write, args=("A - Arrow V", 200)),
            threading.Thread(target=write, args=("B - Arrow H", 200)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10.0)
        log.close()

        rows = rows_of(log.path)[1:]
        assert len(rows) == 400
        assert all(len(row) == len(CSV_COLUMNS) for row in rows)
        assert sorted(log.rows_per_branch.values()) == [200, 200]


class TestContextManager:
    def test_it_opens_and_closes(self, tmp_path):
        path = tmp_path / "quieting.csv"

        with QuietingLog(path, now=lambda: AN_INSTANT) as log:
            log.record(at(0.1), "A", 1.0, gate_open=True)

        assert len(rows_of(path)) == 2

    def test_closing_twice_is_safe(self, tmp_path):
        log = a_log(tmp_path)
        log.open()
        log.close()
        log.close()
