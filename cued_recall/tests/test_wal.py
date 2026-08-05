"""WAL reads, which the admin page polls.

read_all() parsed the entire append-only log into dicts on every stats, budget
and block-history request, so every admin page refresh got slower for as long
as the server stayed up. The replacements have to agree with it exactly.
"""

import json

import pytest

from cued_recall.wal import WAL


@pytest.fixture
def wal(tmp_path):
    w = WAL(tmp_path / "wal.jsonl")
    w.open()
    yield w
    w.close()


def fill(wal, n, event="turn_completed", block_id=""):
    for i in range(n):
        e = {"event": event, "i": i}
        if block_id:
            e["block_id"] = block_id
        wal.write(e)


class TestCount:
    def test_empty(self, wal):
        assert wal.count() == 0

    def test_tracks_writes(self, wal):
        fill(wal, 7)
        assert wal.count() == 7

    def test_counts_an_existing_log_on_open(self, tmp_path):
        path = tmp_path / "wal.jsonl"
        path.write_text('{"event":"a"}\n{"event":"b"}\n', encoding="utf-8")
        w = WAL(path)
        w.open()
        try:
            assert w.count() == 2
            w.write({"event": "c"})
            assert w.count() == 3
        finally:
            w.close()

    def test_unopened_wal_still_counts(self, tmp_path):
        path = tmp_path / "wal.jsonl"
        path.write_text('{"event":"a"}\n\n{"event":"b"}\n', encoding="utf-8")
        assert WAL(path).count() == 2

    def test_agrees_with_read_all(self, wal):
        fill(wal, 25)
        assert wal.count() == len(wal.read_all())


class TestIterAll:
    def test_yields_oldest_first(self, wal):
        fill(wal, 3)
        assert [e["i"] for e in wal.iter_all()] == [0, 1, 2]

    def test_skips_a_torn_line_rather_than_raising(self, wal):
        wal.write({"event": "a"})
        with open(wal.path, "a", encoding="utf-8") as f:
            f.write('{"event": "b", "trunc\n')
        wal.write({"event": "c"})
        # A hard kill mid-write leaves a partial line. Diagnostics are not
        # worth failing an admin request over.
        assert [e["event"] for e in wal.iter_all()] == ["a", "c"]


class TestTailEvents:
    def test_newest_first(self, wal):
        fill(wal, 5)
        assert [e["i"] for e in wal.tail_events(3)] == [4, 3, 2]

    def test_filters_by_event_type(self, wal):
        wal.write({"event": "recall_budget", "i": 0})
        wal.write({"event": "turn_completed", "i": 1})
        wal.write({"event": "recall_budget", "i": 2})
        got = wal.tail_events(10, event="recall_budget")
        assert [e["i"] for e in got] == [2, 0]

    def test_filters_by_block_id(self, wal):
        wal.write({"event": "judge_action", "block_id": "a"})
        wal.write({"event": "judge_action", "block_id": "b"})
        wal.write({"event": "judge_action", "block_id": "a"})
        assert len(wal.tail_events(10, block_id="a")) == 2

    def test_matches_what_a_full_scan_would_have_returned(self, wal):
        for i in range(200):
            wal.write({"event": "recall_budget" if i % 3 == 0 else "other",
                       "i": i})
        expected = [e for e in wal.read_all() if e["event"] == "recall_budget"]
        got = wal.tail_events(50, event="recall_budget")
        assert got == list(reversed(expected[-50:]))

    def test_reads_lines_that_straddle_a_chunk_boundary(self, wal):
        # The reverse reader walks the file in 64 KB chunks; a line split
        # across two of them must still come back whole.
        for i in range(50):
            wal.write({"event": "big", "i": i, "pad": "x" * 4000})
        got = wal.tail_events(50, event="big")
        assert [e["i"] for e in got] == list(range(49, -1, -1))
        assert all(len(e["pad"]) == 4000 for e in got)

    def test_stops_early_instead_of_scanning_the_whole_log(self, wal):
        fill(wal, 5000)
        assert len(wal.tail_events(10)) == 10

    def test_empty_log_and_zero_limit(self, wal):
        assert wal.tail_events(10) == []
        fill(wal, 3)
        assert wal.tail_events(0) == []

    def test_handles_a_log_with_no_trailing_newline(self, tmp_path):
        path = tmp_path / "wal.jsonl"
        path.write_text('{"event":"a"}\n{"event":"b"}', encoding="utf-8")
        assert [e["event"] for e in WAL(path).tail_events(10)] == ["b", "a"]

    def test_survives_non_utf8_bytes(self, tmp_path):
        path = tmp_path / "wal.jsonl"
        with open(path, "wb") as f:
            f.write(b'{"event":"a"}\n')
            f.write(b'\xff\xfe not json\n')
            f.write(b'{"event":"c"}\n')
        assert [e["event"] for e in WAL(path).tail_events(10)] == ["c", "a"]
