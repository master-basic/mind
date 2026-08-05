"""VectorIndex against a real sqlite-vec database.

The two behaviours worth pinning: query() filters by status *after* the KNN
(the sqlite-vec footgun the over-fetch exists for), and blocks_without_vectors()
finds the silent-vector-loss condition that made 57 real blocks unrecallable.
"""

import math

import pytest

from cued_recall.index import VectorIndex

DIM = 4


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


@pytest.fixture
def index(tmp_path):
    ix = VectorIndex(tmp_path, dim=DIM)
    ix.open()
    yield ix
    ix.close()


def add(ix, block_id, vec, status="shelved", *, conversation_id="c1",
        turn_index=0, type_="reasoning", verification="unknown",
        recall_count=0, token_count=10, created_at=1000.0, with_vector=True):
    ix.upsert_block_meta(block_id, type_, status, created_at, conversation_id,
                         turn_index, token_count, verification, recall_count, 0.0)
    if with_vector:
        ix.upsert_vector(block_id, unit(vec))


class TestQuery:
    def test_returns_similar_shelved_blocks_above_threshold(self, index):
        add(index, "near", [1, 0, 0, 0])
        add(index, "mid", [1, 1, 0, 0])
        add(index, "far", [0, 1, 0, 0])
        got = index.query(unit([1, 0, 0, 0]), k=10, threshold=0.5)
        assert [b for b, _ in got] == ["near", "mid"]
        assert got[0][1] == pytest.approx(1.0, abs=1e-5)
        assert got[1][1] == pytest.approx(0.7071, abs=1e-3)

    def test_hot_blocks_are_never_recalled(self, index):
        # A turn's own blocks are 'hot' until the next turn shelves them.
        # Recalling them would inject the current turn into itself.
        add(index, "hot", [1, 0, 0, 0], status="hot")
        add(index, "shelved", [1, 0, 0, 0], status="shelved")
        assert [b for b, _ in index.query(unit([1, 0, 0, 0]), 10, 0.5)] == ["shelved"]

    def test_a_wall_of_hot_blocks_does_not_hide_a_shelved_one(self, index):
        # The footgun in full: sqlite-vec resolves k against block_vec alone,
        # so filtering status in the WHERE clause would return nothing here.
        # The k*50 over-fetch is what keeps "match" visible.
        for i in range(60):
            add(index, f"hot{i}", [1, 0.001 * i, 0, 0], status="hot")
        add(index, "match", [1, 0.5, 0, 0], status="shelved")
        got = index.query(unit([1, 0, 0, 0]), k=4, threshold=0.5)
        assert [b for b, _ in got] == ["match"]

    def test_truncated_blocks_are_recallable(self, index):
        add(index, "t", [1, 0, 0, 0], status="truncated")
        assert index.query(unit([1, 0, 0, 0]), 10, 0.5)

    def test_purged_blocks_are_not(self, index):
        add(index, "p", [1, 0, 0, 0], status="purged")
        assert index.query(unit([1, 0, 0, 0]), 10, 0.5) == []

    def test_k_caps_the_result_count(self, index):
        for i in range(10):
            add(index, f"b{i}", [1, 0.01 * i, 0, 0])
        assert len(index.query(unit([1, 0, 0, 0]), k=3, threshold=0.0)) == 3

    def test_threshold_excludes_dissimilar(self, index):
        add(index, "orthogonal", [0, 1, 0, 0])
        assert index.query(unit([1, 0, 0, 0]), 10, 0.5) == []
        assert index.query(unit([1, 0, 0, 0]), 10, 0.0)

    def test_wrong_dimension_raises_rather_than_matching_nothing(self, index):
        with pytest.raises(ValueError):
            index.query([1.0, 0.0], 4, 0.5)


class TestVectors:
    def test_upsert_replaces_rather_than_duplicating(self, index):
        # vec0 supports neither ON CONFLICT nor INSERT OR REPLACE; the
        # delete-then-insert in upsert_vector is the only working upsert, and
        # a regression here shows up as one block matching twice.
        add(index, "b", [1, 0, 0, 0])
        index.upsert_vector("b", unit([0, 1, 0, 0]))
        got = index.query(unit([0, 1, 0, 0]), 10, 0.9)
        assert [b for b, _ in got] == ["b"]

    def test_delete_vector_keeps_the_metadata_row(self, index):
        add(index, "b", [1, 0, 0, 0])
        index.delete_vector("b")
        assert index.query(unit([1, 0, 0, 0]), 10, 0.5) == []
        assert index.get_meta("b") is not None

    def test_blocks_without_vectors_finds_the_silent_loss(self, index):
        # An embed failure at creation is logged and dropped: the block keeps
        # its text, shows as 'shelved' in the admin table, and can never be
        # retrieved. Status cannot express that, which is why nothing surfaced
        # it until a manual backfill script went looking.
        add(index, "ok", [1, 0, 0, 0])
        add(index, "lost", [0, 0, 0, 0], with_vector=False)
        assert index.blocks_without_vectors() == ["lost"]

    def test_hot_blocks_are_not_reported_as_missing_vectors(self, index):
        # Hot blocks are mid-turn and not yet embedded; reporting them would
        # make the health signal cry wolf on every live conversation.
        add(index, "hot", [0, 0, 0, 0], status="hot", with_vector=False)
        assert index.blocks_without_vectors() == []


class TestMeta:
    def test_upsert_does_not_reset_created_at_or_conversation(self, index):
        add(index, "b", [1, 0, 0, 0], created_at=1000.0)
        index.upsert_block_meta("b", "reasoning", "truncated", 9999.0, "other",
                                7, 5, "accepted", 3, 42.0)
        meta = index.get_meta("b")
        # Only the mutable columns are in the DO UPDATE clause.
        assert meta["status"] == "truncated"
        assert meta["verification"] == "accepted"
        assert meta["recall_count"] == 3
        assert meta["created_at"] == 1000.0
        assert meta["conversation_id"] == "c1"

    def test_increment_recall(self, index):
        add(index, "b", [1, 0, 0, 0])
        index.increment_recall("b", 123.0)
        index.increment_recall("b", 456.0)
        meta = index.get_meta("b")
        assert meta["recall_count"] == 2
        assert meta["last_recalled"] == 456.0

    def test_tag_filter_does_not_match_a_substring_tag(self, index):
        # ",dns," must not match inside ",dns-server," -- the reason tags are
        # stored delimiter-wrapped.
        add(index, "a", [1, 0, 0, 0])
        add(index, "b", [0, 1, 0, 0])
        index.set_tags("a", ["dns"], "a gist")
        index.set_tags("b", ["python"], "b gist")
        items, count = index.list_meta(tag="dns")
        assert count == 1
        assert items[0]["block_id"] == "a"
        assert items[0]["tags"] == ["dns"]

    def test_sort_column_is_whitelisted(self, index):
        # `sort` reaches list_meta straight from an admin query string and is
        # interpolated into the SQL, so anything off the whitelist must fall
        # back rather than reach the database.
        add(index, "b", [1, 0, 0, 0])
        items, _ = index.list_meta(sort="token_count; DROP TABLE blocks")
        assert len(items) == 1

    def test_decay_candidates_never_include_pinned(self, index):
        add(index, "old", [1, 0, 0, 0], created_at=0.0)
        add(index, "pin", [0, 1, 0, 0], created_at=0.0)
        index.set_pinned("pin", True)
        assert index.decay_candidates(purge_age_s=1) == ["old"]

    def test_blocks_due_for_judging_puts_never_judged_first(self, index):
        add(index, "judged", [1, 0, 0, 0], created_at=0.0)
        add(index, "fresh", [0, 1, 0, 0], created_at=0.0)
        index.mark_judged("judged", 1.0)
        due = index.blocks_due_for_judging(min_age_s=1, rejudge_interval_s=1)
        assert due[0] == "fresh"
