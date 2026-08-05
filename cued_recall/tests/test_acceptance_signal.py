"""The recalled-uncontested signal, and its survival across a restart.

A block that was recalled into a turn and then not objected to is the only
positive evidence this system gathers on its own. It lived in a plain dict on
Pipeline, capped at 500 entries and dropped on every restart -- while the decay
rules that consume it kept running. So the reactivation evidence decay depends
on was being erased by a process restart.
"""

import pytest

from cued_recall.index import VectorIndex
from cued_recall.models import Block, BlockStatus, BlockType, Verification
from cued_recall.pipeline import Pipeline
from cued_recall.store import BlockStore
from cued_recall.wal import WAL


def put(pipeline, block_id, turn=0, verification="unknown"):
    b = Block(block_id=block_id, type=BlockType.reasoning,
              status=BlockStatus.shelved, conversation_id="c1",
              turn_index=turn, token_count=10, text="t",
              verification=Verification(verification))
    pipeline.store.put(b)
    pipeline.index.upsert_block_meta(block_id, "reasoning", "shelved",
                                     b.created_at, "c1", turn, 10,
                                     verification, 0, 0.0)
    return b


class TestRecordAndTake:
    def test_round_trip(self, pipeline):
        pipeline.index.record_recall("c1", 3, ["a", "b"], 100.0)
        assert sorted(pipeline.index.take_recalled_into_turn("c1", 3)) == ["a", "b"]

    def test_taking_consumes_it(self, pipeline):
        # Idempotence: a retried turn must not count the same evidence twice
        # and inflate a block's standing.
        pipeline.index.record_recall("c1", 3, ["a"], 100.0)
        assert pipeline.index.take_recalled_into_turn("c1", 3) == ["a"]
        assert pipeline.index.take_recalled_into_turn("c1", 3) == []

    def test_turns_and_conversations_do_not_bleed(self, pipeline):
        pipeline.index.record_recall("c1", 1, ["a"], 100.0)
        pipeline.index.record_recall("c1", 2, ["b"], 100.0)
        pipeline.index.record_recall("c2", 1, ["c"], 100.0)
        assert pipeline.index.take_recalled_into_turn("c1", 1) == ["a"]
        assert pipeline.index.take_recalled_into_turn("c2", 1) == ["c"]

    def test_recording_the_same_block_twice_is_not_two_rows(self, pipeline):
        pipeline.index.record_recall("c1", 1, ["a"], 100.0)
        pipeline.index.record_recall("c1", 1, ["a"], 200.0)
        assert pipeline.index.take_recalled_into_turn("c1", 1) == ["a"]

    def test_empty_is_a_no_op(self, pipeline):
        pipeline.index.record_recall("c1", 1, [], 100.0)
        assert pipeline.index.take_recalled_into_turn("c1", 1) == []

    def test_pruning_drops_only_the_stale(self, pipeline):
        import time
        now = time.time()
        pipeline.index.record_recall("c1", 1, ["old"], now - 30 * 86400)
        pipeline.index.record_recall("c1", 2, ["new"], now)
        assert pipeline.index.prune_turn_recalls(7 * 86400) == 1
        assert pipeline.index.take_recalled_into_turn("c1", 1) == []
        assert pipeline.index.take_recalled_into_turn("c1", 2) == ["new"]


class TestSurvivesRestart:
    def test_a_recall_recorded_before_a_restart_is_still_there_after(
            self, tmp_path, config):
        # The acceptance test from the plan, literally: recall a block, restart
        # the server, start the next turn -- the evidence is still there.
        index = VectorIndex(tmp_path, dim=4)
        index.open()
        wal = WAL(tmp_path / "wal.jsonl")
        wal.open()
        p = Pipeline(config, BlockStore(tmp_path), index, embed=None, wal=wal)
        put(p, "a")
        p._remember_recalled("c1", 0, ["a"])
        wal.close()
        index.close()

        # A whole new process's worth of objects over the same directory.
        index2 = VectorIndex(tmp_path, dim=4)
        index2.open()
        wal2 = WAL(tmp_path / "wal.jsonl")
        wal2.open()
        p2 = Pipeline(config, BlockStore(tmp_path), index2, embed=None, wal=wal2)
        try:
            assert index2.take_recalled_into_turn("c1", 0) == ["a"]
        finally:
            wal2.close()
            index2.close()

    def test_the_500_entry_cap_is_gone(self, pipeline):
        # The old dict dropped its oldest half past 500 turns, silently.
        for turn in range(900):
            pipeline._remember_recalled("c1", turn, [f"b{turn}"])
        assert pipeline.index.take_recalled_into_turn("c1", 0) == ["b0"]
        assert pipeline.index.take_recalled_into_turn("c1", 899) == ["b899"]


class TestApplyAcceptedVerification:
    @pytest.mark.asyncio
    async def test_an_uncontested_recall_is_accepted_and_counted(self, pipeline):
        put(pipeline, "a")
        pipeline._remember_recalled("c1", 0, ["a"])
        await pipeline.apply_accepted_verification("c1", 1)
        meta = pipeline.index.get_meta("a")
        assert meta["verification"] == "accepted"
        assert meta["verification_source"] == "recalled_uncontested"
        assert meta["uncontested_recalls"] == 1

    @pytest.mark.asyncio
    async def test_the_counter_keeps_climbing_after_verification_saturates(
            self, pipeline):
        # Verification is a state and stops moving after the first time; the
        # counter is the gradient decay actually needs.
        put(pipeline, "a")
        for turn in range(3):
            pipeline._remember_recalled("c1", turn, ["a"])
            await pipeline.apply_accepted_verification("c1", turn + 1)
        assert pipeline.index.get_meta("a")["uncontested_recalls"] == 3

    @pytest.mark.asyncio
    async def test_a_corrected_block_earns_nothing(self, pipeline):
        # It was recalled into the very turn that contradicted it. That is not
        # evidence in its favour.
        put(pipeline, "a", verification="corrected")
        pipeline._remember_recalled("c1", 0, ["a"])
        await pipeline.apply_accepted_verification("c1", 1)
        meta = pipeline.index.get_meta("a")
        assert meta["verification"] == "corrected"
        assert (meta["uncontested_recalls"] or 0) == 0

    @pytest.mark.asyncio
    async def test_running_twice_counts_once(self, pipeline):
        put(pipeline, "a")
        pipeline._remember_recalled("c1", 0, ["a"])
        await pipeline.apply_accepted_verification("c1", 1)
        await pipeline.apply_accepted_verification("c1", 1)
        assert pipeline.index.get_meta("a")["uncontested_recalls"] == 1

    @pytest.mark.asyncio
    async def test_the_first_turn_has_no_previous(self, pipeline):
        await pipeline.apply_accepted_verification("c1", 0)
