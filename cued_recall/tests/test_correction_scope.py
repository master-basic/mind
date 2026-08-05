"""What a correction is evidence against.

A wrong answer used to mark every block of the previous turn corrected --
including the `reading` block holding the document the user pasted. Corrected
blocks are dropped from recall and start a purge clock, so the user's own
source material was deleted as punishment for the model misusing it.
"""

from pathlib import Path

import pytest

from cued_recall.config import Config
from cued_recall.index import VectorIndex
from cued_recall.models import Block, BlockStatus, BlockType, Verification
from cued_recall.pipeline import Pipeline
from cued_recall.store import BlockStore
from cued_recall.wal import WAL

EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / "config.example.yaml"


@pytest.fixture
def pipeline(tmp_path):
    config = Config(EXAMPLE_CONFIG)
    store = BlockStore(tmp_path)
    index = VectorIndex(tmp_path, dim=4)
    index.open()
    wal = WAL(tmp_path / "wal.jsonl")
    wal.open()
    # embed is unused by the correction path -- it makes no vector calls.
    p = Pipeline(config, store, index, embed=None, wal=wal)
    yield p
    wal.close()
    index.close()


def put(pipeline, block_id, type_, conversation_id="c1", turn_index=0):
    block = Block(
        block_id=block_id, type=type_, status=BlockStatus.shelved,
        conversation_id=conversation_id, turn_index=turn_index,
        text=f"{type_.value} text", verification=Verification.unknown,
    )
    pipeline.store.put(block)
    pipeline.index.upsert_block_meta(
        block_id, type_.value, block.status.value, block.created_at,
        conversation_id, turn_index, 10, "unknown", 0, 0.0,
    )
    return block


def verification_of(pipeline, block_id):
    meta = pipeline.index.get_meta(block_id)
    block = pipeline.store.get(block_id)
    # The two copies must agree; a divergence here is its own bug.
    assert meta["verification"] == block.verification.value
    return meta["verification"]


class TestMarkCorrected:
    @pytest.mark.asyncio
    async def test_reasoning_and_result_are_corrected(self, pipeline):
        put(pipeline, "r", BlockType.reasoning)
        put(pipeline, "a", BlockType.result)
        await pipeline._mark_corrected(["r", "a"], "pattern")
        assert verification_of(pipeline, "r") == "corrected"
        assert verification_of(pipeline, "a") == "corrected"

    @pytest.mark.asyncio
    async def test_the_pasted_source_is_left_alone(self, pipeline):
        put(pipeline, "r", BlockType.reasoning)
        put(pipeline, "src", BlockType.reading)
        await pipeline._mark_corrected(["r", "src"], "pattern")
        assert verification_of(pipeline, "r") == "corrected"
        assert verification_of(pipeline, "src") == "unknown"

    @pytest.mark.asyncio
    async def test_a_skipped_source_block_is_recorded(self, pipeline):
        # Silently doing less than asked is worse than doing it: the WAL entry
        # is how anyone finds out why a reading block stayed unknown.
        put(pipeline, "src", BlockType.reading)
        await pipeline._mark_corrected(["src"], "model")
        events = [e for e in pipeline.wal.iter_all()
                  if e["event"] == "correction_skipped_source_block"]
        assert [e["block_id"] for e in events] == ["src"]

    @pytest.mark.asyncio
    async def test_the_verdict_source_is_recorded(self, pipeline):
        # _should_purge weighs pattern and model differently, so losing the
        # source would let a 1.5B guess delete a block.
        put(pipeline, "r", BlockType.reasoning)
        await pipeline._mark_corrected(["r"], "model")
        assert pipeline.index.get_meta("r")["verification_source"] == "model"

    @pytest.mark.asyncio
    async def test_a_block_with_no_metadata_is_still_marked(self, pipeline):
        # Fail toward marking: an unknown type is more likely a reasoning or
        # result block than a pasted document, and leaving a wrong answer
        # recallable is the worse of the two failures.
        block = Block(block_id="orphan", type=BlockType.reasoning,
                      status=BlockStatus.shelved, text="x")
        pipeline.store.put(block)
        await pipeline._mark_corrected(["orphan"], "pattern")
        assert pipeline.store.get("orphan").verification == Verification.corrected
