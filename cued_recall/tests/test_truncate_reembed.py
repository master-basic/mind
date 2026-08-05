"""Truncation must not leave a block's vector describing words it no longer has.

The judge rewrites a block's text into a shorter summary. Nothing recomputed
the embedding, so recall went on matching the block by its original wording and
then injected the summary -- and each further rewrite widened the gap.
"""

import pytest

from cued_recall.index import VectorIndex
from cued_recall.judge import Judge
from cued_recall.models import Block, BlockStatus, BlockType
from cued_recall.store import BlockStore
from cued_recall.wal import WAL

DIM = 4


class FakeEmbedder:
    """Returns a vector determined by the text, so a stale vector is visible."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("embed server down")
        # Deterministic and text-dependent: same text -> same vector.
        h = sum(ord(c) for c in text)
        return [1.0, (h % 97) / 97.0, 0.0, 0.0]


@pytest.fixture
def judge(tmp_path, config):
    index = VectorIndex(tmp_path, dim=DIM)
    index.open()
    wal = WAL(tmp_path / "wal.jsonl")
    wal.open()
    j = Judge(config, BlockStore(tmp_path), index, wal, embed=FakeEmbedder())
    yield j
    wal.close()
    index.close()


def seed(judge, type_=BlockType.reasoning, text="the original wording",
         stimulus="the question that produced it"):
    block = Block(type=type_, status=BlockStatus.shelved, text=text,
                  stimulus_text=stimulus, embed_text=text, token_count=100)
    judge.store.put(block)
    judge.index.upsert_block_meta(
        block.block_id, type_.value, "shelved", block.created_at, "c1", 0,
        100, "unknown", 0, 0.0)
    judge.index.upsert_vector(block.block_id, judge.embed.embed(text))
    judge.embed.calls.clear()
    return block


class TestTruncateReembed:
    @pytest.mark.asyncio
    async def test_the_content_channel_follows_the_new_text(self, judge):
        block = seed(judge)
        await judge._truncate_block(block, "the summary", 5)
        assert block.embed_text == "the summary"
        assert judge.store.get(block.block_id).embed_text == "the summary"

    @pytest.mark.asyncio
    async def test_the_vector_is_rebuilt(self, judge):
        block = seed(judge)
        await judge._truncate_block(block, "the summary", 5)
        assert judge.embed.calls, "truncation did not re-embed"

    @pytest.mark.asyncio
    async def test_a_reasoning_block_keeps_its_question(self, judge):
        # stimulus_text is what the judge's own consolidation prompt shows as
        # "Question:". Truncating the note does not change what was asked.
        block = seed(judge)
        await judge._truncate_block(block, "the summary", 5)
        assert block.stimulus_text == "the question that produced it"

    @pytest.mark.asyncio
    async def test_a_result_block_s_stale_stimulus_copy_is_refreshed(self, judge):
        # For result and reading blocks stimulus_text is set to a copy of the
        # block's own text at creation, so after a rewrite it is a copy of text
        # that no longer exists anywhere.
        block = seed(judge, type_=BlockType.result, text="the answer text",
                     stimulus="the answer text")
        await judge._truncate_block(block, "shorter answer", 3)
        assert block.stimulus_text == "shorter answer"

    @pytest.mark.asyncio
    async def test_composite_mode_embeds_the_question_not_the_summary(self, judge):
        block = seed(judge)
        await judge._truncate_block(block, "the summary", 5)
        assert judge.embed.calls == ["the question that produced it"]

    @pytest.mark.asyncio
    async def test_content_mode_embeds_the_summary(self, judge):
        judge.config.embed_source = "content"
        block = seed(judge)
        await judge._truncate_block(block, "the summary", 5)
        assert judge.embed.calls == ["the summary"]

    @pytest.mark.asyncio
    async def test_the_original_wording_is_still_kept(self, judge):
        # Re-embedding must not disturb the one copy of the block's own words.
        block = seed(judge)
        await judge._truncate_block(block, "the summary", 5)
        assert block.original_text == "the original wording"

    @pytest.mark.asyncio
    async def test_a_second_rewrite_keeps_the_true_original(self, judge):
        block = seed(judge)
        await judge._truncate_block(block, "first summary", 5)
        await judge._truncate_block(block, "second summary", 3)
        assert block.original_text == "the original wording"
        assert block.embed_text == "second summary"


class TestReembedFailure:
    @pytest.mark.asyncio
    async def test_a_failed_reembed_keeps_the_old_vector(self, judge):
        # A stale vector still finds the block. Dropping it would make the
        # block unrecallable, which is the worse of the two failures.
        block = seed(judge)
        judge.embed.fail = True
        await judge._truncate_block(block, "the summary", 5)
        assert judge.index.count_blocks_without_vectors() == 0

    @pytest.mark.asyncio
    async def test_a_failed_reembed_is_recorded(self, judge):
        block = seed(judge)
        judge.embed.fail = True
        await judge._truncate_block(block, "the summary", 5)
        errors = [e for e in judge.wal.iter_all() if e["event"] == "reembed_error"]
        assert [e["block_id"] for e in errors] == [block.block_id]

    @pytest.mark.asyncio
    async def test_truncation_still_happens_with_no_embedder_at_all(self, judge):
        # An offline judge run must not be blocked on an embedding server.
        block = seed(judge)
        judge.embed = None
        await judge._truncate_block(block, "the summary", 5)
        assert judge.store.get(block.block_id).text == "the summary"
        assert judge.index.get_meta(block.block_id)["status"] == "truncated"
