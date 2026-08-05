"""Which text of a block goes into the vector index.

A reasoning block's stimulus_text is the question-plus-answer that produced
it, so under the shipped "composite" setting the vector *is* the Q+A pair --
the mechanism behind the trap family scoring 0.841. embed_text is the second
channel: what the block itself says. Four callers pick the source (pipeline at
creation, judge after truncation, the admin restore/import endpoints, the
backfill script) and a disagreement between them would be invisible, so they
all go through one function.
"""

import pytest

from cued_recall.models import Block, BlockStatus, BlockType
from cued_recall.utils import embed_source_text, truncate_tokens


def block(**kw):
    return Block(**{"type": BlockType.reasoning, **kw})


class TestEmbedSourceText:
    def test_composite_prefers_the_stimulus(self):
        b = block(stimulus_text="question + answer", embed_text="own words")
        assert embed_source_text(b, "composite") == "question + answer"

    def test_content_prefers_the_block_s_own_words(self):
        b = block(stimulus_text="question + answer", embed_text="own words")
        assert embed_source_text(b, "content") == "own words"

    def test_default_is_the_shipped_behaviour(self):
        b = block(stimulus_text="question + answer", embed_text="own words")
        assert embed_source_text(b) == embed_source_text(b, "composite")

    def test_content_falls_back_for_a_block_written_before_the_field(self):
        # The whole store predates embed_text. Falling back is what keeps the
        # switch from silently un-indexing every older block.
        b = block(stimulus_text="question + answer", embed_text="")
        assert embed_source_text(b, "content") == "question + answer"

    def test_composite_falls_back_to_content(self):
        b = block(stimulus_text="", embed_text="own words")
        assert embed_source_text(b, "composite") == "own words"

    def test_text_is_the_last_resort(self):
        b = block(stimulus_text="", embed_text="", text="raw")
        assert embed_source_text(b, "content") == "raw"
        assert embed_source_text(b, "composite") == "raw"

    def test_a_block_with_nothing_yields_nothing(self):
        # The caller checks this before embedding: an empty string sent to the
        # embedder is a wasted call and a meaningless vector.
        assert embed_source_text(block(), "content") == ""


class TestConfigValidation:
    def test_default(self, config):
        assert config.embed_source == "composite"
        assert config.embed_token_limit == 1024

    def test_a_typo_fails_at_startup_rather_than_at_recall(self, tmp_path):
        # Silently treating "contents" as "composite" would mean running a
        # measured experiment against the wrong arm without knowing.
        from cued_recall.config import Config
        path = tmp_path / "config.yaml"
        path.write_text("embed_source: contents\n", encoding="utf-8")
        with pytest.raises(ValueError, match="embed_source"):
            Config(path)


class TestBlockCreation:
    @pytest.mark.asyncio
    async def test_every_block_gets_both_channels(self, pipeline, monkeypatch):
        # Token counting would otherwise hit the reasoning server's /tokenize.
        async def fake_count(text):
            return len(text.split())
        monkeypatch.setattr(pipeline, "_count_tokens", fake_count)

        await pipeline._create_blocks(
            full_reasoning="I should check the OCR stack first.",
            full_result="Use Tesseract client-side.",
            user_message="how do I add OCR?",
            reading_content="",
            recall_blocks=[],
            conversation_id="c1",
            turn_index=0,
            response_text="",
        )
        blocks = [pipeline.store.get(b)
                  for b in pipeline.index.block_ids_for_turn("c1", 0)]
        assert blocks
        for b in blocks:
            assert b.embed_text, f"{b.type} block has no content channel"
            # embed_text is the block's own words, never the question.
            assert b.embed_text == truncate_tokens(b.text, 1024)

    @pytest.mark.asyncio
    async def test_a_reasoning_block_s_composite_carries_the_question(
            self, pipeline, monkeypatch):
        async def fake_count(text):
            return len(text.split())
        monkeypatch.setattr(pipeline, "_count_tokens", fake_count)

        await pipeline._create_blocks(
            full_reasoning="Tesseract runs in the browser via WASM.",
            full_result="Use Tesseract client-side.",
            user_message="how do I add OCR?",
            reading_content="",
            recall_blocks=[],
            conversation_id="c1",
            turn_index=0,
            response_text="",
        )
        reasoning = [b for b in
                     (pipeline.store.get(i)
                      for i in pipeline.index.block_ids_for_turn("c1", 0))
                     if b.type == BlockType.reasoning]
        assert reasoning
        for b in reasoning:
            # This is F2 in one assertion: the composite contains the question
            # AND the answer, so a later question sharing either scores against
            # it. The content channel contains neither.
            assert "how do I add OCR?" in b.stimulus_text
            assert "Use Tesseract client-side." in b.stimulus_text
            assert "how do I add OCR?" not in b.embed_text

    @pytest.mark.asyncio
    async def test_blocks_with_no_embeddable_text_are_not_sent_to_the_embedder(
            self, pipeline, monkeypatch):
        # An empty turn is a wasted embed call and a meaningless vector.
        # pipeline.embed is None, so any attempt lands in _embed_and_store's
        # error path and shows up in the WAL -- which is what is asserted.
        async def fake_count(text):
            return len(text.split())
        monkeypatch.setattr(pipeline, "_count_tokens", fake_count)
        await pipeline._create_blocks(
            full_reasoning="", full_result="", user_message="hi",
            reading_content="", recall_blocks=[], conversation_id="c1",
            turn_index=0, response_text="",
        )
        errors = [e for e in pipeline.wal.iter_all()
                  if e["event"] == "embed_store_error"]
        assert errors == []

    @pytest.mark.asyncio
    async def test_a_failed_embed_is_recorded_rather_than_raised(
            self, pipeline, monkeypatch):
        # The complaint from the analysis: block creation must survive an
        # embedding server that is down, but the loss must not be silent.
        async def fake_count(text):
            return len(text.split())
        monkeypatch.setattr(pipeline, "_count_tokens", fake_count)
        await pipeline._create_blocks(
            full_reasoning="some thinking", full_result="some answer",
            user_message="a question", reading_content="", recall_blocks=[],
            conversation_id="c1", turn_index=0, response_text="",
        )
        errors = [e for e in pipeline.wal.iter_all()
                  if e["event"] == "embed_store_error"]
        assert errors
        # And the blocks themselves survived the failure.
        assert pipeline.index.block_ids_for_turn("c1", 0)
