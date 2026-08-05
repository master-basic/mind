"""What the relevance judge is shown for a candidate block.

Shown a block's own words the judge keeps 5 of 6 trap candidates -- material
about a different phase of the same work, which grading_traps.md caught
anchoring a real answer on the wrong stack. Shown the question the block was
written to answer it keeps 0 of 6, and every legitimate recall survives either
way. See evaluate/eval_judge_notes.py.
"""

import pytest

from cued_recall.config import Config
from cued_recall.models import Block, BlockType
from cued_recall.utils import build_stimulus, judge_note_text, truncate_tokens


def reasoning_block(question="how do I add OCR?", answer="Use Tesseract.",
                    text="the think trace", question_text=""):
    return Block(type=BlockType.reasoning, text=text,
                 question_text=question_text,
                 stimulus_text=build_stimulus(question, answer, ""))


class TestJudgeNoteText:
    def test_text_mode_is_the_old_behaviour(self):
        b = reasoning_block(question_text="how do I add OCR?")
        assert judge_note_text(b, "text") == "the think trace"

    def test_question_mode_returns_the_question(self):
        b = reasoning_block(question_text="how do I add OCR?")
        assert judge_note_text(b, "question") == "how do I add OCR?"

    def test_question_mode_never_leaks_the_answer(self):
        # The whole mechanism: a block's words contain the answer, and a note
        # containing the answer is one the judge says yes to.
        b = reasoning_block(question_text="how do I add OCR?",
                            answer="Use client-side Tesseract.js")
        note = judge_note_text(b, "question")
        assert "Tesseract" not in note
        assert "think trace" not in note

    def test_default_is_text(self):
        b = reasoning_block(question_text="q")
        assert judge_note_text(b) == judge_note_text(b, "text")


class TestRetroactiveFallback:
    def test_an_old_reasoning_block_recovers_its_question_from_the_stimulus(self):
        # build_stimulus puts the question first, so a store written before
        # question_text existed gets the benefit with no backfill.
        b = reasoning_block(question="how do I add OCR?",
                            answer="Use Tesseract.", question_text="")
        assert judge_note_text(b, "question") == "how do I add OCR?"

    def test_a_result_block_stimulus_is_not_mistaken_for_a_question(self):
        # A result block's stimulus is a copy of its own text with no "---",
        # so splitting it returns the whole thing. Returning that as "the
        # question" would be the old behaviour wearing a disguise.
        b = Block(type=BlockType.result, text="the full answer text",
                  stimulus_text="the full answer text")
        assert judge_note_text(b, "question") == "the full answer text"

    def test_a_block_with_nothing_falls_back_to_its_text(self):
        b = Block(type=BlockType.reasoning, text="just words",
                  stimulus_text="", question_text="")
        assert judge_note_text(b, "question") == "just words"

    def test_an_explicit_question_beats_the_stimulus_fallback(self):
        b = reasoning_block(question="the old parsed one",
                            question_text="the stored one")
        assert judge_note_text(b, "question") == "the stored one"

    def test_a_whitespace_only_question_is_not_used(self):
        b = reasoning_block(question="the real question", question_text="   ")
        assert judge_note_text(b, "question") == "the real question"


class TestConfig:
    def test_default_is_question(self, config):
        assert config.recall.judge_note == "question"

    def test_a_typo_fails_at_startup(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("recall:\n  judge_note: questions\n", encoding="utf-8")
        with pytest.raises(ValueError, match="judge_note"):
            Config(path)


class TestBlockCreation:
    @pytest.mark.asyncio
    async def test_every_block_stores_the_question(self, pipeline, monkeypatch):
        async def fake_count(text):
            return len(text.split())
        monkeypatch.setattr(pipeline, "_count_tokens", fake_count)
        await pipeline._create_blocks(
            full_reasoning="thinking about OCR",
            full_result="use Tesseract",
            user_message="how do I add OCR?",
            reading_content="", recall_blocks=[],
            conversation_id="c1", turn_index=0, response_text="",
        )
        blocks = [pipeline.store.get(b)
                  for b in pipeline.index.block_ids_for_turn("c1", 0)]
        assert blocks
        for b in blocks:
            assert b.question_text == "how do I add OCR?"
            assert judge_note_text(b, "question") == "how do I add OCR?"
