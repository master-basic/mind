"""Span-level corrections (Phase 4.2 / F4).

A correction used to suppress a whole block from recall. With verifier.spans
on, the verifier's "yes" also carries a quote of the offending part of the
answer; the block stores it (correction_span) and recall admits the block with
the span redacted -- the other claims in it keep being usable, "weight the
stable aspects".

Default off: the yes/no prompt is the one eval_correction.py measured, and the
span quote needs its own eval rows before it is trusted.
"""

import pytest

from cued_recall.models import Block, BlockStatus, BlockType, Verification
from cued_recall.utils import CORRECTION_MARKER, redact_span
from cued_recall.verifier import CorrectionVerifier


def block(tokens, block_id, text=""):
    return Block(block_id=block_id, type=BlockType.result,
                 status=BlockStatus.shelved, token_count=tokens,
                 text=text or f"block {block_id}", question_text=f"q{block_id}",
                 verification=Verification.corrected)


def seed(pipeline, blocks):
    for b in blocks:
        pipeline.store.put(b)
        pipeline.index.upsert_block_meta(
            b.block_id, b.type.value, b.status.value, b.created_at,
            "c1", 0, b.token_count, b.verification.value, 0, 0.0)


class TestRedactSpan:
    def test_exact_match_is_replaced_with_the_marker(self):
        got = redact_span("The port is 8080 and the config lives at /etc/foo",
                          "8080")
        assert "8080" not in got
        assert CORRECTION_MARKER in got
        assert "the config lives at /etc/foo" in got

    def test_case_insensitive_match_still_redacts(self):
        got = redact_span("The port is 8080", "port is 8080")
        assert CORRECTION_MARKER in got
        assert "8080" not in got

    def test_a_paraphrase_is_flagged_instead_of_served(self):
        got = redact_span("The config lives at /etc/foo", "the file is missing")
        assert got.startswith("[reported wrong: the file is missing]")
        assert "/etc/foo" in got

    def test_empty_span_or_text_is_untouched(self):
        assert redact_span("the config lives at /etc/foo", "") == \
            "the config lives at /etc/foo"
        assert redact_span("", "8080") == ""


class TestParseSpan:
    def test_plain_yes(self):
        assert CorrectionVerifier._parse_span("yes", True) == (True, "")

    def test_plain_no(self):
        assert CorrectionVerifier._parse_span("no", False) == (False, "")

    def test_quoted_span_on_the_same_line(self):
        assert CorrectionVerifier._parse_span(
            'yes "the port is 8080"', True) == (True, "the port is 8080")

    def test_span_after_a_separator(self):
        assert CorrectionVerifier._parse_span(
            "yes: the port is 8080", True) == (True, "the port is 8080")

    def test_no_span_when_the_prompt_did_not_ask(self):
        # Old-mode replies are bare yes/no; a trailing word must not become a
        # span that would later redact a substring nobody verified.
        assert CorrectionVerifier._parse_span("yes anything", False) == \
            (True, "")

    def test_first_word_wins(self):
        assert CorrectionVerifier._parse_span(
            "no, the user is asking a follow-up", True) == (False, "")

    def test_garbage_is_none(self):
        assert CorrectionVerifier._parse_span("maybe", True) is None
        assert CorrectionVerifier._parse_span("", True) is None

    def test_old_parse_wrapper_keeps_yes_no_only(self):
        assert CorrectionVerifier._parse("yes") is True
        assert CorrectionVerifier._parse("no") is False
        assert CorrectionVerifier._parse("maybe") is None


class TestPrompt:
    def test_default_prompt_is_the_measured_yes_no_one(self):
        # eval_correction.py scores this exact prompt; it must not change.
        p = CorrectionVerifier._prompt("ans", "msg")
        assert "exact phrase" not in p

    def test_span_prompt_adds_the_quote_instruction(self):
        p = CorrectionVerifier._prompt("ans", "msg", with_span=True)
        assert "exact phrase from the answer" in p


class TestConfig:
    def test_spans_default_off(self, config):
        assert config.verifier.spans is False


class TestMarkCorrectedStoresTheSpan:
    @pytest.mark.asyncio
    async def test_span_is_written_to_the_block_and_the_wal(self, pipeline):
        b = block(10, "a", text="The port is 8080 and the rest is fine")
        seed(pipeline, [b])
        await pipeline._mark_corrected(["a"], "model", "8080")
        got = pipeline.store.get("a")
        assert got.verification == Verification.corrected
        assert got.correction_span == "8080"
        ev = [e for e in pipeline.wal.iter_all()
              if e["event"] == "verification_set"][-1]
        assert ev["span"] == "8080"

    @pytest.mark.asyncio
    async def test_a_later_spanless_remark_never_clears_the_span(self, pipeline):
        b = block(10, "a", text="The port is 8080")
        seed(pipeline, [b])
        await pipeline._mark_corrected(["a"], "model", "8080")
        await pipeline._mark_corrected(["a"], "pattern")
        assert pipeline.store.get("a").correction_span == "8080"


class TestRecallAdmission:
    @pytest.fixture
    def rig(self, pipeline, monkeypatch):
        pipeline.config.recall.budget_tokens = 1000
        pipeline.config.recall.judge_enabled = True

        def set_hits(hits):
            monkeypatch.setattr(pipeline.index, "query",
                                lambda vec, k, thr: hits[:k])

        judged = []

        def set_scores(scores):
            async def fake(question, candidates, keyword_ids=None):
                judged.extend(b.block_id for b, _ in candidates)
                return [(b, sim, scores.get(b.block_id, 0.9))
                        for b, sim in candidates]
            monkeypatch.setattr(pipeline, "_score_by_relevance", fake)

        monkeypatch.setattr(pipeline, "embed",
                            type("E", (), {"embed": staticmethod(
                                lambda t: [0.0, 0.0, 0.0, 1.0])})())
        return pipeline, set_hits, set_scores, judged

    @pytest.mark.asyncio
    async def test_span_corrected_block_stays_out_by_default(self, rig):
        # Old behaviour exactly: corrected is corrected, whole block excluded.
        p, set_hits, set_scores, judged = rig
        b = block(10, "a", text="The port is 8080")
        b.correction_span = "8080"
        seed(p, [b])
        set_hits([("a", 0.90)])
        set_scores({"a": 0.9})

        got = await p.recall_blocks("q")
        assert got == []
        assert judged == []
        budget = [e for e in p.wal.iter_all()
                  if e["event"] == "recall_budget"][-1]
        assert budget["span_corrected"] == 0

    @pytest.mark.asyncio
    async def test_span_corrected_block_is_admitted_with_spans_on(
            self, rig):
        p, set_hits, set_scores, judged = rig
        p.config.verifier.spans = True
        b = block(10, "a", text="The port is 8080 and the rest is fine")
        b.correction_span = "8080"
        seed(p, [b])
        set_hits([("a", 0.90)])
        set_scores({"a": 0.9})

        got = await p.recall_blocks("q")
        assert [b.block_id for b, _ in got] == ["a"]
        assert judged == ["a"]
        budget = [e for e in p.wal.iter_all()
                  if e["event"] == "recall_budget"][-1]
        assert budget["span_corrected"] == 1

    @pytest.mark.asyncio
    async def test_a_corrected_block_without_a_span_stays_out(
            self, rig):
        p, set_hits, set_scores, judged = rig
        p.config.verifier.spans = True
        b = block(10, "a", text="The port is 8080")
        seed(p, [b])
        set_hits([("a", 0.90)])
        set_scores({"a": 0.9})

        got = await p.recall_blocks("q")
        assert got == []
        assert judged == []


class TestRecallShowsTheBlockMinusTheSpan:
    def test_injection_redacts_the_claim_and_keeps_the_rest(self, pipeline):
        # The F4 acceptance case in one assertion: a multi-claim block, one
        # claim corrected. Recall injects the block with the bad claim gone
        # and the other claims intact.
        b = block(10, "a",
                  text="The port is 8080 and the config lives at /etc/foo")
        b.correction_span = "8080"
        text = pipeline.build_recall_injection([(b, 0.9)])
        assert "8080" not in text
        assert CORRECTION_MARKER in text
        assert "/etc/foo" in text

    def test_judge_note_redacts_the_claim(self, pipeline):
        pipeline.config.recall.judge_note = "text"
        b = block(10, "a", text="The port is 8080 and the rest is fine")
        b.correction_span = "8080"
        note = pipeline._judge_note_for(b, keyword=False)
        assert "8080" not in note
        assert CORRECTION_MARKER in note
        assert "the rest is fine" in note

    def test_judge_note_without_a_span_is_unchanged(self, pipeline):
        b = block(10, "a", text="The port is 8080")
        assert pipeline._judge_note_for(b, keyword=False) == "qa"

    def test_question_mode_flags_the_block_when_the_span_is_not_in_it(
            self, pipeline):
        # judge_note "question" shows the question, which never contains the
        # answer's span -- the block is flagged rather than silently serving
        # the refuted claim.
        b = block(10, "a", text="The port is 8080 and the rest is fine")
        b.correction_span = "8080"
        note = pipeline._judge_note_for(b, keyword=False)
        assert note.startswith("[reported wrong: 8080]")
        assert "qa" in note
