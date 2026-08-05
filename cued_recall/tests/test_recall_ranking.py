"""Relevance decides what fits the budget, not nearness.

The budget used to be filled in cosine order *before* the judge ran, so a
candidate the judge was about to reject had already taken a slot, and the slot
was not refilled -- a slightly less similar but genuinely relevant block behind
an oversized or reject-worthy one was silently lost.
"""

import math

import pytest

from cued_recall.models import Block, BlockStatus, BlockType, Verification
from cued_recall.pipeline import Pipeline


def choice(token_probs, content="yes"):
    """A judge response with logprobs, as llama-server returns them."""
    return {
        "message": {"content": content},
        "logprobs": {"content": [{
            "token": content,
            "top_logprobs": [
                {"token": t, "logprob": math.log(p)}
                for t, p in token_probs.items()
            ],
        }]},
    }


class TestScoreFromChoice:
    def test_reads_p_yes_from_the_logprobs(self):
        s = Pipeline._score_from_choice(choice({"yes": 0.75, "no": 0.25}))
        assert s == pytest.approx(0.75)

    def test_renormalises_over_yes_and_no_only(self):
        # The remaining mass sits on tokens that are neither verdict.
        s = Pipeline._score_from_choice(
            choice({"yes": 0.4, "no": 0.1, "Answer": 0.5}))
        assert s == pytest.approx(0.8)

    def test_sums_capitalisation_variants(self):
        # "Yes" and "yes" are the same verdict; on a small model the mass
        # moves between them for no reason worth modelling.
        s = Pipeline._score_from_choice(
            choice({"yes": 0.3, "Yes": 0.3, "no": 0.2, "No": 0.2}))
        assert s == pytest.approx(0.6)

    def test_a_confident_no_scores_near_zero(self):
        s = Pipeline._score_from_choice(choice({"yes": 0.01, "no": 0.99}))
        assert s == pytest.approx(0.01)

    def test_falls_back_to_the_text_when_there_are_no_logprobs(self):
        # A judge server without logprobs must behave exactly as before.
        assert Pipeline._score_from_choice({"message": {"content": "yes"}}) == 1.0
        assert Pipeline._score_from_choice({"message": {"content": "no"}}) == 0.0
        assert Pipeline._score_from_choice({"message": {"content": "No."}}) == 0.0

    def test_an_empty_reply_is_not_a_score(self):
        # None means "no opinion", which the caller turns into a fail-open
        # keep at the floor rather than a confident zero.
        assert Pipeline._score_from_choice({"message": {"content": ""}}) is None
        assert Pipeline._score_from_choice({}) is None

    def test_logprobs_with_no_verdict_tokens_falls_back_to_text(self):
        c = choice({"maybe": 0.6, "perhaps": 0.4}, content="no")
        assert Pipeline._score_from_choice(c) == 0.0


def block(tokens, block_id):
    return Block(block_id=block_id, type=BlockType.reasoning,
                 status=BlockStatus.shelved, token_count=tokens,
                 text=f"block {block_id}", question_text=f"q{block_id}",
                 verification=Verification.unknown)


def seed(pipeline, blocks):
    for b in blocks:
        pipeline.store.put(b)
        pipeline.index.upsert_block_meta(
            b.block_id, b.type.value, b.status.value, b.created_at,
            "c1", 0, b.token_count, "unknown", 0, 0.0)


class TestRankedFill:
    """recall_blocks with the vector and judge stages stubbed."""

    @pytest.fixture
    def rig(self, pipeline, monkeypatch):
        pipeline.config.recall.budget_tokens = 100
        pipeline.config.recall.judge_enabled = True

        def set_hits(hits):
            # (block_id, similarity), in the order index.query would return.
            monkeypatch.setattr(pipeline.index, "query",
                                lambda vec, k, thr: hits[:k])

        def set_scores(scores):
            async def fake(question, candidates, keyword_ids=None):
                return [(b, sim, scores[b.block_id]) for b, sim in candidates]
            monkeypatch.setattr(pipeline, "_score_by_relevance", fake)

        monkeypatch.setattr(pipeline, "embed",
                            type("E", (), {"embed": staticmethod(
                                lambda t: [0.0, 0.0, 0.0, 1.0])})())
        return pipeline, set_hits, set_scores

    @pytest.mark.asyncio
    async def test_a_relevant_block_beats_a_nearer_irrelevant_one(self, rig):
        # The F6 failure in one case: "near" is 0.90 cosine and fills the whole
        # budget, but the judge scores it 0.1. "far" is less similar and
        # genuinely relevant. Before, near took the slot and far was never
        # offered; the judge then dropped near and the turn recalled nothing.
        p, set_hits, set_scores = rig
        seed(p, [block(100, "near"), block(100, "far")])
        set_hits([("near", 0.90), ("far", 0.60)])
        set_scores({"near": 0.10, "far": 0.95})
        got = await p.recall_blocks("a question")
        assert [b.block_id for b, _ in got] == ["far"]

    @pytest.mark.asyncio
    async def test_the_budget_is_spent_in_score_order(self, rig):
        p, set_hits, set_scores = rig
        seed(p, [block(60, "a"), block(60, "b")])
        set_hits([("a", 0.90), ("b", 0.80)])
        set_scores({"a": 0.60, "b": 0.99})
        got = await p.recall_blocks("q")
        # Only one fits. It should be the one that applies, not the nearest.
        assert [b.block_id for b, _ in got] == ["b"]

    @pytest.mark.asyncio
    async def test_a_rejected_block_frees_its_slot(self, rig):
        p, set_hits, set_scores = rig
        seed(p, [block(100, "reject"), block(100, "keep")])
        set_hits([("reject", 0.95), ("keep", 0.55)])
        set_scores({"reject": 0.01, "keep": 0.90})
        got = await p.recall_blocks("q")
        assert [b.block_id for b, _ in got] == ["keep"]

    @pytest.mark.asyncio
    async def test_an_oversized_block_is_skipped_not_stopped_on(self, rig):
        p, set_hits, set_scores = rig
        seed(p, [block(5000, "huge"), block(20, "small")])
        set_hits([("huge", 0.95), ("small", 0.90)])
        set_scores({"huge": 0.99, "small": 0.95})
        got = await p.recall_blocks("q")
        assert [b.block_id for b, _ in got] == ["small"]

    @pytest.mark.asyncio
    async def test_blocks_below_the_floor_are_dropped(self, rig):
        p, set_hits, set_scores = rig
        seed(p, [block(10, "a"), block(10, "b")])
        set_hits([("a", 0.90), ("b", 0.85)])
        set_scores({"a": 0.49, "b": 0.51})
        got = await p.recall_blocks("q")
        assert [b.block_id for b, _ in got] == ["b"]

    @pytest.mark.asyncio
    async def test_a_corrected_block_never_reaches_the_judge(self, rig):
        p, set_hits, set_scores = rig
        seed(p, [block(10, "bad"), block(10, "good")])
        p.index.update_verification("bad", "corrected", "manual")
        judged = []

        async def fake(question, candidates, keyword_ids=None):
            judged.extend(b.block_id for b, _ in candidates)
            return [(b, sim, 0.9) for b, sim in candidates]
        p._score_by_relevance = fake
        set_hits([("bad", 0.99), ("good", 0.70)])
        got = await p.recall_blocks("q")
        assert judged == ["good"]
        assert [b.block_id for b, _ in got] == ["good"]

    @pytest.mark.asyncio
    async def test_a_pin_is_admitted_before_an_equally_scored_unpinned(self, rig):
        # F14 in one case: two blocks, equal relevance, only one fits the
        # budget. The pin used to buy nothing -- the block that sorted first
        # took the slot regardless. Now it is the tie-break.
        p, set_hits, set_scores = rig
        a = block(60, "unpinned")
        b = block(60, "pinned")
        b.pinned = True
        seed(p, [a, b])
        set_hits([("unpinned", 0.90), ("pinned", 0.80)])
        set_scores({"unpinned": 0.90, "pinned": 0.90})
        got = await p.recall_blocks("q")
        assert [x.block_id for x, _ in got] == ["pinned"]

    @pytest.mark.asyncio
    async def test_relevance_still_beats_a_pin(self, rig):
        # The pin is a tie-break, never the primary key: relevance decides
        # what fits, so a clearly more relevant unpinned block still wins the
        # budget over a marginal pinned one.
        p, set_hits, set_scores = rig
        a = block(60, "relevant")
        b = block(60, "pin")
        b.pinned = True
        seed(p, [a, b])
        set_hits([("pin", 0.99), ("relevant", 0.60)])
        set_scores({"pin": 0.55, "relevant": 0.95})
        got = await p.recall_blocks("q")
        assert [x.block_id for x, _ in got] == ["relevant"]

    @pytest.mark.asyncio
    async def test_pin_priority_off_restores_the_old_order(self, rig):
        p, set_hits, set_scores = rig
        p.config.recall.pin_priority = False
        a = block(60, "unpinned")
        b = block(60, "pinned")
        b.pinned = True
        seed(p, [a, b])
        set_hits([("unpinned", 0.90), ("pinned", 0.80)])
        set_scores({"unpinned": 0.90, "pinned": 0.90})
        got = await p.recall_blocks("q")
        assert [x.block_id for x, _ in got] == ["unpinned"]

    @pytest.mark.asyncio
    async def test_a_pin_breaks_ties_without_the_judge(self, rig):
        # The no-judge path ranks by similarity; the pin still breaks an
        # equal-similarity tie in the fill order.
        p, set_hits, set_scores = rig
        p.config.recall.judge_enabled = False
        a = block(60, "unpinned")
        b = block(60, "pinned")
        b.pinned = True
        seed(p, [a, b])
        set_hits([("unpinned", 0.90), ("pinned", 0.90)])
        got = await p.recall_blocks("q")
        assert [x.block_id for x, _ in got] == ["pinned"]

    @pytest.mark.asyncio
    async def test_without_the_judge_similarity_order_is_kept(self, rig):
        p, set_hits, set_scores = rig
        p.config.recall.judge_enabled = False
        seed(p, [block(60, "near"), block(60, "far")])
        set_hits([("near", 0.90), ("far", 0.60)])
        got = await p.recall_blocks("q")
        assert [b.block_id for b, _ in got] == ["near"]

    @pytest.mark.asyncio
    async def test_the_candidate_pool_defaults_to_k(self, rig):
        # Widening it costs one judge call per extra candidate on a
        # single-slot CPU server, so the default must not widen it.
        p, set_hits, set_scores = rig
        asked = {}
        p.index.query = lambda vec, k, thr: asked.setdefault("k", k) and []
        await p.recall_blocks("q")
        assert asked["k"] == p.config.recall.k

    @pytest.mark.asyncio
    async def test_the_multiplier_widens_the_pool(self, rig):
        p, set_hits, set_scores = rig
        p.config.recall.candidate_multiplier = 4
        asked = {}
        p.index.query = lambda vec, k, thr: asked.setdefault("k", k) and []
        await p.recall_blocks("q")
        assert asked["k"] == p.config.recall.k * 4

    @pytest.mark.asyncio
    async def test_the_budget_event_reports_the_new_counters(self, rig):
        p, set_hits, set_scores = rig
        seed(p, [block(100, "a"), block(100, "b")])
        set_hits([("a", 0.90), ("b", 0.80)])
        set_scores({"a": 0.95, "b": 0.10})
        await p.recall_blocks("q")
        ev = [e for e in p.wal.iter_all() if e["event"] == "recall_budget"][-1]
        assert ev["judged"] == 2
        assert ev["rejected_by_judge"] == 1
        assert ev["admitted"] == 1
        assert ev["top_score"] == pytest.approx(0.95)
