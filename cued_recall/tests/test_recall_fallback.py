"""Resilience (Phase 5.2, F10): a cosine floor and a keyword channel.

Two behaviours: the judge can be skipped entirely when the best candidate is
below a floor (the off-topic tax from throughput.md is 1.5-2.2 s of CPU calls
to conclude "nothing here"), and an embed server outage degrades recall to a
gist/tag keyword channel instead of silently recalling nothing.

The floor defaults to 0.0 (off): the plan proposed 0.30 but that sits below
the retrieval threshold (0.48) and can never fire, and the measured corpus has
no safe value -- see config.example.yaml. So these tests exercise the
mechanism with a configured floor rather than asserting a default.
"""

import math

import pytest

from cued_recall.index import VectorIndex
from cued_recall.models import Block, BlockStatus, BlockType, Verification
from cued_recall.pipeline import Pipeline
from cued_recall.utils import distinctive_terms


@pytest.fixture
def index(tmp_path):
    ix = VectorIndex(tmp_path, dim=4)
    ix.open()
    yield ix
    ix.close()


class TestDistinctiveTerms:
    def test_drops_function_words_and_fragments(self):
        got = distinctive_terms("the dns cache ttl settings please")
        assert "the" not in got
        assert "cache" in got
        assert all(len(t) >= 4 for t in got)

    def test_dedupes_and_sorts_longest_first(self):
        got = distinctive_terms(
            "check the dns settings, check the ttl, check the cache")
        assert got == sorted(got, key=lambda w: (-len(w), w))
        assert len(set(got)) == len(got)

    def test_bounded_to_max_terms(self):
        got = distinctive_terms(
            "alpha beta gamma delta epsilon zeta eta theta", max_terms=3)
        assert len(got) == 3

    def test_noise_and_empty_match_nothing(self):
        assert distinctive_terms("a b c") == []
        assert distinctive_terms("") == []

    def test_terms_are_safe_as_like_patterns(self):
        # Distinctive terms must never contain the SQL LIKE wildcards.
        for t in distinctive_terms("100% of the underscore_score please"):
            assert "_" not in t and "%" not in t


def add_kw(index, block_id, gist="", tags=(), status="shelved"):
    index.upsert_block_meta(block_id, "reasoning", status, 1000.0, "c1", 0, 10,
                            "unknown", 0, 0.0)
    if gist or tags:
        index.set_tags(block_id, list(tags), gist)


class TestKeywordQuery:
    def test_matches_against_gist_and_tags(self, index):
        add_kw(index, "by-gist", gist="dns cache ttl tuning")
        add_kw(index, "by-tag", tags=["networking"])
        got = dict(index.keyword_query("dns cache networking", k=10))
        assert "by-gist" in got
        assert "by-tag" in got

    def test_scored_by_fraction_of_terms_matched(self, index):
        add_kw(index, "both", gist="cache settings details")
        add_kw(index, "one", gist="cache server notes")
        got = dict(index.keyword_query("cache settings network", k=10))
        assert got["both"] == pytest.approx(2 / 3)
        assert got["one"] == pytest.approx(1 / 3)

    def test_ranked_by_score_desc(self, index):
        add_kw(index, "one", gist="cache notes")
        add_kw(index, "both", gist="cache settings")
        ids = [b for b, _ in index.keyword_query("cache settings", k=10)]
        assert ids == ["both", "one"]

    def test_only_shelved_and_truncated_are_recalled(self, index):
        add_kw(index, "hot", gist="dns cache ttl", status="hot")
        add_kw(index, "ok", gist="dns cache ttl")
        got = [b for b, _ in index.keyword_query("dns cache ttl", k=10)]
        assert got == ["ok"]

    def test_stopword_only_query_matches_nothing(self, index):
        add_kw(index, "x", gist="dns cache")
        assert index.keyword_query("the and of", k=10) == []

    def test_missing_gist_and_tags_never_match(self, index):
        index.upsert_block_meta("bare", "reasoning", "shelved", 1000.0, "c1",
                                0, 10, "unknown", 0, 0.0)
        assert index.keyword_query("dns cache", k=10) == []


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


class TestFloorShortCircuit:
    """With floor configured, a best candidate below it skips the judge."""

    @pytest.fixture
    def rig(self, pipeline, monkeypatch):
        pipeline.config.recall.budget_tokens = 1000
        pipeline.config.recall.judge_enabled = True
        pipeline.config.recall.floor = 0.60

        def set_hits(hits):
            monkeypatch.setattr(pipeline.index, "query",
                                lambda vec, k, thr: hits[:k])

        judge_calls = []

        def set_scores(scores):
            async def fake(question, candidates, keyword_ids=None):
                judge_calls.append(len(candidates))
                return [(b, sim, scores[b.block_id]) for b, sim in candidates]
            monkeypatch.setattr(pipeline, "_score_by_relevance", fake)

        monkeypatch.setattr(pipeline, "embed",
                            type("E", (), {"embed": staticmethod(
                                lambda t: [0.0, 0.0, 0.0, 1.0])})())
        return pipeline, set_hits, set_scores, judge_calls

    @pytest.mark.asyncio
    async def test_best_below_floor_skips_the_judge(self, rig):
        # The throughput.md tax in one case: top similarity 0.50, which the
        # judge would take 1.5-2.2 s to reject. The floor skips it.
        p, set_hits, set_scores, judge_calls = rig
        seed(p, [block(10, "a")])
        set_hits([("a", 0.50)])
        got = await p.recall_blocks("q")
        assert got == []
        assert judge_calls == []
        ev = [e for e in p.wal.iter_all() if e["event"] == "recall_floor"][-1]
        assert ev["best_similarity"] == pytest.approx(0.50)
        budget = [e for e in p.wal.iter_all()
                  if e["event"] == "recall_budget"][-1]
        assert budget["judged"] == 0
        assert budget["admitted"] == 0

    @pytest.mark.asyncio
    async def test_at_or_above_the_floor_the_judge_still_runs(self, rig):
        p, set_hits, set_scores, judge_calls = rig
        seed(p, [block(10, "a")])
        set_hits([("a", 0.60)])
        set_scores({"a": 0.95})
        got = await p.recall_blocks("q")
        assert [b.block_id for b, _ in got] == ["a"]
        assert judge_calls == [1]

    @pytest.mark.asyncio
    async def test_floor_defaults_to_off(self, pipeline, monkeypatch):
        # The shipped default must preserve the old behaviour exactly.
        assert pipeline.config.recall.floor == 0.0
        pipeline.config.recall.budget_tokens = 1000
        pipeline.config.recall.judge_enabled = True
        seed(pipeline, [block(10, "a")])
        monkeypatch.setattr(pipeline.index, "query",
                            lambda vec, k, thr: [("a", 0.50)])
        monkeypatch.setattr(pipeline, "embed",
                            type("E", (), {"embed": staticmethod(
                                lambda t: [0.0, 0.0, 0.0, 1.0])})())

        async def fake(question, candidates, keyword_ids=None):
            return [(b, sim, 0.95) for b, sim in candidates]
        monkeypatch.setattr(pipeline, "_score_by_relevance", fake)
        got = await pipeline.recall_blocks("q")
        assert [b.block_id for b, _ in got] == ["a"]


class TestEmbedFailureFallback:
    @pytest.fixture
    def rig(self, pipeline, monkeypatch):
        pipeline.config.recall.budget_tokens = 1000
        pipeline.config.recall.judge_enabled = True

        def embed_down(text):
            raise RuntimeError("embed server down")
        monkeypatch.setattr(pipeline, "embed",
                            type("E", (), {"embed": staticmethod(embed_down)})())
        return pipeline

    @pytest.mark.asyncio
    async def test_embed_failure_degrades_to_the_keyword_channel(self, rig,
                                                                 monkeypatch):
        p = rig
        seed(p, [block(10, "kw")])
        monkeypatch.setattr(p.index, "keyword_query",
                            lambda q, k: [("kw", 1.0)])

        async def fake(question, candidates, keyword_ids=None):
            return [(b, sim, 0.9) for b, sim in candidates]
        monkeypatch.setattr(p, "_score_by_relevance", fake)

        got = await p.recall_blocks("q")
        assert [b.block_id for b, _ in got] == ["kw"]
        evs = [e for e in p.wal.iter_all() if e["event"] == "recall_embed_error"]
        assert evs
        budget = [e for e in p.wal.iter_all()
                  if e["event"] == "recall_budget"][-1]
        assert budget["source"] == "keywords"

    @pytest.mark.asyncio
    async def test_keyword_fallback_still_drops_corrected_blocks(self, rig,
                                                                 monkeypatch):
        p = rig
        seed(p, [block(10, "kw")])
        p.index.update_verification("kw", "corrected", "manual")
        monkeypatch.setattr(p.index, "keyword_query",
                            lambda q, k: [("kw", 1.0)])
        judged = []

        async def fake(question, candidates, keyword_ids=None):
            judged.extend(b.block_id for b, _ in candidates)
            return [(b, sim, 0.9) for b, sim in candidates]
        monkeypatch.setattr(p, "_score_by_relevance", fake)

        got = await p.recall_blocks("q")
        assert got == []
        assert judged == []

    @pytest.mark.asyncio
    async def test_embed_failure_with_the_channel_off_returns_nothing(self, rig,
                                                                      monkeypatch):
        p = rig
        p.config.recall.tag_channel = False
        called = []

        def boom(query, k):
            called.append(1)
            return []
        monkeypatch.setattr(p.index, "keyword_query", boom)
        got = await p.recall_blocks("q")
        assert got == []
        assert called == []
