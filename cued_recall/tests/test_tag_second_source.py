"""Phase 5.1 (F3, F11): the tag/gist channel as a second candidate source.

Vector hits miss probes whose wording shares no tokens with a stored block but
whose gist/tags overlap -- the "relevant but semantically distant wording"
miss class. When recall.tag_second_source is on, keyword hits join the judge
pool alongside the vector hits, deduped, and the judge arbitrates both:
keyword-sourced candidates get their gist/tags appended to the note, because
those are the evidence the channel matched on.

The default is off: the vector operating point is a measured result, and the
acceptance rows that would justify the merge (overlap fires recall; overlap
with different content is still rejected) do not exist yet.
"""

import pytest

from cued_recall.models import Block, BlockStatus, BlockType, Verification
from cued_recall.pipeline import Pipeline


def block(tokens, block_id, gist="", tags=()):
    return Block(block_id=block_id, type=BlockType.reasoning,
                 status=BlockStatus.shelved, token_count=tokens,
                 text=f"block {block_id}", question_text=f"q{block_id}",
                 verification=Verification.unknown, gist=gist, tags=list(tags))


def seed(pipeline, blocks):
    for b in blocks:
        pipeline.store.put(b)
        pipeline.index.upsert_block_meta(
            b.block_id, b.type.value, b.status.value, b.created_at,
            "c1", 0, b.token_count, "unknown", 0, 0.0)


class TestConfig:
    def test_second_source_defaults_off(self, config):
        # Old behaviour is vector-only; the merge must not change it silently.
        assert config.recall.tag_second_source is False

    def test_the_channel_itself_defaults_on(self, config):
        assert config.recall.tag_channel is True


class TestMergeKeywordHits:
    def test_vector_hits_first_then_keyword_hits(self):
        out = Pipeline._merge_keyword_hits(
            [("v1", 0.80), ("v2", 0.70)], [("k1", 0.5), ("k2", 0.25)])
        assert [bid for bid, _ in out] == ["v1", "v2", "k1", "k2"]

    def test_a_block_in_both_channels_keeps_the_cosine(self):
        out = dict(Pipeline._merge_keyword_hits(
            [("both", 0.90)], [("both", 1.0), ("kw", 0.5)]))
        assert out == {"both": 0.90, "kw": 0.5}

    def test_keyword_hits_carry_their_match_fraction(self):
        out = Pipeline._merge_keyword_hits([], [("kw", 0.5)])
        assert out == [("kw", 0.5)]


class TestTaxonomyNote:
    def test_appends_tags_and_gist(self):
        b = block(10, "x", gist="dns tuning", tags=["networking"])
        got = Pipeline._taxonomy_note(b, "what was asked")
        assert got.startswith("what was asked")
        assert "Tags: networking" in got
        assert "Gist: dns tuning" in got

    def test_an_untagged_block_leaves_the_note_untouched(self):
        b = block(10, "x")
        assert Pipeline._taxonomy_note(b, "what was asked") == "what was asked"

    def test_tags_without_gist_still_get_appended(self):
        b = block(10, "x", tags=["networking"])
        got = Pipeline._taxonomy_note(b, "what was asked")
        assert "Tags: networking" in got


class TestSecondSourceInRecall:
    @pytest.fixture
    def rig(self, pipeline, monkeypatch):
        pipeline.config.recall.budget_tokens = 1000
        pipeline.config.recall.judge_enabled = True
        pipeline.config.recall.tag_second_source = True

        def set_hits(hits):
            monkeypatch.setattr(pipeline.index, "query",
                                lambda vec, k, thr: hits[:k])

        kw_calls = []

        def set_kw(hits):
            def fake(q, k):
                kw_calls.append(q)
                return hits[:k]
            monkeypatch.setattr(pipeline.index, "keyword_query", fake)

        judged = []

        def set_scores(scores):
            async def fake(question, candidates, keyword_ids=None):
                judged.append(([b.block_id for b, _ in candidates],
                               set(keyword_ids or ())))
                return [(b, sim, scores.get(b.block_id, 0.5))
                        for b, sim in candidates]
            monkeypatch.setattr(pipeline, "_score_by_relevance", fake)

        monkeypatch.setattr(pipeline, "embed",
                            type("E", (), {"embed": staticmethod(
                                lambda t: [0.0, 0.0, 0.0, 1.0])})())
        return pipeline, set_hits, set_kw, set_scores, judged, kw_calls

    @pytest.mark.asyncio
    async def test_keyword_hits_join_the_judge_pool(self, rig):
        # Vector finds "v"; the gist channel alone surfaces "kw". Both go to
        # the judge, and the judge sees which candidate came from which
        # channel so it can weigh the taxonomy evidence.
        p, set_hits, set_kw, set_scores, judged, _ = rig
        seed(p, [block(10, "v"), block(10, "kw", gist="dns tuning")])
        set_hits([("v", 0.80)])
        set_kw([("kw", 1.0)])
        set_scores({"v": 0.9, "kw": 0.9})

        got = await p.recall_blocks("what about the dns settings")
        assert [b.block_id for b, _ in got] == ["v", "kw"]
        assert judged == [(["v", "kw"], {"kw"})]
        budget = [e for e in p.wal.iter_all()
                  if e["event"] == "recall_budget"][-1]
        assert budget["source"] == "vector+keywords"
        assert budget["keyword_candidates"] == 1
        # top_similarity stays a true cosine, not a keyword fraction.
        assert budget["top_similarity"] == pytest.approx(0.80)

    @pytest.mark.asyncio
    async def test_keyword_only_path_still_recalls(self, rig):
        # The acceptance case for the "relevant but semantically distant
        # wording" miss: the probe shares no tokens with the stored block, so
        # nothing clears the cosine threshold -- but the gist overlaps.
        p, set_hits, set_kw, set_scores, judged, _ = rig
        seed(p, [block(10, "kw", gist="dns cache tuning")])
        set_hits([])
        set_kw([("kw", 1.0)])
        set_scores({"kw": 0.9})

        got = await p.recall_blocks("how do I fix the resolver")
        assert [b.block_id for b, _ in got] == ["kw"]
        assert judged == [(["kw"], {"kw"})]
        budget = [e for e in p.wal.iter_all()
                  if e["event"] == "recall_budget"][-1]
        assert budget["source"] == "keywords"
        assert budget["keyword_candidates"] == 1
        assert budget["top_similarity"] is None

    @pytest.mark.asyncio
    async def test_a_block_in_both_channels_is_judged_once(self, rig):
        p, set_hits, set_kw, set_scores, judged, _ = rig
        seed(p, [block(10, "both", gist="dns cache")])
        set_hits([("both", 0.90)])
        set_kw([("both", 1.0)])
        set_scores({"both": 0.9})

        got = await p.recall_blocks("dns cache")
        assert [b.block_id for b, _ in got] == ["both"]
        assert judged == [(["both"], {"both"})]
        budget = [e for e in p.wal.iter_all()
                  if e["event"] == "recall_budget"][-1]
        assert budget["source"] == "vector+keywords"
        assert budget["keyword_candidates"] == 1

    @pytest.mark.asyncio
    async def test_floor_skips_the_keyword_channel(self, rig):
        # The floor's verdict is "this turn is off-topic": pulling keyword
        # hits in anyway would drag the judge back into the tax the floor
        # removes.
        p, set_hits, set_kw, set_scores, judged, kw_calls = rig
        p.config.recall.floor = 0.60
        seed(p, [block(10, "v")])
        set_hits([("v", 0.50)])
        set_scores({"v": 0.9})

        got = await p.recall_blocks("q")
        assert got == []
        assert judged == []
        assert kw_calls == []

    @pytest.mark.asyncio
    async def test_default_off_never_calls_the_keyword_channel(
            self, pipeline, monkeypatch):
        # The shipped default preserves the old vector-only behaviour exactly.
        pipeline.config.recall.budget_tokens = 1000
        pipeline.config.recall.judge_enabled = True
        seed(pipeline, [block(10, "a")])
        monkeypatch.setattr(pipeline.index, "query",
                            lambda vec, k, thr: [("a", 0.90)])
        monkeypatch.setattr(pipeline.index, "keyword_query", lambda q, k: (
            pytest.fail("keyword channel must not run with the source off")))

        async def fake(question, candidates, keyword_ids=None):
            assert keyword_ids is None
            return [(b, sim, 0.9) for b, sim in candidates]
        monkeypatch.setattr(pipeline, "_score_by_relevance", fake)
        monkeypatch.setattr(pipeline, "embed",
                            type("E", (), {"embed": staticmethod(
                                lambda t: [0.0, 0.0, 0.0, 1.0])})())

        got = await pipeline.recall_blocks("q")
        assert [b.block_id for b, _ in got] == ["a"]
