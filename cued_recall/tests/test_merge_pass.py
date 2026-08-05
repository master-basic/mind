"""The abstraction pass: one block derived from several near-identical ones.

This is the only pass that creates a memory rather than editing one, so the
tests are mostly about what it refuses to do. A bad merge invents a
generalisation that was never true and then retires the evidence behind it.
"""

import math
from pathlib import Path

import pytest

from cued_recall.index import VectorIndex
from cued_recall.judge import Judge
from cued_recall.models import Block, BlockStatus, BlockType, Verification
from cued_recall.store import BlockStore
from cued_recall.wal import WAL

DIM = 4
DEADLINE = float("inf")


class FakeEmbedder:
    def embed(self, text):
        h = sum(ord(c) for c in text)
        return [1.0, (h % 97) / 970.0, 0.0, 0.0]


@pytest.fixture
def judge(tmp_path, config):
    index = VectorIndex(tmp_path, dim=DIM)
    index.open()
    wal = WAL(tmp_path / "wal.jsonl")
    wal.open()
    config.judge.merge_enabled = True
    config.judge.merge_min_cluster = 3
    config.judge.merge_min_age_s = 0
    j = Judge(config, BlockStore(tmp_path), index, wal, embed=FakeEmbedder())
    yield j
    wal.close()
    index.close()


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def seed(judge, block_id, vec, text="a note about the DNS cache TTL",
         type_=BlockType.reasoning, age_s=100000, pinned=False,
         verification="unknown"):
    import time
    b = Block(block_id=block_id, type=type_, status=BlockStatus.shelved,
              conversation_id="c1", turn_index=0, token_count=50, text=text,
              embed_text=text, question_text="how do I fix DNS?",
              pinned=pinned, verification=Verification(verification),
              created_at=time.time() - age_s)
    judge.store.put(b)
    judge.index.upsert_block_meta(block_id, type_.value, "shelved",
                                  b.created_at, "c1", 0, 50, verification,
                                  0, 0.0)
    judge.index.set_pinned(block_id, pinned)
    judge.index.upsert_vector(block_id, unit(vec))
    return b


def cluster_of(judge, n=3, prefix="b", text="the DNS cache TTL was raised to 300",
               age_s=100000):
    return [seed(judge, f"{prefix}{i}", [1.0, 0.001 * i, 0, 0], text=text,
                 age_s=age_s)
            for i in range(n)]


def stub_merge(judge, summary="DNS cache TTL is raised to 300 to cut latency"):
    async def fake(blocks):
        return summary
    judge._merge_notes = fake


class TestClustering:
    @pytest.mark.asyncio
    async def test_a_cluster_becomes_one_block(self, judge):
        cluster_of(judge, 3)
        stub_merge(judge)
        out = await judge._merge_pass(DEADLINE)
        assert out["merged_blocks"] == 1
        assert out["retired"] == 3

    @pytest.mark.asyncio
    async def test_below_the_minimum_nothing_happens(self, judge):
        cluster_of(judge, 2)
        stub_merge(judge)
        out = await judge._merge_pass(DEADLINE)
        assert out["merged_blocks"] == 0
        assert all(judge.store.get(f"b{i}").status is BlockStatus.shelved
                   for i in range(2))

    @pytest.mark.asyncio
    async def test_dissimilar_blocks_are_not_a_cluster(self, judge):
        seed(judge, "a", [1, 0, 0, 0])
        seed(judge, "b", [0, 1, 0, 0])
        seed(judge, "c", [0, 0, 1, 0])
        stub_merge(judge)
        out = await judge._merge_pass(DEADLINE)
        assert out["merged_blocks"] == 0

    @pytest.mark.asyncio
    async def test_pinned_blocks_are_never_merged(self, judge):
        # A pin says "keep this exactly", and a merge does not keep it exactly.
        cluster_of(judge, 3)
        seed(judge, "pinned", [1.0, 0.0005, 0, 0], pinned=True)
        stub_merge(judge)
        await judge._merge_pass(DEADLINE)
        assert judge.store.get("pinned").status is BlockStatus.shelved

    @pytest.mark.asyncio
    async def test_young_blocks_are_left_alone(self, judge):
        # A cluster that formed in the last hour is a conversation in
        # progress, not a settled repetition.
        judge.config.judge.merge_min_age_s = 86400
        cluster_of(judge, 3, age_s=60)
        stub_merge(judge)
        assert (await judge._merge_pass(DEADLINE))["merged_blocks"] == 0

    @pytest.mark.asyncio
    async def test_a_young_block_is_not_pulled_in_as_a_member(self, judge):
        # The seed query filters on age; the vector search that finds cluster
        # members does not, so the rule has to be applied to members too.
        judge.config.judge.merge_min_age_s = 86400
        cluster_of(judge, 3, age_s=200000)
        seed(judge, "fresh", [1.0, 0.0005, 0, 0], age_s=10)
        stub_merge(judge)
        await judge._merge_pass(DEADLINE)
        assert judge.store.get("fresh").status is BlockStatus.shelved

    @pytest.mark.asyncio
    async def test_corrected_blocks_are_not_generalised_from(self, judge):
        # Contradicted material must not be laundered into a new memory that
        # carries no correction of its own.
        cluster_of(judge, 3)
        judge.index.update_verification("b0", "corrected", "manual")
        stub_merge(judge)
        out = await judge._merge_pass(DEADLINE)
        # b0 drops out both as a seed and as a member, leaving two -- below
        # the minimum, so nothing is merged.
        assert out["merged_blocks"] == 0
        assert judge.store.get("b0").status is BlockStatus.shelved

    @pytest.mark.asyncio
    async def test_a_corrected_member_is_excluded_but_the_rest_can_merge(self, judge):
        cluster_of(judge, 4)
        judge.index.update_verification("b0", "corrected", "model")
        stub_merge(judge)
        out = await judge._merge_pass(DEADLINE)
        assert out["merged_blocks"] == 1
        assert out["retired"] == 3
        assert judge.store.get("b0").status is BlockStatus.shelved
        ev = [e for e in judge.wal.iter_all() if e["event"] == "blocks_merged"][0]
        assert "b0" not in ev["parents"]

    @pytest.mark.asyncio
    async def test_a_block_is_merged_only_once(self, judge):
        cluster_of(judge, 6)
        judge.config.judge.merge_max_per_pass = 5
        stub_merge(judge)
        out = await judge._merge_pass(DEADLINE)
        retired = [e for e in judge.wal.iter_all()
                   if e["event"] == "block_retired_into_merge"]
        assert len(retired) == len({e["block_id"] for e in retired})

    @pytest.mark.asyncio
    async def test_enabled_by_default_since_the_2026_08_05_measurement(self):
        # The flag shipped off until a real pass was measured
        # (evaluate/eval_merge.py): a good merge that kept every specific and
        # fired recall, and a refusal of the draft that dropped one. On now.
        from cued_recall.config import Config
        cfg = Config((Path(__file__).resolve().parent.parent)
                     / "config.example.yaml")
        assert cfg.judge.merge_enabled is True

    @pytest.mark.asyncio
    async def test_the_off_switch_still_stops_merges(self, judge):
        judge.config.judge.merge_enabled = False
        cluster_of(judge, 3)
        stub_merge(judge)
        assert (await judge._merge_pass(DEADLINE))["merged_blocks"] == 0


class TestTheMergedBlock:
    @pytest.mark.asyncio
    async def test_records_its_parents(self, judge):
        cluster_of(judge, 3)
        stub_merge(judge)
        await judge._merge_pass(DEADLINE)
        ev = [e for e in judge.wal.iter_all() if e["event"] == "blocks_merged"][0]
        merged = judge.store.get(ev["block_id"])
        assert sorted(merged.parents) == ["b0", "b1", "b2"]

    @pytest.mark.asyncio
    async def test_is_conversation_agnostic(self, judge):
        # The point of the block is that it holds across the conversations it
        # came from.
        cluster_of(judge, 3)
        stub_merge(judge)
        await judge._merge_pass(DEADLINE)
        ev = [e for e in judge.wal.iter_all() if e["event"] == "blocks_merged"][0]
        assert judge.store.get(ev["block_id"]).conversation_id == ""

    @pytest.mark.asyncio
    async def test_is_recallable(self, judge):
        cluster_of(judge, 3)
        stub_merge(judge)
        await judge._merge_pass(DEADLINE)
        ev = [e for e in judge.wal.iter_all() if e["event"] == "blocks_merged"][0]
        assert judge.index.get_vector(ev["block_id"]) is not None
        assert judge.index.get_meta(ev["block_id"])["status"] == "shelved"

    @pytest.mark.asyncio
    async def test_the_originals_leave_recall_but_not_existence(self, judge):
        blocks = cluster_of(judge, 3)
        stub_merge(judge)
        await judge._merge_pass(DEADLINE)
        for b in blocks:
            assert judge.index.get_vector(b.block_id) is None
            assert judge.index.get_meta(b.block_id)["status"] == "purged"
            # Restorable: the file and its words are still there.
            assert judge.store.get(b.block_id).text

    @pytest.mark.asyncio
    async def test_the_file_survives_even_with_purge_deletes_file_on(self, judge):
        # Decay is "this stopped being worth keeping"; a merge is "this is
        # better said elsewhere". They must not share a delete.
        judge.config.judge.purge_deletes_file = True
        blocks = cluster_of(judge, 3)
        stub_merge(judge)
        await judge._merge_pass(DEADLINE)
        for b in blocks:
            assert judge.store.get(b.block_id) is not None


class TestRefusals:
    @pytest.mark.asyncio
    async def test_a_merge_no_shorter_than_its_members_is_rejected(self, judge):
        # It saved nothing and added a near-duplicate to the index this pass
        # exists to thin out.
        cluster_of(judge, 3, text="short")
        stub_merge(judge, summary="x" * 5000)
        out = await judge._merge_pass(DEADLINE)
        assert out["clusters"] == 1
        assert out["merged_blocks"] == 0
        assert judge.store.get("b0").status is BlockStatus.shelved

    @pytest.mark.asyncio
    async def test_a_model_failure_leaves_everything_alone(self, judge):
        cluster_of(judge, 3)

        async def fail(blocks):
            return None
        judge._merge_notes = fail
        out = await judge._merge_pass(DEADLINE)
        assert out["merged_blocks"] == 0
        assert all(judge.store.get(f"b{i}").status is BlockStatus.shelved
                   for i in range(3))

    @pytest.mark.asyncio
    async def test_an_unembeddable_merge_is_abandoned(self, judge):
        # A merged block nothing can retrieve is worse than no merge, because
        # the originals would be retired behind it.
        cluster_of(judge, 3)
        stub_merge(judge)

        class Broken:
            def embed(self, text):
                raise RuntimeError("embed server down")
        judge.embed = Broken()
        out = await judge._merge_pass(DEADLINE)
        assert out["merged_blocks"] == 0
        assert [e for e in judge.wal.iter_all()
                if e["event"] == "merge_abandoned"]
        for i in range(3):
            assert judge.store.get(f"b{i}").status is BlockStatus.shelved

    @pytest.mark.asyncio
    async def test_the_deadline_stops_the_pass(self, judge):
        cluster_of(judge, 3)
        stub_merge(judge)
        assert (await judge._merge_pass(0.0))["merged_blocks"] == 0


class TestSpecifics:
    """The checkable half of "keep every specific".

    Asking the model was not enough. Run against the real judge on three
    genuine near-duplicates about DNS latency, the first merge it produced read
    "setting dns.cache_ttl=300 ... reduces the cache TTL from 30 seconds to
    60ms" -- while the originals said the TTL *was* 30s and that latency fell
    from 840ms to 60ms. Two quantities conflated and 840 lost, in a block that
    was about to have its evidence retired behind it.
    """

    def test_finds_numbers_paths_and_identifiers(self, judge):
        found = judge._specifics(
            "set dns.cache_ttl=300 in /etc/resolv-cache.conf, was 30s")
        assert "300" in found
        assert "30" in found
        assert "/etc/resolv-cache.conf" in found
        assert "dns.cache_ttl" in found

    def test_trailing_sentence_punctuation_is_not_part_of_the_specific(self, judge):
        # Otherwise a path at the end of a sentence never matches the same path
        # mid-sentence, and every merge is rejected for losing what it kept.
        assert (judge._specifics("edit /etc/hosts.")
                == judge._specifics("edit /etc/hosts now"))

    def test_a_merge_that_keeps_everything_loses_nothing(self, judge):
        blocks = [Block(text="raise dns.cache_ttl to 300 in /etc/resolv.conf")]
        merged = "dns.cache_ttl=300 in /etc/resolv.conf fixes it"
        assert judge._lost_specifics(blocks, merged) == set()

    def test_a_dropped_number_is_reported(self, judge):
        blocks = [Block(text="latency fell from 840ms to 60ms")]
        assert "840" in judge._lost_specifics(blocks, "latency fell to 60ms")

    def test_specifics_are_pooled_across_all_the_members(self, judge):
        blocks = [Block(text="the port is 8080"), Block(text="the retry is 3")]
        lost = judge._lost_specifics(blocks, "the port is 8080")
        assert lost == {"3"}

    @pytest.mark.asyncio
    async def test_a_merge_that_drops_a_number_is_refused(self, judge):
        cluster_of(judge, 3, text="latency fell from 840ms to 60ms on port 8080")
        stub_merge(judge, summary="latency improved on port 8080")
        out = await judge._merge_pass(DEADLINE)
        assert out["merged_blocks"] == 0
        ev = [e for e in judge.wal.iter_all() if e["event"] == "merge_rejected"]
        assert ev and ev[0]["reason"] == "dropped specifics"
        assert "840" in ev[0]["lost_specifics"]
        # And nothing was retired behind the bad generalisation.
        assert all(judge.store.get(f"b{i}").status is BlockStatus.shelved
                   for i in range(3))


class TestMergePrompt:
    def test_demands_the_specifics_are_kept(self, judge):
        p = judge._merge_prompt(["note one", "note two"])
        # A merge that keeps only the theme is a lossy delete wearing a
        # summary's clothes.
        assert "Keep every specific" in p
        assert "disagree" in p

    def test_asks_for_what_holds_across_them(self, judge):
        p = judge._merge_prompt(["a", "b"])
        assert "in place of all of them" in p
        assert "not what each one said" in p
