"""Judge._should_purge -- the whole source-authority table, row by row.

This is the only function in the system that can destroy a memory, and it is
pure arithmetic over five inputs, so there is no excuse for it to be verified
by reading it. The table it implements is documented in its own docstring; each
row below is one line of that table.
"""

import time
from pathlib import Path

import pytest

from cued_recall.config import Config
from cued_recall.judge import Judge

EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / "config.example.yaml"

DAY = 86400


@pytest.fixture(scope="module")
def judge():
    config = Config(EXAMPLE_CONFIG)
    # _should_purge touches nothing but self.config.judge.
    return Judge(config, store=None, index=None, wal=None)


@pytest.fixture(scope="module")
def cfg(judge):
    return judge.config.judge


class TestPinned:
    def test_a_pin_beats_every_other_rule(self, judge, cfg):
        # Including a manual correction: the pin is the user's other hand.
        for verification in ("unknown", "accepted", "corrected"):
            for source in ("", "manual", "pattern", "model"):
                assert not judge._should_purge(
                    verification, recall_count=0, age=999 * DAY,
                    worthless=True, source=source, pinned=True
                )


class TestCorrectedBySource:
    def test_manual_purges_at_once(self, judge):
        # The user pressed the button. No age gate, no recall protection.
        assert judge._should_purge("corrected", recall_count=99, age=0,
                                   worthless=False, source="manual")

    def test_pattern_purges_only_if_never_recalled_and_past_grace(self, judge, cfg):
        past = cfg.corrected_grace_s + 1
        assert judge._should_purge("corrected", 0, past, False, source="pattern")

    def test_pattern_will_not_purge_inside_the_grace_window(self, judge, cfg):
        assert not judge._should_purge("corrected", 0, cfg.corrected_grace_s - 1,
                                       False, source="pattern")

    def test_pattern_will_not_purge_a_block_that_was_being_used(self, judge, cfg):
        # 17 hand-written regexes with no measured false-positive rate do not
        # get to delete a memory the system was actively recalling.
        assert not judge._should_purge("corrected", 1, 999 * DAY, False,
                                       source="pattern")

    def test_model_gets_no_special_power(self, judge, cfg):
        # The 1.5B classifier falls through to the ordinary bar -- it must
        # clear the age gate and then the utility rule, exactly like an
        # uncorrected block, rather than getting a shortcut of its own.
        assert not judge._should_purge("corrected", 0, cfg.purge_age_s - 1,
                                       False, source="model")
        assert judge._should_purge("corrected", 0, cfg.purge_age_s + 1,
                                   False, source="model")
        # Recently useful, so still kept despite the model's accusation.
        assert not judge._should_purge("corrected", 5, 999 * DAY, False,
                                       source="model",
                                       last_recalled=time.time())

    def test_an_unrecorded_source_is_treated_as_model(self, judge, cfg):
        assert not judge._should_purge("corrected", 5, 999 * DAY, False,
                                       source="", last_recalled=time.time())


class TestOrdinaryDecay:
    def test_one_recall_no_longer_makes_a_block_immortal(self, judge, cfg):
        # The rule this replaced: `recall_count > 0` exempted a block from
        # age-based purging forever, so the store could only grow and "used
        # once, years ago" outranked nothing at all.
        assert judge._should_purge("unknown", 1, 999 * DAY, worthless=True)

    def test_a_single_recall_still_buys_a_month(self, judge, cfg):
        # It is a gradient, not a reversal: one recall earns
        # utility_recall_weight days of idleness before the block goes.
        age = cfg.purge_age_s + 1
        assert not judge._should_purge("unknown", 1, age, worthless=False)

    def test_worthless_goes_earlier_than_ordinary(self, judge, cfg):
        assert cfg.worthless_age_s < cfg.purge_age_s
        age = cfg.worthless_age_s + 1
        assert judge._should_purge("unknown", 0, age, worthless=True)
        assert not judge._should_purge("unknown", 0, age, worthless=False)

    def test_worthless_still_waits_out_its_own_window(self, judge, cfg):
        assert not judge._should_purge("unknown", 0, cfg.worthless_age_s - 1,
                                       worthless=True)

    def test_ordinary_purges_past_purge_age(self, judge, cfg):
        assert judge._should_purge("unknown", 0, cfg.purge_age_s + 1, False)
        assert not judge._should_purge("unknown", 0, cfg.purge_age_s - 1, False)

    def test_accepted_is_not_protected_by_its_verification_alone(self, judge, cfg):
        # "accepted" means recalled-and-uncontested, which is already recorded
        # in recall_count; the verification string itself grants nothing.
        assert judge._should_purge("accepted", 0, cfg.purge_age_s + 1, False)


class TestUtilityGradient:
    """The point of the change: "used once, long ago" and "used weekly" are
    now different states, where before both were simply "used"."""

    def test_recent_use_outranks_old_use(self, judge, cfg):
        old = time.time() - 400 * DAY
        recent = time.time() - 1 * DAY
        age = 500 * DAY
        assert judge._should_purge("unknown", 1, age, False, last_recalled=old)
        assert not judge._should_purge("unknown", 1, age, False,
                                       last_recalled=recent)

    def test_repeated_recall_buys_proportionally_more_time(self, judge, cfg):
        # 1 recall = 30 days of idleness; 10 = 300.
        last = time.time() - 100 * DAY
        age = 500 * DAY
        assert judge._should_purge("unknown", 1, age, False, last_recalled=last)
        assert not judge._should_purge("unknown", 10, age, False,
                                       last_recalled=last)

    def test_an_uncontested_recall_is_worth_more_than_a_bare_one(self, judge, cfg):
        assert (cfg.utility_uncontested_weight
                > cfg.utility_recall_weight)
        last = time.time() - 100 * DAY
        age = 500 * DAY
        assert judge._should_purge("unknown", 2, age, False, last_recalled=last)
        assert not judge._should_purge("unknown", 2, age, False,
                                       uncontested=2, last_recalled=last)

    def test_the_age_gate_still_protects_young_blocks(self, judge, cfg):
        # A quiet week must not empty the store, however unused everything is.
        assert not judge._should_purge("unknown", 0, cfg.purge_age_s - 1, False)
        assert not judge._should_purge("unknown", 0, cfg.worthless_age_s - 1,
                                       worthless=True)

    def test_a_pin_still_beats_everything(self, judge, cfg):
        assert not judge._should_purge("unknown", 0, 999 * DAY, True,
                                       pinned=True)

    def test_the_old_rule_is_still_reachable(self, judge, cfg):
        # Deletion is the one thing the index cannot undo on its own, so the
        # previous behaviour stays switchable.
        cfg.utility_decay = False
        try:
            assert not judge._should_purge("unknown", 1, 999 * DAY, True)
            assert judge._should_purge("unknown", 0, 999 * DAY, False)
        finally:
            cfg.utility_decay = True


class TestUtility:
    def test_never_recalled_is_negative_from_the_start(self, judge):
        assert judge._utility(0, 0, 10 * DAY) < 0

    def test_idle_time_is_measured_from_last_use_not_creation(self, judge, cfg):
        # Otherwise a block that keeps being recalled still ages out on a
        # fixed schedule, which is the bug in a different costume.
        just_used = judge._utility(1, 0, 999 * DAY, time.time())
        never_since = judge._utility(1, 0, 999 * DAY, 0.0)
        assert just_used > 0 > never_since

    def test_uncontested_recalls_add_on_top_of_recalls(self, judge, cfg):
        # One `now` for both, so the idle term is identical and the difference
        # is only the uncontested weight.
        now = time.time()
        base = judge._utility(3, 0, 0, now)
        more = judge._utility(3, 2, 0, now)
        assert more - base == pytest.approx(cfg.utility_uncontested_weight * 2)
