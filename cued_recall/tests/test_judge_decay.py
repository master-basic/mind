"""Judge._should_purge -- the whole source-authority table, row by row.

This is the only function in the system that can destroy a memory, and it is
pure arithmetic over five inputs, so there is no excuse for it to be verified
by reading it. The table it implements is documented in its own docstring; each
row below is one line of that table.
"""

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
        # The 1.5B classifier falls through to the ordinary bar: it must be
        # never-recalled AND old, exactly like an uncorrected block.
        assert not judge._should_purge("corrected", 1, 999 * DAY, False,
                                       source="model")
        assert not judge._should_purge("corrected", 0, cfg.purge_age_s - 1,
                                       False, source="model")
        assert judge._should_purge("corrected", 0, cfg.purge_age_s + 1,
                                   False, source="model")

    def test_an_unrecorded_source_is_treated_as_model(self, judge, cfg):
        assert not judge._should_purge("corrected", 1, 999 * DAY, False, source="")


class TestOrdinaryDecay:
    def test_one_recall_currently_makes_a_block_immortal(self, judge, cfg):
        # Documenting the behaviour, not endorsing it: retrieval is the only
        # evidence the system gathers on its own that a memory is load-bearing,
        # so a single recall exempts a block from age-based purging forever.
        # There is no gradient between "used once, years ago" and "used weekly".
        assert not judge._should_purge("unknown", 1, 999 * DAY, worthless=True)

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
