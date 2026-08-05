"""Token counts, and the one place they are allowed to be guessed.

Block.token_count enforces recall.budget_tokens, judge consolidation gates and
the admin totals. It was once written as len(text.split()) -- a word count,
which measured 36 of 36 sampled blocks low (mean 0.77x, worst 0.52x on code and
markdown), so a nominal 3,000-token recall budget was really spending ~3,900.
The same units bug then reappeared in a different costume as the embed cap, and
sent inputs the embedding server refused with a 400.

These tests pin the boundary: exact where a tokenizer exists, conservative
where one does not, and never a bare word count.
"""

import inspect
import re
from pathlib import Path

import pytest

from cued_recall import pipeline as pipeline_module
from cued_recall.utils import count_tokens, estimate_tokens

PIPELINE_SRC = Path(pipeline_module.__file__).read_text(encoding="utf-8")


class TestCountTokens:
    @pytest.mark.asyncio
    async def test_uses_the_tokenizer_when_it_answers(self, monkeypatch):
        async def exact(text, endpoint, timeout=10):
            return 4242
        monkeypatch.setattr("cued_recall.utils.count_tokens_exact", exact)
        assert await count_tokens("some text", "http://server") == 4242

    @pytest.mark.asyncio
    async def test_falls_back_only_when_the_tokenizer_is_down(self, monkeypatch):
        async def down(text, endpoint, timeout=10):
            return None
        monkeypatch.setattr("cued_recall.utils.count_tokens_exact", down)
        text = "the quick brown fox jumps over the lazy dog " * 10
        assert await count_tokens(text, "http://server") == estimate_tokens(text)

    @pytest.mark.asyncio
    async def test_a_zero_from_the_tokenizer_is_not_a_failure(self, monkeypatch):
        # 0 is falsy. Treating it as "server down" would silently swap in the
        # estimator for empty input and report a nonzero count for nothing.
        async def zero(text, endpoint, timeout=10):
            return 0
        monkeypatch.setattr("cued_recall.utils.count_tokens_exact", zero)
        assert await count_tokens("anything", "http://server") == 0

    @pytest.mark.asyncio
    async def test_empty_text_costs_nothing_and_calls_nobody(self, monkeypatch):
        async def boom(text, endpoint, timeout=10):
            raise AssertionError("should not have been called")
        monkeypatch.setattr("cued_recall.utils.count_tokens_exact", boom)
        assert await count_tokens("", "http://server") == 0


class TestEstimatorNeverReadsLow:
    """The fallback has one job: never hand back a number below the truth.

    Over-counting trims some history early. Under-counting overflows a context
    window (a hard 400 and an empty reply) or an embedding window (a 400 and a
    block that is stored but never recallable).
    """

    @pytest.mark.parametrize("text", [
        "plain english prose about nothing in particular, at some length",
        'const x = {"a": 1, "b": [2, 3]}; // symbol-dense code',
        "Səhv işləmir, düz deyil, yanlış nəticə alınmır.",
        "| col | col |\n|---|---|\n| a | b |",
        "one",
        "   ",
    ])
    def test_estimate_is_at_least_the_word_count(self, text):
        assert estimate_tokens(text) >= len(text.split())

    def test_code_costs_more_than_its_words(self):
        code = 'x={"k":[1,2,3]};\n' * 50
        assert estimate_tokens(code) > len(code.split())

    def test_scales_monotonically(self):
        # EmbeddingClient.fit trims by scaling length against the estimate,
        # which only terminates if longer text never estimates smaller.
        base = "some representative text " * 5
        assert estimate_tokens(base * 2) > estimate_tokens(base)


class TestNoWordCountsOnTheBudgetPath:
    def test_token_count_is_only_ever_set_from_count_tokens(self):
        # The structural version of the bug: any other source for this field
        # is a units mismatch waiting to be measured in production.
        assignments = re.findall(r"token_count=(.+?),?\n", PIPELINE_SRC)
        assert assignments, "no token_count assignments found -- test is stale"
        for a in assignments:
            assert "_count_tokens" in a, f"token_count set from {a.strip()!r}"

    def test_no_split_based_token_count(self):
        assert not re.search(r"token_count\s*=\s*len\(.*split\(\)\)",
                             PIPELINE_SRC)

    def test_word_counts_that_remain_are_named_words(self):
        # Diagnostics may use word counts; they may not call them tokens.
        # "reasoning_tokens"/"result_tokens" held word counts for months.
        for m in re.finditer(r'"(\w+)":\s*len\((.+?)\.split\(\)\)',
                             PIPELINE_SRC):
            field = m.group(1)
            assert not field.endswith("tokens"), (
                f'WAL field "{field}" holds a word count but is named tokens')


class TestEmbedCapAgrees:
    def test_the_embed_cap_uses_the_same_estimator(self):
        # The embed guard and the context guard must not disagree about what a
        # token is; that disagreement is what let 1,024 words of code (2,338
        # tokens) reach a 2,048-token server.
        from cued_recall.embed import EmbeddingClient
        src = inspect.getsource(EmbeddingClient.fit)
        assert "estimate_tokens" in src
