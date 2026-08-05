"""The pure text/token logic every other subsystem sits on.

These functions have no server, no store and no config behind them, so they are
the part of the pipeline that can be pinned down exactly -- and the part whose
silent drift has already cost this project twice (the word-count-as-token-count
bug in backfill_token_counts.py, and the reasoning splitter returning an empty
left half on single-paragraph input).
"""

from cued_recall.utils import (
    BYTES_PER_TOKEN,
    build_stimulus,
    estimate_tokens,
    matches_correction,
    relevance_prompt,
    split_paragraph_boundary,
    truncate_tokens,
)


class TestTruncateTokens:
    def test_under_limit_is_returned_unchanged(self):
        # Not merely equal in content: the original object, whitespace and all.
        text = "one  two\n\nthree"
        assert truncate_tokens(text, 10) == text

    def test_at_limit_is_unchanged(self):
        assert truncate_tokens("a b c", 3) == "a b c"

    def test_over_limit_keeps_the_first_n_words(self):
        assert truncate_tokens("a b c d e", 3) == "a b c"

    def test_empty(self):
        assert truncate_tokens("", 5) == ""

    def test_truncation_normalises_whitespace(self):
        # Documented, not desired: truncation goes through split()/join(), so a
        # truncated block loses its paragraph structure while an untruncated one
        # keeps it. Anything comparing the two must expect that.
        assert truncate_tokens("a\n\nb c", 2) == "a b"


class TestEstimateTokens:
    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0

    def test_takes_the_largest_of_the_three_signals(self):
        text = "hello world " * 50
        by_chars = len(text) / 3.2
        by_words = len(text.split()) * 1.3
        by_bytes = len(text.encode("utf-8")) / BYTES_PER_TOKEN
        assert estimate_tokens(text) == int(max(by_chars, by_words, by_bytes))

    def test_never_under_counts_ascii_prose(self):
        # The failure that matters is reading small: an under-count hands the
        # server a prompt past its window. Word count alone would.
        text = "The quick brown fox jumps over the lazy dog. " * 20
        assert estimate_tokens(text) >= len(text.split())

    def test_azerbaijani_costs_more_than_its_word_count(self):
        # ə/ş/ğ/ı are multi-byte, and the byte signal is what catches them --
        # the whole reason BYTES_PER_TOKEN exists.
        text = "Səhv işləmir, düz deyil, yanlış nəticə alınmır. " * 10
        assert estimate_tokens(text) > len(text.split())


class TestSplitParagraphBoundary:
    def test_splits_on_a_blank_line(self):
        left, right = split_paragraph_boundary("A B C\n\nD E F", 3)
        assert left == "A B C"
        assert right == "D E F"

    def test_falls_back_to_sentences_when_there_is_no_blank_line(self):
        # The bug this guards: a single long paragraph used to come back with
        # an empty left half, and the caller then emitted a zero-token block.
        left, right = split_paragraph_boundary(
            "One two three. Four five six.", 3
        )
        assert left == "One two three."
        assert right == "Four five six."

    def test_short_input_is_all_left(self):
        left, right = split_paragraph_boundary("a b c", 100)
        assert left == "a b c"
        assert right == ""

    def test_left_is_never_empty_for_non_empty_input(self):
        # _split_reasoning loops on `remaining` and only breaks out via a
        # non-empty left; an empty one here is an infinite loop there.
        for text in ["word", "a b c d e f", "Sentence one. Sentence two.",
                     "para one\n\npara two\n\npara three"]:
            left, _ = split_paragraph_boundary(text, 2)
            assert left, f"empty left for {text!r}"

    def test_no_words_are_lost(self):
        text = "alpha beta gamma\n\ndelta epsilon zeta\n\neta theta"
        left, right = split_paragraph_boundary(text, 4)
        assert (left + " " + right).split() == text.split()


class TestBuildStimulus:
    def test_question_and_answer_are_both_present(self):
        s = build_stimulus("what is X?", "X is a thing")
        assert "what is X?" in s
        assert "X is a thing" in s
        assert s.count("---") == 1

    def test_reading_is_a_third_section_only_when_given(self):
        assert build_stimulus("q", "a", "src").count("---") == 2
        assert build_stimulus("q", "a", "").count("---") == 1

    def test_sections_are_truncated_to_512_512_256(self):
        q, a, r = "q " * 600, "a " * 600, "r " * 600
        parts = build_stimulus(q, a, r).split("\n---\n")
        assert [len(p.split()) for p in parts] == [512, 512, 256]


class TestRelevancePrompt:
    def test_bounds_both_sides(self):
        # Distinctive markers: the fixed instruction text around the two slots
        # contains ordinary words, so counting "q"/"n" would count those too.
        p = relevance_prompt("QQQ " * 500, "NNN " * 2000)
        # Judge server window is 8k and several candidates run concurrently.
        assert p.count("QQQ") == 300
        assert p.count("NNN") == 900

    def test_states_the_three_refusal_cases(self):
        p = relevance_prompt("q", "n")
        # These three are exactly what cosine cannot see; if the wording drifts
        # away from them the trap family stops being refused.
        assert "different phase" in p
        assert "shares vocabulary" in p
        assert "different issue" in p


class TestMatchesCorrection:
    def test_plain_match(self):
        assert matches_correction("that is wrong", [r"\bis wrong\b"])

    def test_case_insensitive(self):
        assert matches_correction("THAT IS WRONG", [r"is wrong"])

    def test_no_match(self):
        assert not matches_correction("looks good", [r"is wrong"])

    def test_an_invalid_regex_degrades_to_substring_rather_than_raising(self):
        # correction_patterns is user-editable YAML. A typo there must not take
        # the turn down.
        assert matches_correction("a [b c", ["[b"])
        assert not matches_correction("nothing here", ["[b"])
