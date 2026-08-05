"""EmbeddingClient.fit -- the cap that decides whether a block gets a vector.

An input past the embedding server's window comes back HTTP 400, and
_embed_and_store catches it and moves on, so the block is stored, shows a
healthy status, and can never be retrieved. The guard that was supposed to
prevent that was a 16,000-character cap written for an 8,192-token window,
while this stack runs nomic-embed at 2,048 -- so it never fired, and callers
capping at 1,024 whitespace words sent up to 2,338 tokens.
"""

import pytest

from cued_recall.embed import EmbeddingClient
from cued_recall.utils import estimate_tokens


def client(ctx=2048):
    return EmbeddingClient("http://127.0.0.1:8082", ctx_tokens=ctx)


def fits(text, ctx=2048):
    budget = int(ctx * EmbeddingClient.CTX_MARGIN)
    return estimate_tokens(text) <= budget


class TestFit:
    def test_short_text_is_untouched(self):
        c = client()
        assert c.fit("a short note") == "a short note"

    def test_empty(self):
        assert client().fit("") == ""

    def test_prose_over_the_window_is_cut_to_fit(self):
        c = client()
        text = "the quick brown fox jumps over the lazy dog. " * 400
        out = c.fit(text)
        assert len(out) < len(text)
        assert fits(out)

    def test_code_over_the_window_is_cut_to_fit(self):
        # The case that broke: symbol-dense text costs far more tokens per
        # word, so a word-count cap lets it through and a token budget does not.
        c = client()
        text = 'const x = {"a": 1, "b": [2,3]};\n' * 400
        out = c.fit(text)
        assert fits(out)

    def test_azerbaijani_over_the_window_is_cut_to_fit(self):
        c = client()
        text = "Səhv işləmir, düz deyil, yanlış nəticə alınmır. " * 400
        assert fits(c.fit(text))

    def test_a_thousand_words_of_code_no_longer_goes_out_whole(self):
        # 1,024 words was the shipped truncate_tokens cap for stimulus_text on
        # result and reading blocks, and for embed_text on every block.
        c = client()
        text = " ".join(['x={"k":[1,2,3]};'] * 1024)
        assert len(text.split()) == 1024
        assert not fits(text), "fixture no longer exceeds the window"
        assert fits(c.fit(text))

    def test_a_smaller_window_cuts_harder(self):
        text = "the quick brown fox jumps over the lazy dog. " * 400
        small = client(ctx=512).fit(text)
        large = client(ctx=2048).fit(text)
        assert len(small) < len(large)
        assert fits(small, ctx=512)

    def test_the_margin_leaves_headroom(self):
        # The estimator is conservative but not exact, and the server counts a
        # little scaffolding of its own.
        assert EmbeddingClient.CTX_MARGIN < 1.0

    def test_a_tiny_window_still_returns_something(self):
        # Never an empty string: an empty input embeds to a zero vector, which
        # matches nothing and looks like a healthy block.
        out = client(ctx=16).fit("word " * 5000)
        assert out


class TestDetect:
    def test_a_server_that_will_not_answer_keeps_the_configured_value(self):
        # Port 1 is closed. Falling back beats raising at startup.
        c = EmbeddingClient("http://127.0.0.1:1", ctx_tokens=1234)
        assert c.detect_ctx_tokens() == 1234
        assert c.ctx_tokens == 1234
