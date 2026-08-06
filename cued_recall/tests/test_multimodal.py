"""Images and voice through the middleware.

The chat can attach pictures (sent as OpenAI-style image_url content parts)
and voice recordings (transcribed by the STT backend before they reach the
message). The pipeline must pass image parts through untouched, charge their
~658 tokens each against the context budget, and never try to tokenize an
image as text.
"""

import pytest

from cued_recall.pipeline import IMAGE_TOKENS_PER_IMAGE


def image_part(url="data:image/png;base64,AAAA"):
    return {"type": "image_url", "image_url": {"url": url}}


class TestImageCounting:
    def test_plain_text_has_no_images(self, pipeline):
        m = {"role": "user", "content": "hello"}
        assert pipeline._msg_image_count(m) == 0

    def test_parts_list_counts_image_parts(self, pipeline):
        m = {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            image_part(), image_part()]}
        assert pipeline._msg_image_count(m) == 2

    def test_non_part_content_is_not_counted(self, pipeline):
        m = {"role": "user", "content": "image_url is just text here"}
        assert pipeline._msg_image_count(m) == 0

    def test_token_estimate_charges_images_a_flat_rate(self, pipeline):
        m = {"role": "user", "content": [{"type": "text", "text": "hi"}, image_part()]}
        text_only = pipeline._estimate_tokens("hi")
        assert pipeline._msg_token_estimate(m) == text_only + IMAGE_TOKENS_PER_IMAGE

    def test_image_parts_survive_recall_injection(self, pipeline):
        # _prepend_text splices recalled blocks in front of the first text
        # part; the image part must come through untouched or the model loses
        # the picture it was just asked about.
        m = {"role": "user",
             "content": [{"type": "text", "text": "what is this?"}, image_part()]}
        out = pipeline._prepend_text(m, "recalled> ")
        assert [p["type"] for p in out["content"]] == ["text", "image_url"]
        assert out["content"][0]["text"].startswith("recalled>")
        assert out["content"][1]["image_url"]["url"].startswith("data:image")


@pytest.mark.asyncio
async def test_exact_tokenize_is_skipped_when_images_are_present(pipeline, monkeypatch):
    # /tokenize counts text; an image's ~658 tokens come from the vision
    # encoder and are invisible to it. Sending the text blob anyway would
    # measure low and let an over-budget prompt through to a hard 400.
    called = False

    async def boom(*a, **k):
        nonlocal called
        called = True
        raise AssertionError("must not call the tokenizer")

    monkeypatch.setattr("httpx.AsyncClient", boom)
    msgs = [{"role": "user",
             "content": [{"type": "text", "text": "hi"}, image_part()]}]
    assert await pipeline._exact_prompt_tokens(msgs, {}) is None
    assert not called


@pytest.mark.asyncio
async def test_fit_messages_counts_images_against_the_budget(pipeline):
    # Each image costs a flat ~700 tokens; the drop loop must see that charge
    # or a picture-heavy conversation overflows the window it was sized for.
    def msg(txt):
        return {"role": "user", "content": [{"type": "text", "text": txt}]}

    base = [msg(f"block {i} " + "word " * 200) for i in range(8)]
    limit = 1600

    plain = await pipeline._fit_messages(list(base), {}, limit=limit, use_exact=False)
    with_img = await pipeline._fit_messages(
        list(base) + [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"}, image_part()]}],
        {}, limit=limit, use_exact=False)

    # The picture must cost history it would otherwise keep.
    assert len(with_img) < len(plain)
