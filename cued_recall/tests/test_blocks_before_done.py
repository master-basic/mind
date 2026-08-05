import asyncio
import json

import pytest

from cued_recall import pipeline as pipeline_module


class _FakeResponse:
    status_code = 200
    text = ""

    async def aclose(self):
        pass

    async def aread(self):
        return b""

    async def aiter_lines(self):
        chunks = [
            {"id": "chatcmpl-test", "object": "chat.completion.chunk",
             "created": 0, "model": "m",
             "choices": [{"index": 0, "delta": {"content": "hello there"},
                          "finish_reason": None}]},
            {"id": "chatcmpl-test", "object": "chat.completion.chunk",
             "created": 0, "model": "m",
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        for c in chunks:
            yield "data: " + json.dumps(c)


class _FakeClient:
    """Replaces httpx.AsyncClient: any request returns a canned SSE stream."""

    def __init__(self, response=None):
        self.response = response or _FakeResponse()

    def build_request(self, method, url, json=None, timeout=None):
        return {"method": method, "url": url, "json": json, "timeout": timeout}

    async def send(self, request, stream=False):
        return self.response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def offline_pipeline(pipeline, monkeypatch):
    """A pipeline whose model/tokenize calls are all local and canned."""

    async def fake_count(text):
        return len(text.split())

    async def fake_fit(messages, body=None, **kw):
        return messages

    monkeypatch.setattr(pipeline, "_count_tokens", fake_count)
    monkeypatch.setattr(pipeline, "_fit_messages", fake_fit)
    monkeypatch.setattr(pipeline_module.httpx, "AsyncClient", _FakeClient)
    return pipeline


class TestBlocksBeforeDone:
    @pytest.mark.asyncio
    async def test_disconnect_after_done_keeps_blocks_and_defers_embed(
            self, offline_pipeline):
        # The F8 acceptance test: simulate a client that reads up to [DONE]
        # and disconnects on it, the way Starlette closes the generator when
        # the stream is dropped. Before this fix the persistence ran after
        # [DONE], so the aclose() below meant the turn was never memorized.
        pipeline = offline_pipeline
        body = {
            "stream": True,
            "model": "test",
            "messages": [{"role": "user", "content": "hello there"}],
        }
        result = await pipeline.process_turn(body, "c1", 0)
        gen = result["stream"]

        saw_done = False
        async for chunk in gen:
            if chunk == b"data: [DONE]\n\n":
                saw_done = True
                break
        assert saw_done

        # Disconnect here: the generator is suspended at the [DONE] yield, so
        # everything after it (which used to hold _create_blocks) is skipped.
        await gen.aclose()

        block_ids = pipeline.index.block_ids_for_turn("c1", 0)
        assert block_ids
        assert all(pipeline.store.get(b) is not None for b in block_ids)

        # The post-[DONE] tail really was skipped: no turn_completed event.
        assert not [e for e in pipeline.wal.iter_all()
                    if e["event"] == "turn_completed"]

        # The embed was deferred to a background task scheduled BEFORE [DONE].
        # The fixture has embed=None, so it fails loudly rather than silently
        # -- exactly the surfaced, repairable condition that
        # blocks_missing_vectors reports.
        await asyncio.sleep(0.05)
        errors = [e for e in pipeline.wal.iter_all()
                  if e["event"] == "embed_store_error"]
        assert errors


class TestDeferredEmbeds:
    @pytest.mark.asyncio
    async def test_defer_embeds_persists_blocks_without_embedding(
            self, pipeline, monkeypatch):
        async def fake_count(text):
            return len(text.split())

        monkeypatch.setattr(pipeline, "_count_tokens", fake_count)
        embeddable = await pipeline._create_blocks(
            full_reasoning="some thinking", full_result="some answer",
            user_message="a question", reading_content="", recall_blocks=[],
            conversation_id="c1", turn_index=0, response_text="",
            defer_embeds=True,
        )
        assert embeddable
        assert pipeline.index.block_ids_for_turn("c1", 0)
        # embed is None in the fixture: an inline embed would have written an
        # error event. The deferred path must not touch the embedder.
        errors = [e for e in pipeline.wal.iter_all()
                  if e["event"] == "embed_store_error"]
        assert errors == []

    @pytest.mark.asyncio
    async def test_embed_blocks_runs_the_deferred_embeds(
            self, pipeline, monkeypatch):
        async def fake_count(text):
            return len(text.split())

        monkeypatch.setattr(pipeline, "_count_tokens", fake_count)
        embeddable = await pipeline._create_blocks(
            full_reasoning="some thinking", full_result="some answer",
            user_message="a question", reading_content="", recall_blocks=[],
            conversation_id="c1", turn_index=0, response_text="",
            defer_embeds=True,
        )
        await pipeline._embed_blocks(embeddable)
        # The embedder is None, so the deferred task must record the failure
        # rather than raise -- the same contract as the inline path.
        errors = [e for e in pipeline.wal.iter_all()
                  if e["event"] == "embed_store_error"]
        assert errors

    @pytest.mark.asyncio
    async def test_default_still_embeds_inline(self, pipeline, monkeypatch):
        async def fake_count(text):
            return len(text.split())

        monkeypatch.setattr(pipeline, "_count_tokens", fake_count)
        await pipeline._create_blocks(
            full_reasoning="some thinking", full_result="some answer",
            user_message="a question", reading_content="", recall_blocks=[],
            conversation_id="c1", turn_index=0, response_text="",
        )
        assert pipeline.index.block_ids_for_turn("c1", 0)
