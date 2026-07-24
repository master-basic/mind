import asyncio
import json
import re
import time
import uuid
from typing import AsyncIterator, List, Optional, Tuple

import httpx
import numpy as np

from .config import Config
from .embed import EmbeddingClient
from .index import VectorIndex
from .models import Block, BlockType, BlockStatus, Verification
from .store import BlockStore
from .utils import (
    build_stimulus,
    matches_correction,
    split_paragraph_boundary,
    truncate_tokens,
)
from .wal import WAL


class Pipeline:
    def __init__(
        self,
        config: Config,
        store: BlockStore,
        index: VectorIndex,
        embed: EmbeddingClient,
        wal: WAL,
    ):
        self.config = config
        self.store = store
        self.index = index
        self.embed = embed
        self.wal = wal
        self.think_open = config.think_tags[0]
        self.think_close = config.think_tags[1]
        self.token_sink = None

    async def recall_blocks(self, user_message: str) -> List[Tuple[Block, float]]:
        # Only embed a bounded slice for the recall query: a pasted file can be
        # far larger than the embed model's context.
        query_text = truncate_tokens(user_message, 512)
        # Recall is best-effort: if the embed server errors (e.g. input still too
        # large), skip recall for this turn rather than 500 the whole chat.
        try:
            embed_vec = await asyncio.to_thread(self.embed.embed, query_text)
        except Exception as e:
            self.wal.write({
                "event": "recall_embed_error",
                "error": str(e),
                "timestamp": time.time(),
            })
            return []
        results = await asyncio.to_thread(
            self.index.query,
            embed_vec,
            self.config.recall.k,
            self.config.recall.threshold,
        )
        blocks = []
        total_tokens = 0
        for block_id, sim in results:
            meta = await asyncio.to_thread(self.index.get_meta, block_id)
            if meta and meta.get("verification") == "corrected":
                continue
            block = await asyncio.to_thread(self.store.get, block_id)
            if block is None:
                continue
            total_tokens += block.token_count
            if total_tokens > self.config.recall.budget_tokens:
                break
            blocks.append((block, sim))
        return blocks

    def build_recall_injection(self, blocks: List[Tuple[Block, float]]) -> str:
        if not blocks:
            return ""
        parts = [
            "Prior derivations from earlier sessions. These are advisory. Verify before",
            "reuse; they may contain outdated assumptions.",
        ]
        for block, sim in blocks:
            date_str = time.strftime(
                "%Y-%m-%d", time.localtime(block.created_at)
            )
            parts.append(
                f"[recall {block.block_id[:8]}, similarity {sim:.2f}, {date_str}]"
            )
            parts.append(block.text)
            parts.append("")
        return "\n".join(parts)

    def build_messages(
        self,
        original_messages: list,
        recall_text: str,
    ) -> list:
        if not recall_text:
            return original_messages
        recall_msg = {"role": "system", "content": recall_text}
        return [recall_msg] + original_messages

    def read_messages_from_body(self, body: dict) -> list:
        return body.get("messages", [])

    def get_last_user_message(self, body: dict) -> str:
        messages = self.read_messages_from_body(body)
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        p.get("text", "")
                        for p in content
                        if p.get("type") == "text"
                    ]
                    content = " ".join(text_parts)
                return content
        return ""

    def get_reading_content(self, body: dict) -> str:
        messages = self.read_messages_from_body(body)
        reading_parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if (
                        part.get("type") == "text"
                        and len(part.get("text", "").split()) > 200
                    ):
                        reading_parts.append(part["text"])
            elif isinstance(content, str) and len(content.split()) > 200:
                reading_parts.append(content)
        return "\n\n".join(reading_parts)

    async def forward_stream(
        self, client: httpx.AsyncClient, messages: list, body: dict
    ) -> AsyncIterator[bytes]:
        payload = {**body, "messages": messages, "stream": True}
        async with client.stream(
            "POST",
            f"{self.config.reasoning_endpoint}/v1/chat/completions",
            json=payload,
            timeout=300,
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    yield line.encode() + b"\n\n"

    async def forward_nonstream(
        self, client: httpx.AsyncClient, messages: list, body: dict
    ) -> dict:
        payload = {**body, "messages": messages, "stream": False}
        resp = await client.post(
            f"{self.config.reasoning_endpoint}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    class ThinkSplitter:
        def __init__(self, open_tag: str, close_tag: str):
            self.open_tag = open_tag
            self.close_tag = close_tag
            self.hold = len(open_tag) - 1
            self.buffer = ""
            self.in_think = False
            self.reasoning_parts: List[str] = []
            self.result_parts: List[str] = []

        def feed(self, chunk: str) -> List[tuple[str, str]]:
            self.buffer += chunk
            outputs = []
            while True:
                if not self.in_think:
                    idx = self.buffer.find(self.open_tag)
                    if idx == -1:
                        if len(self.buffer) <= self.hold:
                            break
                        safe = self.buffer[:len(self.buffer) - self.hold]
                        if safe:
                            self.result_parts.append(safe)
                            outputs.append(("result", safe))
                        self.buffer = self.buffer[len(safe):]
                        break
                    before = self.buffer[:idx]
                    if before:
                        self.result_parts.append(before)
                        outputs.append(("result", before))
                    self.buffer = self.buffer[idx + len(self.open_tag):]
                    self.in_think = True
                else:
                    idx = self.buffer.find(self.close_tag)
                    if idx == -1:
                        break
                    content = self.buffer[:idx]
                    if content:
                        self.reasoning_parts.append(content)
                        outputs.append(("think", content))
                    self.buffer = self.buffer[idx + len(self.close_tag):]
                    self.in_think = False
            return outputs

        def flush(self) -> List[tuple[str, str]]:
            outputs = []
            if self.buffer:
                tag = "think" if self.in_think else "result"
                outputs.append((tag, self.buffer))
                if tag == "think":
                    self.reasoning_parts.append(self.buffer)
                else:
                    self.result_parts.append(self.buffer)
                self.buffer = ""
            return outputs

    async def process_turn(
        self,
        body: dict,
        conversation_id: str,
        turn_index: int,
    ) -> dict:
        user_message = self.get_last_user_message(body)
        reading_content = self.get_reading_content(body)
        base_messages = self.read_messages_from_body(body)

        recall_blocks = await self.recall_blocks(user_message)
        recall_text = self.build_recall_injection(recall_blocks)
        augmented_messages = self.build_messages(base_messages, recall_text)

        if body.get("stream", False):
            return await self._process_streaming(
                body, augmented_messages, user_message, reading_content,
                recall_blocks, conversation_id, turn_index,
            )
        else:
            return await self._process_nonstreaming(
                body, augmented_messages, user_message, reading_content,
                recall_blocks, conversation_id, turn_index,
            )

    async def _process_streaming(
        self,
        body: dict,
        augmented_messages: list,
        user_message: str,
        reading_content: str,
        recall_blocks: List[Tuple[Block, float]],
        conversation_id: str,
        turn_index: int,
    ) -> dict:
        return {
            "type": "streaming",
            "stream": self._stream_and_blockify(
                body, augmented_messages, user_message, reading_content,
                recall_blocks, conversation_id, turn_index,
            ),
        }

    async def _stream_and_blockify(
        self,
        body: dict,
        augmented_messages: list,
        user_message: str,
        reading_content: str,
        recall_blocks: List[Tuple[Block, float]],
        conversation_id: str,
        turn_index: int,
    ):
        splitter = self.ThinkSplitter(self.think_open, self.think_close)
        response_text = ""
        # Newer llama.cpp parses <think> out of `content` into a separate
        # `reasoning_content` delta field. Capture both so reasoning blocks are
        # still created when the tags aren't inline.
        reasoning_content_parts: List[str] = []

        async with httpx.AsyncClient() as client:
            payload = {**body, "messages": augmented_messages, "stream": True}
            async with client.stream(
                "POST",
                f"{self.config.reasoning_endpoint}/v1/chat/completions",
                json=payload,
                timeout=300,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    yield line.encode() + b"\n\n"
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta_obj = data.get("choices", [{}])[0].get("delta", {})
                    rc = delta_obj.get("reasoning_content")
                    if rc:
                        reasoning_content_parts.append(rc)
                    delta = delta_obj.get("content", "")
                    if not delta:
                        continue
                    response_text += delta
                    splitter.feed(delta)

                splitter.flush()

        full_reasoning = (
            "".join(reasoning_content_parts) + "".join(splitter.reasoning_parts)
        )
        full_result = "".join(splitter.result_parts)

        await self._create_blocks(
            full_reasoning, full_result, user_message, reading_content,
            recall_blocks, conversation_id, turn_index, response_text,
        )

        if self.token_sink:
            self.token_sink(
                len(full_reasoning.split()) + len(full_result.split())
            )

        self.wal.write({
            "event": "turn_completed",
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "recall_count": len(recall_blocks),
            "reasoning_tokens": len(full_reasoning.split()),
            "result_tokens": len(full_result.split()),
            "timestamp": time.time(),
        })

    async def _process_nonstreaming(
        self,
        body: dict,
        augmented_messages: list,
        user_message: str,
        reading_content: str,
        recall_blocks: List[Tuple[Block, float]],
        conversation_id: str,
        turn_index: int,
    ) -> dict:
        async with httpx.AsyncClient() as client:
            payload = {**body, "messages": augmented_messages, "stream": False}
            resp = await client.post(
                f"{self.config.reasoning_endpoint}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()

        message = result.get("choices", [{}])[0].get("message", {})
        response_text = message.get("content", "") or ""
        reasoning_content = message.get("reasoning_content", "") or ""

        splitter = self.ThinkSplitter(self.think_open, self.think_close)
        splitter.feed(response_text)
        splitter.flush()

        full_reasoning = reasoning_content + "".join(splitter.reasoning_parts)
        full_result = "".join(splitter.result_parts)

        await self._create_blocks(
            full_reasoning, full_result, user_message, reading_content,
            recall_blocks, conversation_id, turn_index, response_text,
        )

        if self.token_sink:
            self.token_sink(
                len(full_reasoning.split()) + len(full_result.split())
            )

        self.wal.write({
            "event": "turn_completed",
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "recall_count": len(recall_blocks),
            "reasoning_tokens": len(full_reasoning.split()),
            "result_tokens": len(full_result.split()),
            "timestamp": time.time(),
        })

        return result

    async def _create_blocks(
        self,
        full_reasoning: str,
        full_result: str,
        user_message: str,
        reading_content: str,
        recall_blocks: List[Tuple[Block, float]],
        conversation_id: str,
        turn_index: int,
        response_text: str,
    ):
        now = time.time()

        reasoning_blocks = []
        if full_reasoning:
            reasoning_blocks = self._split_reasoning(
                full_reasoning, conversation_id, turn_index, now
            )

        result_block = Block(
            type=BlockType.result,
            conversation_id=conversation_id,
            turn_index=turn_index,
            token_count=len(full_result.split()),
            text=full_result,
            stimulus_text=truncate_tokens(full_result, 1024),
            verification=Verification.unknown,
            created_at=now,
        )

        all_blocks = reasoning_blocks + [result_block]

        if reading_content and len(reading_content.split()) > 1000:
            reading_block = Block(
                type=BlockType.reading,
                conversation_id=conversation_id,
                turn_index=turn_index,
                token_count=len(reading_content.split()),
                text=reading_content,
                stimulus_text=truncate_tokens(reading_content, 1024),
                verification=Verification.unknown,
                created_at=now,
            )
            all_blocks.append(reading_block)

        for block in reasoning_blocks:
            stimulus = build_stimulus(user_message, full_result,
                                      reading_content if len(reading_content.split()) > 1000 else "")
            block.stimulus_text = stimulus

        for block in all_blocks:
            await asyncio.to_thread(self.store.put, block)
            await asyncio.to_thread(
                self.index.upsert_block_meta,
                block.block_id, block.type.value, block.status.value,
                block.created_at, block.conversation_id, block.turn_index,
                block.token_count, block.verification.value,
                block.recall_count, block.last_recalled,
            )

        embed_tasks = []
        for block in all_blocks:
            if block.stimulus_text:
                embed_tasks.append(self._embed_and_store(block))
        if embed_tasks:
            await asyncio.gather(*embed_tasks)

        for block, sim in recall_blocks:
            await asyncio.to_thread(
                self.index.increment_recall, block.block_id, now
            )

    def _split_reasoning(
        self, text: str, conversation_id: str, turn_index: int, now: float
    ) -> List[Block]:
        max_tokens = self.config.block_tokens_reasoning
        blocks = []
        remaining = text

        while remaining:
            left, remaining = split_paragraph_boundary(remaining, max_tokens)
            if not left:
                left = remaining
                remaining = ""
            block = Block(
                type=BlockType.reasoning,
                status=BlockStatus.hot,
                conversation_id=conversation_id,
                turn_index=turn_index,
                token_count=len(left.split()),
                text=left,
                verification=Verification.unknown,
                created_at=now,
            )
            blocks.append(block)

        return blocks

    async def _embed_and_store(self, block: Block):
        # Best-effort: a failed embed must not crash block creation. The block
        # is still stored; it just won't be vector-recallable until re-embedded.
        try:
            vec = await asyncio.to_thread(self.embed.embed, block.stimulus_text)
            await asyncio.to_thread(self.index.upsert_vector, block.block_id, vec)
        except Exception as e:
            self.wal.write({
                "event": "embed_store_error",
                "block_id": block.block_id,
                "error": str(e),
                "timestamp": time.time(),
            })

    async def detect_and_apply_correction(
        self, user_message: str, conversation_id: str, turn_index: int
    ):
        if not matches_correction(user_message, self.config.correction_patterns):
            return
        prev_turn = turn_index - 1
        if prev_turn < 0:
            return
        block_ids = await asyncio.to_thread(
            self._find_turn_blocks, conversation_id, prev_turn
        )
        for bid in block_ids:
            await asyncio.to_thread(
                self.index.update_verification, bid, "corrected"
            )
            block = await asyncio.to_thread(self.store.get, bid)
            if block:
                block.verification = Verification.corrected
                await asyncio.to_thread(self.store.put, block)
            self.wal.write({
                "event": "verification_set",
                "block_id": bid,
                "verification": "corrected",
                "timestamp": time.time(),
            })

    async def apply_accepted_verification(
        self, conversation_id: str, turn_index: int
    ):
        prev_turn = turn_index - 1
        if prev_turn < 0:
            return
        block_ids = await asyncio.to_thread(
            self._find_turn_blocks, conversation_id, prev_turn
        )
        for bid in block_ids:
            meta = await asyncio.to_thread(self.index.get_meta, bid)
            if meta and meta.get("verification") == "unknown":
                await asyncio.to_thread(
                    self.index.update_verification, bid, "accepted"
                )
                block = await asyncio.to_thread(self.store.get, bid)
                if block:
                    block.verification = Verification.accepted
                    await asyncio.to_thread(self.store.put, block)

    def _find_turn_blocks(self, conversation_id: str, turn_index: int) -> List[str]:
        all_meta = self.index.list_meta(limit=10000)[0]
        return [
            m["block_id"]
            for m in all_meta
            if m["conversation_id"] == conversation_id
            and m["turn_index"] == turn_index
        ]

    async def shelve_previous_turn(self, conversation_id: str, turn_index: int):
        prev_turn = turn_index - 1
        if prev_turn < 0:
            return
        block_ids = await asyncio.to_thread(
            self._find_turn_blocks, conversation_id, prev_turn
        )
        for bid in block_ids:
            await asyncio.to_thread(self.index.update_status, bid, "shelved")
            block = await asyncio.to_thread(self.store.get, bid)
            if block:
                block.status = BlockStatus.shelved
                await asyncio.to_thread(self.store.put, block)
