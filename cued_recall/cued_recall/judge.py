import asyncio
import json
import re
import time
from typing import Optional

import httpx

from .config import Config
from .index import VectorIndex
from .models import Block, BlockStatus
from .store import BlockStore
from .wal import WAL


class Judge:
    def __init__(
        self,
        config: Config,
        store: BlockStore,
        index: VectorIndex,
        wal: WAL,
    ):
        self.config = config
        self.store = store
        self.index = index
        self.wal = wal
        self.judge_url = config.judge_endpoint.rstrip("/") + "/v1/chat/completions

    async def run_pass(self):
        min_age = self.config.judge.min_age_s
        block_ids = await asyncio.to_thread(
            self.index.oldest_shelved_blocks, min_age, limit=50
        )
        for bid in block_ids:
            block = await asyncio.to_thread(self.store.get, bid)
            if block is None:
                continue
            action, summary = await self._judge_block(block)
            await self._apply_ladder(block, action, summary)
            self.wal.write({
                "event": "judge_action",
                "block_id": bid,
                "action": action,
                "summary_length": len(summary) if summary else 0,
                "timestamp": time.time(),
            })

    async def _judge_block(self, block: Block) -> tuple[str, Optional[str]]:
        meta = await asyncio.to_thread(self.index.get_meta, block.block_id)
        verification = "unknown"
        recall_count = 0
        if meta:
            verification = meta.get("verification", "unknown")
            recall_count = meta.get("recall_count", 0)

        prompt = (
            "You maintain a reasoning archive. Given a derivation, the problem it solved, "
            "whether the answer was accepted, and how often it was recalled, respond with "
            'exactly one JSON object: {"action": "keep" | "truncate" | "purge_candidate", '
            '"summary": "<only when action is truncate: a summary under 400 tokens that '
            'preserves the method, the key facts discovered, and the final conclusion. '
            'Drop dead ends and repetition.>"}\n\n'
            f"Problem/Stimulus:\n{block.stimulus_text}\n\n"
            f"Derivation:\n{block.text}\n\n"
            f"Verification: {verification}\n"
            f"Recall count: {recall_count}"
        )

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    self.judge_url,
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 600,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
        except Exception as e:
            self.wal.write({
                "event": "judge_error",
                "block_id": block.block_id,
                "error": str(e),
                "timestamp": time.time(),
            })
            return "keep", None

        return self._parse_judge_output(content)

    def _parse_judge_output(self, content: str) -> tuple[str, Optional[str]]:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            return "keep", None
        try:
            obj = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return "keep", None

        action = obj.get("action", "keep")
        if action not in ("keep", "truncate", "purge_candidate"):
            action = "keep"
        summary = obj.get("summary") if action == "truncate" else None
        return action, summary

    async def _apply_ladder(self, block: Block, action: str, summary: Optional[str]):
        meta = await asyncio.to_thread(self.index.get_meta, block.block_id)
        verification = meta.get("verification", "unknown") if meta else "unknown"
        recall_count = meta.get("recall_count", 0) if meta else 0
        age = time.time() - block.created_at

        if action == "keep":
            return

        if action == "truncate" and summary:
            block.text = summary
            block.original_len = block.token_count
            block.token_count = len(summary.split())
            block.status = BlockStatus.truncated
            await asyncio.to_thread(self.store.put, block)
            await asyncio.to_thread(
                self.index.update_status, block.block_id, "truncated"
            )
            return

        if action == "purge_candidate":
            can_purge = (
                verification == "corrected"
                or (recall_count == 0 and age > self.config.judge.purge_age_s)
            )
            if can_purge:
                await self._purge_block(block)
            else:
                if summary:
                    block.text = summary
                    block.original_len = block.token_count
                    block.token_count = len(summary.split())
                    block.status = BlockStatus.truncated
                    await asyncio.to_thread(self.store.put, block)
                    await asyncio.to_thread(
                        self.index.update_status, block.block_id, "truncated"
                    )

    async def _purge_block(self, block: Block):
        block.status = BlockStatus.purged
        await asyncio.to_thread(self.store.put, block)
        await asyncio.to_thread(self.store.delete_file, block.block_id)
        await asyncio.to_thread(
            self.index.update_status, block.block_id, "purged"
        )
