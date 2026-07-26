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
        self.judge_url = config.judge_endpoint.rstrip("/") + "/v1/chat/completions"
        self.usage_sink = None
        self._n_ctx: Optional[int] = None

    # The summary alone is specified at up to 400 tokens; 600 for the whole
    # response left no room for the JSON scaffolding around it, so a verbose
    # summary ran past the cap and the closing brace never arrived. Headroom
    # without going so high that every call pays for tokens it won't use --
    # _salvage() recovers the rest if a summary still runs long.
    MAX_TOKENS = 768
    # The judge runs on CPU by design (-ngl 0, to leave VRAM for the reasoning
    # model's KV). Generating a few hundred tokens there is slow enough that
    # the old 120 s tripped on the larger blocks.
    TIMEOUT_S = 300

    async def _judge_n_ctx(self) -> int:
        """The judge server's context window, probed once and cached."""
        if self._n_ctx is not None:
            return self._n_ctx
        self._n_ctx = 8192
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    self.config.judge_endpoint.rstrip("/") + "/props"
                )
                if r.status_code == 200:
                    d = r.json()
                    gs = d.get("default_generation_settings", {}) or {}
                    n = gs.get("n_ctx") or d.get("n_ctx")
                    if n:
                        self._n_ctx = int(n)
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        return self._n_ctx

    @staticmethod
    def _head_tail(text: str, budget_chars: int) -> str:
        """Trim to budget, keeping both ends.

        Head-only truncation would hand the judge the opening of a document
        and ask it to summarise the whole thing. Keeping both ends at least
        shows it where the content started and where it ended up.
        """
        if len(text) <= budget_chars:
            return text
        if budget_chars <= 0:
            return ""
        half = budget_chars // 2
        omitted = len(text) - 2 * half
        return (f"{text[:half]}\n\n[... {omitted:,} characters omitted "
                f"from the middle ...]\n\n{text[-half:]}")

    async def run_pass(self, min_age: Optional[float] = None) -> int:
        # Manual runs pass min_age=0 to judge all shelved blocks now; the
        # automatic (token-triggered) pass uses the configured age gate.
        if min_age is None:
            min_age = self.config.judge.min_age_s
        block_ids = await asyncio.to_thread(
            self.index.oldest_shelved_blocks, min_age, limit=50
        )
        processed = 0
        for bid in block_ids:
            block = await asyncio.to_thread(self.store.get, bid)
            if block is None:
                continue
            action, summary = await self._judge_block(block)
            await self._apply_ladder(block, action, summary)
            processed += 1
            self.wal.write({
                "event": "judge_action",
                "block_id": bid,
                "action": action,
                "summary_length": len(summary) if summary else 0,
                "timestamp": time.time(),
            })
        return processed

    async def _judge_block(self, block: Block) -> tuple[str, Optional[str]]:
        meta = await asyncio.to_thread(self.index.get_meta, block.block_id)
        verification = "unknown"
        recall_count = 0
        if meta:
            verification = meta.get("verification", "unknown")
            recall_count = meta.get("recall_count", 0)

        # The block has to fit the judge's window, and the biggest blocks are
        # exactly the ones worth compressing. Sending them whole made the
        # request 400 (a 64 KB block is ~19,700 tokens against a window of
        # 8,192), which _judge_block caught and turned into "keep" -- so the
        # blocks most in need of truncation were the only ones that could
        # never be truncated, and every pass burned a doomed request on them.
        n_ctx = await self._judge_n_ctx()
        # Reserve the reply plus the instruction preamble, then convert the
        # remainder to characters conservatively.
        avail_tokens = max(512, n_ctx - self.MAX_TOKENS - 400)
        avail_chars = int(avail_tokens * 3.0)
        stim_chars = avail_chars // 4
        stimulus = self._head_tail(block.stimulus_text or "", stim_chars)
        text = self._head_tail(block.text or "", avail_chars - len(stimulus))
        truncated_note = (
            "\n\nNOTE: the derivation above was abridged to fit; summarise what "
            "is shown and say so if it is clearly partial."
            if len(text) < len(block.text or "") else ""
        )

        prompt = (
            "You maintain a reasoning archive. Given a derivation, the problem it solved, "
            "whether the answer was accepted, and how often it was recalled, respond with "
            'exactly one JSON object: {"action": "keep" | "truncate" | "purge_candidate", '
            '"summary": "<only when action is truncate: a summary under 400 tokens that '
            'preserves the method, the key facts discovered, and the final conclusion. '
            'Drop dead ends and repetition.>"}\n\n'
            f"Problem/Stimulus:\n{stimulus}\n\n"
            f"Derivation:\n{text}\n\n"
            f"Verification: {verification}\n"
            f"Recall count: {recall_count}"
            f"{truncated_note}"
        )

        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT_S) as client:
                for attempt in range(2):
                    resp = await client.post(
                        self.judge_url,
                        json={
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.1,
                            "max_tokens": self.MAX_TOKENS,
                        },
                    )
                    if resp.status_code == 200:
                        break
                    # The window can still be exceeded if the estimate was
                    # optimistic. The server reports the real numbers; halve
                    # the block text against them and try once more.
                    if attempt == 0 and self._is_overflow(resp.text):
                        text = self._head_tail(text, len(text) // 2)
                        prompt = prompt.replace(block.text or "", text, 1) \
                            if (block.text or "") in prompt else prompt
                        self.wal.write({
                            "event": "judge_overflow_retry",
                            "block_id": block.block_id,
                            "retry_chars": len(text),
                            "timestamp": time.time(),
                        })
                        continue
                    resp.raise_for_status()
                resp.raise_for_status()
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if self.usage_sink:
                    usage = data.get("usage")
                    if usage:
                        self.usage_sink(usage)
        except Exception as e:
            self.wal.write({
                "event": "judge_error",
                "block_id": block.block_id,
                # str() on an httpx timeout is the empty string, which made a
                # third of the tagger's failures unreadable in the log. Always
                # carry the exception type.
                "error": f"{type(e).__name__}: {e}",
                "timestamp": time.time(),
            })
            return "keep", None

        return self._parse_judge_output(content, block.block_id)

    @staticmethod
    def _salvage(content: str) -> Optional[dict]:
        """Recover action/summary from JSON the model never finished writing."""
        m = re.search(r'"action"\s*:\s*"([a-z_]+)"', content or "")
        if not m:
            return None
        out = {"action": m.group(1)}
        # Take everything after the opening quote of "summary", up to a
        # closing quote that is followed by a comma or brace -- or, when the
        # output was cut off, to the end of what arrived.
        s = re.search(r'"summary"\s*:\s*"(.*?)(?:"\s*[,}]|$)', content or "",
                      re.DOTALL)
        if s:
            text = s.group(1).strip().replace('\\"', '"').replace("\\n", "\n")
            # A salvaged summary replaces the block's text, so it should not
            # end mid-sentence. Cut back to the last sentence that finished,
            # provided that keeps most of what arrived.
            if text and text[-1] not in ".!?":
                cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
                if cut > len(text) * 0.5:
                    text = text[:cut + 1]
            if text:
                out["summary"] = text
        return out

    @staticmethod
    def _is_overflow(raw: str) -> bool:
        try:
            err = (json.loads(raw) or {}).get("error") or {}
        except (json.JSONDecodeError, TypeError):
            return False
        return err.get("type") == "exceed_context_size_error" or bool(
            err.get("n_prompt_tokens") and err.get("n_ctx")
        )

    def _parse_judge_output(self, content: str,
                            block_id: str = "") -> tuple[str, Optional[str]]:
        # Unreadable output falls back to "keep", which is the safe choice but
        # is indistinguishable in the log from the model deliberately keeping
        # a block. Record why, so "100% keep" can be told apart from "the
        # model is emitting garbage".
        def bail(reason):
            self.wal.write({
                "event": "judge_parse_failed",
                "block_id": block_id,
                "reason": reason,
                "sample": (content or "")[:200],
                "timestamp": time.time(),
            })
            return "keep", None

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        obj = None
        if json_match:
            try:
                obj = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                obj = None
        if obj is None:
            # A small model that runs past max_tokens leaves the object
            # unterminated, so `\{.*\}` matches nothing and a well-formed
            # decision gets thrown away. The fields are still readable
            # individually -- salvage them rather than defaulting to "keep".
            obj = self._salvage(content)
        if obj is None:
            return bail("no parseable decision in output")

        action = obj.get("action", "keep")
        if action not in ("keep", "truncate", "purge_candidate"):
            return bail(f"unknown action {action!r}")
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
