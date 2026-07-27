import asyncio
import json
import re
import time
from typing import List, Tuple

import httpx

from .config import Config
from .index import VectorIndex
from .models import Block
from .small_model import SLOTS
from .store import BlockStore
from .taxonomy import TAXONOMY_GROUPS, TAXONOMY_VERSION, validate_gist, validate_tags
from .wal import WAL


class Tagger:
    """Tags a block for human readability the moment it shelves.

    Deliberately separate from Judge: the judge only runs every
    interval_tokens, which would leave blocks unreadable in the admin page
    for potentially hours. This fires per-block, fire-and-forget, so a slow
    or failed call never blocks the request path.
    """

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
        endpoint = config.tagger.endpoint or config.judge_endpoint
        self.tagger_url = endpoint.rstrip("/") + "/v1/chat/completions"
        self.usage_sink = None
        # Tagging is fire-and-forget, one task per block, and shelving arrives
        # in batches (idle sweeps, startup). Unbounded, that stampedes a small
        # CPU-bound model: a third of all tag calls were timing out. The limit
        # now lives in small_model.SLOTS, shared with the correction verifier,
        # because the thing being protected is one server, not one caller.
        self._slots = SLOTS

    async def tag_block(self, block: Block):
        if not self.config.tagger.enabled:
            return
        try:
            async with self._slots:
                gist, tags = await self._call_model(block)
        except Exception as e:
            self.wal.write({
                "event": "tagger_error",
                "block_id": block.block_id,
                # str() on an httpx timeout is "", which made every one of
                # those failures unreadable in the log.
                "error": f"{type(e).__name__}: {e}",
                "timestamp": time.time(),
            })
            return

        gist = validate_gist(gist, self.config.tagger.gist_max_chars)
        tags = validate_tags(tags, self.config.tagger.max_tags)

        await asyncio.to_thread(self.index.set_tags, block.block_id, tags, gist)
        block.gist = gist
        block.tags = tags
        await asyncio.to_thread(self.store.put, block)

        self.wal.write({
            "event": "tagged",
            "block_id": block.block_id,
            "tags": tags,
            "taxonomy_version": TAXONOMY_VERSION,
            "timestamp": time.time(),
        })

    async def _call_model(self, block: Block) -> Tuple[str, List[str]]:
        vocab = "\n".join(
            f"- {group}: {', '.join(tags)}"
            for group, tags in TAXONOMY_GROUPS.items()
        )
        prompt = (
            "Summarize this archived block for a human skimming an admin "
            "dashboard. Respond with exactly one JSON object: "
            '{"gist": "<40 characters or fewer, plain description of what '
            'this block is about>", "tags": [<0 to 3 tags, chosen ONLY from '
            "the fixed list below, nothing else>]}\n\n"
            f"Tag list (grouped, pick tag names only):\n{vocab}\n\n"
            f"Context (what was asked):\n{block.stimulus_text[:1500]}\n\n"
            f"Content:\n{block.text[:2000]}"
        )
        # A 1.5B judge model on CPU can take a while under load; the previous
        # 60 s was tight enough that queued calls tripped it routinely.
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                self.tagger_url,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 150,
                },
            )
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
        return self._parse(content)

    @staticmethod
    def _parse(content: str) -> Tuple[str, List[str]]:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return "", []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return "", []
        gist = obj.get("gist", "") or ""
        tags = obj.get("tags", []) or []
        if not isinstance(tags, list):
            tags = []
        return gist, tags
