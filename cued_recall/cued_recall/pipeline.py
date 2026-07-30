import asyncio
import html as html_lib
import ipaddress
import json
import re
import socket
import time
import uuid
from typing import AsyncIterator, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import numpy as np


def url_block_reason(url: str) -> Optional[str]:
    """Return a reason string if a URL should not be fetched (SSRF guard).

    Blocks non-http(s) schemes and any host that resolves to a loopback,
    private, link-local, reserved, or otherwise internal address — so a model
    (or prompt injection from a fetched page) can't make the middleware hit
    internal services like the llama-servers or cloud metadata endpoints.
    """
    try:
        p = urlparse(url)
    except Exception:
        return "invalid URL"
    if p.scheme not in ("http", "https"):
        return f"scheme '{p.scheme}' not allowed"
    host = p.hostname
    if not host:
        return "missing host"
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return None  # unresolvable — let the HTTP client fail normally
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip.split("%")[0])
        except ValueError:
            continue
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            return f"blocked internal address ({ip})"
    return None

from .config import Config
from .embed import EmbeddingClient
from .index import VectorIndex
from .models import Block, BlockType, BlockStatus, Verification
from .small_model import SLOTS
from .store import BlockStore
from .utils import (
    build_stimulus,
    count_tokens,
    estimate_tokens,
    matches_correction,
    relevance_prompt,
    split_paragraph_boundary,
    truncate_tokens,
    RELEVANCE_SYSTEM,
)
from .wal import WAL

WEB_FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "Fetch content from a URL and return it as text. "
            "Use this to look up documentation, read articles, or "
            "research any topic by fetching its web page content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from",
                }
            },
            "required": ["url"],
        },
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet and return a ranked list of results "
            "(title, URL, and snippet). Use this to find current or factual "
            "information when you don't already know the answer or a specific "
            "URL. After searching, call web_fetch on the most relevant URL to "
            "read the full page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            "required": ["query"],
        },
    },
}


class Pipeline:
    # Applied to the retry budget after a context overflow so the re-fitted
    # prompt is always smaller than the one the server just refused.
    SHRINK_MARGIN = 0.95
    # How far under the limit trimming goes once it has to trim at all. The
    # gap is deliberate slack: see the comment in _fit_messages.
    TRIM_TARGET_RATIO = 0.8
    # Floor on what trimming may leave of the newest user turn. A question cut
    # below this is not worth answering, and a question cut to zero is worse
    # than an error: the model reports that the message is empty, which reads as
    # a broken model rather than a misconfigured budget.
    MIN_QUERY_TOKENS = 256

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
        self.usage_sink = None
        self.tps_sink = None
        self.chat_sink = None
        self.tagger = None
        self.verifier = None
        # Which blocks recall actually served, per (conversation_id, turn).
        # A block that was recalled and then not objected to is the only
        # positive verification signal this system can observe; see
        # apply_accepted_verification. In memory only -- losing it on restart
        # costs nothing but a few unknowns.
        self._recalled_by_turn: dict = {}

    @staticmethod
    def _msg_text(m) -> str:
        c = m.get("content", "")
        if c is None:  # assistant tool-call messages carry content=None
            return ""
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c if isinstance(c, str) else str(c)

    @staticmethod
    def _newest_user_index(msgs: list):
        """Index of the last real user turn, or None.

        A user message whose whole content is a <tool_response> does not
        count -- the chat template applies the same test when deciding
        whether a query exists. Shared by _fit_messages, which must not trim
        this turn away, and build_messages, which anchors recall to it.
        """
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") != "user":
                continue
            text = Pipeline._msg_text(msgs[i]).strip()
            if text.startswith("<tool_response>") and text.endswith("</tool_response>"):
                continue
            return i
        return None

    @staticmethod
    def _prepend_text(msg: dict, prefix: str) -> dict:
        """Put `prefix` in front of a message's text, structure intact.

        Content may be a plain string or a parts list; for a parts list only
        the first text part is touched, so images and other non-text parts
        survive.
        """
        c = msg.get("content", "")
        if isinstance(c, list):
            parts = list(c)
            for i, p in enumerate(parts):
                if isinstance(p, dict) and p.get("type") == "text":
                    parts[i] = {**p, "text": prefix + p.get("text", "")}
                    return {**msg, "content": parts}
            return {**msg, "content": [{"type": "text", "text": prefix}] + parts}
        return {**msg, "content": prefix + (c if isinstance(c, str) else "")}

    def _estimate_tokens(self, text: str) -> int:
        """Conservative token estimate from character and word counts.

        Takes the larger of the two signals on purpose. Under-counting hands
        the server a prompt bigger than its window, which is a hard 400 and an
        empty answer; over-counting only costs some trimmed history.
        """
        return estimate_tokens(text, self.config.chars_per_token,
                               self.config.tokens_per_word)

    async def _count_tokens(self, text: str) -> int:
        """Exact count from the reasoning model's tokenizer, estimate on failure."""
        return await count_tokens(
            text, self.config.reasoning_endpoint,
            self.config.chars_per_token, self.config.tokens_per_word,
        )

    def _payload_overhead_tokens(self, messages: list, body: dict) -> int:
        """Tokens in the request that are NOT message content.

        Tool schemas are the big one: a client like opencode ships a dozen
        JSON-Schema tool definitions in `body["tools"]`, which the chat
        template renders into the prompt. The old budget ignored them
        entirely. Per-message template scaffolding (role headers, turn
        delimiters) is charged at a flat rate per message.
        """
        overhead = len(messages) * 4
        tools = body.get("tools")
        if tools:
            try:
                overhead += self._estimate_tokens(json.dumps(tools))
            except (TypeError, ValueError):
                pass
        return overhead

    @staticmethod
    def _tools_bytes(body: dict) -> int:
        """UTF-8 size of the tool definitions the template will render."""
        tools = body.get("tools")
        if not tools:
            return 0
        try:
            return len(json.dumps(tools).encode("utf-8", "ignore"))
        except (TypeError, ValueError):
            return 0

    async def _exact_prompt_tokens(self, messages: list, body: dict) -> Optional[int]:
        """Tokenize the prompt on the server for an exact count.

        Only worth calling near the limit -- /tokenize costs a fixed ~450 ms
        regardless of input size. Returns None if the server can't be reached,
        in which case the caller keeps its estimate.

        Tools are deliberately excluded from the blob here. They are already
        accounted for in _payload_overhead_tokens, which subtracts them from
        the budget. Including them in both places double-counts their size,
        making the trimmer think far more history must be dropped than actually
        needs to be -- and when the only messages left are the system prompt
        and the latest user turn, the truncation step empties the user message.
        """
        blob = "\n".join(self._msg_text(m) for m in messages)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    f"{self.config.reasoning_endpoint}/tokenize",
                    json={"content": blob},
                )
                if r.status_code != 200:
                    return None
                tokens = r.json().get("tokens")
                if not isinstance(tokens, list):
                    return None
                return len(tokens) + len(messages) * 4
        except (httpx.HTTPError, ValueError, KeyError):
            return None

    async def _fit_messages(self, messages: list, body: Optional[dict] = None,
                            limit: Optional[int] = None,
                            use_exact: bool = True) -> list:
        """Truncate messages to fit the reasoning server's context window.

        Preserves the system message (index 0) and the latest user message.
        Oldest middle messages are dropped first. If a single message exceeds
        the limit, it is hard-truncated.

        `use_exact` controls the server-side recount below. It must be off when
        the caller already expressed `limit` in estimator units (see
        _refit_after_overflow): the recount replaces `total` with real tokens
        while `limit` stays on the estimator scale, and comparing the two
        silently passes a prompt that is over budget.

        Accuracy matters here in a way it did not appear to: the previous
        word-count estimator ran ~41% low on real agent traffic (code and
        markdown), so a 40k-token prompt measured as 24k, sailed under the
        budget untouched, and the server rejected the request outright. Three
        defenses now: a conservative estimate, an exact server-side count when
        that estimate lands near the limit, and -- in the callers -- a retry
        driven by the server's own token count if it still overflows.
        """
        body = body or {}

        def _set_content(m, text):
            c = m.get("content", "")
            if isinstance(c, list):
                # Replace text in first text part, drop others
                new_parts = [{"type": "text", "text": text}]
                return {**m, "content": new_parts}
            return {**m, "content": text}

        if limit is None:
            limit = self.config.max_context_tokens
        overhead = self._payload_overhead_tokens(messages, body)
        limit = max(512, limit - overhead)

        per_msg = [self._estimate_tokens(self._msg_text(m)) for m in messages]
        total = sum(per_msg)

        # Near the limit the estimate is not good enough to bet a hard 400 on.
        # Get the real number and rescale the per-message figures by the same
        # factor so the drop loop below stays consistent with it.
        #
        # The estimate cannot be trusted to decide when it is itself untrusted:
        # measured against this tokenizer it runs 24% low on Azerbaijani and 55%
        # low on base64, and an under-count both inflates the prompt and hides
        # the fact that it did. So the trigger is a fact rather than a guess --
        # every token costs at least one byte, so a payload smaller than the
        # budget in bytes provably cannot exceed it in tokens, and anything
        # larger is worth 450 ms to measure properly. The configured threshold
        # stays as a second trigger for a payload that is compact but dense.
        #
        # Tool schemas count toward those bytes. They are charged to `limit`
        # through _payload_overhead_tokens, which is an ESTIMATE -- so leaving
        # them out of the byte side compared a real measurement against a
        # guess, and a client shipping a dozen JSON-Schema tool definitions
        # could slip past the proof on the strength of that guess.
        payload_bytes = sum(len(self._msg_text(m).encode("utf-8", "ignore"))
                            for m in messages)
        payload_bytes += self._tools_bytes(body)
        must_count = payload_bytes >= limit
        count_method = "estimate"
        if use_exact and (must_count or total > limit * self.config.exact_count_threshold):
            exact = await self._exact_prompt_tokens(messages, body)
            if exact is not None and total > 0:
                scale = exact / total
                per_msg = [int(t * scale) for t in per_msg]
                total = exact
                count_method = "exact"

        # How the budget was decided, so the claim that the estimator is safe
        # here is auditable from the log rather than taken on trust. An
        # "estimate" line whose payload_bytes is under the limit is the proof
        # itself: a token costs at least one byte, so those tokens cannot have
        # exceeded the budget however badly the estimate read.
        self.wal.write({
            "event": "prompt_budget",
            "count_method": count_method,
            "counted_tokens": total,
            "limit": limit,
            "payload_bytes": payload_bytes,
            "byte_proof": payload_bytes < limit,
            "messages": len(messages),
            "trimmed": total > limit,
            "timestamp": time.time(),
        })

        if total <= limit:
            return messages

        result = list(messages)
        sizes = list(per_msg)

        # Trimming rewrites every token after the system prompt, so whatever
        # turn it happens on pays a full re-prefill. Dropping the bare minimum
        # to fit means that is every turn once a conversation reaches the
        # ceiling. Go under by a margin instead and the same single re-prefill
        # buys several turns of byte-identical prefix. The history given up
        # early is exactly what the recall store exists to bring back.
        target = int(limit * self.TRIM_TARGET_RATIO)

        # Drop oldest middle messages first. This loop always claimed to keep
        # everything "between system and latest user", but nothing enforced the
        # user half: only index 0 and the final message were protected. In a
        # long agent session the user turns sit in the middle while the tail is
        # assistant/tool traffic, so trimming could remove every user message --
        # and the template then raises "No user query found in messages.",
        # which llama.cpp returns as a 400 that kills the turn. Only reachable
        # once trimming actually started removing things (see SHRINK_MARGIN).
        # Sizes are popped in lockstep so the running total stays on the same
        # scale as `total` -- which may have been replaced by an exact
        # server-side count above.
        while len(result) > 2 and total > target:
            keep = self._newest_user_index(result)
            victim = next((j for j in range(1, len(result) - 1) if j != keep), None)
            if victim is None:
                break        # only the protected turn is left to give up
            total -= sizes.pop(victim)
            result.pop(victim)

        # If nothing is left to drop and it still does not fit, start shrinking
        # messages in place -- cheapest to lose first: old turns, then the system
        # prompt, then the newest user turn, and that one only down to a floor.
        #
        # It used to truncate result[-1] unconditionally, which is usually the
        # newest user turn. The drop loop above carefully protects that turn;
        # this step then erased it anyway, because keep_tokens goes to zero as
        # soon as the overshoot exceeds the message's own size. A two-token "hi"
        # against a budget blown by an agentic client's tool schemas came out as
        # content="", and the model answered that the message was empty -- the
        # one failure mode with no error anywhere to point at it.
        #
        # Each shrink drops from the FRONT, keeping the tail: for a pasted
        # document or a tool result, the end is usually the part being asked
        # about.
        if total > limit and result:
            protected = self._newest_user_index(result)
            order = sorted((j for j in range(1, len(result)) if j != protected),
                           key=lambda j: sizes[j], reverse=True)
            if protected != 0:
                order.append(0)
            if protected is not None:
                order.append(protected)

            for j in order:
                if total <= limit:
                    break
                text = self._msg_text(result[j])
                if not text or sizes[j] <= 0:
                    continue
                floor = self.MIN_QUERY_TOKENS if j == protected else 0
                keep_tokens = max(floor, sizes[j] - (total - limit))
                if keep_tokens >= sizes[j]:
                    continue          # already smaller than its own floor
                # Convert the token allowance back into characters at this
                # message's own measured density, rounding down so we never
                # leave more than the budget allows.
                keep_chars = int(len(text) * (keep_tokens / sizes[j]))
                result[j] = _set_content(result[j], text[len(text) - keep_chars:])
                total -= sizes[j] - keep_tokens
                sizes[j] = keep_tokens

        if total > limit:
            # The system prompt and the current question do not fit together, so
            # no amount of trimming will help. Silently sending a blank question
            # is how that stayed invisible; say what is actually wrong instead.
            raise RuntimeError(
                f"prompt does not fit: {total:,} tokens of messages against a "
                f"{limit:,}-token budget with nothing left to trim. The "
                f"reasoning server's window is too small for this client's "
                f"system prompt and tool schemas -- raise --reasoning-ctx."
            )

        # Trimming can orphan a `tool` result whose preceding assistant
        # tool_calls message was dropped, which llama.cpp rejects. Drop any
        # leading tool messages left just after the system prompt.
        while len(result) > 1 and result[1].get("role") == "tool":
            result.pop(1)
        return result

    @staticmethod
    def _parse_upstream_error(raw: str) -> dict:
        """Pull the useful bits out of a llama.cpp error body.

        A context overflow reports the numbers we need to recover:
        {"error":{"code":400,"type":"exceed_context_size_error",
                  "n_prompt_tokens":37245,"n_ctx":32768,"message":"..."}}
        """
        out = {"message": (raw or "")[:300], "overflow": False,
               "n_prompt_tokens": None, "n_ctx": None}
        try:
            err = (json.loads(raw) or {}).get("error") or {}
        except (json.JSONDecodeError, TypeError):
            return out
        if err.get("message"):
            out["message"] = str(err["message"])[:300]
        npt, nctx = err.get("n_prompt_tokens"), err.get("n_ctx")
        if err.get("type") == "exceed_context_size_error" or (npt and nctx):
            out["overflow"] = True
            out["n_prompt_tokens"] = npt
            out["n_ctx"] = nctx
        return out

    async def _refit_after_overflow(self, messages: list, body: dict,
                                    info: dict) -> list:
        """Re-trim using the server's own token count.

        The last line of defence. Whatever the estimator believed, the server
        has now stated exactly how many tokens the prompt was and how many it
        can take, so scale the budget by the ratio it just reported.
        """
        n_ctx = info.get("n_ctx") or 0
        n_prompt = info.get("n_prompt_tokens") or 0
        reserve = self.config.context_reserve_tokens
        target = max(512, int(n_ctx) - reserve) if n_ctx else self.config.max_context_tokens

        # Express the target in whatever units the estimator uses, via the
        # ratio between its estimate and the server's real count.
        real_target = target
        est = sum(self._estimate_tokens(self._msg_text(m)) for m in messages)
        if n_prompt and est:
            # SHRINK_MARGIN keeps the retry strictly smaller than the payload
            # that was just rejected. Without it the converted budget can land
            # above `est` -- it did: a 53,525-token prompt was refused and the
            # retry budget came out at 54,918, so nothing was dropped and the
            # identical payload went back out and was refused again.
            target = max(512, int(target * (est / n_prompt) * self.SHRINK_MARGIN))

        self.wal.write({
            "event": "context_overflow_retry",
            "n_prompt_tokens": n_prompt,
            "n_ctx": n_ctx,
            # Two scales: the real token target, and that same target expressed
            # in estimator units (what _fit_messages actually compares against).
            "target_real_tokens": real_target,
            "retry_limit_estimator_units": target,
            "estimator_total": est,
            "timestamp": time.time(),
        })
        # use_exact=False: `target` is in estimator units, and the server-side
        # recount would put `total` back on the real-token scale, so the two
        # would no longer be comparable.
        return await self._fit_messages(messages, body, limit=target,
                                        use_exact=False)

    @staticmethod
    def _usage_total(usage: dict) -> Optional[int]:
        if not usage:
            return None
        total = usage.get("total_tokens")
        if total is None:
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if pt is None and ct is None:
                return None
            total = (pt or 0) + (ct or 0)
        return total

    def _report_usage(self, usage: dict):
        if self.usage_sink:
            self.usage_sink(usage)

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
        # Why a candidate did not make it into the prompt was invisible: the
        # turn log recorded only how many blocks were admitted. Counted here so
        # the admin panel can show budget pressure -- a high skipped_oversized
        # against a low admitted count means recall.budget_tokens is starving
        # the injection rather than the index failing to find anything.
        skipped_corrected = skipped_missing = skipped_oversized = 0
        for block_id, sim in results:
            meta = await asyncio.to_thread(self.index.get_meta, block_id)
            if meta and meta.get("verification") == "corrected":
                skipped_corrected += 1
                continue
            block = await asyncio.to_thread(self.store.get, block_id)
            if block is None:
                skipped_missing += 1
                continue
            # Skip, don't stop. A single oversized hit (a pasted document can
            # be several times the budget on its own) used to break the loop
            # on the first iteration, returning zero memories for the turn
            # even though smaller, equally relevant blocks sat right behind
            # it. Pass over what doesn't fit and keep filling the budget.
            if total_tokens + block.token_count > self.config.recall.budget_tokens:
                skipped_oversized += 1
                continue
            total_tokens += block.token_count
            blocks.append((block, sim))

        rejected_by_judge = 0
        if self.config.recall.judge_enabled and blocks:
            kept = await self._filter_by_relevance(query_text, blocks)
            rejected_by_judge = len(blocks) - len(kept)
            blocks = kept
            total_tokens = sum(b.token_count for b, _ in blocks)

        self.wal.write({
            "event": "recall_budget",
            "candidates": len(results),
            "admitted": len(blocks),
            "rejected_by_judge": rejected_by_judge,
            "skipped_corrected": skipped_corrected,
            "skipped_missing": skipped_missing,
            "skipped_oversized": skipped_oversized,
            "tokens_used": total_tokens,
            # Carried per event so a historical turn shows the budget that was
            # actually in force when it ran.
            "budget_tokens": self.config.recall.budget_tokens,
            "top_similarity": round(results[0][1], 4) if results else None,
            "timestamp": time.time(),
        })
        return blocks

    async def _filter_by_relevance(
        self, question: str, blocks: List[Tuple[Block, float]]
    ) -> List[Tuple[Block, float]]:
        """Drop candidates the small model says do not apply to this question.

        Cosine similarity answers "is this about the same subject", which is
        not the question that matters. Measured on the evaluate/ corpus, a
        block about phase 1 of a problem scores 0.841 against a question about
        phase 2, and a block sharing only vocabulary scores 0.708 -- both above
        the 0.62 threshold, both wrong to inject.

        Fail-open throughout: a timeout or an unparseable reply keeps the
        block. The failure this guards against is a mildly irrelevant note in
        the prompt; the failure it must not cause is losing a relevant one
        because a CPU model was busy.
        """
        async def verdict(pair):
            block, sim = pair
            try:
                # Through the shared semaphore: the judge server has one slot,
                # so a gather() of four calls outside this queue is the pile-up
                # that once timed out a third of all tag calls.
                async with SLOTS:
                    async with httpx.AsyncClient(
                        timeout=self.config.recall.judge_timeout_s
                    ) as client:
                        resp = await client.post(
                            self.config.judge_endpoint.rstrip("/")
                            + "/v1/chat/completions",
                            json={
                                "messages": [
                                    {"role": "system",
                                     "content": RELEVANCE_SYSTEM},
                                    {"role": "user",
                                     "content": relevance_prompt(
                                         question, block.text)},
                                ],
                                "temperature": 0,
                                "max_tokens": 4,
                            },
                        )
                        resp.raise_for_status()
                        content = (resp.json().get("choices", [{}])[0]
                                   .get("message", {}).get("content", ""))
            except Exception as e:
                self.wal.write({
                    "event": "recall_judge_error",
                    "block_id": block.block_id,
                    # str() on an httpx timeout is "", so carry the type.
                    "error": f"{type(e).__name__}: {e}",
                    "timestamp": time.time(),
                })
                return True
            text = (content or "").strip().lower()
            if text.startswith("no"):
                return False
            # Anything that is not a clear "no" keeps the block.
            return True

        verdicts = await asyncio.gather(*[verdict(p) for p in blocks])
        return [p for p, keep in zip(blocks, verdicts) if keep]

    def build_recall_injection(self, blocks: List[Tuple[Block, float]]) -> str:
        if not blocks:
            return ""
        parts = [
            "The user has told you things in earlier conversations. Below is what was",
            "recovered from that history -- treat it as true and use it directly to",
            "answer the current message. Do not say you lack access to this",
            "information; you have it, right here. If a recalled item conflicts with",
            "something said in the CURRENT conversation, the current conversation",
            "wins. If it's a technical derivation rather than a stated fact, re-verify",
            "it before relying on it, since it may be outdated.",
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

    def build_messages(self, original_messages: list, recall_text: str) -> list:
        """Attach the recall note to the newest user message.

        Where this goes is a performance decision, not a formatting one.
        Recalled blocks are retrieved per query, so they differ from turn to
        turn. They used to be merged into the front of the client's system
        message, which meant the first tokens of the prompt changed on every
        request and invalidated llama.cpp's entire KV prefix behind them.
        Measured on this stack: 43,345-token prompts re-prefilled from scratch
        at f_keep 0.12, 107 s of prompt processing per turn to produce a reply
        that decoded in 1.4 s. Anchoring recall to the last user message leaves
        the system prompt and the whole conversation history byte-identical
        between turns, so the cache holds and only the recall block and the new
        message are prefilled.

        It cannot be a second system message. Qwen3.5's chat template renders
        messages[0] as the system block and raises 'System message must be at
        the beginning.' for any later one, which llama.cpp returns as a 400 --
        that is what used to break agent CLIs like opencode, which always ship
        their own system prompt. Folding into the user turn sidesteps the rule
        rather than working around it, and costs nothing in protection:
        _fit_messages keeps the newest user message, so the recall text is as
        safe from trimming here as it was at index 0.
        """
        if not recall_text:
            return original_messages
        idx = self._newest_user_index(original_messages)
        if idx is None:
            # No user turn to anchor to -- a system-only or tool-only body.
            # Fall back to the old front-merge rather than drop the recall.
            first = original_messages[0] if original_messages else None
            if first and first.get("role") == "system":
                client_text = self._extract_text(first.get("content"))
                merged = {**first, "content": (recall_text + "\n\n" + client_text
                                               if client_text else recall_text)}
                return [merged] + original_messages[1:]
            return [{"role": "system", "content": recall_text}] + original_messages
        out = list(original_messages)
        out[idx] = self._prepend_text(out[idx], recall_text + "\n\n")
        return out

    @staticmethod
    def _extract_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        return str(content or "")

    @staticmethod
    def _html_to_text(raw: str) -> str:
        raw = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
        text = re.sub(r"(?s)<[^>]+>", " ", raw)
        text = html_lib.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
        return text.strip()

    @staticmethod
    async def _fetch_url(url: str) -> str:
        reason = await asyncio.to_thread(url_block_reason, url)
        if reason:
            return f"Refused to fetch {url}: {reason}"
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "CuedRecall/1.0 web-fetch-tool"},
            )
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "json" in ct:
                return json.dumps(resp.json(), indent=2)[:8000]
            resp.encoding = resp.encoding or "utf-8"
            if "html" in ct:
                return Pipeline._html_to_text(resp.text)[:20000]
            return resp.text[:20000]

    # ---- Web search -------------------------------------------------------

    def _backend_usable(self, name: str) -> bool:
        ws = self.config.web_search
        if name == "searxng":
            return bool(ws.searxng_url)
        if name in ("brave", "serper"):
            return bool(ws.key_for(name))
        return name == "duckduckgo"  # needs no credentials

    def _search_chain(self) -> List[str]:
        """Backends to try in order: the configured one, then any others.

        DuckDuckGo is throttled unpredictably, so a configured paid backend
        should be able to cover for it and vice versa rather than letting one
        bad response become "no results".
        """
        ws = self.config.web_search
        chain = []
        chosen = (ws.backend or "duckduckgo").lower()
        if self._backend_usable(chosen):
            chain.append(chosen)
        if ws.fallback:
            for name in ("brave", "serper", "searxng", "duckduckgo"):
                if name not in chain and self._backend_usable(name):
                    chain.append(name)
        return chain or ["duckduckgo"]

    async def _run_backend(self, name: str, query: str, n: int) -> Tuple[list, bool]:
        """Returns (results, blocked)."""
        if name == "searxng":
            return await self._search_searxng(query, n), False
        if name == "brave":
            return await self._search_brave(query, n), False
        if name == "serper":
            return await self._search_serper(query, n), False
        return await self._search_duckduckgo_checked(query, n)

    async def _web_search(self, query: str) -> str:
        ws = self.config.web_search
        n = max(1, ws.max_results)
        results: list = []
        blocked = False
        for name in self._search_chain():
            try:
                results, blocked = await self._run_backend(name, query, n)
            except Exception as e:
                self.wal.write({
                    "event": "web_search_error",
                    "backend": name,
                    "error": str(e),
                    "timestamp": time.time(),
                }) if self.wal else None
                results, blocked = [], True
                continue
            if results:
                break
        if blocked and not results:
            # Distinguish "the engine refused us" from "nothing matched". A
            # bland "no results" reads to the model as bad luck, so it retries
            # the same dead backend until it burns every tool round -- which is
            # exactly what an anti-bot block looks like from the inside.
            tried = ", ".join(self._search_chain())
            return (
                f"web_search is UNAVAILABLE: every configured backend ({tried}) "
                "failed or was refused, so no query can succeed right now. Do "
                "NOT retry this tool or rephrase the query -- the result will "
                "be identical. Either answer from your own knowledge and say "
                "the information may be out of date, or ask the user to "
                "configure a working search backend (brave, serper, or "
                "searxng) under web_search in config.yaml."
            )
        if not results:
            return f"No search results for: {query}"
        lines = [f"Search results for '{query}':", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            if r.get("snippet"):
                lines.append(f"   {r['snippet']}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _strip_html(s: str) -> str:
        s = re.sub(r"<[^>]+>", "", s or "")
        return html_lib.unescape(s).strip()

    @staticmethod
    def _ddg_unwrap(href: str) -> str:
        # DuckDuckGo wraps result URLs as //duckduckgo.com/l/?uddg=<encoded>
        if href.startswith("//"):
            href = "https:" + href
        try:
            qs = parse_qs(urlparse(href).query)
            if "uddg" in qs:
                return unquote(qs["uddg"][0])
        except Exception:
            pass
        return href

    async def _search_duckduckgo_checked(self, query: str, n: int) -> Tuple[list, bool]:
        """Search, and report whether the engine blocked us.

        DuckDuckGo answers scraped requests with HTTP 202 and an anti-bot
        interstitial that contains none of the result markup, which is
        indistinguishable from "no matches" unless checked explicitly.
        """
        try:
            html, status = await self._fetch_ddg_html(query)
        except Exception:
            return [], True
        results = self._parse_ddg_html(html, n)
        if results:
            return results, False
        blocked = status != 200 or "result__a" not in html
        return [], blocked

    async def _fetch_ddg_html(self, query: str) -> Tuple[str, int]:
        """Fetch DDG's HTML endpoint, retrying a throttled response.

        DuckDuckGo rate-limits scraped requests rather than blocking outright:
        it answers with HTTP 202 and an anti-bot interstitial, then serves the
        same query normally a moment later. Backing off here keeps a transient
        throttle from surfacing to the model as "no results" -- which used to
        make it retry immediately and throttle itself harder.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            ),
        }
        text, status = "", 0
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            for attempt in range(3):
                resp = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query, "kl": "us-en"},
                    headers=headers,
                )
                resp.encoding = "utf-8"  # DDG serves UTF-8; avoid mojibake
                text, status = resp.text, resp.status_code
                if status == 200 and "result__a" in text:
                    return text, status
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
        return text, status

    async def _search_duckduckgo(self, query: str, n: int) -> list:
        html, _ = await self._fetch_ddg_html(query)
        return self._parse_ddg_html(html, n)

    def _parse_ddg_html(self, html: str, n: int) -> list:
        link_re = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
        snip_re = re.compile(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
        links = link_re.findall(html)
        snippets = snip_re.findall(html)
        results = []
        for i, (href, title) in enumerate(links[:n]):
            results.append({
                "title": self._strip_html(title),
                "url": self._ddg_unwrap(href),
                "snippet": self._strip_html(snippets[i]) if i < len(snippets) else "",
            })
        return results

    async def _search_searxng(self, query: str, n: int) -> list:
        base = self.config.web_search.searxng_url.rstrip("/")
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            resp = await client.get(base + "/search",
                                    params={"q": query, "format": "json"})
            resp.raise_for_status()
            data = resp.json()
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": r.get("content", "")}
            for r in (data.get("results") or [])[:n]
        ]

    async def _search_brave(self, query: str, n: int) -> list:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                # Brave caps count at 20 and rejects larger values outright.
                params={"q": query, "count": min(max(n, 1), 20)},
                headers={
                    "X-Subscription-Token": self.config.web_search.key_for("brave"),
                    "Accept": "application/json",
                    # Brave's API documents gzip as required.
                    "Accept-Encoding": "gzip",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            # Titles and descriptions come back with <strong> highlight markup.
            {"title": self._strip_html(r.get("title", "")),
             "url": r.get("url", ""),
             "snippet": self._strip_html(r.get("description", ""))}
            for r in ((data.get("web") or {}).get("results") or [])[:n]
        ]

    async def _search_serper(self, query: str, n: int) -> list:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": n},
                headers={"X-API-KEY": self.config.web_search.key_for("serper"),
                         "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            {"title": r.get("title", ""), "url": r.get("link", ""),
             "snippet": r.get("snippet", "")}
            for r in (data.get("organic") or [])[:n]
        ]

    # ---- Tool plumbing ----------------------------------------------------

    # Tools this middleware implements and executes itself. Anything else in a
    # tool_call belongs to the client (opencode's bash/read/edit/...) and must
    # be forwarded to it, never executed here.
    OWN_TOOLS = {"web_fetch", "web_search"}

    def _inject_tools(self, body: dict) -> dict:
        """Add our web tools -- but only for plain chat clients.

        If the caller supplied its own tools it is an agentic client that
        executes tools on its side (and already ships its own web fetch and
        search). Injecting ours there would put tools in the prompt that the
        client cannot run and that we would have to intercept, so leave an
        agentic client's tool set exactly as it sent it.
        """
        client_tools = body.get("tools") or []
        if client_tools:
            return body
        tools = []
        tools.append(WEB_FETCH_TOOL)
        ws = getattr(self.config, "web_search", None)
        if ws and ws.enabled:
            tools.append(WEB_SEARCH_TOOL)
        return {**body, "tools": tools}

    @staticmethod
    def _client_owns_tools(body: dict) -> bool:
        return bool(body.get("tools"))

    async def _handle_tool_calls(self, messages: list, tool_calls: list) -> list:
        if not tool_calls:
            return []
        results = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments", "{}") or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                if name == "web_fetch":
                    url = args.get("url", "")
                    content = "No URL provided" if not url else (await self._fetch_url(url))[:8000]
                elif name == "web_search":
                    query = args.get("query", "")
                    content = "No query provided" if not query else await self._web_search(query)
                else:
                    content = f"Unknown tool: {name}"
            except Exception as e:
                content = f"{name} error: {e}"
            results.append({"tool_call_id": tc.get("id"), "role": "tool", "content": content})
        return results

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
        """Long-form material the user supplied on THIS turn.

        Scoped to the newest user message for two reasons. Scanning every
        role swept up the client's own system prompt -- agent CLIs ship
        multi-thousand-word prompts -- and archived it as a memory once per
        turn. Scanning every turn's history re-archived the same pasted
        document on each subsequent turn of the conversation, so one paste
        became a block per turn thereafter. Neither is user knowledge worth
        recalling, and both crowd the vector index with near-duplicates.
        """
        messages = self.read_messages_from_body(body)
        last_user = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg
                break
        if last_user is None:
            return ""

        reading_parts = []
        content = last_user.get("content", "")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
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

    _FALLBACK_TOOL_CALL_RE = re.compile(
        r"<tool_call>\s*<function=([\w\-]+)>(.*?)</function>\s*</tool_call>",
        re.DOTALL,
    )
    _FALLBACK_PARAM_RE = re.compile(
        r"<parameter=([\w\-]+)>(.*?)</parameter>", re.DOTALL,
    )

    @classmethod
    def _parse_fallback_tool_calls(cls, text: str) -> List[dict]:
        """Parse a Hermes-style textual tool call some fine-tunes fall back to
        instead of emitting a properly structured tool_calls delta -- notably
        abliterated/uncensored merges, whose alignment-removal training often
        degrades structured-output adherence as a side effect. Recognizing
        this means the call still executes instead of leaking as raw markup.
        """
        calls = []
        for m in cls._FALLBACK_TOOL_CALL_RE.finditer(text):
            name, body = m.group(1), m.group(2)
            args = {pm.group(1): pm.group(2).strip()
                    for pm in cls._FALLBACK_PARAM_RE.finditer(body)}
            calls.append({"name": name, "arguments": json.dumps(args)})
        return calls

    class ToolCallFallbackFilter:
        """Buffers content deltas to intercept a mis-emitted textual tool
        call before it ever reaches the client, the same way ThinkSplitter
        holds back a partial <think> tag split across SSE chunks.
        """
        OPEN_TAG = "<tool_call>"
        CLOSE_TAG = "</tool_call>"

        def __init__(self):
            self.buffer = ""
            self.hold = len(self.OPEN_TAG) - 1

        def feed(self, chunk: str) -> Tuple[str, List[dict]]:
            """Returns (text_safe_to_release, parsed_fallback_calls)."""
            self.buffer += chunk
            idx = self.buffer.find(self.OPEN_TAG)
            if idx == -1:
                if len(self.buffer) <= self.hold:
                    return "", []
                safe = self.buffer[:len(self.buffer) - self.hold]
                self.buffer = self.buffer[len(safe):]
                return safe, []
            close_idx = self.buffer.find(self.CLOSE_TAG, idx)
            if close_idx == -1:
                # Might still be mid-call -- release anything before the open
                # tag, keep the rest buffered until it closes (or the round
                # ends and flush() releases it as plain text instead).
                before = self.buffer[:idx]
                self.buffer = self.buffer[idx:]
                return before, []
            end = close_idx + len(self.CLOSE_TAG)
            full_match = self.buffer[idx:end]
            before = self.buffer[:idx]
            self.buffer = self.buffer[end:]
            calls = Pipeline._parse_fallback_tool_calls(full_match)
            if not calls:
                # Looked like a tool call but didn't parse -- don't swallow it.
                before += full_match
            return before, calls

        def flush(self) -> str:
            rest = self.buffer
            self.buffer = ""
            return rest

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

        # Inject web_fetch / web_search tools so the LLM can research. Skipped
        # when the client brought its own tools (see _inject_tools).
        body_with_tools = self._inject_tools(body)

        # Hard rule: for "search"/"latest"/etc. queries, force a web_search on
        # the first turn instead of trusting the model to decide. Only valid
        # when we actually injected web_search -- forcing tool_choice for a
        # tool that isn't in the request would make llama.cpp reject it.
        force_search = (
            not self._client_owns_tools(body)
            and self._should_force_search(user_message)
        )

        if body.get("stream", False):
            return await self._process_streaming(
                body_with_tools, augmented_messages, user_message, reading_content,
                recall_blocks, conversation_id, turn_index, force_search,
            )
        else:
            return await self._process_nonstreaming(
                body_with_tools, augmented_messages, user_message, reading_content,
                recall_blocks, conversation_id, turn_index, force_search,
            )

    def _should_force_search(self, text: str) -> bool:
        ws = getattr(self.config, "web_search", None)
        if not ws or not ws.enabled or not getattr(ws, "force_search", False):
            return False
        t = text or ""
        for pat in ws.force_patterns:
            try:
                if re.search(pat, t, re.IGNORECASE):
                    return True
            except re.error:
                if pat.lower() in t.lower():
                    return True
        return False

    @staticmethod
    def _force_search_choice() -> dict:
        return {"type": "function", "function": {"name": "web_search"}}

    async def _process_streaming(
        self,
        body: dict,
        augmented_messages: list,
        user_message: str,
        reading_content: str,
        recall_blocks: List[Tuple[Block, float]],
        conversation_id: str,
        turn_index: int,
        force_search: bool = False,
    ) -> dict:
        return {
            "type": "streaming",
            "stream": self._stream_and_blockify(
                body, augmented_messages, user_message, reading_content,
                recall_blocks, conversation_id, turn_index, force_search,
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
        force_search: bool = False,
    ):
        splitter = self.ThinkSplitter(self.think_open, self.think_close)
        response_text = ""
        reasoning_content_parts: List[str] = []
        t0 = time.time()
        token_count = 0
        MAX_TOOL_ROUNDS = 5

        # Strict OpenAI-compatible clients (the AI SDK used by opencode, etc.)
        # expect every chunk to carry the full envelope -- id/object/created/
        # model and a choices[0].index -- not just a bare delta. Reuse the
        # upstream id/model when llama.cpp provides them so the whole stream
        # is internally consistent.
        env = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "model": body.get("model") or "cued-recall",
            "created": int(t0),
            "role_sent": False,
        }
        upstream_finish = None
        # An agent client decides when to auto-compact from the usage it sees.
        # This stream reported none: usage arrived from llama.cpp, went to the
        # admin metrics sink, and was dropped. opencode therefore believed the
        # conversation cost nothing, never compacted, and let it grow until
        # _fit_messages silently truncated history or the server refused the
        # prompt outright.
        usage_acc = {"prompt_tokens": 0, "completion_tokens": 0}

        def sse(delta: dict, finish_reason=None, usage=None) -> bytes:
            # OpenAI sends role on the first delta of the message.
            if delta and not env["role_sent"]:
                delta = {"role": "assistant", **delta}
                env["role_sent"] = True
            chunk = {
                "id": env["id"],
                "object": "chat.completion.chunk",
                "created": env["created"],
                "model": env["model"],
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }],
            }
            if usage:
                chunk["usage"] = usage
            return f"data: {json.dumps(chunk)}\n\n".encode()

        # Multi-round tool loop: stream each model turn to the client; if it
        # requests tools, run them, append the results, and stream the next
        # turn — so chains like web_search -> web_fetch -> answer complete
        # (the old code did a single non-streaming follow-up and stopped).
        messages = list(augmented_messages)
        async with httpx.AsyncClient() as client:
            for _round in range(MAX_TOOL_ROUNDS):
                tool_calls_by_index = {}
                round_content = ""
                tool_fallback_filter = self.ToolCallFallbackFilter()
                reasoning_fallback_filter = self.ToolCallFallbackFilter()
                try:
                    fitted = await self._fit_messages(messages, body)
                except RuntimeError as e:
                    # _fit_messages raises when nothing more can be trimmed.
                    # Uncaught, that exception reaches the client as a bare
                    # connection drop -- the generator is already the body of
                    # a 200 StreamingResponse, so there is no status code left
                    # to carry an error, and the stream just ends with no
                    # explanation. Same visible-error treatment as an upstream
                    # HTTP failure, so this doesn't become the next silent
                    # empty-response bug.
                    yield sse({"content": f"\n\n[{e}]"})
                    upstream_finish = "stop"
                    break

                # An upstream failure here used to be invisible. The old code
                # went straight to `async for line in resp.aiter_lines()` with
                # no status check, and llama.cpp reports errors as a plain JSON
                # body -- which never starts with "data: ", so every line was
                # skipped by the SSE filter. The round then ended with no
                # content and no tool calls, and the client received a
                # perfectly well-formed EMPTY stream. Check the status, retry
                # once on a context overflow using the server's own token
                # count, and otherwise surface the error to the client.
                resp = None
                stream_error = None
                for attempt in range(2):
                    payload = {
                        **body, "messages": fitted, "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    # Hard rule: force web_search on the first round only.
                    if force_search and _round == 0:
                        payload["tool_choice"] = self._force_search_choice()
                    try:
                        r = await client.send(
                            client.build_request(
                                "POST",
                                f"{self.config.reasoning_endpoint}/v1/chat/completions",
                                json=payload,
                                timeout=300,
                            ),
                            stream=True,
                        )
                    except httpx.HTTPError as e:
                        # Only HTTP *status* failures were handled here. A
                        # transport failure -- a wedged llama.cpp slot never
                        # answers, so the read times out waiting for response
                        # headers -- escaped this generator, killed the
                        # StreamingResponse mid-flight, and reached the client
                        # as a dropped connection with a traceback in the log.
                        # Report it the way a status failure is reported.
                        # str() on an httpx timeout is empty, hence the type
                        # name (see Judge._consolidate for the same trap).
                        detail = str(e) or type(e).__name__
                        self.wal.write({
                            "event": "upstream_transport_error",
                            "error": f"{type(e).__name__}: {detail}",
                            "round": _round,
                            "attempt": attempt,
                            "timestamp": time.time(),
                        })
                        stream_error = (
                            f"{type(e).__name__} talking to the reasoning "
                            f"server ({detail}). It may be wedged -- /slots "
                            f"will hang too if so."
                        )
                        break
                    if r.status_code == 200:
                        resp = r
                        break
                    await r.aread()
                    await r.aclose()
                    info = self._parse_upstream_error(r.text)
                    self.wal.write({
                        "event": "upstream_error",
                        "status": r.status_code,
                        "overflow": info["overflow"],
                        "message": info["message"],
                        "round": _round,
                        "attempt": attempt,
                        "timestamp": time.time(),
                    })
                    if info["overflow"] and attempt == 0:
                        fitted = await self._refit_after_overflow(
                            messages, body, info
                        )
                        continue
                    stream_error = f"HTTP {r.status_code}: {info['message']}"
                    break

                if stream_error:
                    yield sse({"content": f"\n\n[upstream model error - {stream_error}]"})
                    upstream_finish = "stop"
                    break

                try:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if data.get("usage"):
                            self._report_usage(data["usage"])
                            u = data["usage"]
                            # Prompt tokens are the LAST round's context
                            # occupancy, not a sum: every tool round re-sends
                            # the whole conversation, so adding them up would
                            # report several times the real context use.
                            if u.get("prompt_tokens"):
                                usage_acc["prompt_tokens"] = u["prompt_tokens"]
                            usage_acc["completion_tokens"] += (
                                u.get("completion_tokens") or 0
                            )
                        # Adopt upstream's id/model so the stream we emit is
                        # consistent with what actually generated it.
                        if data.get("id") and _round == 0:
                            env["id"] = data["id"]
                        if data.get("model"):
                            env["model"] = data["model"]
                        choices = data.get("choices") or []
                        delta_obj = choices[0].get("delta", {}) if choices else {}
                        if choices and choices[0].get("finish_reason"):
                            upstream_finish = choices[0]["finish_reason"]

                        delta_tool_calls = delta_obj.get("tool_calls")
                        if delta_tool_calls:
                            for tc in delta_tool_calls:
                                idx = tc.get("index", 0)
                                entry = tool_calls_by_index.setdefault(
                                    idx, {"id": "", "name": "", "arguments": ""})
                                if tc.get("id"):
                                    entry["id"] = tc["id"]
                                fn = tc.get("function", {})
                                if fn.get("name"):
                                    entry["name"] = fn["name"]
                                if fn.get("arguments"):
                                    entry["arguments"] += fn["arguments"]
                            continue  # tool_call chunks aren't shown to the client

                        rc = delta_obj.get("reasoning_content")
                        if rc:
                            # This model emits its fallback textual tool calls
                            # inside the reasoning stream, not the content
                            # stream, so the same filter has to run here or the
                            # raw <tool_call> markup reaches the user and the
                            # call never executes.
                            rc_out, rc_calls = reasoning_fallback_filter.feed(rc)
                            for i, fc in enumerate(rc_calls):
                                key = f"rfallback-{len(tool_calls_by_index)}-{i}"
                                tool_calls_by_index[key] = {
                                    "id": f"fallback-{uuid.uuid4().hex[:8]}",
                                    "name": fc["name"], "arguments": fc["arguments"],
                                }
                            if rc_out:
                                reasoning_content_parts.append(rc_out)
                                yield sse({"reasoning_content": rc_out})
                        delta = delta_obj.get("content", "")
                        if not delta:
                            continue
                        released, fallback_calls = tool_fallback_filter.feed(delta)
                        for i, fc in enumerate(fallback_calls):
                            idx = f"fallback-{len(tool_calls_by_index)}-{i}"
                            tool_calls_by_index[idx] = {
                                "id": f"fallback-{uuid.uuid4().hex[:8]}",
                                "name": fc["name"], "arguments": fc["arguments"],
                            }
                        if released:
                            response_text += released
                            round_content += released
                            token_count += len(released.split())
                            splitter.feed(released)
                            yield sse({"content": released})

                    # Round's SSE stream ended -- release whatever the filter
                    # was still holding (either it never closed, meaning it
                    # was never really a tool call, or a trailing tag split
                    # right at the response boundary).
                    trailing = tool_fallback_filter.flush()
                    if trailing:
                        response_text += trailing
                        round_content += trailing
                        token_count += len(trailing.split())
                        splitter.feed(trailing)
                        yield sse({"content": trailing})
                    rc_trailing = reasoning_fallback_filter.flush()
                    if rc_trailing:
                        reasoning_content_parts.append(rc_trailing)
                        yield sse({"reasoning_content": rc_trailing})
                finally:
                    await resp.aclose()

                # No tools requested -> this was the final answer.
                if not tool_calls_by_index:
                    break

                calls = list(tool_calls_by_index.values())

                # Calls for tools we don't implement belong to the client
                # (opencode's bash/read/edit/...). Forward them and stop:
                # the client executes them and sends the results back as a
                # new request. Running them through _handle_tool_calls here
                # would answer every one with "Unknown tool", which the model
                # then retries -- burning MAX_TOOL_ROUNDS inference passes and
                # leaving the client believing no tool was ever called.
                foreign = [c for c in calls if c["name"] not in self.OWN_TOOLS]
                if foreign:
                    for i, c in enumerate(foreign):
                        yield sse({"tool_calls": [{
                            "index": i,
                            "id": c["id"] or f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {"name": c["name"],
                                         "arguments": c["arguments"]},
                        }]})
                    upstream_finish = "tool_calls"
                    break

                names = ", ".join(c["name"] for c in calls if c["name"]) or "tool"
                # Surface activity in the Reasoning panel (not stored as a block).
                note = f"\n[running {names}…]\n"
                yield sse({"reasoning_content": note})

                tool_results = await self._handle_tool_calls(
                    messages,
                    [{"id": c["id"], "function": {"name": c["name"], "arguments": c["arguments"]}}
                     for c in calls],
                )
                assistant_msg = {
                    "role": "assistant",
                    "content": round_content or None,
                    "tool_calls": [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in calls
                    ],
                }
                messages = messages + [assistant_msg] + tool_results

        splitter.flush()

        # Terminating chunk: an empty delta carrying finish_reason. Strict
        # OpenAI-compatible clients use this -- not [DONE] -- to finalize the
        # assembled message; without it the stream ends with no completion
        # signal and the client can drop everything it accumulated.
        usage_out = None
        if usage_acc["prompt_tokens"] or usage_acc["completion_tokens"]:
            usage_out = {
                "prompt_tokens": usage_acc["prompt_tokens"],
                "completion_tokens": usage_acc["completion_tokens"],
                "total_tokens": (usage_acc["prompt_tokens"]
                                 + usage_acc["completion_tokens"]),
            }
        # Carried on the finish_reason chunk rather than only in the
        # empty-choices chunk below, so a client that stops reading at
        # finish_reason still gets the numbers it needs to decide on
        # compaction.
        yield sse({}, finish_reason=upstream_finish or "stop", usage=usage_out)
        # The OpenAI-spec placement: a final chunk with no choices, sent only
        # when the client actually asked for usage, since a client that did
        # not ask may index choices[0] unconditionally.
        if usage_out and (body.get("stream_options") or {}).get("include_usage"):
            yield (f"data: {json.dumps({'id': env['id'], 'object': 'chat.completion.chunk', 'created': env['created'], 'model': env['model'], 'choices': [], 'usage': usage_out})}\n\n").encode()
        yield b"data: [DONE]\n\n"

        elapsed = time.time() - t0
        if self.tps_sink:
            self.tps_sink(token_count, elapsed)

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

        # Plain transcript for the history sidebar. Blocks can't serve this:
        # they are split, summarised, and eventually purged by the judge.
        if self.chat_sink:
            self.chat_sink(conversation_id, user_message, full_result)

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
        force_search: bool = False,
    ) -> dict:
        # Tool-call loop: keep calling the LLM until no more tool_calls (max 5 rounds)
        messages = list(augmented_messages)
        for _round in range(5):
            t0 = time.time()
            async with httpx.AsyncClient() as client:
                fitted = await self._fit_messages(messages, body)
                # Same overflow recovery as the streaming path: retry once
                # against the token count the server itself reported before
                # giving up and raising.
                for attempt in range(2):
                    payload = {**body, "messages": fitted, "stream": False}
                    # Hard rule: force web_search on the first round only.
                    if force_search and _round == 0:
                        payload["tool_choice"] = self._force_search_choice()
                    try:
                        resp = await client.post(
                            f"{self.config.reasoning_endpoint}/v1/chat/completions",
                            json=payload,
                            timeout=300,
                        )
                    except httpx.HTTPError as e:
                        # Same exposure as the streaming path: only status
                        # failures were handled, so a wedged server surfaced as
                        # an unhandled transport exception. Record it before it
                        # leaves, so the WAL shows a deadlock rather than
                        # nothing at all.
                        detail = str(e) or type(e).__name__
                        self.wal.write({
                            "event": "upstream_transport_error",
                            "error": f"{type(e).__name__}: {detail}",
                            "round": _round,
                            "attempt": attempt,
                            "streaming": False,
                            "timestamp": time.time(),
                        })
                        raise RuntimeError(
                            f"reasoning server unreachable: {type(e).__name__}"
                            f" ({detail})"
                        ) from e
                    if resp.status_code == 200:
                        break
                    info = self._parse_upstream_error(resp.text)
                    self.wal.write({
                        "event": "upstream_error",
                        "status": resp.status_code,
                        "overflow": info["overflow"],
                        "message": info["message"],
                        "round": _round,
                        "attempt": attempt,
                        "timestamp": time.time(),
                    })
                    if info["overflow"] and attempt == 0:
                        fitted = await self._refit_after_overflow(
                            messages, body, info
                        )
                        continue
                    break
                resp.raise_for_status()
                result = resp.json()
            elapsed = time.time() - t0

            self._report_usage(result.get("usage") or {})

            usage = result.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")
            if completion_tokens is None:
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                completion_tokens = len(content.split())
            if self.tps_sink:
                self.tps_sink(completion_tokens, elapsed)

            message = result.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                break

            # Tools we don't implement belong to the client -- hand the whole
            # response back untouched so it can run them, instead of replying
            # "Unknown tool" to itself for MAX rounds. See the streaming path.
            if any(tc.get("function", {}).get("name") not in self.OWN_TOOLS
                   for tc in tool_calls):
                return result

            # Execute tool calls and append results
            tool_results = await self._handle_tool_calls(messages, tool_calls)
            messages.append(message)
            messages.extend(tool_results)
        else:
            # If loop exhausted, take whatever the LLM gave us
            pass

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

        # Plain transcript for the history sidebar. Blocks can't serve this:
        # they are split, summarised, and eventually purged by the judge.
        if self.chat_sink:
            self.chat_sink(conversation_id, user_message, full_result)

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
            reasoning_blocks = await self._split_reasoning(
                full_reasoning, conversation_id, turn_index, now
            )

        result_block = Block(
            type=BlockType.result,
            conversation_id=conversation_id,
            turn_index=turn_index,
            token_count=await self._count_tokens(full_result),
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
                token_count=await self._count_tokens(reading_content),
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

        self._remember_recalled(
            conversation_id, turn_index,
            [b.block_id for b, _ in recall_blocks],
        )

    def _remember_recalled(self, conversation_id: str, turn_index: int,
                           block_ids: List[str]):
        if not block_ids:
            return
        self._recalled_by_turn[(conversation_id, turn_index)] = block_ids
        # Only the immediately previous turn is ever read back, so this is a
        # short tail, not a history. Bound it so a long-running server does not
        # accumulate one entry per turn forever.
        if len(self._recalled_by_turn) > 500:
            for key in list(self._recalled_by_turn)[:250]:
                del self._recalled_by_turn[key]

    async def _split_reasoning(
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
                token_count=await self._count_tokens(left),
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
    ) -> bool:
        """Pattern-matched corrections. Runs BEFORE the turn, deliberately.

        Marking the previous turn corrected here is what stops the block the
        user is objecting to from being recalled into the very turn where they
        object to it. The check is a regex, so it costs nothing; the model's
        second opinion runs after the reply instead, in
        verify_correction_with_model.

        Returns whether it matched, so the caller can skip the model when the
        answer is already known.
        """
        if not matches_correction(user_message, self.config.correction_patterns):
            return False
        prev_turn = turn_index - 1
        if prev_turn < 0:
            return False
        block_ids = await asyncio.to_thread(
            self._find_turn_blocks, conversation_id, prev_turn
        )
        await self._mark_corrected(block_ids, "pattern")
        return True

    async def verify_correction_with_model(
        self, user_message: str, conversation_id: str, turn_index: int
    ):
        """Second opinion from the small model, after the reply has gone out.

        Only reached when the patterns did not match, which is the case that
        needs help: across 395 stored blocks the pattern list had fired zero
        times. Never runs on the request path -- the classifier is a CPU-bound
        round trip, and verification applies to the previous turn's blocks
        anyway, so nothing waits on it.
        """
        if self.verifier is None:
            return
        prev_turn = turn_index - 1
        if prev_turn < 0:
            return
        block_ids = await asyncio.to_thread(
            self._find_turn_blocks, conversation_id, prev_turn
        )
        if not block_ids:
            return
        previous_answer = await self._previous_answer_text(block_ids)
        if not previous_answer:
            return

        verdict = await self.verifier.is_correction(previous_answer, user_message)
        # None means the model failed or hedged. That is not "no" -- leave the
        # blocks alone rather than recording a judgement nobody made.
        if verdict is True:
            await self._mark_corrected(block_ids, "model")

    async def _previous_answer_text(self, block_ids: List[str]) -> str:
        """The answer the user is responding to, for the correction classifier.

        Prefers the result block: that is what the user actually saw. Reasoning
        blocks hold the think trace, which the user never read and which would
        make the classifier judge the wrong text.
        """
        fallback = ""
        for bid in block_ids:
            block = await asyncio.to_thread(self.store.get, bid)
            if block is None:
                continue
            if block.type == BlockType.result:
                return block.text
            if not fallback:
                fallback = block.text
        return fallback

    async def _mark_corrected(self, block_ids: List[str], source: str):
        for bid in block_ids:
            # The source is recorded because it decides how much weight the
            # verdict carries: a pattern match can get a block deleted
            # outright, a model guess cannot. See Judge._should_purge.
            await asyncio.to_thread(
                self.index.update_verification, bid, "corrected", source
            )
            block = await asyncio.to_thread(self.store.get, bid)
            if block:
                block.verification = Verification.corrected
                await asyncio.to_thread(self.store.put, block)
            self.wal.write({
                "event": "verification_set",
                "block_id": bid,
                "verification": "corrected",
                "source": source,
                "timestamp": time.time(),
            })

    async def apply_accepted_verification(
        self, conversation_id: str, turn_index: int
    ):
        """Mark blocks accepted when something actually happened to them.

        This used to upgrade every block of the previous turn the moment a
        next turn arrived, which is why 309 of 395 stored blocks claim to be
        "accepted": the conversation continuing is not evidence of anything,
        and it upgraded blocks nobody had looked at.

        A block that was RECALLED into a turn and then not objected to is a
        real event -- it was put in front of the model, it shaped an answer,
        and the user carried on. That is the only positive signal available
        here, so it is the only one used.
        """
        prev_turn = turn_index - 1
        if prev_turn < 0:
            return
        block_ids = self._recalled_by_turn.pop(
            (conversation_id, prev_turn), []
        )
        for bid in block_ids:
            meta = await asyncio.to_thread(self.index.get_meta, bid)
            # Anything already corrected stays corrected: a later turn that
            # happens to recall it must not launder that away.
            if meta and meta.get("verification") == "unknown":
                await asyncio.to_thread(
                    self.index.update_verification, bid, "accepted",
                    "recalled_uncontested"
                )
                block = await asyncio.to_thread(self.store.get, bid)
                if block:
                    block.verification = Verification.accepted
                    await asyncio.to_thread(self.store.put, block)
                self.wal.write({
                    "event": "verification_set",
                    "block_id": bid,
                    "verification": "accepted",
                    "source": "recalled_uncontested",
                    "timestamp": time.time(),
                })

    def _find_turn_blocks(self, conversation_id: str, turn_index: int) -> List[str]:
        all_meta = self.index.list_meta(limit=10000)[0]
        return [
            m["block_id"]
            for m in all_meta
            if m["conversation_id"] == conversation_id
            and m["turn_index"] == turn_index
        ]

    async def _shelve_block_id(self, block_id: str):
        await asyncio.to_thread(self.index.update_status, block_id, "shelved")
        block = await asyncio.to_thread(self.store.get, block_id)
        if block:
            block.status = BlockStatus.shelved
            await asyncio.to_thread(self.store.put, block)
            # Tag at shelve time, not at the next judge pass (which may be
            # hours away) -- tagging exists so the admin page is readable
            # now. Fire-and-forget: a slow/failed tag call must not hold
            # up the request that triggered this shelve.
            if self.tagger and not block.tags:
                asyncio.create_task(self.tagger.tag_block(block))

    async def shelve_previous_turn(self, conversation_id: str, turn_index: int):
        prev_turn = turn_index - 1
        if prev_turn < 0:
            return
        block_ids = await asyncio.to_thread(
            self._find_turn_blocks, conversation_id, prev_turn
        )
        for bid in block_ids:
            await self._shelve_block_id(bid)

    async def shelve_stale_hot_blocks(self, min_idle_s: float, limit: int = 200) -> int:
        """Shelve 'hot' blocks whose conversation was never continued.

        Normal shelving only fires when the NEXT turn arrives in the same
        conversation, so a single "remember this" message with no follow-up
        would otherwise sit unrecallable indefinitely. This is the idle-timeout
        safety net -- called periodically, not just at startup.
        """
        block_ids = await asyncio.to_thread(
            self.index.hot_blocks_older_than, min_idle_s, limit
        )
        for bid in block_ids:
            await self._shelve_block_id(bid)
        return len(block_ids)
