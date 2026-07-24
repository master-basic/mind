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
from .store import BlockStore
from .utils import (
    build_stimulus,
    matches_correction,
    split_paragraph_boundary,
    truncate_tokens,
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

    def _fit_messages(self, messages: list) -> list:
        """Truncate messages to fit within max_context_tokens.

        Preserves the system message (index 0) and the latest user message.
        Oldest middle messages are dropped first. If a single message exceeds
        the limit, it is hard-truncated.
        """
        def _msg_tokens(m):
            c = m.get("content", "")
            if c is None:  # assistant tool-call messages carry content=None
                return 0
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return len(c.split())

        def _set_content(m, text):
            c = m.get("content", "")
            if isinstance(c, list):
                # Replace text in first text part, drop others
                new_parts = [{"type": "text", "text": text}]
                return {**m, "content": new_parts}
            return {**m, "content": text}

        limit = self.config.max_context_tokens
        total = sum(_msg_tokens(m) for m in messages)
        if total <= limit:
            return messages

        result = list(messages)
        # Drop oldest middle messages (between system and latest user) first
        while len(result) > 2 and total > limit:
            drop_idx = 1  # skip system at 0
            dropped = result.pop(drop_idx)
            total -= _msg_tokens(dropped)

        # If still over, truncate the last user message
        if total > limit and result:
            last = result[-1]
            c = last.get("content", "") or ""
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            words = c.split()
            excess = total - limit
            if len(words) > excess:
                truncated = " ".join(words[excess:])
                result[-1] = _set_content(last, truncated)
            else:
                result[-1] = _set_content(last, "")

        # Trimming can orphan a `tool` result whose preceding assistant
        # tool_calls message was dropped, which llama.cpp rejects. Drop any
        # leading tool messages left just after the system prompt.
        while len(result) > 1 and result[1].get("role") == "tool":
            result.pop(1)
        return result

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

    def build_messages(self, original_messages: list, recall_text: str) -> list:
        if not recall_text:
            return original_messages
        recall_msg = {"role": "system", "content": recall_text}
        return [recall_msg] + original_messages

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

    async def _web_search(self, query: str) -> str:
        ws = self.config.web_search
        n = max(1, ws.max_results)
        backend = (ws.backend or "duckduckgo").lower()
        if backend == "searxng" and ws.searxng_url:
            results = await self._search_searxng(query, n)
        elif backend == "brave" and ws.api_key:
            results = await self._search_brave(query, n)
        elif backend == "serper" and ws.api_key:
            results = await self._search_serper(query, n)
        else:
            results = await self._search_duckduckgo(query, n)
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

    async def _search_duckduckgo(self, query: str, n: int) -> list:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            ),
        }
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            resp = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "kl": "us-en"},
                headers=headers,
            )
            resp.raise_for_status()
            resp.encoding = "utf-8"  # DDG serves UTF-8; avoid mojibake in titles
            html = resp.text
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
                params={"q": query, "count": n},
                headers={"X-Subscription-Token": self.config.web_search.api_key,
                         "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": r.get("description", "")}
            for r in ((data.get("web") or {}).get("results") or [])[:n]
        ]

    async def _search_serper(self, query: str, n: int) -> list:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": n},
                headers={"X-API-KEY": self.config.web_search.api_key,
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

    def _inject_tools(self, body: dict) -> dict:
        tools = body.get("tools") or []
        existing_names = {t.get("function", {}).get("name") for t in tools if t.get("type") == "function"}
        if "web_fetch" not in existing_names:
            tools.append(WEB_FETCH_TOOL)
        ws = getattr(self.config, "web_search", None)
        if ws and ws.enabled and "web_search" not in existing_names:
            tools.append(WEB_SEARCH_TOOL)
        return {**body, "tools": tools}

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

        # Inject web_fetch tool so the LLM can use it for research
        body_with_tools = self._inject_tools(body)

        if body.get("stream", False):
            return await self._process_streaming(
                body_with_tools, augmented_messages, user_message, reading_content,
                recall_blocks, conversation_id, turn_index,
            )
        else:
            return await self._process_nonstreaming(
                body_with_tools, augmented_messages, user_message, reading_content,
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
        reasoning_content_parts: List[str] = []
        t0 = time.time()
        token_count = 0
        MAX_TOOL_ROUNDS = 5

        # Multi-round tool loop: stream each model turn to the client; if it
        # requests tools, run them, append the results, and stream the next
        # turn — so chains like web_search -> web_fetch -> answer complete
        # (the old code did a single non-streaming follow-up and stopped).
        messages = list(augmented_messages)
        async with httpx.AsyncClient() as client:
            for _round in range(MAX_TOOL_ROUNDS):
                tool_calls_by_index = {}
                round_content = ""
                payload = {
                    **body, "messages": self._fit_messages(messages), "stream": True,
                    "stream_options": {"include_usage": True},
                }
                async with client.stream(
                    "POST",
                    f"{self.config.reasoning_endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                ) as resp:
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
                        choices = data.get("choices") or []
                        delta_obj = choices[0].get("delta", {}) if choices else {}

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
                            reasoning_content_parts.append(rc)
                            yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': rc}}]})}\n\n".encode()
                        delta = delta_obj.get("content", "")
                        if not delta:
                            continue
                        response_text += delta
                        round_content += delta
                        token_count += len(delta.split())
                        splitter.feed(delta)
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': delta}}]})}\n\n".encode()

                # No tools requested -> this was the final answer.
                if not tool_calls_by_index:
                    break

                calls = list(tool_calls_by_index.values())
                names = ", ".join(c["name"] for c in calls if c["name"]) or "tool"
                # Surface activity in the Reasoning panel (not stored as a block).
                note = f"\n[running {names}…]\n"
                yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': note}}]})}\n\n".encode()

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

        # Yield [DONE]
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
        # Tool-call loop: keep calling the LLM until no more tool_calls (max 5 rounds)
        messages = list(augmented_messages)
        for _ in range(5):
            t0 = time.time()
            async with httpx.AsyncClient() as client:
                payload = {**body, "messages": self._fit_messages(messages), "stream": False}
                resp = await client.post(
                    f"{self.config.reasoning_endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                )
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
