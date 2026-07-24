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
        self.tagger = None

    def _fit_messages(self, messages: list) -> list:
        """Truncate messages to fit within max_context_tokens.

        Preserves the system message (index 0) and the latest user message.
        Oldest middle messages are dropped first. If a single message exceeds
        the limit, it is hard-truncated.
        """
        # Word count understates real tokens by roughly a third, so a limit
        # applied to raw word counts lets prompts through that overflow the
        # server's actual context window. Scale to an estimated token count so
        # max_context_tokens means what its name says.
        ratio = self.config.tokens_per_word

        def _msg_tokens(m):
            c = m.get("content", "")
            if c is None:  # assistant tool-call messages carry content=None
                return 0
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return int(len(c.split()) * ratio)

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
            # `total`/`limit` are token estimates; convert the overshoot back
            # into words before slicing, rounding up so we never undershoot.
            excess_words = int((total - limit) / ratio) + 1
            if len(words) > excess_words:
                truncated = " ".join(words[excess_words:])
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

        def sse(delta: dict, finish_reason=None) -> bytes:
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
                payload = {
                    **body, "messages": self._fit_messages(messages), "stream": True,
                    "stream_options": {"include_usage": True},
                }
                # Hard rule: force web_search on the first round only.
                if force_search and _round == 0:
                    payload["tool_choice"] = self._force_search_choice()
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
        yield sse({}, finish_reason=upstream_finish or "stop")
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
        force_search: bool = False,
    ) -> dict:
        # Tool-call loop: keep calling the LLM until no more tool_calls (max 5 rounds)
        messages = list(augmented_messages)
        for _round in range(5):
            t0 = time.time()
            async with httpx.AsyncClient() as client:
                payload = {**body, "messages": self._fit_messages(messages), "stream": False}
                # Hard rule: force web_search on the first round only.
                if force_search and _round == 0:
                    payload["tool_choice"] = self._force_search_choice()
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
