# Bug Report: Web-search turns that never end, end empty, or leak raw reasoning

**Date:** 2026-08-06
**Branch:** semantic-memory
**Affected component:** `cued_recall/cued_recall/pipeline.py` — `_process_streaming` /
`_stream_and_blockify` (multi-round tool loop), `ToolCallFallbackFilter`,
`_web_search` backend chain.
**Environment:** Python 3.14, uvicorn middleware on :8000, llama-server
(Qwen3.6-35B-A3B-UD-Q4_K_XL, `--jinja`, `-np 1`, ctx 65536) on :8080.

## Symptom

A chat turn that should use `web_search` behaves as one of:

1. **Never ends.** The model emits reasoning tokens indefinitely (measured:
   2.2 MB / 9,194 SSE chunks in 420 s+ with no `finish_reason`), so the client
   times out and shows "send a message again".
2. **Ends empty.** After a few search tool rounds the stream stops with
   `finish_reason: "tool_calls"` and **no answer content** — the client shows
   an empty assistant turn and asks the user to send the message again.
3. **Raw thinking wall.** The user sees the model's entire reasoning stream
   (including repeated identical planning paragraphs) as the "answer",
   concluding with a wrong date.
4. **Wrong facts.** The model answers from a stale/hallucinated search snippet
   ("current UTC time is 14:05 on February 20, 2026" from a junk page) instead
   of the real date (2026-08-06), then re-searches repeatedly in a loop.

## Reproduction

1. Send a plain-chat prompt that requires the current date, e.g. "what date is
   today" (web tools are injected only for plain clients, `_inject_tools`).
2. The reasoning model Qwen3.6 narrates tool use in **prose** inside its
   thinking trace ("Action: `web_search`…", "Let's do the search."), emitting
   the structured `<tool_call>` markup only intermittently.
3. Each time the fallback parser misses the prose, no search runs; the model
   keeps reasoning about searching, repeating the same paragraph verbatim.

## Root-cause analysis

### R1 — Tool calls live in the reasoning channel, and detection is markup-fragile

`Qwen3.6-35B-A3B-UD` via this llama.cpp stack emits **no structured
`tool_calls` deltas**. Every tool request must be caught by
`ToolCallFallbackFilter` parsing `<tool_call>`-style markup out of
`reasoning_content` (pipeline.py ~1998-2014, `_parse_fallback_tool_calls`
~1649). In practice the model mostly writes prose about tools instead of the
markup. When the markup is absent:
- no tool executes;
- the round ends with reasoning-only;
- `not tool_calls_by_index` → the loop breaks and the turn "finishes" with
  whatever the think trace was — which the client has already been streaming
  as `reasoning_content`.

Result: a search-driven turn depends on the model emitting exact markup in
the right channel. The middleware has no independent trigger for the search
and no way to tell "thinking about searching" from "searching".

### R2 — No bound on reasoning, no repetition guard

The upstream payload set no `max_tokens` (payload built at pipeline.py
~1887), and the middleware streams `reasoning_content` verbatim with no
limit and no loop detection. The model's degenerate repetition (the same
"Query: current date and time / I will fetch timeanddate.com" paragraph
appears verbatim dozens of times) is therefore free to stream for minutes.
A finish_reason never arrives while llama.cpp keeps sampling, so nothing in
the middleware ever ends the round. This is the direct cause of symptom 1.

### R3 — Tool-round exhaustion ends the turn with an empty answer

The tool loop was hard-capped at `MAX_TOOL_ROUNDS = 5` (pipeline.py:1800,
pre-fix). When a model keeps requesting tools (search → unsatisfying result →
search again), all five rounds are consumed, the loop falls through, and the
final chunk carries `finish_reason: "tool_calls"` with **no content** —
symptom 2. Raising the cap to 20 (as considered) does not fix this; it only
makes the empty turn slower, since every round re-sends the whole
conversation and appends an assistant tool_call message + a tool result.

### R4 — Reasoning leaks to the client as the answer

`reasoning_content` deltas are streamed to the client raw (pipeline.py
~2012-2014). `ThinkSplitter` only re-splits content containing the configured
think tags; the OpenAI-compat reasoning channel is not suppressed for plain
clients. For a reasoning-heavy model, the visible "answer" is the think wall
(symptom 3) and the actual `content` is empty. The transcript shows the user
receiving exactly this wall, ending in a wrong date.

### R5 — Search snippets are stale/unreliable for factual lookups

For "current date and time" the DDG/ddgs chain returned a snippet from a junk
page ("grokipedia.com" claiming 14:05 on Feb 20, 2026) that the model
believed, then contradicted against the recall block date (2026-08-06,
injected from the store), and spiraled into re-search loops. The middleware
feeds snippets verbatim with no freshness judgment; for date/time queries the
correct data source is a clock API (e.g. keyless worldtimeapi.org), not a
search scrape. The injected recall metadata date also poisons the model's
reasoning about "today".

### R6 — No per-round output cap interacts with R2/R3

Even with a cap (the fix in progress, default 8192), a reasoning model can
burn the entire cap inside the think phase. My 1024-token test produced
`finish_reason: "length"` with zero content even after the forced-answer
round — the model spent the budget on thinking and was cut before writing
the answer. A forced answer round must therefore also **disable the think
phase** (`chat_template_kwargs: {"enable_thinking": false}` — verified
working against the running :8080 server, returns a plain answer in ~1 s)
or the forced round still ends empty.

## Changes already made (in progress, uncommitted)

- `config.py`: added `max_completion_tokens` (default 8192) and
  `max_tool_rounds` (default 5), both documented.
- `pipeline.py` streaming path: per-round `max_tokens` cap; tool loop
  converted to a bounded while-loop; when the budget is exhausted or a round
  is truncated at `length` with no content, one forced plain-text round
  (`tool_choice: "none"`) is run so the turn cannot end empty.
- `pipeline.py` non-streaming path: same cap and forced-answer treatment.
- Pending: forced round must set `enable_thinking: false` (R6); a
  reasoning-repetition guard (R2); a clock API backend for date/time queries
  (R5).

## Proposed fixes (priority order)

1. **Forced answer round with thinking disabled** (fixes R3+R6, the
   "send message again" bug): in the forced round send
   `chat_template_kwargs: {"enable_thinking": false}` alongside
   `tool_choice: "none"`, so the model writes content immediately instead of
   re-spending the cap on thinking.
2. **Reasoning repetition guard** (fixes R1/R2 symptom 1): track the
   reasoning stream in the middleware; on N repeated identical
   paragraphs/tokens (e.g. an exact 300-token block seen 3 times), stop the
   round and go straight to the forced no-think answer round.
3. **Clock API for date/time queries** (fixes R5): detect
   current-date/time intent (the existing `force_patterns` machinery) and
   answer from a keyless time API before/without a web search scrape.
4. **Hide reasoning from plain clients** (fixes R4): for clients without
   `reasoning_content` support expectations, suppress the raw channel (or
   only surface it behind a flag) so the wall of thinking is not the visible
   answer.
5. Optional: prompt-side steering — after the first successful `web_search`
   result, tell the model (tool-result footer) not to re-search the same
   intent.

## Verification notes

- Live repro before fixes: same prompt streamed 420 s+ without terminating.
- After cap fix (cap=1024): stream terminates (109 s) but ends empty
  (`length`, no content) — confirming R6.
- `enable_thinking: false` verified working on the live llama-server:
  direct request returned `content: "OK"`, 2 tokens, `finish_reason: "stop"`.
