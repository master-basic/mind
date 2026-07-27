# Cued Recall Architecture

## Overview

Cued Recall is a local AI proxy that sits between an OpenAI-compatible chat client and `llama-server`, providing persistent reasoning memory via a block lifecycle — blocks are created from model responses, shelved for future recall, then either rewritten shorter by a smaller judge model or forgotten by a decay rule.

## Four-Process Topology

```
                     ┌──────────────────────────────────────────────────────────┐
                     │                   Middleware (port 8000)                  │
                     │  FastAPI proxy: /v1/chat/completions, admin GUI, tools   │
                     │  Core: pipeline.py, judge.py, tagger.py                  │
                     └────┬──────────┬──────────────┬───────────────────────────┘
                          │          │              │
                     ┌────▼───┐ ┌───▼──────┐ ┌─────▼────────┐
                     │Reasoning│ │  Judge    │ │  Embedding    │
                     │Port 8080│ │ Port 8081 │ │ Port 8082     │
                     │Main LLM │ │1.5B model│ │nomic-embed-text│
                     │8B model │ │(CPU, no  │ │(CPU, dim=768) │
                     │(GPU)    │ │ VRAM)    │ │               │
                     └─────────┘ └──────────┘ └───────────────┘
```

Four server processes managed by `run.py`:

| Server | Default Model | Port | Hardware |
|---|---|---|---|
| Reasoning | Qwen3-8B (configurable) | 8080 | GPU |
| Judge | Qwen2.5-1.5B-Instruct-Q4_K_M | 8081 | CPU (`-ngl 0`) |
| Embedding | nomic-embed-text-v1.5 | 8082 | CPU |
| Middleware | This project | 8000 | CPU |

---

## Block Lifecycle

The central abstraction is a **Block** — a chunk of reasoning or result text with metadata.

### States

```
                  ┌──────────┐
                  │   hot    │  ← created fresh, not yet recallable
                  └────┬─────┘
                       │ next turn arrives OR idle timeout (15 s)
                       ▼
                  ┌──────────┐
           ┌─────▶│ shelved  │  ← recallable by vector search
           │      └────┬─────┘
           │           │ judge pass (after idle_trigger_s of quiet, min_age: 1 hr)
           │           ▼
           │      ┌───────────┐
           │      │ truncated │  ← text replaced by judge's summary
           │      └─────┬─────┘
           │            │ can be re-judged again later
           │            ▼
           │      ┌───────────┐
           └──────│ truncated │  ← cycle continues
                  └───────────┘

    purged: status flipped and embedding dropped, so the block is
            unrecallable but recoverable; the msgpack file is deleted
            only when purge_deletes_file is set
```

### Transitions

1. **hot → shelved**: `shelve_previous_turn()` runs when the next user message arrives in a conversation. Falls back to `hot_sweep_loop` (every ~8 s, 15 s timeout) for abandoned conversations.

2. **shelved → purged**: `Judge._decay_sweep()` runs first on every pass, on `index.decay_candidates()`. Pure arithmetic, no model call.

3. **shelved → truncated**: `Judge.run_pass()` then walks `index.blocks_due_for_judging()` — least-recently-judged first, `max_per_pass` at a time — and asks the small model to rewrite the ones that qualify.

4. **truncated → re-judged**: Truncated blocks remain recallable and are reconsidered once `rejudge_interval_s` has passed.

### Consolidation vs. decay

The two are deliberately separate, and only one of them uses the model.

**Decay** is arithmetic over facts the index already records, so it costs a query:

| Condition | Result |
|---|---|
| `verification == "corrected"` | Purge at once, at any age — a wrong answer is harmful if recalled again. |
| Never recalled, older than `purge_age_s` | Purge. Retrieval is the only evidence available that a memory is load-bearing. |
| Model found nothing reusable, never recalled, older than `worthless_age_s` | Purge. |

Purging sets `status = purged` and drops the block's row from `block_vec`, which is enough to make it unrecallable and can be undone. The msgpack file survives unless `purge_deletes_file` is set.

**Consolidation** is the model's only job. A block reaches it only if it is not purged, has been recalled fewer than `keep_recall_count` times, is of a type in `consolidate_types` (default: `reasoning` only), and is at least `consolidate_min_tokens` long. The model is asked for one thing — rewrite this shorter — with an empty rewrite meaning "nothing here worth keeping". There is no action label for it to get wrong.

Three guards sit between its answer and the archive:

| Guard | Why |
|---|---|
| `_is_copied_opening` | A 2,091-token block came back as its own first two sentences, verbatim. That is not compression, it is 98% loss, and the size check waves it through *because* it is small. |
| `MIN_SHRINK` (0.8) | A rewrite must be at least 20% smaller to be worth losing the original wording. |
| `original_text` | The pre-rewrite text is kept on first truncation, so a small model's paraphrase is never the only copy. |

Why only `reasoning` blocks: a think trace is mostly scaffolding and compresses ~90% with nothing lost. `result` blocks are the answer the user actually saw and are already dense; `reading` blocks are pasted or fetched source material. Measured against this store, a 1.5B model handed either returns a topic sentence — a 618-token status report came back as *"A project status report detailing core features, bugs fixed, and next steps."* No wording fixed that. Those types are left to decay instead.

### History

The judge previously asked the model for a three-way `keep`/`truncate`/`purge_candidate` verdict while giving it no criteria for any of the three; the criteria lived only in the code that overrode its answer. It returned `keep` 142 times out of 142. Two further faults hid behind that: blocks average 80 tokens, so most had nothing to compress, and a pass took the oldest 50 shelved blocks that `keep` then left shelved — so every pass re-read the same 50, and 345 of 395 blocks had never been judged at all.

---

## Data Flow: One Chat Turn

```
User sends /v1/chat/completions
│
├─ detect_and_apply_correction()
│   Matches the user message against correction_patterns (17 anchored
│   regexes, EN + AZ). If it hits, marks the previous turn's blocks
│   verification="corrected", source="pattern".
│   Runs BEFORE the turn on purpose: it is what stops the block being
│   objected to from being recalled into the objection itself.
│
├─ shelve_previous_turn()
│   Finds blocks from previous turn in same conversation
│   Sets status from hot → shelved (makes them recallable)
│   Fires fire-and-forget Tagger to assign tags + gist
│
├─ recall_blocks()
│   Embeds user message via Embedding server (port 8082)
│   Queries block_vec (sqlite-vec) for top-k cosine similarity matches
│   Filters: skip corrected blocks, skip oversized (> budget_tokens=3000)
│   Returns 0-4 blocks
│
├─ build_recall_injection()
│   Formats recalled blocks into a text prefix with similarity scores
│
├─ build_messages()
│   Prepends recall text to the latest user message in the chat history
│
├─ forward_stream() / forward_nonstream()
│   Sends full message list to reasoning LLM (port 8080)
│   Streams response back to client, parsing <think>...</think> tags
│
├─ _create_blocks()
│   Splits reasoning content at paragraph boundaries (~8000 tokens each)
│   Creates 1 result block for the visible response
│   Optionally creates reading block for long user messages
│   Writes each block: store.put() → .msgpack file
│                      index.upsert_block_meta() → SQLite row
│                      _embed_and_store() → vector in block_vec
│
├─ apply_accepted_verification()
│   Marks accepted only those previous-turn blocks that recall actually
│   SERVED and that were not then objected to. A turn merely happening is
│   not evidence — that version left 309 of 395 blocks claiming "accepted".
│
├─ verify_correction_with_model()   [fire-and-forget, only if no pattern hit]
│   Few-shot yes/no on the small model for the phrasings the patterns miss
│   ("it returns 404 when I try that"). Marks corrected with source="model",
│   which is trusted for recall exclusion but NOT for immediate purging.
│   Never awaited: it is a CPU-bound round trip and applies to the previous
│   turn's blocks anyway.
│
└─ _accumulate_judge_tokens()
    Adds word count of reasoning + result to a running total, and stamps
    last_turn_at. Does not trigger anything itself — consolidate_loop in
    main.py fires a pass once there is new material AND the machine has been
    quiet for idle_trigger_s.
```

---

## Storage Layer

### BlockStore (store.py)

- Location: `{store_path}/blocks/{block_id}.msgpack`
- Serialization: msgpack (binary, compact, no schema enforcement)
- Atomic writes: temp file + `os.replace`
- 442 files currently on disk (some may be orphaned)

### VectorIndex (index.py)

- Location: `{store_path}/index.db`
- Engine: SQLite + sqlite-vec extension
- Tables:
  - `blocks` — metadata: block_id, type, status, created_at, conversation_id, turn_index, token_count, verification, verification_source, recall_count, last_recalled, judged_at, tags, gist
  - `block_vec` — virtual table with float[dim] embedding, cosine distance
- Concurrency: threading.Lock around all writes
- 395 metadata rows, all `shelved`, none `truncated` or `purged`

### WAL (wal.py)

- Location: `{store_path}/wal.jsonl`
- Append-only JSONL audit trail for all lifecycle events
- Events: `turn_completed`, `recall_budget`, `tagged`, `judge_pass`, `judge_action`, `judge_error`, `judge_parse_failed`, `judge_overflow_retry`, `judge_schema_unsupported`, `verification_set`, `verifier_error`, `idle_shelve`, `startup_shelve`, `admin_kv_clear`
- `judge_action` carries the outcome per block: `purge`, `truncate`, `keep_recalled`, `skip_type`, `skip_small`, `worthless_kept`, `summary_was_copied`, `summary_not_shorter`, `no_decision`. Counting these is how you tell "the judge is working and there is nothing to do" apart from "the judge is broken" — the distinction the old all-`keep` log could not make.

---

## Supporting Services

### Tagger (tagger.py)

- Runs **fire-and-forget** at shelve time, not during judge pass
- Uses judge model (or dedicated endpoint) to assign tags + gist
- Tags validated against fixed TAXONOMY vocabulary (73 tags across 8 groups)

### CorrectionVerifier (verifier.py)

- Answers one yes/no question: is this user message reporting that the previous answer was wrong?
- Few-shot, because the bare instruction did not work. Asked in plain prose the model scored 6/14 on a hand-built set — worse than answering "no" every time, and it said yes to ordinary follow-ups. With seven examples it scores 13/14.
- Its one remaining miss reads "now do the same for the firewall" as a complaint, which is why a `source="model"` correction cannot purge a block outright.

### small_model.py

One `asyncio.Semaphore(2)`, shared by the tagger and the verifier. The judge server is CPU-only with a single slot (`-np 1`), so the limit belongs to the server, not to each caller — throttling separately still let several callers pile onto one slot, and a third of all tag calls were timing out.

### EmbeddingClient (embed.py)

- Calls embed server's `/v1/embeddings` endpoint
- Hard cap at 16,000 chars per input
- L2-normalizes vectors for cosine similarity search
- Blocks are embedded at creation time (not at query time)

### Tool System

- `web_search`: 4 backends (DuckDuckGo, Brave, Serper, SearXNG) with fallback chain
- `web_fetch`: SSRF-guarded URL fetcher, HTML-to-text
- Tools forwarded from client (reasoning model has its own tool definitions)
- Force-search heuristics auto-detect search-intent queries

---

## Author's notes on architecture understanding

This document was reverse-engineered from reading every source file in `cued_recall/`. The codebase has no formal architecture document — the closest is the ASCII diagram in `README.md`. Key insights from reading code rather than docs:

- **The block lifecycle** is the backbone, but its three phases (creation, shelving, judging) are spread across `pipeline.py`, `main.py`, and `judge.py` with no single document connecting them.
- **The judge splits forgetting from compressing** — decay is arithmetic over age and recall count and runs without a model call; the model only ever rewrites. An earlier design asked a 1.5B model for a three-way verdict with no criteria attached and got `keep` 142 times out of 142.
- **Token counting is inconsistently** word-count-based throughout (`len(text.split())`), which undercounts compared to the model's actual tokenizer by ~23% on average.
- **Streaming and non-streaming** paths in `pipeline.py` share ~70% of logic but are separate methods with subtle differences.
