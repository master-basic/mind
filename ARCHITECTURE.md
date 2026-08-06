# Cued Recall Architecture

## Overview

Cued Recall is a local AI proxy that sits between an OpenAI-compatible chat client and `llama-server`, providing persistent reasoning memory via a block lifecycle — blocks are created from model responses, shelved for future recall, then either rewritten shorter by a smaller judge model or forgotten by a decay rule.

## Five-Process Topology

```
   run.py (launcher + supervisor)
     │  sizes VRAM, starts the servers, watches for a wedged queue
     ▼
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
                     │9B/35B   │ │(CPU, no  │ │(CPU, dim=768) │
                     │(GPU)    │ │ VRAM)    │ │               │
                     └─────────┘ └──────────┘ └───────────────┘
```

Five server processes managed by `run.py`:

| Server | Default Model | Port | Hardware |
|---|---|---|---|
| Reasoning | Qwen3.5-9B-Q5_K_M (catalog of 6) | 8080 | GPU, context autosized |
| Judge | Qwen2.5-1.5B-Instruct-Q4_K_M | 8081 | CPU (`-ngl 0`) |
| Embedding | nomic-embed-text-v1.5 | 8082 | CPU |
| Speech-to-text | whisper.cpp whisper-server (large-v3-turbo-q8_0, 99 languages) | 8083 | CPU |
| Middleware | This project | 8000 | CPU |

`run.py` is not only a starter. It sizes the reasoning server's context window
and MoE expert split from free VRAM before launch (see **VRAM planning**), keeps
each server's stdout in `logs/{name}.log`, and stays resident afterwards as a
watchdog (see **Staying up**). All three llama servers run `-np 1`: this build
defaults to 4 slots over one shared KV pool, so four slots each advertising the
full window really compete for it — a 16,001-token prompt was refused once other
slots held context, while 60,000 succeeds on a single slot.

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

    merged: a NEW block, derived by the abstraction pass from ≥ 3
            near-identical shelved blocks. type="result",
            conversation_id="" (holds across conversations), parents
            field lists the originals. Its members are retired to purged
            -- the reversible state below -- and the merged block takes
            their place in recall. The file of every original survives,
            even when purge_deletes_file is set: merging is "better said
            elsewhere", and must not share a delete with forgetting.

    purged: status flipped and embedding dropped, so the block is
            unrecallable but recoverable; the msgpack file is deleted
            only when purge_deletes_file is set. POST /admin/blocks/restore
            brings it back and re-embeds it.

    pinned: orthogonal to status, set only by hand. A pinned block is
            skipped by both decay and consolidation at any age, so it
            keeps its own words indefinitely.
```

### Retention guarantees

Three of them, in the order they are checked:

1. **A pinned block is never purged and never rewritten.** Nothing automatic
   can override it; `POST /admin/blocks/{id}/pin` is the only way in or out.
2. **A block that keeps being recalled stays.** Recalls and uncontested
   recalls are converted into days of life (utility), spent against time
   since the block was last *useful* — so a block recalled every week
   outlives one recalled once eighteen months ago, which is exactly what the
   old rule could not say. (That old rule — "recalled once, ever → immortal" —
   made the store only ever grow, and was replaced 2026-08-05.)
3. **Purging is reversible.** It flips a status and drops the search vector;
   the msgpack file survives unless `purge_deletes_file` is set, and restore
   re-embeds from it.

The pin exists because guarantee 2 only covers memories the system has already
needed. A note that will matter next month has no recall count to protect it,
and until there was a pin there was no way for a user to say so.

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
| Pinned | Never purged, at any age, for any reason below. |
| `corrected`, source `manual` | Purge at once. The user pressed the button. |
| `corrected`, source `pattern` | Stop recalling it at once; purge only if never recalled and older than `corrected_grace_s`. |
| `corrected`, source `model` | No special power — must clear the ordinary bar below. |
| Older than `purge_age_s` AND utility at/below `utility_floor` | Purge. Utility = recalls × `utility_recall_weight` + uncontested recalls × `utility_uncontested_weight`, as days of life earned, spent against time since the block was last *recalled* (not written). The age gate comes first: a block cannot be spent down before it is old enough to matter. |
| Model found nothing reusable, never recalled, older than `worthless_age_s` | Purge. |

The old rule — `recall_count > 0` made a block permanently exempt from
age-based purging — could say only "used" and "never used". A block recalled
once eighteen months ago outranked one recalled weekly, and the store could
only grow. `report_decay.py` prints what the next pass would purge under
either rule, writing nothing; run it before a pass on a store you care about.

The three correction sources are weighted apart because they are not equally
reliable, and the expensive direction is a false positive. Measured on a
34-row hand-labelled set (`evaluate/eval_correction.py`), the 17 anchored
patterns score precision 0.87, recall 0.76, and a **false-positive rate of
0.12** — two of seventeen ordinary messages read as complaints. A detector that
wrong should not be able to delete anything, so a pattern hit now hides a
memory for a day rather than destroying it. The verifier arm was never
recorded until 5 Aug 2026: on 38 labelled rows (the set grew from 34) it
scores precision 0.59, recall 1.00, **false-positive rate 0.78** — the 1.5B
answers "yes" even to the prompt's own canonical negative example ("can you
also show the uninstall command?", which the few-shot labels "no"). An
earlier "13/14" in this document no longer describes the live server; the
model-sourced verdicts are treated as the weakest of the three sources,
which is why a model correction cannot purge a block outright.

Purging sets `status = purged` and drops the block's row from `block_vec`, which is enough to make it unrecallable and can be undone. The msgpack file survives unless `purge_deletes_file` is set.

**Consolidation** is the model's only job. A block reaches it only if it is not purged, has been recalled fewer than `keep_recall_count` times, is of a type in `consolidate_types` (default: `reasoning` only), and is at least `consolidate_min_tokens` long. The model is asked for one thing — rewrite this shorter — with an empty rewrite meaning "nothing here worth keeping". There is no action label for it to get wrong.

Three guards sit between its answer and the archive:

| Guard | Why |
|---|---|
| `_is_copied_opening` | A 2,091-token block came back as its own first two sentences, verbatim. That is not compression, it is 98% loss, and the size check waves it through *because* it is small. |
| `MIN_SHRINK` (0.8) | A rewrite must be at least 20% smaller to be worth losing the original wording. |
| `original_text` | The pre-rewrite text is kept on first truncation, so a small model's paraphrase is never the only copy. |

And it terminates. `max_truncate_count` (default 2) caps how many times any
block may be rewritten, ever — each round paraphrases the last, and `MIN_SHRINK`
guarantees there is little left to win after the first two. `max_pass_seconds`
caps the pass itself: a pass that runs out of time stops without marking the
remainder judged, so `judged_at` ordering makes the next one resume where the
clock cut it off rather than start over. Neither bound depends on the shrink
arithmetic happening to converge.

Why only `reasoning` blocks: a think trace is mostly scaffolding and compresses ~90% with nothing lost. `result` blocks are the answer the user actually saw and are already dense; `reading` blocks are pasted or fetched source material. Measured against this store, a 1.5B model handed either returns a topic sentence — a 618-token status report came back as *"A project status report detailing core features, bugs fixed, and next steps."* No wording fixed that. Those types are left to decay instead.

### The abstraction pass (merging)

The store records every episode and never reduces many into one — a question
asked three times leaves three near-identical blocks, each spending judge
budget and prompt tokens to say the same thing. `Judge._merge_pass`, the only
pass that *creates* a memory rather than editing one, closes that:

1. `index.merge_candidates` returns seeds older than `merge_min_age_s` (a
   week: a cluster that formed in the last hour is a conversation in
   progress, not a settled repetition) of the judge's `consolidate_types`.
2. Each seed is queried against the vector index at `merge_cluster_sim`
   (0.90 — well above `recall.threshold` 0.48: that one asks "is this
   relevant", this one asks "is this the same thing said twice"). Paraphrases
   do not reach it; near-identical ground does.
3. A cluster of `merge_min_cluster` (3) or more eligible members — the same
   eligibility test as the seeds, re-applied to members because the vector
   search filters on status alone, so a pinned, corrected or still-warm block
   could otherwise be pulled in and retired behind a merge it was never
   eligible for — is sent to the judge to write one note that holds across
   them.
4. The draft is refused unless it keeps every specific. `_lost_specifics`
   pools numbers, paths and dotted identifiers across the members and
   compares against the draft; the first real merge ever produced,
   "dns.cache_ttl=300 … reduces the cache TTL from 30 seconds to 60ms",
   conflated two quantities and lost the 840ms — so asking the model to keep
   them is not enough, it is now *checked*. A refused cluster writes a
   `merge_rejected` WAL event and leaves everything untouched.
5. A draft that passes is stored as a new block: `type="result"`,
   `conversation_id=""` (it holds across the conversations it came from),
   `parents` linking the members. The originals are retired to `purged` —
   status flipped, vector dropped, **file always kept, even when
   `purge_deletes_file` is on** — so a bad generalization can be undone
   block by block.

The guard is deliberately asymmetric. A refusal costs nothing (the originals
stay shelved); an acceptance retires evidence behind a generalization that was
never true. Measured live against a snapshot copy (`evaluate/eval_merge.py`,
2026-08-05): a genuine nginx-timeout family merged to a correct 18-token note
that kept every specific and fired recall for a related-but-new probe, and a
DNS-latency family whose draft dropped `dns.cache_ttl` was refused. That
measurement is why `judge.merge_enabled` defaults to `true`; the flag still
switches the pass off entirely.

`merge_max_per_pass` bounds how many merges one pass performs (each retires
several blocks), and a block retired into a merge is never eligible again in
the same pass.

### History

The judge previously asked the model for a three-way `keep`/`truncate`/`purge_candidate` verdict while giving it no criteria for any of the three; the criteria lived only in the code that overrode its answer. It returned `keep` 142 times out of 142. Two further faults hid behind that: blocks average 80 tokens, so most had nothing to compress, and a pass took the oldest 50 shelved blocks that `keep` then left shelved — so every pass re-read the same 50, and 345 of 395 blocks had never been judged at all.

---

## Data Flow: One Chat Turn

```
User sends /v1/chat/completions
│
├─ mode == "passthrough"?  →  forwarded verbatim to the reasoning server, and
│   nothing below this line runs. The branch sits at the very top because
│   everything under it -- the correction scan, the shelve, process_turn -- is
│   memory work. Toggled from Live → Mode in the admin page, persisted in
│   mode.json in the store. For coding agents: their prompt is already what it
│   should be, and augmenting it degrades the code that comes back.
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
│   Filters: skip corrected blocks (a span-corrected block may still
│   enter if verifier.spans is on, with the reported span redacted from
│   both the judge's note and the injection), skip oversized
│   (> budget_tokens=3000)
│   Then, if recall.judge_enabled: _filter_by_relevance() asks the small
│   model whether each survivor actually applies to THIS question, in
│   parallel but through the shared small-model semaphore. Fail-open.
│   The judge is shown the note recall.judge_note names -- default
│   "question", the block's originating question, which measured as the
│   one thing that refused every trap family member (see Evaluation).
│   Candidates are ranked by the judge's P(yes) and the budget is spent
│   down that ranked list (Phase 2: score, rank, fill), not in cosine
│   order. Pins break ties (recall.pin_priority).
│   If the embed server errors, the query falls back to keyword_query
│   over gist/tags (recall.tag_channel) instead of recalling nothing.
│   Returns 0-4 blocks. These, and only these, are the blocks "served" for
│   the turn: recall_count is incremented for exactly this set, and it is
│   what apply_accepted_verification reads back next turn.
│
├─ build_recall_injection()
│   Formats recalled blocks into a text prefix with similarity scores
│
├─ build_messages()
│   Prepends recall text to the latest user message in the chat history.
│   Not to the system message: recall changes every turn, so putting it at
│   the front invalidates llama.cpp's whole KV prefix behind it. See
│   "Prompt assembly and the KV prefix" below.
│
├─ _fit_messages()
│   Trims history to the prompt budget, keeping the system message and the
│   newest user turn. Estimates first, recounts on the server when the
│   payload is big enough in bytes to possibly exceed the window.
│
├─ forward_stream() / forward_nonstream()
│   Sends full message list to reasoning LLM (port 8080)
│   Streams response back to client, parsing <think>...</think> tags
│   Reports usage (prompt/completion tokens) back to the client, so an agent
│   client can decide when to compact, and to ctx_usage for the admin page.
│   On a context-overflow 400, re-fits from the server's own n_prompt_tokens
│   and retries once (wal: context_overflow_retry).
│
├─ chat_sink()
│   Appends the exchange to chats.db, the plain transcript store the history
│   sidebar reads. Independent of the block lifecycle.
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
│   SERVED — the set recorded by _remember_recalled above, i.e. what
│   survived the budget filter and the relevance judge and was put in
│   front of the model — and that were not then objected to. A turn merely
│   happening is not evidence; that version left 309 of 395 blocks
│   claiming "accepted".
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

## Prompt assembly and the KV prefix

Where the recall note goes is a performance decision, not a formatting one.
Recalled blocks are retrieved per query, so they differ from turn to turn.
Merging them into the front of the system message changed the first tokens of
every request and invalidated llama.cpp's entire cached prefix behind them:
measured on this stack, 43,345-token prompts re-prefilled from scratch at
`f_keep 0.12` — 107 s of prompt processing to produce a reply that decoded in
1.4 s.

`build_messages` anchors the note to the newest real user message instead
(`_newest_user_index`, which skips a turn whose whole content is a
`<tool_response>`, the same test the chat template applies). The system prompt
and all prior history stay byte-identical between turns, so only the recall note
and the new message are prefilled. It cannot be a second system message: Qwen3.5's
template raises *"System message must be at the beginning."* for any later one,
which llama.cpp returns as a 400 — that is what used to break agent CLIs that
ship their own system prompt.

### Counting tokens

Three different jobs, three different methods, all in `utils.py`:

| Where | Method | Why |
|---|---|---|
| Block `token_count` | `count_tokens` → the server's `/tokenize`, estimator on failure | Word counts measured 36 of 36 sampled blocks low (mean 0.77×, worst 0.52× on code). The admin `tokens` column read 42% under, and since the same field enforces `recall.budget_tokens`, a 3,000-token budget was really spending ~3,900. |
| Prompt budget, normal case | `estimate_tokens` — the largest of chars/3.2, words×1.3, bytes/3.0 | Cheap. Every tie goes to the larger number: over-counting trims some history, under-counting hands the server a prompt bigger than its window and the reply is cut off mid-JSON. |
| Prompt budget, near the limit | `_exact_prompt_tokens` → `/tokenize` | ~450 ms flat, so only worth it close to the ceiling. |

The trigger for the exact recount is a fact, not a guess. The estimator cannot
be trusted to decide when it is itself untrustworthy — it runs 24% low on
Azerbaijani and 55% low on base64, and an under-count both inflates the prompt
and hides that it did. Every token costs at least one byte, so a payload smaller
than the budget *in bytes* provably cannot exceed it in tokens; anything larger
is measured properly. `exact_count_threshold` remains as a second trigger for a
payload that is compact but dense.

This is why the estimator's error on non-English text cannot overflow the
window, and it is a proof rather than a hope — but only if the byte count
covers everything the template will render. Tool schemas are charged to the
budget through `_payload_overhead_tokens`, which is itself an estimate, so
leaving them out of the byte side compared a measurement against a guess: a
client shipping a dozen JSON-Schema tool definitions could clear the proof on
the strength of it. `_tools_bytes` closes that.

Every turn writes a `prompt_budget` WAL record — `count_method`, `counted_tokens`,
`limit`, `payload_bytes`, `byte_proof` — so which path decided a given prompt is
readable afterwards instead of being asserted here.

Byte-based estimation exists for the same reason: characters per token range from
2.54 (Azerbaijani) to 4.50 (English prose) against this tokenizer, while bytes per
token hold between 3.11 and 4.50 — the extra UTF-8 bytes of `ə/ş/ğ/ı/ö/ü/ç` are
exactly what the tokenizer is paying for.

Trimming also goes *under* the budget by a margin rather than dropping the
minimum that fits. Any trim rewrites every token after the system prompt and
costs a full re-prefill; going under by a margin buys several turns of identical
prefix from that one re-prefill. The history given up early is exactly what the
recall store exists to bring back.

---

## VRAM planning (run.py)

One pool of memory, so one decision, made before any server starts:

1. Read the GGUF metadata for KV geometry (`kv_bytes_per_token`) and, for an
   MoE, the per-layer expert tensor sizes.
2. Charge what must be VRAM-resident: weights, the embedding model, a CUDA
   context per GPU process, a safety fraction, and a transient-buffer reserve.
   For an MoE under `--cpu-moe`, only the non-expert tensors count — a 19.9 GB
   file needs a fraction of that resident, which is what lets a 35B-A3B run on a
   12 GB card.
3. Whatever is left is the KV budget → context size, rounded to 4096, capped at
   the model's trained context and at `MAX_AUTO_CTX` (so a full window stays a
   few minutes of prefill, not tens).
4. Spend the leftover on experts. Once the window is fixed, the KV cache has a
   fixed size and any remaining budget sits idle. `_plan_expert_split` picks the
   smallest `--n-cpu-moe N` whose GPU-side tail still fits — generation is
   bandwidth-bound on expert reads, so moving them from PCIe to VRAM is what the
   spare memory can buy.

Every unknown resolves toward the known-good default: no GPU, unreadable
metadata, a computed context under the floor, or another instance already
holding the port all fall back to the hardcoded `--ctx-size` rather than a guess.
`--reasoning-ctx N` and `--reasoning-n-cpu-moe N` pin either decision.

Both numbers are estimates — VRAM can be taken between measuring and loading —
so a reasoning server that fails to come up is retried once with ground given:
the expert split first (it frees gigabytes of weight, where halving the window
frees only KV), then the context halved. On recovery the middleware's prompt
budget is rewritten from the context that actually loaded.

`run.py` then writes what it resolved into `run_settings.txt`
(`MODEL_<n>_NAME`, `MODEL_<n>_CTX`, `JUDGE_NAME`, `EMBED_NAME`), because
`run.bat` asks "reuse these settings?" before Python starts and otherwise has
only a choice number to show. Context is keyed per model: autosizing puts a 9B
dense and a 35B MoE in different places, so a single figure would go stale the
moment the model changed.

---

## Staying up: the wedge watchdog

A llama.cpp slot can be lost while the process stays alive. The HTTP threads
keep answering, but everything routed through the inference queue blocks forever
at 0% CPU. `/health` and `/props` are served without touching that queue, so they
stay fast — that split *is* the signature, and it is what makes the failure
detectable at all.

`ServerSupervisor` (in `run.py`) probes every 15 s: `/health` must answer, then
`/slots` must answer within 10 s (healthy, it answers in 0.03–0.67 s even under
load; a 501 from a server without slot support still counts as a reply). Three
consecutive strikes — about 45 s of blocked queue — and it restarts that server.
The bar is deliberately high because a restart aborts whatever is generating.

The restart belongs to the launcher, not the middleware: it owns the process
handles, and a restart from anywhere else orphans a multi-GB process when the
launcher exits. So `/admin/server/restart` writes a name into the file named by
`CUED_RECALL_RESTART_FILE` and the supervisor picks it up on its next tick;
without a launcher the endpoint returns 503 rather than pretending it worked.
Clearing the KV cache is not an alternative — `/admin/kv/clear` reaches the
server through `/slots`, which is exactly what a wedge blocks.

Related: server stdout goes to a file, not a `PIPE`. Nothing ever read that
pipe, so once the OS buffer filled (a few hundred requests) llama-server's next
write blocked forever and froze the server with no log to show for it. A file
also means there is a server-side log to read when a slot does wedge.

---

## Storage Layer

### BlockStore (store.py)

- Location: `{store_path}/blocks/{block_id}.msgpack`
- Serialization: msgpack (binary, compact, no schema enforcement)
- Atomic writes: temp file + `os.replace`

### What a block is: three texts

A block carries three different strings, and mixing them up has cost this
project twice:

- `text` — the block's own words: the think trace or the answer as stored.
- `embed_text` — the text the block is *retrieved* by. Written on every
  block; `config.embed_source` (`composite` by default) picks what feeds the
  vector index — the classic question+answer composite, or the block's own
  content. Measured on this corpus (evaluate/eval_retrieval.py), swapping to
  content lowers trap similarity from 0.756 to 0.698 but every other family
  by about as much, so exact-vs-trap separation gets *worse*: the claimed
  mechanism is not there, and `embed_source` stays `composite`. Both texts
  are always stored, so a larger corpus can reverse that with a re-embed
  (backfill) rather than a migration. The widened corpus (53 rows, 5 Aug
  2026) refutes it again: in content mode exact-vs-trap separation narrows
  from 0.140 to 0.110 and crosslingual's mean top-sim (0.594) falls *below*
  the trap family's (0.698), so content embedding can no longer even recall
  the crosslingual family without also firing the traps.
- `stimulus_text` — the "question" side of the block: the user message it
  was written to answer (for a reasoning block) or its own text (for a
  result/reading block). This is where `judge_note_text` recovers the
  originating question for the relevance judge, and it is what a
  merge-generalized block falls back to (its `question_text` is empty).

### VectorIndex (index.py)

- Location: `{store_path}/index.db`
- Engine: SQLite + sqlite-vec extension
- Tables:
  - `blocks` — metadata: block_id, type, status, created_at, conversation_id, turn_index, token_count, verification, verification_source, recall_count, last_recalled, judged_at, tags, gist
  - `block_vec` — virtual table with float[dim] embedding, cosine distance
  - `turn_recalls` — which blocks were served into which turn, recorded at
    recall time and *consumed* on the next turn's verification, so the
    uncontested-recall evidence survives a restart (Phase 3.3)
- Concurrency: threading.Lock around all writes
- Analytics queries live here too, not in the router: `growth_by_day`,
  `token_histogram`, `recall_effectiveness` — they are what the admin Memory tab
  draws.

### ChatStore (chats.py)

- Location: `{store_path}/chats.db` (SQLite, WAL mode) — `conversations` + `messages`
- Plain transcripts, appended per turn, so a conversation can be picked up later
  from the chat UI's history sidebar
- Deliberately never touched by the block lifecycle. Blocks are the memory
  system's own representation — split, tagged, summarised, eventually purged —
  and a poor source for replaying a conversation: the user's message survives
  only inside a reasoning block's stimulus, truncated to 512 words and glued to
  the reply. Deleting a transcript therefore keeps its blocks; forgetting the
  chat is not the same as forgetting what was learned in it.
- Included in snapshots and restored alongside the store and index

### WAL (wal.py)

- Location: `{store_path}/wal.jsonl`
- Append-only JSONL audit trail for all lifecycle events
- Turn/recall: `turn_completed`, `prompt_budget`, `recall_budget`, `recall_embed_error`, `recall_judge_error`, `context_overflow_retry`, `upstream_error`, `upstream_transport_error`, `embed_store_error`, `web_search_error`, `chat_record_error`
- Lifecycle: `tagged`, `tagger_error`, `verification_set`, `verifier_error`, `idle_shelve`, `startup_shelve`
- Judge: `judge_pass`, `judge_action`, `judge_error`, `judge_parse_failed`, `judge_overflow_retry`, `judge_schema_unsupported`
- Merge: `blocks_merged` (with parents and before/after token counts), `merge_rejected` (reason and the lost specifics), `block_retired_into_merge`, `merge_abandoned`
- Admin: `admin_verify`, `admin_pin`, `admin_restore`, `admin_delete_blocks`, `admin_import`, `admin_kv_clear`, `admin_server_restart`

`judge_pass` carries the counters a pass produced — `elapsed_s`, `visited`,
`model_calls`, `purged`, `truncated`, `stopped_early`. Most visits cost two DB
reads and no generation (a rewrite needs type `reasoning`, ≥ `consolidate_min_tokens`,
and fewer than `keep_recall_count` recalls), so `model_calls` against `visited`
is what separates "the judge keeps up with the store" from "the judge is falling
behind it", and a run of `stopped_early` says the store has outgrown
`max_pass_seconds`.
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
- Size guard is the server's own window: reads `n_ctx` from `/props` at
  startup (nomic-embed runs 2,048) and `fit()` trims by characters against a
  conservative token estimate with a 10% margin. The old 16,000-character cap
  was chosen for an 8,192-token window and sat four times above the real
  limit, so oversized inputs reached the server, came back HTTP 400, and were
  swallowed by `_embed_and_store`'s except — leaving blocks stored, listed,
  and permanently unrecallable. A failed embed is logged (`embed_store_error`)
  and repairable via `backfill_missing_vectors.py`.
- L2-normalizes vectors for cosine similarity search
- Blocks are embedded at creation time (not at query time)

### Tool System

- `web_search`: 4 backends (DuckDuckGo, Brave, Serper, SearXNG) with fallback chain
- `web_fetch`: SSRF-guarded URL fetcher, HTML-to-text
- Tools forwarded from client (reasoning model has its own tool definitions)
- Force-search heuristics auto-detect search-intent queries

### sysinfo.py

Host GPU/CPU telemetry for the admin page, via `nvidia-smi` and `psutil`.
Deliberately standalone — it imports nothing from the pipeline, index, or store,
so a probing failure can never reach the chat path. Every function is total:
it returns empty on failure rather than raising, because the page polls it every
5 s and one bad poll must not take the page down. `nvidia-smi` results are cached
for 2 s so a user mashing Refresh cannot stack calls, and on Windows it is spawned
with `CREATE_NO_WINDOW` so each poll does not pop a console.

---

## Admin surface

Three tabs, because a single page polled everything at once whether or not it
was on screen:

| Tab | Shows | Endpoints |
|---|---|---|
| Live | Memory/passthrough switch, per-server context usage, GPU/system telemetry, uptime, block stats, throughput | `/admin/mode`, `/admin/models`, `/admin/system`, `/admin/stats`, `/admin/tps`, `/admin/wedge` |
| Memory | Memory health, per-turn recall budget decisions, token distribution, store growth, most-recalled blocks | `/admin/stats/budget`, `/admin/stats/distribution`, `/admin/stats/growth`, `/admin/stats/recall` |
| Blocks | Paged, filterable block table; pin toggle per row, bulk restore and delete | `/admin/blocks`, `/admin/blocks/{id}`, `/admin/blocks/{id}/pin`, `/admin/blocks/restore`, `/admin/blocks/delete` |

Two details worth keeping:

- **Refresh cadence follows the clock that changes the data.** A block's status
  flips hot → shelved `hot_shelve_timeout_s` after its last message, so `/health`
  returns that value and the Blocks table refreshes on it rather than on a
  duplicated constant.
- **Prefill and decode are reported apart.** The middleware's own ring times
  whole requests, which conflates the two: a reply that spends 107 s prefilling
  and 1.4 s generating reads as slow generation and makes a healthy model look
  broken. `rates_from_metrics` reads llama.cpp's counters instead, as
  token-weighted lifetime averages, so a burst of tiny requests cannot drag them
  around. No traffic yet yields `null`, which the page renders as a dash and an
  explanation rather than a confident zero.

---

## Evaluation (evaluate/)

Split in two on purpose, because conflating them produces numbers that mean
nothing:

| Harness | Question | Cost |
|---|---|---|
| `eval_retrieval.py` | Does the right block come back? | Seconds, deterministic, no generation. Sweeps `recall.threshold` from 0.30 to 0.94 and writes `retrieval_sweep.csv`. `--judge` adds the relevance filter and prints both arms side by side. `--fake` self-tests with synthetic vectors and no servers. |
| `eval_correction.py` | How often does correction detection fire on something that was not a correction? | Instant with `--no-model`. Scores the shipped patterns and the shipped classifier against a labelled set; `--from-chats` mines candidate rows out of a live `chats.db` for labelling. |
| `eval_throughput.py` | What does the memory layer cost per turn? | Minutes. Same prompts direct to `:8080` and through `:8000`, streamed, reporting TTFT and decode rate apart — the middleware's work is all pre-first-token, so decode should be flat and TTFT carries the cost. |
| `eval_e2e.py` | Does having the block actually help? | Slow and noisy. A/B: baseline goes straight to `:8080`, treatment through the middleware, store wiped and re-warmed per repeat, `temperature: 0`, fixed seed, `--repeats 3`. |
| `eval_merge.py` | Is the abstraction pass safe to run? | Copies a snapshot store, seeds near-duplicate families, runs the real `_merge_pass` against the copy, and prints the `blocks_merged` / `merge_rejected` events plus a recall probe. Never touches a live store. |
| `eval_judge_notes.py` | What is the judge shown, and what does that cost? | Scores the shipped note (the block's own text) against candidate rewordings on the corpus pairs, isolating the false-fire defect the sweep cannot see. |

Both evals import the prompt the middleware actually sends
(`utils.relevance_prompt`, `CorrectionVerifier._prompt`) rather than a copy. A
harness that scores its own private wording measures nothing about the system,
and a copy drifts the first time either prompt is tuned.

`analyse.py` pairs each probe against itself across arms and bootstraps a CI —
between-prompt variance is enormous (some questions produce 800 tokens of
reasoning, others 12,000), so comparing group means across 20 prompts would
drown any real effect.

The corpus (`corpus.jsonl`) is the actual work. Every family carries adversarial
members across nine relation types: `exact` and `paraphrase` must recall,
`crosslingual` (Azerbaijani) must recall, `trap` (same vocabulary, different
answer — phase 1 vs phase 2) *should* fire but must not be blindly reused,
`trap-asym` (same entities, inverted direction — the stork/baby class, added
5 Aug 2026) likewise, `tag-same` (no shared wording, taxonomy tags overlap)
must recall via the gist/tag channel, and `distractor` (high lexical overlap,
unrelated), `tag-diff` (tags overlap, content differs) and `control` must not
fire. The false-fire rate is the number this exists to measure; trap-family
answers are hand-graded, since no script can catch a model anchoring
confidently on a recalled block that did not apply.

What the sweep shows: recall and false-fire trade against each other with no
clean separation — at the shipped threshold of 0.62, recall is 0.96 and the
false-fire rate is 0.55; false fires only reach zero around 0.86, where recall
has already fallen to 0.58.

That is what `recall.judge_enabled` is for. The first measured version of the
sweep showed the judge taking the false-fire rate to **0.00 at every
threshold** — and that number was an artefact of the harness, which showed the
judge each seed's *prompt* as the note rather than a real block. Shown a real
block's own text (as the shipped judge then read), it kept 5 of 6 traps, and
false-fire at the shipped operating point is 0.64. The judge could say "does
not apply" to a phase-1 note against a phase-2 question only when it was shown
something that did not contain the answer material.

The fix is `recall.judge_note` (default `question`): the judge now reads the
**question that produced the block**, recovered from `stimulus_text`, falling
back to the block's text when no question exists. Measured end to end on the
real representation: false-fire drops 0.64 → **0.09**, traps refused 6/6, and
every legitimate recall — exact, paraphrase, crosslingual — survives. The
residual false fires are the risk stated plainly: a block whose originating
question differed but whose content happens to answer the new question is now
refused; `recall.judge_note: text` restores the old behaviour.

The widened corpus (53 rows, 5 Aug 2026) sharpens the picture. At the 0.62
operating point with the question-note judge, false-fire is **0.00** and
recall 0.61 — but the asymmetric trap class (`trap-asym`: same entities,
direction inverted, mean top-sim 0.866 — the stork/baby class) survives the
judge: the 1.5B refuses 4 of 6, and the 2 that leak are the ones where
"about" cannot tell the direction ("the restart command is wrong…" against a
phase-1 note about the same service; a phase-3 "add OCR" question against an
OCR-era note). The old trap family still refuses 6/6, and `tag-diff`
probes (tags overlap, content differs) false-fire 0/3; `tag-same` recalls
3/3, two of them only because the gist/tag keyword channel added the right
block when embedding could not (top-sim 0.566/0.524).

The more useful consequence: with false fires handled by the second stage, the
embedding threshold no longer has to suppress them, and at 0.48 the crosslingual
(Azerbaijani) family recalls 6 of 6 against 3 of 6 at the 0.70 the embedding
would otherwise need.

So both defaults moved together, and they only make sense together:
`judge_enabled` on, `threshold` down to 0.48. Turning the judge off without
raising the threshold back toward 0.62 leaves the worst of both — a low bar and
nothing checking what clears it. See `evaluate/benchmark.md` for the full table
and the trap-family caveat.

It is not free, and the price is latency rather than tokens. Measured on live
traffic (`evaluate/throughput.md`), a turn with 3–4 candidates spends
**1.5–2.2 s** on judge calls before the reasoning model is asked anything —
3–4 requests to a CPU 1.5B, two at a time through the shared slot semaphore.
A turn with no candidates pays none of it. The cost falls entirely on
time-to-first-token; decode is unaffected, because the judge has finished
before generation starts.

The obvious inefficiency is that the cost is identical whether the judge admits
everything or nothing, and on an off-topic question it is always nothing —
three CPU calls to conclude the store has nothing to say. `recall.floor`
exists to skip the judge below a cosine floor, but it ships **off (0.0)**:
the measured corpus has no safe value anywhere (control family mean top-sim
0.4996 sits against crosslingual's 0.6408, with n=6 per family), and the
plan's proposed 0.30 cannot fire at all — it sits below the 0.48 retrieval
threshold, so no candidate ever has a best similarity under it. The widened
corpus (53 rows, 5 Aug 2026) confirms it with a second, stronger number: the
new `trap-asym` family's mean top-sim is 0.866, and crosslingual's composite
top-sim is 0.841, so a floor high enough to skip the judge for the traps
would strand a legitimate family. The mechanism
is exercised by tests at 0.60, and 0.60 remains the candidate to revisit if
a third family ever lands between the control and crosslingual means.

---

## Notes on the shape of the codebase

Five things about this code are worth knowing before changing it, because none of
them are visible from any single file:

- **The block lifecycle** is the backbone, but its three phases (creation, shelving, judging) are spread across `pipeline.py`, `main.py`, and `judge.py`, and no one file shows the whole path.
- **The judge splits forgetting from compressing** — decay is arithmetic over age and recall count and runs without a model call; the model only ever rewrites. An earlier design asked a 1.5B model for a three-way verdict with no criteria attached and got `keep` 142 times out of 142.
- **Token counting** was word-count-based throughout (`len(text.split())`) and ran well under the model's own tokenizer. Block counts and the prompt budget now use `/tokenize` where it matters; see **Prompt assembly and the KV prefix** for which path uses which method and why.
- **Streaming and non-streaming** paths in `pipeline.py` share ~70% of logic but are separate methods with subtle differences.
- **Placement decisions live in the launcher, not the config.** Context size, the MoE expert split and their fallbacks are computed in `run.py` from GGUF metadata and free VRAM, then written into `config.yaml` on every launch — so reading `config.yaml` alone will not tell you why a window is the size it is.
