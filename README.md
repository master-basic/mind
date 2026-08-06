# Cued Recall Memory Middleware

A local AI proxy with semantic memory. Sits between any OpenAI-compatible chat client and `llama-server`, providing persistent reasoning memory, web search, tool execution, and block lifecycle management.

---

## Architecture

```
Client ──▶ Cued Recall Middleware ──▶ llama-server (reasoning)
                          │
                    ┌─────┴──────┐
                    │  Block Store │  msgpack + SQLite + sqlite-vec
                    └─────────────┘
                          │
                    ┌─────┴──────┐
                    │ Judge Model │  consolidation (rewrite) + correction check
                    └─────────────┘
                          │
                    ┌─────┴──────┐
                    │    Decay    │  forgetting, by age and recall count
                    └─────────────┘
                          │
                    ┌─────┴──────┐
                    │  Embedding  │  semantic retrieval keys
                    └─────────────┘
                          │
                    ┌─────┴──────┐
                    │ Transcripts │  plain chat history, never touched by decay
                    └─────────────┘
```

Four server processes managed by the launcher:
- **Reasoning** — main LLM (Qwen3.5-9B by default, catalog of 6 role models: fast assistant, vision+voice, fast thinker, MoE abliterated, coding)
- **Judge** — rewrites long think traces; also tags blocks and classifies corrections (Qwen2.5-1.5B, CPU-only)
- **Embedding** — vector retrieval keys (nomic-embed-text-v1.5)
- **Middleware** — FastAPI proxy (this project)

`run.py` sizes the reasoning server's context window and MoE expert split from
free VRAM and the model's own GGUF metadata before launch, then stays resident
as a watchdog: a llama.cpp slot can be lost while the process keeps answering
`/health`, and only the launcher owns the handles needed to restart it.

How it got from an idea to this, including what broke on the way:
**[CHANGELOG.md](CHANGELOG.md)**.

Memory upkeep is split in two, deliberately. **Consolidation** — turning a long
derivation into a short one — needs to understand the text, so the judge model
does it. **Forgetting** is a function of age, recall count and whether the
answer was corrected, all of which are recorded exactly, so it is arithmetic
and runs without a model call. See **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Features & status

**Working** means implemented and used in daily runs. **Measured** means there
is a number in [evaluate/benchmark.md](evaluate/benchmark.md) behind the claim.
They are different words on purpose: most of this table is the first and not the
second.

| Feature | Status | Details |
|---|---|---|
| OpenAI-compatible chat proxy | Working | Streaming + non-streaming, `/v1/chat/completions`, `/v1/models` |
| Reasoning/result split | Working | ` thinking` / ` response` tag parsing during streaming |
| Semantic recall | Working, measured | Cosine retrieval plus a relevance filter. Embedding alone at 0.62 recalls 0.96 with a **false-fire rate of 0.55**; that is why it no longer runs alone. End to end it fires 6/6 on exact, paraphrase and Azerbaijani probes |
| Vector backfill | Working | Blocks are embedded once, at creation, and a failure there was logged and never retried -- so a block written while the embedding server was busy stayed `shelved`, kept its text, and was invisible to recall forever. 57 of 1,812 blocks in one real store were in that state. `backfill_missing_vectors.py` finds and re-embeds them |
| Semantic reranker | Working, measured, **on by default** | Second stage asks the small model whether a candidate applies — and reads the **question that produced the block**, not the block's own words: shown the answer text the judge kept 5 of 6 traps (the false-fire **0.00** originally recorded was a harness artefact that fed it the seed prompt); shown the question it refuses all 6, with every legitimate recall surviving. See `recall.judge_note`. Disable with `recall.judge_enabled: false` — and raise `threshold` back toward 0.62 if you do |
| Manual retention (pin) | Working | A pinned block is never purged and never rewritten, at any age |
| Restore purged blocks | Working | Purging was always reversible on disk; `POST /admin/blocks/restore` re-embeds and brings it back |
| KV-prefix-safe injection | Working | Recall is anchored to the newest user turn, so the cached prefix survives the turn |
| Exact token accounting | Working | Block counts and near-limit prompt budgets use the model's `/tokenize`; conservative estimator otherwise |
| Block lifecycle | Working | hot → shelved → truncated/purged; passes sweep forward, oldest-unjudged first |
| Tagging system | Working | Auto-tags blocks with gist + tags at shelve time |
| Correction detection | Working, measured | 17 anchored patterns (EN + AZ) plus a few-shot classifier. Patterns: precision 0.87, recall 0.76, **false-positive rate 0.12** on 34 hand-labelled rows; live verifier (5 Aug, 38 rows): precision 0.59, **FPR 0.78** — the 1.5B says "yes" even to its own few-shot negative ("can you also show the uninstall command?") |
| Span-level corrections | Working, off by default | `verifier.spans: true` — when a correction fires, the verifier also quotes the exact phrase from the answer that is wrong, and recall redacts that span from the block instead of suppressing the whole block. Off: the yes/no classifier is the measured one. Live measurement (5 Aug, 38 rows): span-mode on the 1.5B produces bare `yes` with empty span — the span machinery never engages on live output; the 4.2 fixture measures 0/3 pass |
| Decay | Working, measured | **Utility decay**, not immortality: recalls and uncontested recalls earn days of life, spent against time since the block was last useful; a block recalled once eighteen months ago no longer outranks one recalled weekly. Purge needs the utility floor hit *and* the age gate. No model call. Reversible by default. See the retention guarantees below |
| Merge (abstraction pass) | Working, measured, **on by default** | Derives one gist block from ≥ 3 near-identical older blocks (cosine ≥ 0.90), links it to its `parents`, and retires the originals reversibly. A merge must keep every number, path and identifier — a draft that drops one is refused (`merge_rejected`), which has now caught real losses twice. `judge.merge_enabled: false` turns the pass off. `evaluate/eval_merge.py` is the repeatable live measurement |
| Consolidation | Working | Judge rewrites long think traces only; guarded against copied openings and non-shrinking rewrites; capped at `max_truncate_count` rewrites per block |
| Bounded judge pass | Working, measured | Wall-clock ceiling per pass; a pass that runs out of time resumes where it stopped. Measured: 163 blocks visited in 7.3 s, 1 model call |
| Idle consolidation | Working | Passes run after a quiet period, not mid-conversation |
| Web search | Working | 4 backends: DuckDuckGo (free), Brave, Serper, SearXNG |
| Web fetch | Working | SSRF-guarded, HTML-to-text, JSON detection |
| Tool calling | Working | Own tools (web_search, web_fetch) + client tool forwarding |
| Model catalog | Working | 6 role models with interactive menu; launch details remembered per model |
| VRAM autosizing | Working | Context window sized from free VRAM + GGUF KV geometry; leftover VRAM spent on MoE expert layers (`--n-cpu-moe`) |
| MoE support | Working | 17–20 GB A3B models on a 12 GB card, experts in system RAM |
| Wedge watchdog | Working | Detects a llama.cpp server whose `/health` answers but whose inference queue is blocked, and restarts it |
| Server logs | Working | Each llama-server's stdout in `logs/{name}.log` |
| Built-in chat UI | Working | Streaming, reasoning pane, file upload, image attach, voice recording (STT), Stop button |
| Chat history | Working | Durable transcripts in `chats.db` with a sidebar; independent of the block lifecycle |
| Admin web GUI | Working | Tabbed: Live (context, GPU, throughput), Memory (analytics), Blocks (table) |
| GPU/system telemetry | Working | Per-GPU and per-process usage, uptime, launch plan |
| Memory analytics | Working | Store growth, token distribution, recall effectiveness, per-turn budget decisions |
| Throughput reporting | Working | Prefill and decode speeds reported separately from llama.cpp's own counters |
| Usage reporting | Working | `prompt_tokens`/`completion_tokens` returned to the client so agents can compact |
| Export/import blocks | Working | JSON export/import with re-embedding |
| Snapshots | Working | Periodic + on-shutdown, persistent across reboots; blocks, index and transcripts |
| WAL event log | Working | JSONL audit trail for all lifecycle events |
| RAM disk support | Working | ImDisk (Windows) / tmpfs (Linux), auto-mount |
| Multi-backend search fallback | Working | Tries configured backends in chain on failure |
| Force search heuristics | Working | Auto-detects search-like queries |
| Retrieval benchmark | Working | Threshold sweep over a hand-built corpus, with false-fire rate |
| Throughput benchmark | Working, measured | Direct vs through the middleware. Decode unchanged; **+1.7 s time-to-first-token**, nearly all of it the relevance judge |
| End-to-end benchmark | Working; cost measured, effect not yet | Three arms, so the memory is separated from the tool definitions the middleware injects. **Cost is measured**: +1,334 prompt tokens and +6.4 s a turn. **Whether it shortens the reasoning trace is still open** — median +395 chars but the interval spans zero (CI [-604, +1012]) on 9 usable pairs, one model, a 6-block store. Underpowered, not negative. Of four hand-graded traps, two were helped, one anchored on the stale block, one was noise |
| KV cache management | In progress | Clear endpoint exists; slot save/restore is Phase 2 |
| Retry/circuit breaking | Not started | External calls (search, embed, upstream) fail soft and are logged, but nothing backs off or trips |
| Multi-user/isolation | Not started | Single-user alpha |
| Authentication/TLS | Not started | Loopback by default; `--host` can open it to a LAN, with nothing guarding it |

### What the memory will and will not delete

| Guarantee | |
|---|---|
| A pinned block | Never purged, never rewritten, at any age |
| A block still earning its keep | Recalls and uncontested recalls convert into days of life; a block keeps being recalled stays. A block recalled once long ago and never again does not |
| Any purge | Reversible — status flip plus vector drop, file kept unless `purge_deletes_file` |
| A regex-matched correction | Stops being recalled at once; can only purge after `corrected_grace_s`, and never if the block was ever recalled |

`purge_age_s` is 3 days, and read on its own that sounds alarming. It applies
only to blocks that are unpinned and old enough to be past it, and that have
spent the life their recalls earned — and every purge can be restored
afterwards.

---

## Quick start

Full step-by-step: **[INSTALLATION.md](INSTALLATION.md)**

```
git clone https://github.com/master-basic/mind.git
cd mind
run.bat                          # Windows — double-click or run in terminal
python run.py                    # Linux
```

First run offers a model menu, downloads 3 models (~9 GB with the default
reasoning model and its vision projector), prints a launch plan showing what
runs where and with how much context, then starts 5 servers (reasoning, judge,
embedding, speech-to-text, middleware). Chat UI at `http://127.0.0.1:8000`,
admin at `/admin`. Later runs reuse the remembered answers without asking.

### Images and voice in the chat

- **Images** — 🖼️ in the composer attaches pictures; they travel as OpenAI-style
  `image_url` content parts through the middleware and are seen by any
  vision-capable catalog model (choices 1, 2 and 4 ship with an mmproj
  projector). The memory pipeline stays text-only: images are answered but not
  embedded.
- **Voice** — 🎤 records the mic (MediaRecorder), re-encodes the audio in the
  browser as 16 kHz mono WAV, and POSTs it to the middleware's `/v1/stt`,
  which proxies it to the whisper.cpp server that `run.py` starts on port 8083
  (`whisper-server` + `ggml-large-v3-turbo-q8_0.bin` — 99 languages including
  Russian and Azerbaijani — auto-downloaded from the `ggerganov/whisper.cpp`
  repo into `C:\llama\whisper\models\`; pass `--skip-stt` to disable). The
  chat page has a transcription-language selector next to the mic
  (Auto / Русский / Azərbaycanca / English); Auto detects the language per
  recording, and pinning a language is the reliable fix for mixed-speech
  turns. The transcript is inserted into your message as `[You said: …]`, so
  spoken words go through the same memory pipeline as typed ones. `--stt-model`
  swaps the model: `ggml-large-v3-q5_0.bin` (1 GB) is the full non-turbo
  large-v3 — a bit more accurate on short utterances but ~2× slower — and
  `ggml-large-v3.bin` (2.9 GB, fp16) is the slowest and most accurate. For
  short turns in a low-resource language, pin the language in the selector —
  auto-detect can mis-pick (e.g. Azerbaijani → Turkish/Russian).

  If the whisper.cpp **CUDA build** is unpacked to `C:\llama\whisper\cuda\`
  (from `whisper-cublas-*-bin-x64.zip`), `run.py` prefers it automatically:
  transcription drops from seconds to ~0.3 s and even the fp16 model is
  interactive. Its ~1.1 GB of VRAM is charged against the reasoning window up
  front (the context autosizer subtracts it), and `--stt-cpu` forces the CPU
  build instead.

### Reaching it from another machine

The middleware listens on loopback by default, so only the machine running it
can connect. `--host 0.0.0.0` binds every interface instead, and the startup
banner then prints the LAN address to point clients at:

```
python run.py --host 0.0.0.0
```

`run.bat` asks for the same thing interactively and remembers the answer in
`run_settings.txt` (`HOST=`). Windows Firewall still has to allow the port:

```
netsh advfirewall firewall add rule name="Cued Recall 8000" dir=in action=allow protocol=TCP localport=8000
```

The three llama.cpp servers follow the same address: bind the middleware to the
network and the reasoning server (8080), judge (8081) and embedding (8082)
servers are reachable too, which is what a benchmark run from another machine
needs (`evaluate/` scripts talk to those ports directly). They serve the raw
models with no memory layer in front, so pass `--no-expose-backends` to keep
them on loopback and publish only the middleware.

The firewall rule has to cover whichever ports were opened — the startup banner
prints the exact command for the set it actually published.

There is no authentication and no TLS anywhere in this stack. Anyone who can
reach port 8000 can read and write the memory store. Bind it to a network you
trust, not to the internet.

---

## API endpoints

All routes are on the middleware (default `127.0.0.1:8000`, or the machine's LAN
address when started with `--host 0.0.0.0`).

### Chat

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `POST` | `/v1/chat/completions` | Chat completion with streaming, recall, tools |
| `GET` | `/chat` | Built-in chat UI |
| `GET` | `/health` | Health check (also reports `hot_shelve_timeout_s`) |

### Chat history

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/chats` | List conversations (paginated, filterable by source) |
| `GET` | `/chats/{id}` | Full transcript |
| `PATCH` | `/chats/{id}` | Rename a conversation |
| `DELETE` | `/chats/{id}` | Delete the transcript — derived memory blocks are kept |

### Admin

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/admin` | Admin web GUI (Live / Memory / Blocks tabs) |
| `GET` | `/admin/blocks` | List blocks (filterable, paginated) |
| `GET` | `/admin/blocks/{id}` | Full block details + WAL history |
| `POST` | `/admin/blocks/{id}/verify` | Set verification (accepted/corrected) |
| `POST` | `/admin/blocks/{id}/pin` | Pin/unpin — a pinned block never decays or gets rewritten |
| `POST` | `/admin/blocks/restore` | Bring purged blocks back and re-embed them |
| `POST` | `/admin/blocks/delete` | Batch delete blocks (this one is not reversible) |
| `GET` | `/admin/tags` | Tag taxonomy + counts |
| `POST` | `/admin/judge/run` | Force judge pass |
| `POST` | `/admin/kv/clear` | Clear KV caches |
| `GET` | `/admin/stats` | Block counts, file counts |
| `GET` | `/admin/stats/growth` | Blocks created per day |
| `GET` | `/admin/stats/distribution` | Token-size histogram |
| `GET` | `/admin/stats/recall` | Recall effectiveness + most-recalled blocks |
| `GET` | `/admin/stats/budget` | Recent per-turn recall budget decisions |
| `GET` | `/admin/tps` | Request-level ring buffer + server prefill/decode rates |
| `GET` | `/admin/export` | Export blocks as JSON |
| `POST` | `/admin/import` | Import blocks from JSON |
| `GET` | `/admin/models` | Per-server status + context usage |
| `GET` | `/admin/system` | GPU/CPU telemetry, uptime |
| `GET` | `/admin/wedge` | Is a server's inference queue blocked? |
| `POST` | `/admin/server/restart` | Ask the launcher to restart a llama server (503 without one) |
| `GET` | `/admin/mode` | Current request mode (`memory` or `passthrough`) |
| `POST` | `/admin/mode` | Switch mode — `{"mode": "memory"\|"passthrough"}` |

**Passthrough mode.** Live → Mode in the admin GUI has an Enable/Disable switch.
Disabled turns `:8000` into a plain proxy to the reasoning server: the request is
forwarded byte-for-byte, nothing is recalled or injected, nothing is stored, and the
idle judge pass stays out of the way. It exists for coding agents, whose prompts are
already exactly what they should be — anything this middleware adds to them makes the
output worse. The choice is written to `mode.json` in the store, so it survives a
restart.

### Utility

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/v1/fetch` | URL fetch (SSRF-guarded) |

---

## Launcher CLI

```
python run.py [options]
run.bat [options]          # Windows
```

| Option | Description |
|--------|-------------|
| `--storage PATH` | Root for models + blocks. Sticky across runs |
| `--snapshot PATH` | Snapshot backup location (persistent, not on RAM disk) |
| `--models-cache DIR` | Keep models here; copy to working dir each run |
| `--download-to DIR` | Download to this directory first, then copy to the working dir |
| `--reasoning-model PATH` | Explicit reasoning GGUF path |
| `--reasoning-choice N` | Pick model from catalog by number |
| `--model-menu` | Force model selection menu |
| `--reasoning-ctx N\|auto` | Context window. `auto` sizes it from free VRAM and the model's KV cost |
| `--reasoning-cpu-moe` | Force expert offload (auto-detected for A3B/MoE models) |
| `--reasoning-n-cpu-moe N\|auto` | How many layers keep their experts in RAM. `auto` spends leftover VRAM on the rest |
| `--judge-model PATH` | Explicit judge GGUF path |
| `--embed-model PATH` | Explicit embedding GGUF path |
| `--reasoning-port N` | Reasoning server port (default: 8080) |
| `--judge-port N` | Judge server port (default: 8081) |
| `--embed-port N` | Embedding server port (default: 8082) |
| `--middleware-port N` | Middleware port (default: 8000) |
| `--skip-reasoning` / `--skip-judge` / `--skip-embed` | Skip that server |
| `--no-tmpfs` | Local directories instead of tmpfs (Windows default) |
| `--llama-bin PATH` | Path to `llama-server` |
| `--no-download` | Fail if models not already present |
| `--dry-run` | Print plan without executing |
| `--help` | Full usage |

---

## Configuration

Settings live in `cued_recall/config.yaml` (gitignored, auto-created from `config.example.yaml` on first run). Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `block_tokens_reasoning` | 8000 | Max tokens per reasoning block |
| `max_context_tokens` | 26624 | Prompt budget. Rewritten by `run.py` from the context it actually served |
| `context_reserve_tokens` | 4096 | Held back for the reply, including the think trace |
| `tokens_per_word` | 1.3 | Word→token multiplier (estimator) |
| `chars_per_token` | 3.2 | Char→token divisor (estimator) |
| `exact_count_threshold` | 0.6 | Above this fraction of the budget, tokenize the prompt on the server instead of estimating |
| `hot_shelve_timeout_s` | 15 | Seconds before abandoned convos are shelved |
| `recall.k` | 4 | Top blocks to retrieve |
| `recall.threshold` | 0.48 | Cosine similarity threshold. Low on purpose — the judge below does the rejecting. Raise toward 0.62 if you disable it |
| `recall.budget_tokens` | 3000 | Max injected recall tokens |
| `recall.judge_enabled` | true | Second-stage relevance filter on the small model |
| `recall.judge_timeout_s` | 5.0 | Per candidate; a timeout keeps the block |
| `recall.judge_note` | question | What the judge reads as the note: the block's originating question (measured: refuses all 6 old traps, false-fire 0.09; on the widened corpus 5 Aug: trap-asym leaks 2/6 where "about" cannot tell direction) or its own text (old behaviour, false-fire 0.64) |
| `recall.judge_score_floor` | 0.5 | Relevance score below which a candidate is dropped |
| `recall.candidate_multiplier` | 1 | How many candidates the judge scores, as a multiple of k |
| `recall.floor` | 0.0 | Cosine floor below which the judge stage is skipped; **off** — confirmed on the widened corpus (5 Aug): `trap-asym` mean top-sim 0.866, crosslingual 0.841, no safe floor exists that doesn't strand legitimate recalls |
| `recall.tag_channel` | true | gist/tag keyword channel; serves as the embed-failure fallback |
| `recall.tag_second_source` | false | Same channel as a second candidate source on the normal path; off — acceptance rows now exist and pass (tag-same 3/3, tag-diff 0/3, 5 Aug); wiring decision pending on the next PR |
| `recall.pin_priority` | true | Whether a pin breaks ties in the ranked recall fill |
| `judge.interval_tokens` | 20000 | New material before a pass is worth running |
| `judge.idle_trigger_s` | 300 | Quiet time before a consolidation pass starts |
| `judge.sweep_interval_s` | 21600 | Sweep at least this often even with no new material, so decay still happens in a quiet week |
| `judge.purge_age_s` | 259200 | Never recalled and older than this (3 days) → purge |
| `judge.worthless_age_s` | 172800 | Shorter deadline for blocks the model found nothing reusable in |
| `judge.corrected_grace_s` | 86400 | A pattern-matched correction cannot purge before this, and never if the block was recalled |
| `judge.max_truncate_count` | 2 | Rewrites allowed per block, ever |
| `judge.max_pass_seconds` | 600 | Wall-clock ceiling on one pass |
| `judge.max_per_pass` | 200 | Blocks visited in one pass |
| `judge.consolidate_min_tokens` | 600 | Below this, the model is not called at all |
| `judge.consolidate_types` | `[reasoning]` | Block types the model may rewrite |
| `judge.keep_recall_count` | 3 | Recalled this often → keep verbatim, never compress |
| `judge.rejudge_interval_s` | 604800 | Leave a block alone this long after judging it |
| `judge.purge_deletes_file` | false | Purging is reversible unless this is set |
| `judge.utility_decay` | true | Decay by earned utility (recalls → days of life) rather than "ever recalled" immortality |
| `judge.utility_recall_weight` | 30.0 | Days of idleness one recall earns |
| `judge.utility_uncontested_weight` | 60.0 | Extra days for an uncontested recall |
| `judge.utility_floor` | 0.0 | Utility at or below which a block purges (once past the age gate) |
| `judge.merge_enabled` | true | The abstraction pass: ≥ 3 near-identical blocks → one gist block, originals retired reversibly. On by default since the 2026-08-05 live measurement |
| `judge.merge_cluster_sim` | 0.90 | Similarity at which two blocks count as the same ground |
| `judge.merge_min_cluster` | 3 | How many near-duplicates before generalising is worth a model call |
| `judge.merge_min_age_s` | 604800 | How settled a cluster must be (a week) |
| `judge.merge_max_per_pass` | 5 | Merges per pass |
| `tagger.enabled` | true | Auto-tag at shelve time |
| `verifier.enabled` | true | Ask the small model about corrections the patterns miss |
| `verifier.spans` | false | Span-level corrections: the verifier quotes the offending phrase, recall redacts it instead of suppressing the block; off — the yes/no prompt is the measured one |
| `web_search.backend` | duckduckgo | Search backend |
| `web_search.brave_api_key` | — | Brave Search API key |
| `web_search.serper_api_key` | — | Serper.dev API key |
| `web_search.searxng_url` | — | Self-hosted SearXNG URL |

---

## Project structure

```
mind/
├── run.bat                          # Windows launcher (interactive, remembers answers)
├── run.py                           # Python orchestrator: VRAM planning, launch, watchdog
├── run_settings.txt                 # What the last launch resolved (read by run.bat)
├── ramdisk_setup.bat                # Windows RAM disk utility
├── INSTALLATION.md                  # Step-by-step setup guide
├── ARCHITECTURE.md                  # Block lifecycle, recall, consolidation vs decay
├── logs/                            # Per-server llama-server stdout
├── snapshots/                       # Block store backups (persistent)
├── evaluate/                        # Retrieval + end-to-end benchmarks
│   ├── benchmark.md                 # Method notes and results
│   ├── corpus.jsonl                 # Hand-built probe corpus, 9 relation types
│   ├── corrections.jsonl            # Labelled correction/not-correction rows
│   ├── eval_retrieval.py            # Threshold sweep (no generation), --judge arm
│   ├── eval_correction.py           # Precision/recall/false-positive rate for corrections
│   ├── eval_throughput.py           # Direct vs through-the-middleware: TTFT and tok/s
│   ├── throughput.md                # Throughput report
│   ├── eval_e2e.py                  # A/B: direct vs through the middleware
│   ├── eval_merge.py                # Live merge-pass measurement (Phase 3.1 decision)
│   ├── eval_judge_notes.py          # What the judge is shown, and the false-fire cost
│   ├── analyse.py                   # Paired analysis + bootstrap CI
│   ├── inspect_blocks.py            # Look at what is actually stored
│   ├── retrieval_sweep.csv          # Latest sweep output
│   └── semantic_judge_plan.md       # Planned two-stage recall filter
└── cued_recall/
    ├── config.yaml                  # Active config (gitignored)
    ├── config.example.yaml          # Template, auto-copied
    ├── report_decay.py              # What the next judge pass would purge, and why
    ├── backfill_missing_vectors.py  # Re-embed blocks whose vector failed at creation
    ├── backfill_token_counts.py     # Re-count stored blocks with the real tokenizer
    └── cued_recall/
        ├── main.py                  # FastAPI app, all routes
        ├── pipeline.py              # Recall, blockify, tools, web search
        ├── config.py                # Config classes
        ├── models.py                # Block schema
        ├── store.py                 # Msgpack block store
        ├── index.py                 # SQLite + sqlite-vec, analytics queries
        ├── chats.py                 # Durable chat transcripts
        ├── embed.py                 # Embedding client
        ├── judge.py                 # Consolidation + decay
        ├── verifier.py              # Yes/no correction classifier
        ├── tagger.py                # Gist + tags at shelve time
        ├── small_model.py           # Shared queue for the CPU judge server
        ├── sysinfo.py               # GPU/CPU telemetry (standalone by design)
        ├── router.py                # Admin API routes
        ├── taxonomy.py              # Tag taxonomy
        ├── wal.py                   # Write-ahead log
        ├── utils.py                 # Token counting, stimulus builder, correction matcher
        └── static/
            ├── admin.html           # Admin web GUI
            └── chat.html            # Built-in chat UI
```

---

## Benchmarks

Two harnesses, kept apart because conflating them produces numbers that mean
nothing.

**Retrieval** — does the right block come back? No generation, deterministic,
runs in seconds:

```bash
python evaluate/eval_retrieval.py --endpoint http://127.0.0.1:8082
```

Sweeps `recall.threshold` from 0.30 to 0.94 over `corpus.jsonl` and writes
`retrieval_sweep.csv` with a recall rate *and* a false-fire rate at each step.
`--fake` self-tests the harness with synthetic vectors and no servers running.

The corpus is hand-written from real recurring work, with adversarial members in
every family: `exact`, `paraphrase` and `crosslingual` (Azerbaijani) must recall;
`trap` (same vocabulary, different answer) should fire but must not be reused
blindly; `trap-asym` (same entities, direction inverted — the stork/baby class)
likewise; `tag-same` (no shared wording, taxonomy tags overlap) must recall via
the gist/tag channel; `distractor`, `tag-diff` (tags overlap, content differs)
and `control` must not fire.

Add `--judge` to run the second-stage relevance filter as a second arm and print
both columns side by side. That is the number that decides whether
`recall.judge_enabled` should be on.

**Correction detection** — how often does it fire on something that was not a
correction? Needs no servers:

```bash
python evaluate/eval_correction.py --no-model
```

Drop `--no-model` to score the classifier too. `--from-chats <chats.db>` mines
real user messages for labelling, which is the only way the negative half stops
reflecting only what someone thought to test.

**Merge pass** — is the abstraction pass safe to run? Copies a snapshot store
and runs the real `_merge_pass` against it with seeded near-duplicates, reading
the `blocks_merged` / `merge_rejected` events and probing recall of the merged
block. Never touches a live store:

```bash
python evaluate/eval_merge.py [--store snapshots/latest]
```

**Throughput** — what does the memory layer cost per turn? Runs the same
prompts straight at `llama-server` and again through the middleware:

```bash
python evaluate/eval_throughput.py --repeats 3
```

Reports time-to-first-token, decode tokens/sec and prompt size **separately**,
because a single tok/s number is what once hid a turn spending 107 s on prefill
and 1.4 s generating. Everything the middleware does — embedding the query,
vector search, the relevance judge — lands in TTFT; decode should be flat,
since the middleware is not in that path. `--cold` clears the KV cache before
every request to compare prefill instead of steady state. Results:
**[evaluate/throughput.md](evaluate/throughput.md)**.

**End-to-end** — does having the block help? Slow and noisy, so `--repeats 3`
with paired analysis:

```bash
python evaluate/eval_e2e.py --repeats 3
```

Baseline goes straight to `:8080`, treatment through the middleware, store wiped
and re-warmed between repeats. `analyse.py` pairs each probe against itself and
bootstraps a CI. Answer quality is graded by hand on purpose — no script catches
a model anchoring confidently on a recalled block that did not apply.

---

## License

MIT
