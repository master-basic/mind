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
- **Reasoning** — main LLM (Qwen3.5-9B by default, catalog of 6 including 35B-A3B MoE)
- **Judge** — rewrites long think traces; also tags blocks and classifies corrections (Qwen2.5-1.5B, CPU-only)
- **Embedding** — vector retrieval keys (nomic-embed-text-v1.5)
- **Middleware** — FastAPI proxy (this project)

`run.py` sizes the reasoning server's context window and MoE expert split from
free VRAM and the model's own GGUF metadata before launch, then stays resident
as a watchdog: a llama.cpp slot can be lost while the process keeps answering
`/health`, and only the launcher owns the handles needed to restart it.

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
| Semantic recall | Working, measured | Cosine retrieval plus a relevance filter. Embedding alone at 0.62 recalls 0.96 with a **false-fire rate of 0.55**; that is why it no longer runs alone |
| Semantic reranker | Working, measured, **on by default** | Second stage asks the small model whether a candidate applies. False-fire rate **0.00** at every threshold. Its only recall cost is the trap family. Disable with `recall.judge_enabled: false` — and raise `threshold` back toward 0.62 if you do |
| Manual retention (pin) | Working | A pinned block is never purged and never rewritten, at any age |
| Restore purged blocks | Working | Purging was always reversible on disk; `POST /admin/blocks/restore` re-embeds and brings it back |
| KV-prefix-safe injection | Working | Recall is anchored to the newest user turn, so the cached prefix survives the turn |
| Exact token accounting | Working | Block counts and near-limit prompt budgets use the model's `/tokenize`; conservative estimator otherwise |
| Block lifecycle | Working | hot → shelved → truncated/purged; passes sweep forward, oldest-unjudged first |
| Tagging system | Working | Auto-tags blocks with gist + tags at shelve time |
| Correction detection | Working, measured | 17 anchored patterns (EN + AZ) plus a few-shot classifier. Patterns: precision 0.87, recall 0.76, **false-positive rate 0.12** on 34 hand-labelled rows |
| Decay | Working | Purges on correction, or never-recalled past a cutoff. No model call. Reversible by default. See the retention guarantees below |
| Consolidation | Working | Judge rewrites long think traces only; guarded against copied openings and non-shrinking rewrites; capped at `max_truncate_count` rewrites per block |
| Bounded judge pass | Working, measured | Wall-clock ceiling per pass; a pass that runs out of time resumes where it stopped. Measured: 163 blocks visited in 7.3 s, 1 model call |
| Idle consolidation | Working | Passes run after a quiet period, not mid-conversation |
| Web search | Working | 4 backends: DuckDuckGo (free), Brave, Serper, SearXNG |
| Web fetch | Working | SSRF-guarded, HTML-to-text, JSON detection |
| Tool calling | Working | Own tools (web_search, web_fetch) + client tool forwarding |
| Model catalog | Working | 6 reasoning models with interactive menu; launch details remembered per model |
| VRAM autosizing | Working | Context window sized from free VRAM + GGUF KV geometry; leftover VRAM spent on MoE expert layers (`--n-cpu-moe`) |
| MoE support | Working | 17–20 GB A3B models on a 12 GB card, experts in system RAM |
| Wedge watchdog | Working | Detects a llama.cpp server whose `/health` answers but whose inference queue is blocked, and restarts it |
| Server logs | Working | Each llama-server's stdout in `logs/{name}.log` |
| Built-in chat UI | Working | Streaming, reasoning pane, file upload, Stop button |
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
| End-to-end benchmark | Harness only | A/B script + paired analysis; results are hand-graded, not yet published |
| KV cache management | In progress | Clear endpoint exists; slot save/restore is Phase 2 |
| Retry/circuit breaking | Not started | External calls (search, embed, upstream) fail soft and are logged, but nothing backs off or trips |
| Multi-user/isolation | Not started | Single-user alpha |
| Authentication/TLS | Not started | Open on localhost only |

### What the memory will and will not delete

| Guarantee | |
|---|---|
| A pinned block | Never purged, never rewritten, at any age |
| A block that has ever been recalled | Never purged by the age cutoff |
| Any purge | Reversible — status flip plus vector drop, file kept unless `purge_deletes_file` |
| A regex-matched correction | Stops being recalled at once; can only purge after `corrected_grace_s`, and never if the block was ever recalled |

`purge_age_s` is 3 days, and read on its own that sounds alarming. It applies
only to blocks that are unpinned, were never once retrieved, and can be
restored afterwards.

---

## Quick start

Full step-by-step: **[INSTALLATION.md](INSTALLATION.md)**

```
git clone https://github.com/master-basic/mind.git
cd mind
run.bat                          # Windows — double-click or run in terminal
python run.py                    # Linux
```

First run offers a model menu, downloads 3 models (~8 GB with the default
reasoning model), prints a launch plan showing what runs where and with how much
context, then starts 4 servers. Chat UI at `http://127.0.0.1:8000`, admin at
`/admin`. Later runs reuse the remembered answers without asking.

---

## API endpoints

All routes are on the middleware (default `127.0.0.1:8000`).

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
| `tagger.enabled` | true | Auto-tag at shelve time |
| `verifier.enabled` | true | Ask the small model about corrections the patterns miss |
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
│   ├── corpus.jsonl                 # Hand-built probe corpus, 6 relation types
│   ├── corrections.jsonl            # Labelled correction/not-correction rows
│   ├── eval_retrieval.py            # Threshold sweep (no generation), --judge arm
│   ├── eval_correction.py           # Precision/recall/false-positive rate for corrections
│   ├── eval_e2e.py                  # A/B: direct vs through the middleware
│   ├── analyse.py                   # Paired analysis + bootstrap CI
│   ├── inspect_blocks.py            # Look at what is actually stored
│   ├── retrieval_sweep.csv          # Latest sweep output
│   └── semantic_judge_plan.md       # Planned two-stage recall filter
└── cued_recall/
    ├── config.yaml                  # Active config (gitignored)
    ├── config.example.yaml          # Template, auto-copied
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
blindly; `distractor` and `control` must not fire.

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
