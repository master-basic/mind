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
```

Four server processes managed by the launcher:
- **Reasoning** — main LLM (Qwen3-8B, configurable catalog of 6 models)
- **Judge** — rewrites long think traces; also tags blocks and classifies corrections (Qwen2.5-1.5B, CPU-only)
- **Embedding** — vector retrieval keys (nomic-embed-text-v1.5)
- **Middleware** — FastAPI proxy (this project)

Memory upkeep is split in two, deliberately. **Consolidation** — turning a long
derivation into a short one — needs to understand the text, so the judge model
does it. **Forgetting** is a function of age, recall count and whether the
answer was corrected, all of which are recorded exactly, so it is arithmetic
and runs without a model call. See **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Features & status

| Feature | Status | Details |
|---|---|---|
| OpenAI-compatible chat proxy | Working | Streaming + non-streaming, `/v1/chat/completions`, `/v1/models` |
| Reasoning/result split | Working | ` thinking` / ` response` tag parsing during streaming |
| Semantic recall | Working | Cosine-similarity retrieval, advisory injection, budget-limited |
| Block lifecycle | Working | hot → shelved → truncated/purged; passes sweep forward, oldest-unjudged first |
| Tagging system | Working | Auto-tags blocks with gist + tags at shelve time |
| Correction detection | Working | 17 anchored patterns (EN + AZ), plus a few-shot yes/no classifier for what they miss |
| Decay | Working | Purges on correction, or never-recalled past a cutoff. No model call. Reversible by default |
| Consolidation | Working | Judge rewrites long think traces only; guarded against copied openings and non-shrinking rewrites |
| Idle consolidation | Working | Passes run after a quiet period, not mid-conversation |
| Web search | Working | 4 backends: DuckDuckGo (free), Brave, Serper, SearXNG |
| Web fetch | Working | SSRF-guarded, HTML-to-text, JSON detection |
| Tool calling | Working | Own tools (web_search, web_fetch) + client tool forwarding |
| Model catalog | Working | 6 reasoning models with interactive menu |
| Admin web GUI | Working | Context usage, stats, block table, export/import, KV clear |
| Export/import blocks | Working | JSON export/import with re-embedding |
| Snapshots | Working | Periodic + on-shutdown, persistent across reboots |
| WAL event log | Working | JSONL audit trail for all lifecycle events |
| RAM disk support | Working | ImDisk (Windows) / tmpfs (Linux), auto-mount |
| Multi-backend search fallback | Working | Tries configured backends in chain on failure |
| Force search heuristics | Working | Auto-detects search-like queries |
| KV cache management | In progress | Clear endpoint exists; slot save/restore is Phase 2 |
| Multi-user/isolation | Not started | Single-user alpha |
| Authentication/TLS | Not started | Open on localhost only |

---

## Quick start

Full step-by-step: **[INSTALLATION.md](INSTALLATION.md)**

```
git clone https://github.com/master-basic/mind.git
cd mind
run.bat                          # Windows — double-click or run in terminal
python run.py                    # Linux
```

First run downloads 3 models (~7 GB), starts 4 servers, and opens the middleware at `http://127.0.0.1:8000`.

---

## API endpoints

All routes are on the middleware (default `127.0.0.1:8000`).

### Chat

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `POST` | `/v1/chat/completions` | Chat completion with streaming, recall, tools |
| `GET` | `/chat` | Built-in chat UI |
| `GET` | `/health` | Health check |

### Admin

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/admin` | Admin web GUI |
| `GET` | `/admin/blocks` | List blocks (filterable, paginated) |
| `GET` | `/admin/blocks/{id}` | Full block details + WAL history |
| `POST` | `/admin/blocks/{id}/verify` | Set verification (accepted/corrected) |
| `POST` | `/admin/blocks/delete` | Batch delete blocks |
| `GET` | `/admin/tags` | Tag taxonomy + counts |
| `POST` | `/admin/judge/run` | Force judge pass |
| `POST` | `/admin/kv/clear` | Clear KV caches |
| `GET` | `/admin/stats` | Block counts, file counts |
| `GET` | `/admin/tps` | Tokens-per-second ring buffer |
| `GET` | `/admin/export` | Export blocks as JSON |
| `POST` | `/admin/import` | Import blocks from JSON |
| `GET` | `/admin/models` | Per-server status + context usage |

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
| `--reasoning-model PATH` | Explicit reasoning GGUF path |
| `--reasoning-choice N` | Pick model from catalog by number |
| `--model-menu` | Force model selection menu |
| `--reasoning-cpu-moe` | Force `--cpu-moe` for MoE models |
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
| `max_context_tokens` | 26624 | Prompt budget (derived from `--ctx-size`) |
| `tokens_per_word` | 1.3 | Word→token multiplier |
| `hot_shelve_timeout_s` | 15 | Seconds before abandoned convos are shelved |
| `recall.k` | 4 | Top blocks to retrieve |
| `recall.threshold` | 0.62 | Cosine similarity threshold |
| `recall.budget_tokens` | 3000 | Max injected recall tokens |
| `judge.interval_tokens` | 20000 | New material before a pass is worth running |
| `judge.idle_trigger_s` | 300 | Quiet time before a consolidation pass starts |
| `judge.purge_age_s` | 259200 | Never recalled and older than this (3 days) → purge |
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
├── run.bat                          # Windows launcher
├── run.py                           # Python orchestrator
├── ramdisk_setup.bat                # Windows RAM disk utility
├── INSTALLATION.md                  # Step-by-step setup guide
├── ARCHITECTURE.md                  # Block lifecycle, recall, consolidation vs decay
├── snapshots/                       # Block store backups (persistent)
└── cued_recall/
    ├── config.yaml                  # Active config (gitignored)
    ├── config.example.yaml          # Template, auto-copied
    └── cued_recall/
        ├── main.py                  # FastAPI app, all routes
        ├── pipeline.py              # Recall, blockify, tools, web search
        ├── config.py                # Config classes
        ├── models.py                # Block schema
        ├── store.py                 # Msgpack block store
        ├── index.py                 # SQLite + sqlite-vec
        ├── embed.py                 # Embedding client
        ├── judge.py                 # Consolidation + decay
        ├── verifier.py              # Yes/no correction classifier
        ├── tagger.py                # Gist + tags at shelve time
        ├── small_model.py           # Shared queue for the CPU judge server
        ├── router.py                # Admin API routes
        ├── taxonomy.py              # Tag taxonomy
        ├── wal.py                   # Write-ahead log
        ├── utils.py                 # Stimulus builder, correction matcher
        └── static/
            ├── admin.html           # Admin web GUI
            └── chat.html            # Built-in chat UI
```

---

## License

MIT
