# Cued Recall Memory Middleware

A semantic block memory layer that sits between any OpenAI-compatible chat client and a local `llama.cpp` server. It separates reasoning from results, archives reasoning as retrievable blocks, and recalls prior reasoning when a similar problem appears.

## Architecture

```
Client ──▶ Cued Recall Middleware ──▶ llama-server (reasoning model)
                          │
                    ┌─────┴──────┐
                    │  Block Store │  (tmpfs + SQLite + sqlite-vec)
                    └─────────────┘
                          │
                    ┌─────┴──────┐
                    │ Judge Model │  (small LLM for truncation decisions)
                    └─────────────┘
```

Three `llama-server` instances:
- **Reasoning model** (e.g. Qwen3, GLM-4, R1 distills) — main generation, emits `<think>` tags
- **Judge model** (4B class, Q4) — truncation and purging decisions
- **Embedding model** (Qwen3-Embedding-0.6B or nomic-embed-text) — stimulus key vectors

## Status

**Alpha implementation — complete and buildable.**

### Implemented

- [x] OpenAI-compatible FastAPI proxy (`POST /v1/chat/completions`)
- [x] Streaming and non-streaming passthrough
- [x] `<think>` tag parsing and reasoning/result split
- [x] Block store on tmpfs (msgpack files, one per block)
- [x] SQLite metadata index + `sqlite-vec` vector index
- [x] Embedding client (HTTP to llama-server embedding endpoint)
- [x] Recall pipeline: embed user message → query vector index → inject advisory system message
- [x] Blockify: split reasoning on paragraph boundaries at 8k tokens
- [x] Shelve: flip hot → shelved after one turn
- [x] Correction detection via regex patterns
- [x] Verification signal (accepted/corrected)
- [x] Judge pass: background task calls small LLM for truncate/purge decisions with safety ladder
- [x] WAL (write-ahead event log)
- [x] Snapshot/restore (tmpfs survives reboots via periodic snapshots to NVMe)
- [x] Admin routes: list/get blocks, override verification, force judge pass, stats
- [x] Advisory framing on recalled blocks
- [x] Purge safety: no block with `recall_count > 0` or non-corrected verification is ever purged

### In progress / alpha scope

- [ ] KV-cache slot save/restore (Phase 2 — specified, not attempted)
- [ ] Cross-lingual paraphrase robustness (depends on embedding model choice)
- [ ] Production hardening (rate limiting, auth, config validation)

## Windows RAM Disk (optional, recommended)

Models and the block store benefit from fast storage. On Windows, create a RAM disk:

```
ramdisk_setup.bat
```

Prompts for drive letter, size, block size, and volume label. Requires [ImDisk Toolkit](https://sourceforge.net/projects/imdisk-toolkit/) (free, open-source).

Then run the launcher with:
```
run.bat --storage R:\cued_recall
```

## Quick start

```bash
# Install
pip install -e cued_recall/

# Edit config.yaml to point at your llama-server instances

# Run
cued-recall
# or
python -m cued_recall.main config.yaml
```

On Windows:
```
run.bat
```

## Configuration

See [`cued_recall/config.yaml`](cued_recall/config.yaml) for all defaults. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `listen` | `127.0.0.1:8000` | Middleware listen address |
| `reasoning_endpoint` | `http://127.0.0.1:8080` | Main reasoning model |
| `judge_endpoint` | `http://127.0.0.1:8081` | Judge model |
| `embed_endpoint` | `http://127.0.0.1:8082` | Embedding model |
| `recall.k` | `4` | Top-k blocks to retrieve |
| `recall.threshold` | `0.62` | Cosine similarity threshold |
| `recall.budget_tokens` | `3000` | Max tokens injected per request |
| `block_tokens_reasoning` | `8000` | Max tokens per reasoning block |

## Admin API

| Route | Method | Description |
|---|---|---|
| `/admin/blocks` | GET | List blocks (filter by status, type, conversation_id) |
| `/admin/blocks/{id}` | GET | Full block + WAL history |
| `/admin/blocks/{id}/verify` | POST | Set verification (`accepted`/`corrected`) |
| `/admin/judge/run` | POST | Force a judge pass |
| `/admin/stats` | GET | Block counts, disk usage, recall hit rate |

## License

MIT
