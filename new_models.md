# Plan: 6-role model lineup + vision & voice in the middleware chat

Status: approved (STT upgraded to large-v3-turbo-q8_0 legacy .bin with language selector; STT -> text path; Q5_K_M kept as default).
Owner: model research round, 5 Aug 2026.

## Goal

Ship the researched lineup (exactly 6 catalog models, one per role) and add two
chat features: **image upload** (the model sees the picture) and **voice
recording** (mic -> STT -> text), both flowing through the existing middleware
(`/v1/chat/completions` on port 8000).

## Final lineup (names per user spec — 6 roles, 6 models)

| # | Role name | Model | Quant / size | Age | Catalog |
|---|-----------|-------|--------------|-----|---------|
| 1 | Fast assistant | Qwen3.5-9B | Q5_K_M 6.6 GB (+mmproj) | Feb 2026 | default, now vision-capable |
| 2 | Vision, Voice assistant | Gemma 4 12B | Q4_K_M ~6.7 GB (+mmproj) | 2026 | new entry |
| 3 | Fast thinker | Qwen3.5-4B, thinking on | UD-Q4_K_XL ~3.1 GB | Feb 2026 | new entry |
| 4 | Vision, Voice assistant (large) | Gemma 4 26B-A4B | Q4_K_M 16.8 GB (+mmproj+MTP), --cpu-moe | 2026 | relabelled pre-existing entry |
| 5 | Aggressive | Qwen3.5-35B-A3B Abliterated | Q4_K_S 19.9 GB, --cpu-moe | 2025-26 | pre-existing, role-named |
| 6 | Coding | Qwen3.6-35B-A3B | UD-Q4_K_XL 19.4 GB, --cpu-moe | 2026 | pre-existing, role-named |

The catalog is exactly these six; the former 9B heretic/abliterated variant
entries were dropped to keep it one model per role.

## Steps

### 1. run.py `REASONING_CATALOG` (run.py:42-89)

- Catalog = exactly 6 entries, one per role (9B heretic/abliterated variants
  dropped). Choice 1 keeps Q5_K_M as default and gains an `mmproj` extra
  (Qwen3.5-9B is natively multimodal; the default model becomes vision-capable).
- Entry 2 — Gemma 4 12B: Q4_K_M ~6.7 GB, `moe: False`, `extras: [mmproj]`
  (mirror choice 5's pattern, run.py:77-80).
- Entry 3 — Qwen3.5-4B: `unsloth/Qwen3.5-4B-GGUF` UD-Q4_K_XL ~3.1 GB, no mmproj.
  Thinking mode: verify forcing mechanism against installed llama-server
  (Qwen3 thinking is template/tool driven — resolve at impl; smoke-test shows a
  thinking trace).
- Entry 4 — the pre-existing Gemma4-26B-A4B entry, relabelled "Vision, Voice
  assistant (large)". Entry 5/6 keep their models, gain "Aggressive"/"Coding"
  role names.

### 2. Vision support (image upload)

- UI `static/chat.html`: new image button; `FileReader.readAsDataURL`; user
  message content becomes a parts array
  `[{"type":"text","text":…},{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}]`;
  thumbnail chip in composer; history stores the parts array.
- Middleware pass-through: `_prepend_text` already preserves non-text parts
  (pipeline.py:238-254). Verify `process_turn`/`build_messages` don't strip them.
- Token accounting: `_estimate_tokens` must count image parts (~658 tok/img for
  Gemma-3-style tiles) or the window overflows -> hard 400.
- Wiring: run.py already passes `--mmproj` from `extras` (run.py:1733-1739);
  autosizer already counts mmproj VRAM (run.py:1168-1171).
- Persistence: `_msg_text`/`get_last_user_message` tolerate parts (extract first
  text part); `chats.record_turn` stores parts alongside text; `/chats/{id}`
  reload round-trips parts (JSON-safe).
- Memory pipeline stays text-only: images are seen in-conversation, not embedded.

### 3. Voice support (STT)

- UI `static/chat.html`: mic button -> `MediaRecorder` -> WebM/ogg -> POST
  `multipart` to `/v1/stt` -> transcript appended to user text as
  `[You said: …]` -> normal text path (memory indexes it too).
- Middleware `main.py`: new `POST /v1/stt` endpoint proxying audio to the STT
  backend; WAL log entry; config key for the STT endpoint.
- STT backend: llama-server with `--whisper-model whisper-small` + `--stt`
  (verify support in installed build; whisper.cpp server fallback if absent).
  New port; tiny VRAM; `moe: False`.

### 4. Tests

- Parts forwarding: chat with `image_url` content reaches the reasoning server
  untouched; token estimate includes image tokens.
- `_msg_text` / `get_last_user_message` with parts lists; correction scanner
  ignores parts safely.
- `/v1/stt` endpoint with mocked backend; transcript lands in the user message.
- Full suite run (295 passed pre-upstream-pull; upstream added
  `test_time_backend.py` — confirm no regressions).

### 5. Docs

- README.md catalog table + vision/voice usage.
- CHANGELOG.md entry.
- This plan is new_models.md.

## Verification (manual)

- `run.bat` menu shows 6 role-named choices; select 1, 2, 3 each ->
  `/props` + one chat turn (role 3 shows thinking trace; roles 1, 2 & 4 accept
  an image).
- Image chat: attach -> model describes it. Voice: record -> transcript appears
  -> answered.
- VRAM: choice 2 ~9-11 GB @32k with mmproj (autosizer confirms).
