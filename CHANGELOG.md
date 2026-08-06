# Changelog — from idea to working system

Six days, 74 commits, 23–28 July 2026. This is the honest version: what the
idea was, what happened when it met a real GPU and a real conversation, and
what it can actually do now. The dated sections are how it got here; the last
two are what it is.

---

## The idea

A reasoning model rederives the same conclusions forever. Ask it about ADVPN
routing on Monday and again on Thursday, and Thursday costs the same thousands
of think-tokens Monday did — the model has no way to notice it already worked
this out.

So keep the work. Split each turn into **blocks** — the think trace, the answer,
anything long that was pasted in — store them with an embedding, and when a new
question looks semantically close to an old one, hand the old reasoning back
before the model starts thinking. Cued recall, in the psychological sense: not
a search engine, a cue that brings back what was already worked out.

Two design commitments from the start, both of which survived:

- **It sits in front of any OpenAI-compatible client**, so it works with
  whatever you already use rather than being an app you have to live in.
- **It runs entirely local** — llama.cpp, a 9B reasoning model, a 1.5B model for
  housekeeping, an embedding model. Nothing leaves the machine.

---

## 23 July — the alpha (9 commits)

Block lifecycle, msgpack store, SQLite + sqlite-vec index, FastAPI proxy,
admin GUI, a launcher, and a RAM-disk setup script for Windows. It ran.

Then the first three real bugs, all in the part that decides what a block *is*:
the think splitter, conversation identity, and cosine distance being computed
against unnormalized vectors. Recall cannot be better than its splitting.

## 24 July — contact with reality (24 commits, the busiest day)

The day the design met the hardware and the clients.

- **Embedding dimension mismatch 500'd every single chat.** The index was built
  for 1024 dimensions, the model produced 768.
- **Models on a RAM disk, downloaded to HDD first** — re-downloading 7 GB after
  every reboot is not a workflow.
- Model placement got deliberate: reasoning weights and KV on the GPU, judge on
  CPU so it never competes for VRAM, embedding KV in RAM.
- A built-in chat UI, so the system could be used without a third-party client.
- Web search and web fetch as tools, with tool-call forwarding for clients that
  bring their own.
- Snapshots started failing on Windows shutdown with PermissionError. Patched
  with a retry. *(This came back on the 28th, and the patch was not the fix.)*

## 25 July — becoming a well-behaved server (9 commits)

Strict clients rejected the SSE stream, so the chunks became fully
OpenAI-conformant. `/v1/models` was added because agent CLIs call it during
setup and 404 before you can chat. Client-owned tool calls stopped being
executed as "Unknown tool" and started being forwarded. Block tagging with a
fixed taxonomy, export/import.

The theme of the day: the memory system was fine, and everything *around* it
was what made it unusable.

## 26 July — the machine, honestly (7 commits)

Context size stopped being a guess and became a calculation from free VRAM and
the model's own KV geometry. A GPU/system panel. Chat transcripts moved into
their own store, separate from blocks, because blocks are a poor way to replay
a conversation — the user's message survives only inside a reasoning block's
stimulus, truncated.

## 27 July — the honesty pass (21 commits)

The day the project stopped trusting itself, and the most valuable day in it.

**The judge had never done anything.** It answered `keep` 142 times out of 142.
395 blocks, none truncated, none purged. Three faults hid behind one symptom: a
1.5B model was given a three-way choice with no criteria (the criteria lived in
the code that overrode its answer anyway); blocks average 80 tokens against a
prompt asking for a summary "under 400"; and each pass re-read the same oldest
50 blocks that `keep` left in place, so 345 of 395 had never been looked at.

The fix was to stop asking a small model to do two jobs. **Forgetting is
arithmetic** — age, recall count, correction status, all recorded exactly, so it
needs no model at all. **Compressing needs to understand the text**, so that is
the only thing the model is asked to do, with one question and no label to get
wrong.

**Verification was dead upstream of all of it.** Five fixed phrases had never
matched anything, while 309 blocks claimed "accepted" on no evidence beyond the
conversation continuing. Patterns widened to 17 and anchored; acceptance now
requires a real event.

**Recall was destroying the KV cache.** The recall note was merged into the
front of the system prompt, so the first tokens changed every turn and llama.cpp
re-prefilled everything behind them: 43,345-token prompts at `f_keep 0.12`,
107 seconds of prefill to produce a reply that decoded in 1.4. Anchoring the
note to the newest user message instead leaves the whole prefix byte-identical.

Also: a wedge watchdog, after discovering a llama.cpp slot can be lost while the
process stays alive — `/health` answers fast while everything through the
inference queue blocks forever. And MoE support that spends leftover VRAM on
expert layers, so a 35B-A3B runs on a 12 GB card.

## 28 July — measurement (4 commits)

A benchmark, an outside review, and the answers to it.

Built a hand-written corpus with adversarial members in every family, and
measured retrieval properly — including the number nobody publishes, the
**false-fire rate**. It was bad: at the shipped threshold, recall 0.96 and false
fires 0.55.

A review then called four things blocking. Three were right: no way to keep a
memory by hand, a regex able to delete one, an unbounded judge loop, an
unmeasured reranker. One was wrong — "always use /tokenize" — because the exact
recount already fires whenever the payload exceeds the budget *in bytes*, and a
token costs at least one byte, so an under-estimate provably cannot overflow.
The real hole there was narrower: tool schemas were charged as an estimate while
the proof measured only message text.

All four closed, plus the one the review understated: a pattern-matched
correction could purge a block at any age regardless of how often it had been
recalled. Now it hides the block for a day instead, and can never touch one
that was ever retrieved.

Last, the bug that explains a scare. A live store came up empty with 158 blocks
sitting in the snapshot: `latest` had an ACL the account could not read, and
both restore paths gate on `Path.exists()` — which answers False for "denied"
exactly as it does for "absent". Restore did nothing and said nothing. The
snapshot writer had also been deleting `latest` *before* writing its
replacement, and swallowing every failure silently. Both fixed; an unreadable
snapshot is now loud, and the swap keeps the old copy until the new one lands.

---

## 29 July — 5 August — the semantic-memory work (16 behaviour-changing commits)

`semantic-mind.md` and `update_plan.md` (committed 28–29 July) took the
measurement habit further: every behaviour change ships behind a config flag
with the old value as default, and every headline number is a measurement, not
an assertion. What got built, in commit order:

- **A pytest seam** (Phase 0.1) — the pure logic had no tests at all. 295 now.
- **Missing-vector health signal** (Phase 0.2) — `blocks_missing_vectors` on
  `/admin/stats`, so the backfill need is visible instead of discovered.
- **Kill the O(n) per-turn scans** (Phase 7.2) — `_find_turn_blocks` walked
  `list_meta(limit=10000)` up to four times a turn, and was silently *wrong*
  above 10,000 blocks; the WAL stats scans went the same way.
- **Correction scoped by block type** (Phase 4.1) — a wrong answer marked
  *every* previous-turn block corrected, including the pasted source document
  the model merely misused; the user's own material was being deleted for the
  model's mistake.
- **Consistent "user's question"** (Phase 6.2) — recall embedded the tool
  payload as the query while the blocks it found were spliced in before it,
  because the agent's turn ends with a `<tool_response>`.
- **`embed_text` split from `stimulus_text`** (Phase 1.1, 6.1) — as a switch,
  not a migration; truncation re-embeds now, so a rewritten block's vector
  stops describing text it no longer holds.
- **Embed inputs capped by tokens, not words** (not in the plan) — 1,024
  words of code measured 2,338 tokens against a 2,048-token embedder: an HTTP
  400, swallowed, and the block stored unrecallable forever.
- **The `embed_source` measurement** (Phase 0.3) — and it refuted Phase 1:
  embedding from content alone does not separate traps from legitimate
  recalls. The plan's migration was refused; `embed_source` stays `composite`.
- **Show the judge the question, not the answer** (roadmap item 4) — the
  false-fire 0.00 recorded on 28 July was a harness artefact. Shown a real
  block's text, the judge kept 5 of 6 traps; shown the originating question,
  it refuses 6/6, false-fire 0.64 → 0.09. This was the plan's roadmap item 4
  and the largest single defect found.
- **Spend the budget on relevance, not nearness** (Phase 2) — candidates are
  ranked by the judge's P(yes) (free: +2.4 ms reading logprobs) and the budget
  spent down that order, not in cosine order.
- **Graded utility decay** (Phase 3.2, 3.3) — "recalled once, ever" was
  immortality; recalls now earn days of life. And the acceptance signal that
  feeds it was an in-memory dict lost on restart; it is durable now.
- **The merge pass** (Phase 3.1) — one block derived from ≥ 3 near-identical
  ones, originals retired reversibly, and a verifier that refuses any merge
  that drops a number, path or identifier — which caught the first real draft
  conflating 840ms with the TTL. **On by default 2026-08-05** after a live
  measurement (`evaluate/eval_merge.py`): a real family merged correctly and
  fired recall; a draft that dropped `dns.cache_ttl` was refused.
- **Blocks off the response path** (Phase 7.1) — blocks were created after
  `[DONE]` was yielded, so a client disconnecting on the completion signal
  silently dropped the turn from memory.
- **Recall floor + embed-failure fallback** (Phase 5.2) — the floor ships off
  (no safe value exists yet; the plan's 0.30 cannot fire below the 0.48
  threshold), and when the embed server errors, recall now degrades to a
  gist/tag keyword channel instead of the whole store vanishing.
- **Tag/gist as a second candidate source** (Phase 5.1) — ships off: the
  acceptance rows now exist and pass (tag-same 3/3, tag-diff 0/3, 5 Aug 2026);
  wiring decision pending on the next PR. The channel itself is on, because
  it is the embed-failure fallback.
- **Span-level corrections** (Phase 4.2) — off by default: when the verifier
  says a claim is wrong, it also quotes the offending phrase, and recall
  redacts just that span instead of suppressing the whole block. Live
  measurement (5 Aug, 38 rows): the 1.5B produces bare `yes` with empty span
  even in span mode; the 4.2 fixture measures 0/3 pass — stays off.
- **Pin priority in the budget** (Phase 7.3) — a pin was exempt from decay
  but bought nothing at retrieval; it is now the tie-break in the ranked fill.

Progress is recorded in `update_implement.md` (§1–§19a), which is also where
the two places the plan was wrong — Phase 1's premise, and Phase 3.1's
`truncated` retirement — are documented.

## 6 August — the model lineup and multimodal chat

Researched a role-based GGUF lineup (newest model ≤ 8 months old, all fitting a
12 GB GPU, MoE experts in system RAM) and landed it in the catalog:

| # | Role | Model | Size |
|---|------|-------|------|
| 1 | Fast assistant | Qwen3.5-9B Q5_K_M (+mmproj, now vision-capable) | 6.6 GB |
| 2 | Vision, Voice assistant | Gemma 4 12B Q4_K_M +mmproj | ~6.7 GB |
| 3 | Fast thinker | Qwen3.5-4B UD-Q4_K_XL (thinking mode) | ~3.1 GB |
| 4 | Vision, Voice assistant (large) | Gemma 4 26B-A4B Q4_K_M +mmproj +MTP (MoE) | 16.8 GB |
| 5 | Aggressive | Qwen3.5-35B-A3B Abliterated (MoE) | 19.9 GB |
| 6 | Coding | Qwen3.6-35B-A3B (MoE) | 19.4 GB |

The catalog was renumbered from the pre-existing six to exactly these six role
models; the 9B heretic/abliterated variants are gone, and the Gemma 4 26B-A4B
entry now carries the role-4 label instead of its old name.

Extras resolution got a `save_as` key: unsloth names every projector
`mmproj-BF16.gguf`, so the first launch of Gemma 4 12B silently reused the
Qwen3.5-9B projector already in the cache and llama-server died with
`mismatch between text model (n_embd = 3840) and mmproj (n_embd = 4096)`.
Choices 1 and 2 now stage their projectors under unique names
(`mmproj-qwen3.5-9b-…` / `mmproj-gemma4-12b-…`).

Voice chat: Chrome records `audio/webm;codecs=opus`, which whisper.cpp cannot
demux (`failed to decode audio data from memory buffer` → HTTP 400). The chat
page now decodes the recording in the browser (`decodeAudioData` →
`OfflineAudioContext`) and re-encodes it as 16 kHz mono PCM WAV before the
`/v1/stt` upload.

The transcription model is no longer whisper-small: `run.py` serves
`ggml-large-v3-turbo-q8_0.bin` (834 MiB, 99 languages, ru/az included) by
default with `-t 16`, and downloads it on first use. whisper.cpp only reads
the legacy ggml `.bin` format — it has no GGUF loader at all — so the GGUF
mirror downloads (handy-computer, 845 MiB + 1.6 GiB) were dead weight that
the server rejected with `invalid model data (bad magic)`, while the old
small `.bin` kept loading and silently served as the fallback. The resolver
now fetches the real legacy-format file from `ggerganov/whisper.cpp` (the
`ggml-org` namespace is gated; the old alias is open); `--stt-model` switches
files (`ggml-large-v3-q5_0.bin`, 1 GB full large-v3, ~2× slower, and the
fp16 `ggml-large-v3.bin`, 2.9 GB, for maximum accuracy). Russian
transcription went from garbled to near-verbatim on the test audio;
Azerbaijani went from misheard ("Azərbaycan" → "Ağız etəbəyət səm") to exact
("Azərbaycan."); on-device Russian improves ~10× versus small. The full
large-v3 variants measured identical on degraded test clips (fp16 included)
at 2–2.5× the turbo latency, so turbo stays the default; short low-resource
utterances remain the hardest case — pinning the language in the chat
selector beats auto-detect there (auto picked Turkish/Russian in tests).

A CUDA whisper-server (`whisper-cublas-12.4.0-bin-x64.zip` unpacked to
`C:\llama\whisper\cuda\`) is now used automatically when present: the whole
model runs on the RTX 4070, transcription drops from 4–10 s to ~0.2–0.5 s,
and the fp16 large-v3 (3.1 GB) becomes interactive. The stt process starts
*before* llama-server so it claims its VRAM first, and the context autosizer
charges the whisper footprint (~1.1 GB for turbo q8_0, or ~3.4 GB for fp16)
to the reasoning KV budget so the window still fits the card.
`--stt-cpu` forces the CPU build; the CPU path is unchanged otherwise.
On the GPU build the server also runs beam search (`-bs 5`, ~free there) and
the chat page records at 128 kbps opus instead of Chrome's default ~32-64
(less loss before the 16 kHz PCM decode), both aiming at short-phrase
accuracy.

Two whisper-server quirks surfaced and are handled: `-l auto` at launch makes
it detect the language but still transcribe in English (v1.9.x bug), so the
launcher no longer passes it — the middleware forwards the chat page's
per-request language instead (default `auto`, which detects correctly). The
chat page gained a language selector next to the mic
(Auto / Русский / Azərbaycanca / English); pinning a language is the reliable
fix for turns that mix or start in a low-resource language.

Speech-to-text is now resilient in two ways: a corrupt or half-downloaded
model file is detected by magic bytes and re-downloaded instead of crashing
the server (`invalid model data (bad magic)`), and a dead stt process no
longer brings the whole stack down — voice input is disabled and the rest
keeps running, instead of stopping all four servers.

Researched and rejected for the fast-thinker slot: Phi-4-mini-reasoning and
SmolLM3-3B (too old), GLM-4.7-Flash (too big for the role), Nemotron 3 Nano 4B
(fails the community reasoning gauntlet that Qwen3.5-4B passes), Qwen3.6-6.7B
(existence unverified — Qwen3.6 ships from 27B up).

The chat UI went multimodal:

- **Images** — 🖼️ attaches pictures as OpenAI-style `image_url` content parts;
  the pipeline passes non-text parts through untouched (`_prepend_text` was
  already parts-safe), charges each image a flat 700 tokens against the context
  budget (`_msg_token_estimate`, `IMAGE_TOKENS_PER_IMAGE`), and skips the
  server-side exact token count when images are present (a tokenizer cannot see
  vision-encoder tokens). Memory stays text-only: images are seen, not embedded.
- **Voice** — 🎤 records with MediaRecorder, POSTs to a new `/v1/stt` endpoint,
  which proxies multipart audio to a whisper.cpp `whisper-server` that `run.py`
  launches on 8083 (whisper-small q8_0, CPU; `--skip-stt` disables; missing
  binary/model gets a pointed error instead of a silent no-op). The transcript
  lands in the message as `[You said: …]` and flows through the normal text
  pipeline, memory included.
- `run.bat`'s echo menu now matches the catalog (it had drifted to a Hermes3.6
  entry the catalog no longer had).

Tests: `tests/test_multimodal.py` (7 tests: image counting, flat-rate token
charge, parts surviving recall injection, no-tokenize-with-images, budget
trimming with pictures). Suite: **317 passed, 1 skipped**.

## What it does, with numbers

Measured, not asserted — see [evaluate/benchmark.md](evaluate/benchmark.md):

| | |
|---|---|
| Recall, embedding only @ 0.62 | 0.96 recall, **0.55 false-fire** |
| Recall, with the relevance judge, note = question | 0.75 recall, **0.09 false-fire**, traps refused 6/6 — the 0.00 originally recorded was a harness artefact that showed the judge a seed prompt, not a real block |
| Widened corpus (53 rows, 5 Aug), judged @ 0.62 | 0.61 recall, **0.00 false-fire**; old traps 6/6, trap-asym leaks 2/6 (direction invisible to "about"); tag-same 3/3 via channel, tag-diff 0/3 |
| Correction verifier (live, 5 Aug, n=38) | precision 0.59, recall 1.00, **FPR 0.78** — the 1.5B says "yes" even to its own few-shot negative; span mode 0/3 pass (bare `yes`, empty span) |
| Crosslingual (Azerbaijani) recall | 3/6 → **6/6**, once the judge let the threshold drop to 0.48 |
| Correction patterns | precision 0.87, recall 0.76, false-positive rate 0.12 (n=34) |
| Judge pass on a 164-block store | 163 blocks visited in **7.3 s**, 1 model call |
| Consolidation on think traces | 76–94% smaller, 7 of 8 keeping the concrete facts |

And what it is made of: ~5,500 lines of Python in the middleware, ~1,400 in the
launcher, five processes, four models, one GPU.

Working end to end: an OpenAI-compatible proxy with streaming and tools; a
built-in chat UI with history, image attachment and voice transcription; a
semantic recall with a second-stage relevance
filter that reads the originating question; a block lifecycle with
consolidation, utility decay, pinning and restore; a merge pass that derives
one block from near-identical ones; span-level corrections; correction
detection in English and Azerbaijani; a tabbed admin page
with GPU telemetry and memory analytics; VRAM-aware launching with MoE expert
splitting; a wedge watchdog; snapshots; and six evaluation harnesses.

## What it does not do

Single user. No authentication, no TLS. It binds loopback by default and
`--host 0.0.0.0` will serve a LAN, but nothing stands in front of that: reaching
the port is the whole authorization model. No retry or circuit
breaking on external calls — they fail soft and get logged, nothing backs off.
KV slot save/restore is unfinished. The end-to-end benchmark has a harness and
no published results, because the grading that matters there is done by hand.

The corpus is 53 rows and the correction set is 38, both hand-written. They
bound the shape of the problem, not the rate.

## What the six days actually taught

Every serious bug here was the same bug wearing different clothes: **a
component reported success while doing nothing.** The judge said `keep`. The
verifier matched nothing and everything read as accepted. Recall injected
memories and destroyed the cache that made them affordable. A snapshot restored
nothing and logged nothing. The wedged server answered `/health` in 30 ms.

None of them threw an exception. All of them were found by asking for a number
and not getting one — which is why the WAL now records why a decision was made
and not just that it was, why passes report what they cost, and why the feature
table separates "implemented" from "measured".
