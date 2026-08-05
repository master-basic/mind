# Update Plan — Cued Recall, from the semantic-mind.md analysis

Companion to `semantic-mind.md`. That document is the *why* (each finding is
mapped to Yee, Jones & McRae 2018 Ch. 9); this one is the *what, where, and
how to verify*. Every item references the finding it addresses (F1–F14), the
exact files/functions, the change, the config flag, and the acceptance test.

## Working principles

- **One behavior change per PR, always behind a config flag with the old
  value as default.** This project has a measured, adversarial eval culture
  (`evaluate/benchmark.md`); each change must be shown against it, not
  asserted.
- **Never change the retrieval corpus semantics without re-running the
  sweep.** The operating point (threshold 0.48, k=4, judge on) is a measured
  result, not a taste.
- **The store is user data.** All destructive changes (purge, re-embed,
  merge) go through the reversible status-flip pattern the codebase already
  uses, and never delete files unless `purge_deletes_file`.
- **No model changes.** These are all middleware/index/config/evals changes.
  No new dependencies unless a phase says so.
- **Add the missing unit test seam.** The project has excellent integration
  evals but no `pytest` suite; Phase 0 adds one around the pure logic so the
  O(n) and budget changes don't have to be verified by hand each time.

---

## Phase 0 — Foundation: test seam, health signal, baseline (prereq for all)

**Files:** new `cued_recall/tests/`, `pyproject.toml` (add `pytest`), `index.py`,
`router.py`

### 0.1 Unit-test seam
- Add `pytest` + `pytest-asyncio` to dev deps.
- Cover the pure functions that everything else sits on, so later phases can
  refactor with a safety net:
  - `utils.split_paragraph_boundary`, `truncate_tokens`, `estimate_tokens` vs
    `count_tokens_exact` agreement,
  - `judge._should_purge` for every row of its source-authority table
    (manual > pattern > model; recall>0 protection; pin exemption),
  - `index.query` status filtering and k×50 over-fetch,
  - `build_stimulus` composition (512/512/256 truncation),
  - `taxonomy.validate_tags` / `validate_gist`.
- Acceptance: `pytest` green; `eval_retrieval --fake` still self-tests.

### 0.2 Missing-vector health signal (F7)
- `index.blocks_without_vectors()` already exists (used by the backfill
  script). Surface it in `/admin/stats` (`router.py:247 stats`) as
  `blocks_missing_vectors` and a `vector_backfill_needed` boolean, and log it
  in the WAL once per judge pass when nonzero.
- Acceptance: on the known 1,812-block store, stats reports the missing
  count; the number is visible without running the manual backfill.

### 0.3 Baseline run
- Record a baseline of `eval_retrieval` sweep, `eval_correction`, and one
  `eval_throughput` run into `evaluate/baseline_*.json` so every phase's
  before/after is a diff against a file, not a memory.
- Acceptance: files exist and are committed with the Phase 0 PR.

---

## Phase 1 — Representation: embed reasoning blocks from their own text (F2)

**The single highest-leverage change** in the whole analysis: the store's
dominant block type (reasoning) is embedded from a **question+answer
composite** (`build_stimulus`, pipeline.py:1886-1889), so its vector *is*
the Q+A pair and any phase-2 question sharing entities scores 0.84 against
phase-1 material. Fix the representation, and the traps stop scoring high in
the first place instead of depending on the judge to catch them.

### 1.1 Separate "embed text" from "prompt context"
Currently `stimulus_text` serves two jobs: (a) the vector source
(`_embed_and_store` embeds `block.stimulus_text`), and (b) the "Question:"
context handed to the judge's consolidation prompt (`judge._user_prompt`,
judge.py:367, fed `stimulus` from `block.stimulus_text` at 455). Those must
diverge:

- Add `embed_text: str = ""` to `Block` (models.py). Set it at creation:
  - reasoning blocks → `truncate_tokens(full_reasoning, 1024)` — the think
    trace itself,
  - result blocks → `truncate_tokens(full_result, 1024)`,
  - reading blocks → `truncate_tokens(reading_content, 1024)`.
- Keep `stimulus_text` exactly as it is today **for the judge's prompt only**
  (it is the "what was asked" context and is correct for that).
- `_embed_and_store` (pipeline.py:1956) embeds `block.embed_text or
  block.stimulus_text` (fallback so old blocks/backfills don't break).
- `backfill_missing_vectors.py` and `router.py:203` (re-embed endpoint) use
  the same fallback.
- `index.upsert_vector` unchanged.

Acceptance (this is the measurable one):
- Re-run the retrieval sweep. The `trap` family's mean top-sim must drop
  materially (target: below the 0.48 operating threshold for all existing
  trap rows), while `exact`/`paraphrase`/`crosslingual` recall stays ≥ 0.90
  at the operating point with the judge on.
- `grading_traps.md` cases: the RECALL-ON answer must no longer anchor on the
  seed's stack (the `ocr1-trap` tesseract.js failure).

### 1.2 Re-embed the existing store
- Extend `backfill_missing_vectors.py` into `backfill_reembed.py` (or add a
  `--reembed-from embed_text` mode): re-embed every shelved/truncated block
  from its `embed_text` field, writing only vectors (status untouched).
- Run it once as a migration. Vectors change; that is the point.

### 1.3 Judge prompt: make the trap an explicit asymmetry question (F2)
- Re-word `utils.relevance_prompt` (utils.py:97) so the judge distinguishes
  "is this block **about** the user's question" from "does it merely share
  vocabulary/entities with it." Keep the shipped `RELEVANCE_SYSTEM`
  semantics but add the asymmetry instruction.
- Add the trap wording to `evaluate/` so it's measured, not vibes: new
  `relation: "trap"` rows where seed and probe share entities but differ in
  direction (the chapter's stork/baby class).
- Acceptance: with `--judge`, `false_fire_rate` at operating threshold stays
  0.00 while `recall_rate` for `should_recall=true` does not drop.

---

## Phase 2 — Retrieval order: judge first, budget second (F6)

Today the budget is filled in geometric order *before* the judge runs
(pipeline.py:567-592), so a reject-worthy or oversized block consumes a slot
that a judge-approved block behind it never gets. Fix the order:

### 2.1 Score, then rank, then fill
- Change `_filter_by_relevance` to return a **score** (0–1 or
  "yes/relevant" with a confidence), not a boolean. Keep fail-open behavior
  (timeout → keep, scored low).
- In `recall_blocks`:
  1. query top-K (raise internal K to ~`k*4` for the judge stage),
  2. judge-score the candidate pool,
  3. sort by score desc,
  4. fill `budget_tokens` from that ranked list (still skipping oversized),
  5. report `skipped_oversized`, `rejected_by_judge`, `tokens_used` in the
     WAL `recall_budget` event as today.
- This is a small-step toward the chapter's multi-hub organization: cosine
  is the first pass, relevance is the arbiter of *what fits*.

Acceptance:
- The eval corpus keeps identical recall/false-fire at operating point.
- `evaluate/throughput.md` TTFT delta must not grow beyond +100 ms vs
  baseline (the judge still runs on the same candidate count; we're only
  reordering).

### 2.2 Token-count honesty
- The `backfill_token_counts.py` docstring documents the historical ~42%
  understatement and the fix. Make it permanent: add a unit test that
  `count_tokens` falls back to `estimate_tokens` only when the tokenizer is
  down, and that `token_count` is never written as `len(text.split())`
  anywhere in the pipeline (grep for `split())` in `_stream_and_blockify`).
- Acceptance: `pytest` asserts the two counting paths agree within 5% on the
  eval corpus prompts.

---

## Phase 3 — Abstraction: prototype formation, graded decay, persistent signal (F1, F5, F13)

This is the "missing half" from the meta-observation: the system recalls
episodes but never generalizes across them.

### 3.1 Prototype/abstraction pass (F1)
- Add an opt-in judge stage, `judge.merge_enabled` (default off in the first
  PR, on after measured).
- In `judge.run_pass`, after `_decay_sweep`:
  - cluster candidates by embedding similarity ≥ 0.90 (reuse `index.query`
    per candidate, or a coarse `tag` group),
  - when a cluster has ≥ N members (config, default 3) that are all older
    than `min_age_s`, ask the consolidation model to write a single merged
    gist block (`type="result"`, `gist` set, `text` = the merged
    generalization, `conversation_id=""` so it is conversation-agnostic),
  - mark the originals `truncated` (reversible) and link via a new
    `parents` field on the merged block.
- This is the prototype-over-exemplars the chapter's complementary-learning
  account describes; the merged block is a *category-level* memory while the
  originals are still recoverable.
- Acceptance: seeded with 3 near-duplicate blocks (eval fixture), a pass
  produces one merged block and marks the originals truncated; recall of the
  merged block fires for a related-but-new probe the originals also would
  have matched.

### 3.2 Graded utility, not immortality (F5)
Replace the all-or-nothing `recall_count > 0 → immortal` rule
(`_should_purge`, judge.py:361) with a utility score:
- `utility = w_recall * recall_count + w_uncontested * uncontested_recalls -
  w_age * age - w_corrected * correction_strikes` (weights in config).
- Purge rule becomes: purge when `utility` below a floor AND `age >
  purge_age_s`. `keep_recall_count` gates *consolidation* only (as today),
  it no longer gates *purging*.
- A block recalled once and then never again still becomes stale; one
  recalled weekly stays.

### 3.3 Persist the recalled-uncontested signal (F13)
- `apply_accepted_verification` (pipeline.py:2064) reads `_recalled_by_turn`,
  an in-memory dict capped at 500 and lost on restart. Fold the evidence into
  durable block metadata instead:
  - on recall, write `last_recalled` (already exists) and bump a new
    `uncontested_recalls` counter on the *next* turn only if the block was
    recalled into it and not corrected,
  - drop `_recalled_by_turn` (or keep only as a fast-path cache of the
    durable state).
- This is exactly the reactivation-strengthening signal the decay logic
  should run on; it must survive restart.
- Acceptance: recall a block, restart the server, then start a new turn in
  the same conversation — the block is still marked `accepted` and its
  counter increments.

---

## Phase 4 — Correction granularity: stop punishing the source (F4)

A wrong answer marks *every* block of the previous turn corrected, including
the pasted `reading` source block the model merely misused
(`detect_and_apply_correction` → `_find_turn_blocks` → `_mark_corrected`).

### 4.1 Scope correction by block type (quick win)
- In `detect_and_apply_correction` and `verify_correction_with_model`, mark
  only `reasoning` and `result` blocks corrected; leave `reading` blocks
  `unknown` (they can still be consolidated but are no longer poisoned by
  the model's misuse of them).
- Acceptance: corrected turn containing a reading block → reading block
  verification remains `unknown`, result/reasoning are `corrected`.

### 4.2 Span-level corrections (bigger)
- Teach the verifier (`verifier.py`, `_prompt`/`_parse`) to output a span or
  short paraphrase of the offending part, not just yes/no.
- Store it on the block (`correction_span: str = ""`). On recall, the judge
  injects the block *minus* the span (or with a "this part was corrected"
  marker) rather than suppressing the whole block.
- This maps to the chapter's "weight the stable aspects": 90% of a block that
  is right keeps being usable.
- Acceptance: eval fixture where the correction targets one claim in a
  multi-claim block → recall injects the block with the bad claim flagged,
  and the other claims survive.

---

## Phase 5 — Taxonomy as a second retrieval channel (F3, F10, F11)

`tagger.py` writes 40-char gist + up to 3 taxonomy tags per block; retrieval
never reads them (`recall_blocks` queries vectors only; the judge sees
`block.text`). The taxonomy is the project's only non-vector,
category-level representation and it is dead weight.

### 5.1 Tag/gist into recall (F3, F11)
- In `recall_blocks`, after the vector stage, use `tag` overlap as a
  *second candidate source*: blocks whose tags match the probe's tags (or
  whose gist contains a probe keyword) enter the judge pool alongside vector
  hits.
- `_filter_by_relevance` gets `block.gist` and `block.tags` in the prompt so
  the judge can arbitrate between the two channels.
- This fixes the "relevant but semantically distant wording" miss class and
  gives recall a feature-based code in the chapter's sense.
- Acceptance: add eval rows where probe wording shares no tokens with the
  seed but the tags/gist overlap → recall fires; and rows where tags overlap
  but content differs → judge rejects (no false-fire increase).

### 5.2 Resilience fallback (F10)
- Implement the documented-but-missing similarity floor: if the best cosine
  is below a config `recall.floor` (default 0.30), skip the judge call
  entirely (the 1.5–2.2 s off-topic tax disappears).
- If the embed server errors during `recall_blocks`, fall back to the
  tag/keyword channel alone instead of returning `[]` (currently
  pipeline.py:546-552 fails open to zero recall).
- Acceptance: with the embed server stopped, recall still returns
  tag-matched blocks; with it up, an obviously-off-topic turn makes zero
  judge calls (WAL `recall_budget` shows `judge_calls=0`).

---

## Phase 6 — Drift & data integrity (F7, F9)

### 6.1 Recompute vectors when text changes (F7)
- `judge._truncate_block` (judge.py:615) replaces `block.text` but never
  touches the vector. After truncation, re-embed from `embed_text` (the new
  summary, or keep the original `embed_text` if truncation keeps the
  original text as the durable copy — see 1.1).
- Correction (`_mark_corrected`) does not change text, so no re-embed there,
  but a corrected block should have its vector dropped (it is suppressed
  from recall by status today; making the vector absent too is belt-and-
  braces and matches `delete_vector`'s role in purge).
- Acceptance: truncate a block, then verify `index.blocks_without_vectors()`
  does not include it and its vector reflects the new text (spot check via
  admin block view).

### 6.2 Consistent "user's question" (F9)
- `get_last_user_message` (pipeline.py:1053) returns the last `user` message
  verbatim, including `<tool_response>`-wrapped turns, while `_newest_user_index`
  (pipeline.py:164) deliberately skips them. Make the recall query and the
  injection anchor agree: reuse the `_newest_user_index` logic in
  `get_last_user_message`.
- Acceptance: an agentic-client fixture with a trailing `<tool_response>`
  message embeds the *same* text as the message recall is injected into.

---

## Phase 7 — Robustness (F8, F12, F14)

### 7.1 Blocks off the response path (F8)
- Streaming: `_create_blocks` runs after `[DONE]` (pipeline.py:1665→1676).
  If the client disconnects at `[DONE]`, Starlette closes the generator and
  the turn is never memorized.
- Fix: write the blocks + metadata (the fast, local part) *before* yielding
  `[DONE]`, and defer only the embedding (`_embed_and_store`) to a
  background task. With Phase 0.2's missing-vector health signal and the
  backfill path, a dropped embed becomes a surfaced, repairable condition
  rather than silent memory loss.
- Acceptance: simulate a disconnect right after `[DONE]` (test fixture
  closes the SSE stream) → blocks exist in the store; vectors may be absent
  but `blocks_missing_vectors` reports them.

### 7.2 Kill the O(n) per-turn scans (F12)
- `_find_turn_blocks` (pipeline.py:2106) loads all 10,000 metas and filters
  in Python, and is called up to 4× per turn. `index.db` has no index on
  `(conversation_id, turn_index)`.
- Add the index; rewrite `_find_turn_blocks` as a parameterized query.
- WAL: `read_all()` (wal.py:23) is read whole by router.py:109,128,250 on
  every admin request. Add `read_tail(n)` / a persisted cursor so stats and
  history don't re-parse the whole log.
- Acceptance: `pytest` with a 10k-block fixture shows `_find_turn_blocks`
  issues one indexed query (use SQLite `EXPLAIN QUERY PLAN`); admin stats on
  a growing WAL does not re-read the file.

### 7.3 Pins and recall (F14)
- Pinned blocks are exempt from decay but still lose budget priority. In the
  Phase 2 ranked fill, give pinned blocks priority (and a config knob
  `recall.pin_priority`, default on).
- Acceptance: with budget pressure, a pinned block is admitted before
  equally-scored unpinned ones.

---

## Cross-cutting

- **Config surface** (config.example.yaml): new keys — `block.embed_token_limit`,
  `judge.merge_enabled`, `judge.merge_cluster_sim`, `judge.merge_min_cluster`,
  `judge.utility_weights`, `recall.floor`, `recall.tag_channel`,
  `recall.pin_priority`. All default to current behavior.
- **Eval corpus**: extend `evaluate/corpus.jsonl` with asymmetric `trap`
  rows (F2), tag-overlap rows (F5), multi-claim correction rows (F4). Regrade
  `grading_traps.md`.
- **Docs**: `README.md` (features), `ARCHITECTURE.md` (embed_text vs
  stimulus_text, merge pass, utility decay), `CHANGELOG.md`.
- **Ordering rationale**: 0 → 1 (representation is the root of the worst
  behavior) → 2 (retrieval correctness) → 3 (the missing abstraction) → 4–7
  (granularity, channels, integrity, robustness). Phases 1, 2, 3 change
  measured behavior; each must land with its eval diff attached.

## Explicit non-goals

- No change to the embedding model, judge model, or their servers.
- No new vector store; sqlite-vec stays.
- No multi-user/authentication work (out of scope for a single-user stack).
- No LLM-sourced abstraction that overwrites source blocks irreversibly —
  merges are reversible (originals truncated, not purged).
