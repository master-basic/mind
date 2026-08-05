# Implementation progress — `semantic-mind.md` / `update_plan.md`

Branch `semantic-memory`, off `master` at b876983. Companion to
`semantic-mind.md` (the *why*) and `update_plan.md` (the *what*). This file is
the *what actually happened*, including the two places the plan turned out to
be wrong.

Status as of 2026-08-05. All model servers up (8080 reasoning / 8081 judge /
8082 embed); every number below is measured, not projected.

---

## Headline

Two of the plan's conclusions did not survive contact with a measurement, and
one defect nobody had ranked turned out to be the largest.

1. **Phase 1 (embed reasoning blocks from their own text) is dead.** Its
   premise — that the Q+A composite is what makes traps score 0.841, so
   changing the representation stops them scoring high — is not supported.
   `embed_source` stays `composite`.
2. **The judge's false-fire 0.00 was an artefact of the eval harness.** Shown
   a real block instead of a seed prompt, the judge kept 5 of 6 traps. This is
   roadmap item 4, it is much larger than the analysis estimated, and it is
   now fixed.
3. **A live silent-data-loss bug was found**: embed inputs were capped in
   words against a server that counts tokens, so code-heavy blocks were
   getting HTTP 400 and being stored unrecallable.

---

## Done

| # | Commit | Plan item | Finding |
|---|---|---|---|
| 1 | `5d321c8` Add a pytest seam over the pure logic | Phase 0.1 | — |
| 2 | `b5d49f3` Surface blocks that lost their embedding | Phase 0.2 | F7 |
| 3 | `f763f50` Stop scanning the whole store, and the whole log, per request | Phase 7.2 | F12 |
| 4 | `852d136` Stop punishing the source for the model's mistake | Phase 4.1 | F4 |
| 5 | `a4a379e` Make recall and injection agree on what the user asked | Phase 6.2 | F9 |
| 6 | `d0c90fb` Split what a block says from the question that produced it | Phase 1.1, 6.1 | F2, F7 |
| 7 | `f33fdfb` Cap embed inputs by tokens, not by a guess | *not in plan* | new |
| 8 | `82c4ef7` Measure the two embed_source settings, and find the judge does not hold | Phase 0.3 | F2 refuted |
| 9 | *(pending commit)* Show the judge the question, not the answer | Phase 4 (roadmap item 4) | the real defect |

149 unit tests, no servers needed, under six seconds.

### 1 — Test seam (Phase 0.1)

`cued_recall/tests/`, `pytest` in an optional `dev` extra. Covers the pure
logic every other subsystem sits on: the paragraph/sentence splitter including
the empty-left case that would loop forever in `_split_reasoning`, the
three-signal token estimate, `build_stimulus` composition, `index.query`
filtering status *after* the KNN (with 60 hot blocks in front of a shelved one
— the case the k×50 over-fetch exists for), every row of the `_should_purge`
source-authority table, and the taxonomy validators.

### 2 — Missing-vector health signal (F7)

`index.count_blocks_without_vectors()`, reported by `/admin/stats` as
`blocks_missing_vectors` / `vector_backfill_needed`, shown as a red card in the
admin Stats grid, and written to the WAL once per judge pass. The query already
existed for the backfill script; nothing in the running system called it.

### 3 — O(n) scans (F12)

`_find_turn_blocks` answered "which blocks belong to conversation X, turn Y?"
with `list_meta(limit=10000)` plus a Python filter, up to four times per user
turn, against a table with no index on `(conversation_id, turn_index)`. It was
also **wrong** above 10,000 blocks — the limit silently truncated the scan,
which the analysis had not spotted. Now one indexed query, with
`EXPLAIN QUERY PLAN` asserted in the tests.

`WAL.read_all()` parsed the whole append-only log on every `/admin/stats`,
`/stats/budget` and `/blocks/{id}` request. Replaced by `count()` (maintained
from `open()` and each write), `tail_events()` (scans backwards in chunks,
stops when it has enough) and `iter_all()` (streams, skips a torn final line).

### 4 — Correction scope (F4)

A correction marked every block of the previous turn corrected, including the
`reading` block holding the document the user pasted. Corrected blocks drop out
of recall and start a purge clock, so the user's own source material was being
deleted for the model's misuse of it. Now scoped to `reasoning` and `result`,
with a `correction_skipped_source_block` WAL event so doing less than asked is
visible. An unrecognised block is still marked — leaving a wrong answer
recallable is the worse failure.

### 5 — The user's question (F9)

`_newest_user_index` skips a user message whose whole content is a
`<tool_response>`; `get_last_user_message` and `get_reading_content` did not.
For an agentic client, whose turn *ends* with such a message, recall embedded
the tool payload as the query while the blocks it found were spliced into the
message before it — query and injection target were different texts. Both now
anchor on `_newest_user_index`.

### 6 — `embed_text` as a second channel (F2, F7)

The plan wanted the composite replaced and the store re-embedded as a one-way
migration. That is a one-way door on user data resting on an untested premise,
so it shipped as a switch instead: `Block.embed_text` written on every block
regardless of config, `config.embed_source` choosing which text feeds the
index, default unchanged, and one shared `utils.embed_source_text()` for all
four embed sites (creation, judge re-embed, admin restore/import, backfill) —
a disagreement between them would be invisible. A bad `embed_source` raises at
startup rather than silently reading as `composite`.

**That decision is the reason the measurement in §8 was possible at all**, and
the measurement then said not to switch.

Truncation also re-embeds now (F7): the judge rewrote a block's text and left
its vector describing the old wording. A reasoning block keeps its stimulus
(the question did not change); a result or reading block, whose stimulus is a
copy of its own text, has that copy refreshed. A failed re-embed keeps the
stale vector and logs `reembed_error` — a stale vector still finds the block,
an absent one loses it.

### 7 — Embed inputs capped by tokens (not in the plan)

Found while building the sweep, and live in `master`.

`EmbeddingClient`'s only size guard was `MAX_CHARS = 16000`, chosen for "an
8192-token window". This stack runs nomic-embed at **2048**, so the guard sat
four times above the real limit and never fired. Callers truncate with
`truncate_tokens`, which counts whitespace *words*, and the two units diverge
worst on exactly the content most worth keeping — measured against the live
server, **1,024 words of a code-heavy think trace is 2,338 tokens**. That is an
HTTP 400, `_embed_and_store` catches it and moves on, and the block is stored,
looks healthy in the admin table, and is unrecallable forever.

Not hypothetical for the shipped config: result and reading blocks already set
`stimulus_text = truncate_tokens(text, 1024)`, so any code- or markdown-heavy
answer over roughly 900 words has been failing to embed.

`EmbeddingClient` now reads the real window from the server's `/props` at
startup and `fit()` trims by characters against the conservative token
estimator with a 10% margin, never returning empty. The redundant `[:2000]`
character caps in the backfill and the two admin re-embed paths are gone, so a
repaired block carries the same vector a healthy one would.

**It also corrects the record.** `backfill_missing_vectors.py` concluded that
the 57 content-holding blocks re-embedding "on the first attempt" showed the
original failure was transient. The script was truncating to 2,000 *characters*
(~500 tokens) while the pipeline sent 1,024 words — it succeeded because it was
sending a smaller text. That docstring now says so.

### 8 — The `embed_source` measurement (Phase 0.3, and F2 refuted)

The instrument had to be fixed first. `eval_retrieval.py` used each seed's
*prompt* on both sides of the pipeline — as the text embedded to represent a
block, and as the note shown to the judge. No block looks like that. So the
sweep measured question-to-question similarity and the store's real geometry
had never been swept.

- `make_seed_blocks.py` runs each seed once against the reasoning model with no
  middleware and records the think trace, the answer and the tagger's gist
- `eval_retrieval.py --key-source {prompt,composite,content}` builds keys with
  the middleware's own `build_stimulus`/`truncate_tokens`, capped by the
  shipped `EmbeddingClient.fit` so it cannot score a representation production
  is unable to produce
- `--judge-note {text,question}`, `--json` for before/after diffs

At threshold 0.48, k=4, judge on, judge shown `block.text` as
`_filter_by_relevance` does:

| | prompt (harness) | composite (shipped) | content (proposed) |
|---|---|---|---|
| recall | 0.75 | 0.96 | 0.96 |
| false-fire | **0.00** | **0.64** | **0.45** |
| trap fired | 0/6 | 5/6 | 5/6 |
| distractor fired | 0/6 | 5/6 | 4/6 |
| control fired | 0/5 | 2/5 | 1/5 |

**Verdict on Phase 1:** `content` lowers trap similarity 0.756 → 0.698 — still
far above the 0.48 threshold — and lowers every other relation by about as
much, so exact-minus-trap separation gets *worse*, 0.140 → 0.109. The claimed
mechanism is not there. `content` is modestly better on false fires on n=11,
not enough to move a measured operating point. `embed_source` stays
`composite`; both texts remain stored, so a larger corpus can reverse it with a
re-embed.

Two harness bugs fixed on the way: the embedding cache was keyed on the
*number* of corpus rows, so any experiment varying key text at constant row
count silently scored the previous run's vectors; and the sweep table printed a
box-drawing character un-encodable in cp1252, crashing every redirected run.
`retrieval_sweep.csv` came back byte-identical to the committed one — the check
that none of this moved the default path.

### 9 — Show the judge the question, not the answer (roadmap item 4)

`eval_judge_notes.py` isolates the defect. Holding the pair fixed and varying
only what the judge is shown as the note:

| note | chars (ocr1) | legitimate recall | traps leaked |
|---|---|---|---|
| the answer | 8,463 | 18/18 | **6/6** |
| the think trace (production) | 10,627 | 18/18 | **5/6** |
| question **and** content | ~10,800 | 18/18 | **5/6** |
| the tagger's 40-char gist | 40 | 17/18 | **0/6** |
| **the originating question** | 208 | **18/18** | **0/6** |

Answer and trace leak identically despite 25% different length, so it is not
size. It is that a real block *contains the answer material*, and the shipped
prompt asks whether the note "would change or improve the answer" — for a
phase-1 note against a phase-2 question about the same codebase, that is
honestly yes. `both` proves the question must **replace** the content, not
accompany it: content dominates and the benefit vanishes.

Rewording the prompt does not fix it. Two candidates were scored; one collapsed
recall to 3/18, the other changed nothing. Tuning further against six trap
examples would be fitting noise.

End-to-end on the real representation (`embed_source=composite`, threshold
0.48, judge on):

| | judge note = text (today) | judge note = question |
|---|---|---|
| recall | 0.96 | 0.75 |
| **false-fire** | **0.64** | **0.09** |
| exact / paraphrase / crosslingual | 6/6, 6/6, 6/6 | 6/6, 6/6, 6/6 |
| trap fired | 5/6 | **0/6** |
| distractor fired | 5/6 | 1/6 |
| control fired | 2/5 | 0/5 |

The recall drop is entirely the trap family, which the corpus labels
`should_recall: true` and which `grading_traps.md` showed causes real anchoring
on the wrong stack. Every legitimate recall survives.

Implemented as `Block.question_text` (written on every block type) plus
`recall.judge_note`, **defaulting to `question`**. It works retroactively with
no backfill: a reasoning block's `stimulus_text` already begins with the
question, so `judge_note_text` parses it back out, and anything that cannot be
resolved to a question falls back to the block's text — the old behaviour.

**The untested risk, stated plainly:** a block whose originating question
differed but whose content happens to answer the new one is now refused. The
corpus has no relation family covering that miss class. If recall starts
missing things it used to find, `recall.judge_note: text` is the first knob to
try.

---

## Not done

Blocked on nothing — these are simply next.

| Plan item | Why not yet |
|---|---|
| **Phase 2** — judge-scored budget (F6) | Wants re-thinking after §9. Ranking by a judge score only helps if the score is trustworthy; it now is far more trustworthy than it was this morning, so this is the natural next item. |
| **Phase 3** — prototype merging, graded decay, persistent acceptance (F1, F5, F13) | The speculative phase, and the one the standing decision-gate note argues against starting before continuity is shown to justify its cost. 3.2 (graded decay) and 3.3 (persist the acceptance signal) are cheap and independent of the LLM-merging part; 3.1 is not. |
| **Phase 4.2** — span-level corrections (F4) | Needs new output from `verifier.py` (a span, not a yes/no). Phase 4.1 landed. |
| **Phase 5** — tag/gist retrieval channel, similarity floor (F3, F10, F11) | Worth revisiting: §9 measured the gist at 17/18 recall and 0/6 trap leakage, which is the first evidence that the taxonomy carries real signal rather than being decoration. The similarity floor (5.2) is independent and cheap. |
| **Phase 7.1** — block persistence off the response path (F8) | Straight durability fix, needs no eval. |
| **Phase 7.3** — pin priority in the budget (F14) | Depends on Phase 2's ranked fill. |

## Corrections to the source documents

Recorded here because both documents are committed and now partly wrong:

- `semantic-mind.md` §5 item 1 and §6 ("the single most consequential design
  detail is that its dominant memory type is *represented* by the question that
  produced it") — measured and not supported. The composite is not what makes
  traps fire.
- `semantic-mind.md` F2's "cheap" option — re-wording the judge's relevance
  check — was tried twice and does not work. The fix is changing *what the
  judge is shown*, not how it is asked.
- `semantic-mind.md` F7's account of vector loss as transient embed failures is
  incomplete: a deterministic word-vs-token truncation bug was producing it.
- `evaluate/benchmark.md`'s false-fire 0.00 is annotated in place rather than
  edited away, under "The false-fire figure above does not describe the running
  system".
- `update_plan.md` Phase 1.2 ("Run it once as a migration. Vectors change; that
  is the point.") — not done, and should not be.

## Reproducing

```
cd cued_recall && python -m pytest
cd ../evaluate
python make_seed_blocks.py
python eval_retrieval.py --key-source composite --judge --json baseline_composite.json
python eval_retrieval.py --key-source content   --judge --json baseline_content.json
python eval_retrieval.py --key-source composite --judge --judge-note question \
                         --json baseline_composite_qnote.json
python eval_judge_notes.py --all-notes
python eval_judge_notes.py --variants
```

Every sweep above is committed as `evaluate/baseline_*.json` so the next phase
is a diff against a file rather than against somebody's memory of a terminal.
