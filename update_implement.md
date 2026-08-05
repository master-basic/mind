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
| 9 | `6ecd9c3` Show the judge the question, not the answer | Phase 4 (roadmap item 4) | the real defect |
| 10 | `79b7526` Spend the recall budget on relevance, not on nearness | Phase 2.1, 2.2 | F6 |
| 11 | `5116538` Give decay a gradient, and stop losing the evidence it runs on | Phase 3.2, 3.3 | F5, F13 |
| 12 | `372b965` Derive one block from several, and refuse the merge when it garbles them | Phase 3.1 | F1 |

232 unit tests, no servers needed, under ten seconds.

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

### 10 — Relevance decides the budget (Phase 2)

The budget was filled in cosine order *before* the judge ran, so a candidate
the judge was about to reject had already taken a slot and the slot was never
refilled. The judge could remove admitted blocks but never influence which ones
fit. Now: resolve candidates → score the whole set → rank → spend the budget
down the ranked list.

**The score is real, and free.** The judge server already computes logprobs;
reading P(yes) over the first token costs **+2.4 ms a call** (n=24) and turns a
verdict into an ordering. It separates with a wide empty band:

| relation | min | mean | max |
|---|---|---|---|
| exact | 0.945 | 0.979 | 0.996 |
| paraphrase | 0.928 | 0.978 | 0.998 |
| crosslingual | 0.899 | 0.964 | 0.988 |
| **trap** | **0.012** | **0.051** | **0.119** |

So `judge_score_floor: 0.5` sits in empty space, and is the same decision the
old yes/no parse made — **verified identical on all 24 corpus pairs**. A server
that rejects or omits logprobs is detected once and falls back to the text,
scoring 1.0/0.0, which is exactly the previous behaviour.

**Where the plan contradicted itself.** Phase 2.1 asks for a `k*4` candidate
pool; its own acceptance criterion requires TTFT within +100 ms. The judge
server is single-slot CPU at ~55 ms a call, so `k*4` costs ~0.7 s — the two
cannot both hold. Resolved by defaulting `candidate_multiplier` to **1**: this
reorders the set the judge already saw rather than enlarging it, so the
worst-case judge call count is unchanged at `k`. Widening is a knob with its
cost documented beside it.

Driven against the live stack:

| turn | latency | admitted | judged | rejected | top score |
|---|---|---|---|---|---|
| same task | 572 ms | the block | 1 | 0 | 0.990 |
| phase-2 trap | 590 ms | **nothing** | 2 | **2** | — |
| off topic | 10 ms | nothing | 0 | 0 | — |

The trap row is the `grading_traps.md` failure, refused outright.

**Phase 2.2 — token-count honesty.** Tests pin that `count_tokens` uses the
tokenizer when it answers and the estimator only when it does not; that a `0`
from the tokenizer is not mistaken for a failure; and that the estimator never
reads below a word count on prose, code, Azerbaijani or markdown. A structural
test asserts `Block.token_count` is only ever set from `_count_tokens` — any
other source is a units mismatch waiting to happen, and that bug has now cost
this project twice (once as the recall budget, once as the embed cap).

Two mislabelled counters fixed while there: the WAL's `reasoning_tokens` /
`result_tokens` held whitespace word counts (nothing reads them, so renamed to
`_words`, with a test that fails any field named `tokens` holding a word
count), and `token_sink` fed `judge.interval_tokens` a word count — ~30% low on
prose and worse on code, so the judge was waiting for materially more material
than the number claimed.

### 11 — Decay gets a gradient, and keeps its evidence (Phase 3.2, 3.3)

Done in that order because a decay rule needs a signal that survives a restart
before it is worth scoring against.

**3.3.** A block recalled into a turn and then not objected to is the only
positive evidence this system gathers on its own — and it lived in a dict on
`Pipeline`, capped at 500 entries and dropped on every restart, while the decay
rules consuming it kept running. It moves into the index as a `turn_recalls`
table plus an `uncontested_recalls` counter. Taking a turn's record *consumes*
it, so a retried turn cannot count the same evidence twice; a block already
marked corrected earns nothing, since it was recalled into the very turn that
contradicted it; and the counter keeps climbing after `verification` saturates
at "accepted", because a state cannot express "used weekly" and a counter can.

**3.2.** `recall_count > 0` made one recall, ever, a permanent exemption from
age-based purging: the system could say "used" and "never used" and nothing
between, so a block recalled once eighteen months ago outranked one recalled
every week, and the store could only grow. Utility is now recalls and
uncontested recalls converted into days of life earned, spent against time
since the block was last *useful* rather than since it was written — otherwise
a block that keeps being recalled still ages out on a fixed schedule, which is
the same bug in a different costume. The age gate still comes first, and a pin
still exempts outright.

`report_decay.py` runs the real rule against the real store and prints what a
pass would purge and why, writing nothing. On the live 396-block store the old
and new rules currently agree exactly — 37 blocks, all corrected — so the
change is a no-op there today. Worth being able to check before a pass rather
than after.

Three tests in `test_judge_decay.py` asserted the old immortality rule. They
documented behaviour rather than endorsing it, and were rewritten rather than
deleted.

### 12 — The abstraction pass, and what it got wrong (Phase 3.1)

The missing half of the whole analysis: the store records every episode and
never reduces many into one. The pass clusters blocks above
`merge_cluster_sim`, asks for one note that holds across them, stores it with a
`parents` link, and retires the originals reversibly — status flipped, vector
dropped, **file always kept even when `purge_deletes_file` is on**. Decay is
"this stopped being worth keeping"; a merge is "this is better said elsewhere";
they must not share a delete.

**Off by default, and the first real run showed why.** Three genuine
near-duplicates about DNS latency merged to:

> "setting `dns.cache_ttl=300` … reduces the cache TTL **from 30 seconds to
> 60ms**"

The originals said the TTL *was* 30s and that first-lookup *latency* fell from
840ms to 60ms. Two quantities conflated and 840 lost — a generalisation that
was never true, about to have its evidence retired behind it.

So "keep every specific" is now **checked rather than requested**.
`_lost_specifics` pools numbers, paths and dotted identifiers across the
members and refuses any merge that drops one. Re-run, the same cluster is
rejected for `dropped specifics` and all three originals stay shelved. It
cannot catch the conflation — only the dropped number — but a merge that loses
a number is not one to trust with the rest.

Three bugs the tests caught, all the same shape: `index.query` filters on
status alone, so every rule that stopped a block being a *seed* had to be
applied again to cluster *members*. Without it a pinned, corrected or
still-warm block could be pulled in and retired behind a merge it was never
eligible for — a pin especially, for which being merged is the one thing it
must prevent. Two more in my own first cut: the size guard compared against the
largest single member (rejecting every real merge — it must be the members'
*combined* size), and a refused cluster was re-reached from each of its members
for another generation each.

Verified live: a good merge takes 88 tokens to 40 in 1.9 s; the bad one is
refused.

---

## Not done

Blocked on nothing — these are simply next.

| Plan item | Why not yet |
|---|---|
| **Phase 3.1 — deciding whether to turn merging on** | Built and defaulted off. Before enabling it on a real store: snapshot, set `merge_enabled: true`, run a pass, and read the `blocks_merged` and `merge_rejected` WAL events. The one merge measured so far was factually wrong and correctly refused; that is a sample of one. |
| **Phase 4.2** — span-level corrections (F4) | Needs new output from `verifier.py` (a span, not a yes/no). Phase 4.1 landed. |
| **Phase 5** — tag/gist retrieval channel, similarity floor (F3, F10, F11) | Worth revisiting: §9 measured the gist at 17/18 recall and 0/6 trap leakage, which is the first evidence that the taxonomy carries real signal rather than being decoration. The similarity floor (5.2) is independent and cheap. |
| **Phase 7.1** — block persistence off the response path (F8) | Straight durability fix, needs no eval. |
| **Phase 7.3** — pin priority in the budget (F14) | Unblocked now: Phase 2's ranked fill is the hook it needed. A pin would raise a block's effective score at fill time. |

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
- `update_plan.md` Phase 2.1 is internally inconsistent: it asks for a `k*4`
  candidate pool and then requires TTFT within +100 ms, which on a single-slot
  CPU judge at ~55 ms a call cannot both hold. Implemented as a reorder of the
  existing pool, with widening as a documented knob.
- `update_plan.md` Phase 3.1 says to mark merged originals `truncated`. That
  status is still recallable (`index.query` accepts `shelved` and `truncated`),
  so the merged block and all its members would compete for the same budget —
  the opposite of the intent. They are retired to `purged` instead, which is
  the codebase's existing reversible "out of recall, still on disk" state.
- `semantic-mind.md` F1 assumes the abstraction is worth having once built. The
  one merge measured against the real judge was factually wrong. The pass ships
  off, with a verifier that catches dropped specifics; whether a 1.5B model can
  generalise safely at all is still open.

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
