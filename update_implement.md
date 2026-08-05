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

### Every plan item, and where it stands

`update_plan.md` in full, so "what is left" is a lookup rather than a reading.

| Plan item | State | Where |
|---|---|---|
| 0.1 unit-test seam | **done** | §1 |
| 0.2 missing-vector health signal | **done** | §2 |
| 0.3 baseline run | **done** | §8 |
| 1.1 embed_text vs stimulus_text | **done**, as a switch not a swap | §6 |
| 1.2 re-embed the store as a migration | **refused** — measured, and the premise is wrong | §8 |
| 1.3 asymmetry wording for the judge | **tried, failed** — the fix was the note, not the wording | §9 |
| 1.3 asymmetric trap rows in the corpus | **not done** | below |
| 2.1 score, rank, then fill | **done** | §10 |
| 2.2 token-count honesty | **done** | §10 |
| 3.1 prototype/abstraction pass | **done, shipped off** | §12 |
| 3.2 graded utility decay | **done** | §11 |
| 3.3 persist the acceptance signal | **done** | §11 |
| 4.1 scope correction by block type | **done** | §4 |
| 4.2 span-level corrections | **not done** | below |
| 5.1 tag/gist as a retrieval channel | **not done** | below |
| 5.2 similarity floor + embed-failure fallback | **done** — by deepseek, 2026-08-05 | §14 |
| 6.1 re-embed when text changes | **done** | §6 |
| 6.2 consistent "user's question" | **done** | §5 |
| 7.1 blocks off the response path | **done** — by deepseek, 2026-08-05 | §15 |
| 7.2 kill the O(n) scans | **done** | §3 |
| 7.3 pin priority in the budget | **done** — by deepseek, 2026-08-05 | §13 |
| *cross-cutting* config surface | **partly** — keys for shipped phases only | below |
| *cross-cutting* eval corpus rows | **not done** | below |
| *cross-cutting* README / ARCHITECTURE / CHANGELOG | **not done** | below |

### Findings, F1–F14

| | Finding | State |
|---|---|---|
| F1 | no prototype layer | **done** — merge pass, off by default (§12) |
| F2 | symmetric retrieval / composite embedding | **measured, refuted**; the real defect was the judge's note (§8, §9) |
| F3 | one flat level of representation | **open** — Phase 5.1 |
| F4 | correction is all-or-nothing | **half** — turn scope fixed (§4); span-level open (4.2) |
| F5 | decay is arithmetic, not utility | **done** (§11) |
| F6 | budget filled by geometry before the judge | **done** (§10) |
| F7 | index/vector drift and silent vector loss | **done**, and the cause was not what F7 said (§2, §6, §7) |
| F8 | blocks created after `[DONE]` | **done** — by deepseek, 2026-08-05 (§15) |
| F9 | inconsistent "last user message" | **done** (§5) |
| F10 | single embedding space, no fallback | **done** — by deepseek, 2026-08-05 (§14) |
| F11 | tags/gist written but never used | **open** — Phase 5.1, and §9 found the gist carries real signal |
| F12 | per-turn O(n) scans | **done** (§3) |
| F13 | acceptance signal is in-memory | **done** (§11) |
| F14 | pins do not help retrieval | **done** — by deepseek, 2026-08-05 (§13) |

### Config keys added

All shipped with the measured value as the default. Documented in
`config.example.yaml` next to the number that justifies them.

| Key | Default | What it decides |
|---|---|---|
| `embed_source` | `composite` | which text feeds the vector index (§8) |
| `embed_token_limit` | `1024` | first-pass word cap on either embed text |
| `embed_ctx_tokens` | `2048` | embedder window; overwritten from `/props` at startup (§7) |
| `recall.judge_note` | `question` | what the judge reads as the note (§9) |
| `recall.candidate_multiplier` | `1` | how many candidates the judge scores (§10) |
| `recall.judge_score_floor` | `0.5` | relevance score below which a candidate is dropped (§10) |
| `judge.recall_record_ttl_s` | `604800` | how long an unconsumed recall record is kept (§11) |
| `judge.utility_decay` | `true` | decay by earned utility rather than "ever recalled" (§11) |
| `judge.utility_recall_weight` | `30.0` | days of idleness one recall earns |
| `judge.utility_uncontested_weight` | `60.0` | extra days for an uncontested recall |
| `judge.utility_floor` | `0.0` | utility at or below which a block purges |
| `judge.merge_enabled` | `false` | the abstraction pass (§12) |
| `judge.merge_cluster_sim` | `0.90` | similarity at which two blocks are the same ground |
| `judge.merge_min_cluster` | `3` | how many near-duplicates before generalising |
| `judge.merge_min_age_s` | `604800` | how settled a cluster must be |
| `judge.merge_max_per_pass` | `5` | merges per pass |
| `recall.pin_priority` | `true` | whether a pin breaks ties in the ranked recall fill (§13, by deepseek) |
| `recall.floor` | `0.0` | cosine floor below which the judge is skipped; **off** — the plan's 0.30 cannot fire below the 0.48 threshold and no safe value is measured yet (§14, by deepseek) |
| `recall.tag_channel` | `true` | the gist/tag keyword channel; used as the embed-failure fallback (§14, by deepseek) |

The plan also names `recall.floor` and `recall.tag_channel`; those now exist —
see §14 for why `floor` shipped disabled.

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

### 13 — Pin priority in the budget (Phase 7.3, by deepseek)

Continued on 2026-08-05 by deepseek; recorded here for later audit.

F14 in one line: a pin was exempt from decay but bought nothing at retrieval.
The budget was spent down the ranked list in `recall_blocks`, so an
equally-scored unpinned block that sorted first could take the last slot and a
pinned block behind it was skipped as oversized.

The fix makes the pin the **tie-break** in the ranked fill, not the primary
key: `kept.sort(key=lambda t: (score, int(block.pinned)), reverse=True)` (and
`(similarity, pinned)` on the no-judge path). Relevance still decides what
fits — a 0.95 unpinned block still beats a 0.55 pinned one — a pin only
decides between equals. That was a deliberate reading of the acceptance
criterion ("a pinned block is admitted before equally-scored unpinned ones")
and keeps Phase 2's measured relevance-first property intact; the alternative
("all pinned first") would displace a genuinely relevant block on an
irrelevant pin.

`recall.pin_priority`, default `true`, restores the old order exactly when
off. Documented in `config.example.yaml`.

Four tests in `test_recall_ranking.py`: a pin is admitted before an
equally-scored unpinned under budget pressure; relevance still beats a pin;
`pin_priority: false` restores the old cosine order; and a pin breaks ties on
the no-judge path too. `pytest` green at 236.

### 14 — The floor, and the keyword channel (Phase 5.2, by deepseek)

Continued on 2026-08-05 by deepseek; recorded here for later audit.

**The floor is a measurement hostage, so it shipped off.** The plan proposes
`recall.floor: 0.30`. That number cannot fire: it sits below the shipped
retrieval threshold (0.48), and `index.query` already refuses everything below
the threshold, so no candidate ever has a best similarity under 0.30 — the
judge-skip is unreachable. Same self-contradiction as Phase 2.1's `k*4`.

Worse, the measured corpus has no safe value *anywhere*. `baseline_composite`
per-relation mean top-sim: control (off-topic, should not recall) **0.4996**,
crosslingual (the weakest legitimate family) **0.6408**, distractor 0.6409,
trap 0.7559. A floor high enough to remove the off-topic tax (which live
traffic measured at 0.49–0.57, throughput.md) lands right against crosslingual
— with n=6 and only means published, I will not pick the number that can
silently amputate a legitimate family. `recall.floor` therefore defaults to
**0.0 (off)**: the mechanism ships (best cosine below floor → no judge call, a
`recall_floor` WAL event, nothing recalled), tests exercise it at 0.60, and
0.60 is documented in `config.example.yaml` as the candidate to confirm on the
widened corpus before enabling.

**The embed-failure fallback ships on.** When the embed server errors,
`recall_blocks` used to return `[]` — the whole store silently vanishing for
the turn. Now `index.keyword_query` (new) matches the query's distinctive words
against each block's gist and tags — both already columns in the index, so no
block file is opened and a per-turn fallback stays off the O(store) path. The
keyword candidates flow through the exact same judge / ranked-budget path as
vector candidates, so a CPU 1.5B relevance model still arbitrates what an
embed outage surfaces. `utils.distinctive_terms` drops function words and
fragments (a stopword query matches nothing, deliberately), keeps the longest
words first, and is bounded to characters safe for SQL LIKE. `recall.tag_channel`
(default `true`) gates it; `recall_budget` events now carry `source:
"vector"|"keywords"` so an outage recall is visible in the admin panel rather
than indistinguishable from a normal empty one.

Phase 5.1 reuses `keyword_query` and `tag_channel` as the second candidate
source in the normal path — the fallback was the excuse to build the channel,
not its purpose.

Seventeen tests in `test_recall_fallback.py` (terms, keyword query scoring and
status filtering, floor short-circuit incl. the "at the floor judge still runs"
boundary and the default-off guarantee, fallback on/off, corrected-block
filtering on the fallback path). `pytest` green at 253.

### 15 — Blocks off the response path (Phase 7.1 / F8, by deepseek)

Continued on 2026-08-05 by deepseek; recorded here for later audit.

`_create_blocks` ran after `[DONE]` was yielded, so a client disconnecting on
receiving `[DONE]` cancelled the generator and the turn was never memorized —
the very latest episode silently dropped. The fix splits `_create_blocks`
into a durable part and a deferred part:

- `_create_blocks(..., defer_embeds=True)` writes the blocks + index metadata
  (the fast, local part) and returns the embeddable blocks. In the streaming
  path this now runs *before* the finish_reason / `[DONE]` chunks, so a client
  that stops reading at the completion signal is guaranteed the turn is in the
  store.
- The embedding — a network call that can take seconds — is handed to
  `asyncio.create_task(self._embed_blocks(...))`, scheduled *before* `[DONE]`
  so it survives the generator being closed on disconnect. A dropped embed is
  recorded (`embed_store_error`) and surfaces under `blocks_missing_vectors`,
  repairable via the backfill path, rather than becoming silent memory loss —
  exactly the F8 acceptance wording.
- The non-streaming path is untouched: `_create_blocks` is already awaited
  before the response returns, so it keeps the default inline embed.

No config flag: this changes *when* the same writes happen, not what is
stored, and there is no old behaviour a user would want to keep (reverting to
it means the memory-loss bug). The only observable difference is a few ms of
local sqlite writes added before the completion chunks.

The post-`[DONE]` tail (`tps_sink`, `token_sink`, `chat_sink`,
`turn_completed`) stays where it was: those are not memory writes, and a
disconnect dropping them is acceptable.

Four tests in `test_blocks_before_done.py`: the acceptance test drives the
real streaming generator through `process_turn` with a canned upstream
(httpx patched), reads up to `[DONE]`, calls `agen.aclose()` the way Starlette
does on disconnect, and asserts the blocks are in the store, the post-`[DONE]`
tail really was skipped (no `turn_completed`), and the deferred embed failed
loudly against the `embed=None` fixture. Plus `defer_embeds=True` persists
without touching the embedder, `_embed_blocks` runs the deferred embeds with
the same fail-recorded contract as inline, and the default still embeds
inline. `pytest` green at 257.

---

## Not done

Blocked on nothing — these are simply next. Roughly in the order I would take
them.

### Code

| Plan item | Size | Why not yet |
|---|---|---|
| **5.1** tag/gist as a retrieval channel (F3, F11) | medium | Newly interesting: §9 measured the gist at 17/18 recall and **0/6** trap leakage, the first evidence the taxonomy carries real signal rather than being decoration. §14's `keyword_query` is the channel; 5.1 wires it into the normal path as a second candidate source. |
| **4.2** span-level corrections (F4) | large | Needs new output from `verifier.py` — a span, not a yes/no — so it is a model-output change, not a plumbing one. |
| **3.1** decide whether to enable merging | judgement | Built and off. Snapshot, set `merge_enabled: true`, run a pass, read the `blocks_merged` and `merge_rejected` events. The one merge measured was factually wrong and correctly refused — a sample of one. |

### Evals and docs (the plan's cross-cutting section)

Both were in `update_plan.md` from the start and neither has been touched. They
are listed separately because they are not features and are easy to lose.

| Item | State |
|---|---|
| **Corpus rows** — asymmetric `trap` rows (1.3), tag-overlap rows (5.1), multi-claim correction rows (4.2) | Not done. `corpus.jsonl` is unchanged at 6 seeds / 35 probes. **This is the binding constraint on everything measured so far**: every headline number in §8–§10 rests on n=6 per relation, and both the judge-note fix and the merge verifier were validated against that same small set. Widening the corpus is worth more than the next feature. |
| **Regrade `grading_traps.md`** | Not done. It records the RECALL-ON answers anchoring on the seed's wrong stack; §9 and §10 should have changed that, and re-running it is the end-to-end confirmation the retrieval sweep cannot give. |
| **`README.md`** — features | Not done. It still describes the similarity floor as "Not implemented" (correct) but says nothing about `judge_note`, `embed_source` or utility decay. |
| **`ARCHITECTURE.md`** — embed_text vs stimulus_text, merge pass, utility decay | Not done. The three sections the plan names are exactly the three concepts a reader now cannot get from it. |
| **`CHANGELOG.md`** | Not done. Twelve behaviour-changing commits are unlisted. |

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
python report_decay.py               # what a judge pass would purge, and why
python report_decay.py --old-rule    # what the pre-utility rule would purge
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
