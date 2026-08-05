# Cued Recall — analysis against Yee, Jones & McRae (2018), Ch. 9 "Semantic Memory"

This document is a critique of the Cued Recall memory middleware from the
standpoint of the semantic-memory literature, primarily the chapter by Yee,
Jones & McRae (2018). It is intentionally conceptual: it is not a bug list
but a mapping of the system onto what cognitive science says a semantic
memory *is* and what it is *for*, followed by concrete improvement
directions. No project code was modified to produce this.

---

## 1. What the system is (as memory)

Cued Recall stores *blocks*: verbatim (later judge-truncated) text slices of
the reasoning model's think-trace, result, and pasted reading material.
Blocks move hot → shelved → truncated/purged. Retrieval is: embed the current
user message → KNN in one flat vector space (k=4, cos sim ≥ 0.48) → a 1.5B
judge re-ranks candidate blocks for relevance → matched blocks are spliced
into the newest user message before the model sees it. A correction loop
flags blocks the model said were wrong and suppresses them; a decay counter
purges blocks that are old and never recalled.

## 2. The chapter's core thesis vs. the system

The chapter's central claim is that semantic memory exists for **abstraction
and generalization**: it captures regularities that hold *across* episodes
and lets you act on new instances without re-experiencing the old ones.
Abstraction (pulling out what is common) and generalization (applying it) are
two sides of the same mechanism, and the mechanism is statistical — a memory
of what *tends* to be true.

Cued Recall has the *storage* and *retrieval* halves but is missing the
**abstraction half**. It records episodes (turn-level think-traces) and
recalls them, but it never reduces many episodes into one generalized
representation. This is the single largest conceptual gap, and most of the
specific findings below are consequences of it.

---

## 3. Findings, mapped to chapter themes

### F1. Exemplar memory only — no prototype layer (prototype vs. exemplar)

The chapter contrasts exemplar (store every instance) with prototype
(store the central tendency) representations. Cued Recall is purely
exemplar: every turn spawns fresh blocks, near-identical derivations are
stored repeatedly, and nothing collapses them.

Consequences actually observed in this project:
- The `get_reading_content` docstring already describes the symptom: one
  pasted document became "a block per turn thereafter … crowd[ing] the
  vector index with near-duplicates." The fix there was to stop *creating*
  duplicates, not to *merge* them.
- A repeated question yields N near-identical blocks; KNN happily returns
  several, each consuming the judge budget and prompt tokens, so the recall
  is redundant while off-topic but related material is crowded out.
- There is no "this is the same fact as that block" step at write time.

Chapter-grounded direction: add an **abstraction/consolidation pass** that
clusters similar blocks (embedding similarity above a higher threshold, or
hash of judge gists) and, when a cluster has enough members, emits a *merged
prototype* block ("users have reported X, and the model has concluded Y in
repeated forms") — the judge already has summarization capability, but it is
currently used only to shrink a single block, never to generalize across
blocks. This maps directly to the chapter's "deriving regularities across
episodes" (McClelland et al.'s complementary learning systems in the
chapter's §"When and where is abstraction done?").

### F2. Retrieval is symmetric; asymmetric relations are exactly the trap failure

The chapter notes geometric/distributional models represent relations
symmetrically (the famous *stork → baby* vs *baby → stork* example; topic
models, unlike spatial models, can capture asymmetric relations).

The project's eval `trap` family is this phenomenon in disguise: phase‑1
material (a question, later its answer, both containing the same entities)
fires at high cosine on a phase‑2 question because the embedding space
cannot represent "the answer to this" ≠ "this." The 0.84-trap case is a
textbook symmetric-similarity false positive.

Mechanism (verified in `pipeline.py`/`utils.py`), and it is worse than the
flat-vector story: a **reasoning** block is embedded not from its own text
but from `build_stimulus(user_message, full_result, reading)` — the
*question-plus-answer composite* (pipeline.py:1886-1889, `_embed_and_store`
embeds `block.stimulus_text`). The vector that represents a reasoning block
*is* the Q+A pair. A phase‑2 question shares entities with the phase‑1
answer, so the phase‑1 reasoning block — whose embedding literally contains
the phase‑1 answer — scores 0.84 against it. The trap is not merely
"geometric similarity can't see direction"; the store's dominant block type
is embedded as the very relation that must not be symmetric.

`grading_traps.md` shows the consequence on real runs: `ocr1-trap` injected
**2,005 tokens** of the seed's client-side Tesseract stack and the RECALL-ON
answer then built a *server-side* OCR endpoint with `tesseract.js`/`express`
— anchored to the seed's wrong stack. That is the failure mode the chapter
predicts for a symmetric memory: recall supplies related-but-wrong material
and the generator anchors on it.

Chapter-grounded direction: two options, in increasing ambition.
1. Cheap: retain the judge but give it the *relation* to check — instruct it
   to answer "is this block **about** the user's current question, or merely
   sharing vocabulary/entities with it?" — turning the trap check into an
   explicit asymmetry question. (The judge is already the only thing holding
   the traps back; it deserves the explicit framing.)
2. Representational: stop embedding reasoning blocks as Q+A composites and
   embed the reasoning text itself, so a block is recalled for *what it
   says*, not for the question that produced it. Keep the composite only as a
   second channel if needed. This directly removes the mechanism that makes
   the traps score high. A further step toward the chapter's multi-hub
   organization (ATL/angular-gyrus hubs) is retrieving in two spaces
   (question-similarity and answer-similarity) and requiring both.

### F3. One flat level of representation — no hierarchy of abstraction

The chapter reviews evidence for a posterior→anterior gradient of
abstraction (mid-level features, then semantic hubs) and describes
representations at multiple granularities (feature → object → category).
Cued Recall stores everything at one granularity: the verbatim slice. The
only "abstraction" is lossy shortening by the judge, which *reduces* a block
rather than *re-classifying* it.

Chapter-grounded direction: give each block a small family:
`raw → gist (judge) → tag/category (tagger)` and let retrieval choose the
level. Currently `tagger.py` computes per-block tags that are written but
not used in retrieval. Tags are the project's closest existing analog to a
feature-based / propositional code (the chapter's "feature-based models"
that can represent shared attributes across dissimilar exemplars) — the
natural improvement is a **second retrieval signal**: recall candidates by
embedding similarity *and* by tag overlap, and let the judge arbitrate. This
would also fix the "relevant but semantically distant wording" miss class.

### F4. Retrieval = compute-everything-every-time; no per-attribute weighting

The chapter describes feature-weighted representations (features weighted by
their distinctiveness/correlations — e.g., Cree et al.). Cued Recall's
retrieval has no notion of *which features of a block matter*. A block that
is 90% right and 10% wrong is either recalled wholesale or (after
correction) suppressed wholesale — the middle ground, "recall it but flag
the wrong part," doesn't exist. The CorrectionVerifier is all-or-nothing per
block.

Worse, correction is scoped to the *turn*, not the *offending block*:
`detect_and_apply_correction` finds every block of the previous turn
(reasoning + result + reading) via `_find_turn_blocks` and `_mark_corrected`
marks them **all** corrected. A wrong answer therefore marks the source
*reading* block — material the user pasted, which the model misused — as
corrupted too, even though the source itself was never in error. That is
punishing the exemplar for the retrieval's mistake, the exact inverse of the
chapter's "weight the stable aspects" prescription.

Chapter-grounded direction: when a correction identifies the wrong part of a
block, store the correction as a *feature-level* patch rather than poisoning
the whole block — e.g., a suppressed span that the judge strips on recall, or
a `corrected_text` alternative that is returned instead of the raw text.
Today the verifier is a bare yes/no classifier (`verifier.py:37`) with no
span output, so this requires teaching the small model to also point at the
offending part (a span or paraphrase), not just vote. This mirrors the
chapter's point that semantic knowledge updates by weighting the *stable*
aspects of a concept (the dog's shape persists; its current fur length
doesn't).

### F5. Decay is arithmetic; semantic memory should decay by *utility*

Current decay is `age + recall_count` arithmetic. The chapter's functional
account would predict decay should track **predictive utility**: the
knowledge that keeps being *used and found true* should strengthen; the
knowledge that is *used and contradicted* should weaken faster. Cued Recall
has half of this (recalled-but-never-corrected strengthens the counter) but
no mechanism to *promote* blocks that consistently survive — no analog of
systems consolidation, where regular reactivation moves a memory from
episodic (details) to semantic (gist). The judge's truncation loop is the
only place anything moves.

A further wrinkle visible in the decay code (`_should_purge`, judge.py:322):
the "recalled → protected" rule is all-or-nothing — `recall_count > 0`
exempts a block from age-based purging entirely, and `keep_recall_count = 3`
exempts it from consolidation too. So a single recall makes a block
effectively immortal: it is never summarized further and never purged, no
matter how old or how redundant. The intended "utility" signal has no
gradient. In the chapter's terms the system has no way to express "this
memory was used once and is now stale" — only "used" vs "never used."

### F6. Recall budget is filled by geometry, before the judge ever runs

`recall_blocks` iterates candidates in similarity order and *skips* blocks
that don't fit the token budget — and only *after* the budget is filled does
`_filter_by_relevance` (the judge) run on the admitted set (pipeline.py:567-
592). Two consequences:

1. A block the judge will reject has already consumed its budget slot; the
   budget is not refilled from below, so a slightly-less-similar but
   judge-approved block behind an oversized or reject-worthy one is silently
   lost.
2. The budget is filled against geometric similarity, not against the
   relevance signal the system already computes — the judge's verdicts
   cannot influence *which* candidates fit, only which admitted ones are
   dropped. This is the feature-correlation problem in miniature: a huge
   irrelevant block dominates the budget while smaller relevant ones starve.

Related, independently discovered in `backfill_token_counts.py`: for blocks
written before the token-count fix, `token_count` was `len(text.split())` — a
word count. Sampled against the real tokenizer it understated tokens by ~42%
in aggregate (mean 0.77x, worst 0.52x on code/markdown). Because the same
field enforces `recall.budget_tokens`, a nominal 3,000-token budget actually
spent ~3,900. The count is now corrected via tokenizer, but the historical
data and the "word-count vs token-count" mismatch class are worth noting:
every arithmetic invariant in this system (budget, purge age, `MIN_SHRINK`)
is only as sound as `token_count` is.

Chapter-grounded direction: run the judge on the top-K candidates *first*,
score them (not yes/no), and fill the budget from the ranked-by-relevance
list.

### F7. Index/vector drift and silent, permanent vector loss

When the judge truncates a block, the *text* changes but the *embedding* is
not recomputed (`_truncate_block` only writes text/token_count/status;
judge.py:615-629). Two quiet mismatches: (a) recall is keyed on
`stimulus_text`, injected text is the current (truncated) version — the two
can describe different content; (b) after truncation the injected text and
the vector that found it diverge further. Combined with F2, note the
asymmetry: `stimulus_text` is set once at creation (for result/reading
blocks it is the truncated text itself; for reasoning blocks it is the Q+A
composite) and *never updated*, so a block's vector reflects the block as it
was born, not as it is now.

Worse, vector loss is **silent and permanent by design**. `backfill_missing_
vectors.py` documents that on a 1,812-block store, 729 shelved blocks had no
vector — 672 were empty debris from the fixed empty-turn bug, but **57 held
real content and were invisible to recall forever** until the manual script
re-embedded them. `_embed_and_store` logs a WAL error and moves on; nothing
in the runtime ever retries. There is no "blocks with no vector" health
check in the running system (the admin table shows status `shelved` whether
or not a vector exists). The system's *representation* of a memory can
therefore quietly stop existing while the memory appears healthy. The
chapter's point applies with force: the representation *is* the memory; if
the vector is gone, no amount of text on disk makes it recallable.

Chapter-grounded direction: recompute the vector whenever text changes, and
surface/retry missing vectors as a first-class health signal rather than a
manual backfill script.

### F8. Blocks are created on the request path, after `[DONE]`

In the streaming path, `_create_blocks` runs *after* the final `[DONE]`
chunk is yielded (pipeline.py:1665→1676). If the client disconnects on
receiving `[DONE]`, Starlette closes the generator and the persistence work
after the yield never runs — the turn is silently never memorized. Even in
the happy path, the user sees a complete answer before the memory write is
guaranteed; a crash in that window loses the turn's memory. (Non-streaming
path is fine: `_create_blocks` is awaited before return.) This is not a
semantic-memory issue per se, but a memory system that can silently drop the
most recent episode undermines the episodic→semantic feed the chapter
describes.

### F9. Inconsistent "last user message" for recall vs. anchoring

`get_last_user_message` (pipeline.py:1053) returns the last `user` role
message verbatim, including `<tool_response>`-wrapped turns, whereas
`_newest_user_index` deliberately skips those. So for agentic clients the
recall query can be keyed on a tool response while injection is anchored on
a different message. The recall query and the recall target disagree about
what the user asked.

### F10. Single embedding space / single model family

The chapter surveys the five distributional-model families and the point
that different corpora/tasks need different representational choices.
Cued Recall pins everything to one embedder (`nomic-embed-text-v1.5`) and
one similarity space. Cross-lingual recall is measured to work only because
the embedder happens to align it (eval: 6/6 crosslingual at threshold
0.48) — that is a property of the model, not the system. The chapter's
multi-cue argument (semantic memory integrates multiple sources; abstract
concepts especially benefit from linguistic/situational info) suggests a
cheap resilience win: a second, independent retrieval channel (tag/keyword
match + the existing embedder) so recall doesn't fail open when the embed
server wedges — the README already admits "Not implemented" for a similarity
floor, i.e. current behavior is *always call the judge*, which is exactly a
single point of failure with a 1.5–2.2 s latency cost per off-topic turn.

### F11. Tags/gist are written but never used in retrieval

`tagger.py` produces a 40-char gist + up to 3 taxonomy tags per block; they
are stored (`index.set_tags`) and shown in the admin page. Retrieval never
reads them: `recall_blocks` queries the vector index only, and
`_filter_by_relevance` feeds `block.text` to the judge, not the gist or
tags. So the taxonomy — the project's only non-vector, category-level
representation, the closest thing it has to a *feature-based* code in the
chapter's sense — is decoration. It is not used to filter, rank, or fuse
recall, and a tag/keyword match could also have been the F10 fallback
channel for free. This is the concrete instance of F3's "no second channel."

### F12. Per-turn O(n) metadata scans

`_find_turn_blocks` (pipeline.py:2106-2113) loads **all** block metadata via
`list_meta(limit=10000)` and filters in Python, per call. It is invoked
multiple times per turn: `shelve_previous_turn`, `detect_and_apply_correction`,
`verify_correction_with_model`, and `apply_accepted_verification`. On a
store of thousands of blocks that is a full-table scan of `index.db` plus a
Python filter several times per user turn — an O(store) read for an O(1)
question ("which blocks belong to conversation X, turn Y?"). `index.db` has
no index on `(conversation_id, turn_index)`, so the query could not be made
efficient even if it wanted to. The admin `/admin/stats` and the WAL
`read_all()` (router.py:109,128,250) load the entire WAL JSON file on every
stats/event/history request — unbounded as the WAL grows.

### F13. The "accepted" (recalled-uncontested) signal is in-memory and small

`apply_accepted_verification` reads `_recalled_by_turn`, a plain dict capped
at 500 entries and never persisted (pipeline.py:1918-1928, 2082). If the
server restarts between a recall and the next turn, that turn's "the model
recalled these blocks and the user didn't object" evidence is lost. And the
cap means older entries silently drop. This is exactly the kind of
consolidation signal the chapter cares about (reactivation = strengthening),
and it is ephemeral by construction. The `keep_recall_count`/purge logic in
F5 therefore runs on a signal that is partially erased on every restart.

### F14. Pins protect from decay but not from recall-count starvation

Pinned blocks are exempt from decay and consolidation, but the recall
counter (`recall_count`) still governs budget and recall statistics; a pin
does not, for example, make a block preferentially recalled or protected
from being crowded out by similar unpinned blocks at budget-fill time. In
the chapter's terms: "important" (stable, must-not-forget) is not a
retrieval-relevant dimension — it only disables forgetting.

---

## 4. What the system gets right (per the chapter)

- **Retrieval-based abstraction, MINERVA-style.** The chapter describes
  retrieval-based models that compute a derived representation at *retrieval
  time* rather than storing it. Cued Recall's judge-then-inject is a
  primitive instance of this, and the two-stage (KNN + rerank) design
  (`semantic_judge_plan.md`) is sound architecture for it.
- **Distinctness of episodic vs. semantic.** Transcripts (`chats.db`) are
  kept separate from blocks and are never decayed; blocks are. The system
  already has the right *filing cabinet*; it just doesn't do the
  abstraction that should fill the semantic drawer.
- **Correction is a real signal.** The chapter's account of how semantic
  knowledge is refined (disambiguation, expertise, contradiction) maps onto
  the CorrectionVerifier loop. Its weakness is granularity (F4), not
  existence.
- **Pins / protected blocks** acknowledge that some memories must not decay —
  the chapter's "stable knowledge" requirement.
- **The decay-by-recall signal** is a legitimate proxy for "utility" and is
  on the right track toward F5 (though its all-or-nothing form is F5's
  complaint).
- **Retrieval over-fetch and status filtering** (`index.query` over-fetches
  k×50 then filters by status in Python) is a correct handling of a genuine
  sqlite-vec footgun.
- **Source-weighted correction authority** (`_should_purge` table: manual >
  pattern > model) is a sensible treatment of an unreliable classifier
  signal — the system knows the difference between a trusted and a guessed
  correction.

---

## Changelog

- **v2 (second pass).** Verified every claim against source. Major additions:
  - F2: reasoning blocks are embedded from the *Q+A composite*
    (`build_stimulus`), not their own text — the mechanism behind the 0.84
    trap scores; plus `grading_traps.md` evidence of real anchoring.
  - F4: correction is scoped to the whole previous turn — the reading/source
    block is marked corrected alongside the wrong answer; the verifier has
    no span output, so feature-level correction needs new model output.
  - F5: `recall_count > 0` makes a block immortal; `keep_recall_count = 3`
    halts consolidation; no gradient.
  - F6: budget is filled by geometry *before* the judge runs; rejected
    blocks are not refilled; token-count understatement (~42%) history.
  - F7: truncation never recomputes the vector; missing vectors are silent
    and permanent (729/1812 blocks affected, 57 with real content).
  - F11 (new): tags/gist written but never used in retrieval.
  - F12 (new): O(n) per-turn metadata scans (`_find_turn_blocks`), full-WAL
    reads per admin request.
  - F13 (new): the recalled-uncontested "accepted" signal is in-memory,
    capped at 500, lost on restart.
  - F14 (new): pins disable forgetting but do not improve retrieval.
  - Roadmap expanded to 11 items, re-prioritized around F2's mechanism fix.

---

## 5. Prioritized improvement roadmap

1. **Embed reasoning blocks from their own text, not Q+A composites (F2).**
   This removes the mechanism that makes the trap family score 0.84 against
   phase‑2 questions. Highest-leverage change: it fixes the store's dominant
   block type at the representation level.
2. **Abstraction pass (F1, F5).** Cluster near-duplicate blocks at write or
   during judge consolidation; when a cluster is mature, emit a merged gist
   block and let members decay. This is the missing "prototype."
3. **Budget by relevance, not geometry (F6).** Ask the judge for a score
   (not yes/no) on the top-K candidates, rank, then fill the token budget
   from the ranked list; refill slots freed by rejected blocks.
4. **Asymmetry-aware trap handling (F2).** Re-word the judge's relevance
   check to distinguish "about this" from "shares vocabulary/entities with
   this"; measure trap precision separately in the eval corpus.
5. **Move block persistence off the response path (F8).** Fire-and-forget
   the `_create_blocks` task with the WAL as the durable receipt, or await
   persistence before emitting `[DONE]`.
6. **Feature-level corrections (F4).** Use `verifier.py`'s span info to
   store a corrected span/alternative instead of binary block suppression.
7. **Make the taxonomy a retrieval channel (F3, F11, F10).** Use tagger
   gist/tags as a second, judge-arbitrated retrieval signal — it is the
   only non-vector representation in the system and is currently dead
   weight; it is also the natural resilience fallback if the embed server
   wedges.
8. **Consistency and drift (F9, F7).** Make `get_last_user_message` and
   `_newest_user_index` agree about what "the user's question" is; recompute
   the embedding whenever block text changes (truncation, correction); treat
   missing vectors as a first-class, surfaced health signal with retry,
   since vector loss is silent and permanent today.
9. **Fix the O(n) per-turn scans (F12).** Index `index.db` on
   `(conversation_id, turn_index)`, replace `_find_turn_blocks` with a real
   query, and stop reading the whole WAL per admin request.
10. **Persist and size the acceptance signal (F13).** Store
    `_recalled_by_turn` (or fold "recalled then uncontested" into the block
    metadata) so restart doesn't erase the reactivation evidence the decay
    logic depends on.
11. **Give decay a gradient (F5).** Replace the all-or-nothing
    `recall_count > 0` immortality rule with a utility score (recalls,
    uncontested-recall, age, correction history) so a once-recalled memory
    can still become stale.

## 6. Meta-observation

The system is architecturally a *retrieval-based exemplar model with a
statistical embedder and a learned reranker* — which is, in the chapter's
taxonomy, a hybrid of the passive-co-occurrence and retrieval-based
families, plus a small slice of the predictive family (the judge). What is
absent is the *latent-abstraction* and *Bayesian* families — i.e., exactly
the mechanisms the chapter argues do the abstraction work. In one sentence:
**Cued Recall is a well-built episodic store wearing semantic memory's
clothes; the missing piece is generalization across stored episodes.** And
the single most consequential design detail is that its dominant memory type
(the reasoning block) is *represented* by the question that produced it, not
by what it actually says — the memory system is indexing its input instead
of its content.
