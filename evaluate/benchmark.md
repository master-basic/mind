# Benchmark design

Two harnesses, kept apart on purpose, because conflating them produces numbers
that mean nothing.

**Retrieval** — does the right block come back? No generation involved, so it is
deterministic, runs in seconds, and can be swept across thresholds. This is what
answers the question of where `recall.threshold` belongs.

**End to end** — does having the block actually help? Slow, noisy, and needs
repeats before any of it means anything.

Retrieval comes first: it is cheap, it produces the curve, and if retrieval is
bad there is nothing downstream worth measuring.

```
python eval_retrieval.py --endpoint http://127.0.0.1:8082
```

Five seconds for a threshold sweep from 0.30 to 0.94 with a recall rate and a
false-fire rate at each step, plus `retrieval_sweep.csv` to plot. `--fake`
self-tests the harness with a bag-of-words embedder and no servers running; the
shape it produces is the check that the harness measures what it claims —
`exact` scores 1.00, `trap` scores high because it shares most of its
vocabulary, and `crosslingual` collapses to noise.

## The corpus is the actual work

The scripts are mechanical. What makes an eval honest is the corpus, and every
family needs adversarial members. Six relation types:

| relation | requirement |
|---|---|
| `exact` | identical restatement. Must recall — if this fails something is broken |
| `paraphrase` | same problem, different words. Must recall. This is the real target |
| `crosslingual` | same problem in Azerbaijani. Must recall |
| `trap` | same vocabulary, different answer — phase 1 solved, phase 2 not; digest auth against basic auth. Retrieval firing here is *correct*; blindly reusing the old conclusion is the failure. This family is the whole reason to hand-grade |
| `distractor` | high lexical overlap, unrelated problem. Must not fire |
| `control` | nothing to do with anything. Must not fire |

Most published work in this area measures only whether recall fires. The number
nobody reports is the **false-fire rate**, and it is the one that decides whether
the rest is credible. The distractor prompts are written deliberately to share
surface words with a real family while meaning something entirely different —
that is what separates an embedding evaluation from a vibe check.

Twenty-five rows is enough to start; ~60 (10 families × 6) before publishing
anything.

## Where the prompts come from

Written by hand, from real recurring work — not as a fallback but because it is
the better option. Correctness can be graded, which is the hard part of any eval,
and the distribution matches the traffic the system actually serves: this store
holds infrastructure debugging, so a math benchmark would measure the wrong thing
entirely. The seeded families come from problems that genuinely recur here —
ADVPN routing, DKIM selectors, nginx failover, rclone mounts, digest auth.

For an auto-graded objective layer on top, MBPP or HumanEval are the right
choice: Python problems with unit tests, so correctness needs no human. Grouping
them into families by technique (dynamic programming, string parsing, recursion)
gives reasoning that really is reusable within a family. GSM8K is the wrong
choice — arithmetic reasoning does not transfer the way this design assumes.

One rule worth stating: **do not use a larger model to generate the paraphrases
unmonitored.** Model paraphrases share embedding-space structure with the
original in ways human rewording does not, and the result is an inflated recall
rate. Write them by hand, or generate and then rewrite each one.

## Protocol details that decide whether the numbers mean anything

**The baseline arm bypasses the middleware entirely** — point it at `:8080`
directly. No config flag, no code path that might differ. Two URLs, same model,
same seed.

**Wipe the store between repeats.** The tmpfs blocks, the SQLite index and the
snapshot all have to go, or repeat 2 starts warm and the numbers drift.

**`temperature: 0` and a fixed seed**, while expecting residual nondeterminism
from llama.cpp batching — hence `--repeats 3` and medians rather than means.

**Paired analysis, not group means.** `analyse.py` pairs each probe against
itself across arms. Between-prompt variance is enormous — some questions produce
800 tokens of reasoning, others 12,000 — so comparing group averages across 20
prompts would drown any real effect. The bootstrap CI is flagged with `*` only
when it excludes zero.

## The metrics that matter

| metric | where | what it tells you |
|---|---|---|
| recall rate against false-fire rate | retrieval | the threshold answer |
| mean top similarity per relation | retrieval | whether crosslingual is near firing or nowhere close |
| Δ `think_chars`, paired | e2e | the "saves N% of reasoning" claim |
| Δ latency | e2e | whether recall overhead eats the savings |
| trap-family answer quality | hand-graded | anchoring failures |

The last row has no script because it cannot have one. The trap-family answers
have to be read: if the model recalls the phase-1 block and confidently answers
about phase 1 when the question was about phase 2, that is the real limitation of
the design, and no counter will surface it.

---

# Results

## Retrieval, embedding only

`retrieval_sweep.csv`, nomic-embed-text-v1.5, k=4. Recall and false fires do not
separate cleanly: false fires reach zero only at 0.86, where recall has already
fallen to 0.58. At the shipped threshold of 0.62 recall is 0.96 and the
false-fire rate is 0.55 — better than half the prompts that should retrieve
nothing retrieve something.

That is the finding, not a footnote to it. It is why `recall.judge_enabled`
exists, and why claiming semantic recall "works" on the strength of the recall
column alone would be dishonest.

## Retrieval, with the semantic judge

Measured 2026-07-28, Qwen2.5-1.5B-Instruct on CPU as the relevance judge:

```
python eval_retrieval.py --endpoint http://127.0.0.1:8082 --judge
```

| threshold | recall | false-fire | recall +judge | false-fire +judge |
|---|---|---|---|---|
| 0.48 | 1.00 | 0.82 | 0.75 | **0.00** |
| 0.62 (shipped) | 0.96 | 0.55 | 0.71 | **0.00** |
| 0.70 | 0.88 | 0.18 | 0.63 | **0.00** |
| 0.86 | 0.58 | 0.00 | 0.50 | 0.00 |

The judge takes the false-fire rate to zero at every threshold. It is not
trading recall away at random to get there — the entire recall cost is one
family:

| relation | fired without judge | fired with judge |
|---|---|---|
| exact (6) | 6 | 6 |
| paraphrase (6) | 6 | 6 |
| crosslingual (6) | 6 at 0.48 | 6 |
| trap (6) | 6 | **0** |
| distractor (6) | 2–6 | **0** |
| control (5) | 0–5 | **0** |

Every legitimate recall survives. Every distractor and control is refused. The
judge rejects the whole trap family, and the corpus labels those
`should_recall: true`, which is where the 0.96 → 0.71 comes from.

Whether that is a loss is a real question, not a rounding error. A trap probe
asks about phase 2 while the stored block concluded something about phase 1; the
corpus counts retrieval as correct there because the block *is* about the same
subject, and expects the model not to reuse the conclusion blindly. The judge
reads the same pair and says it does not apply. That is the failure mode the
trap family exists to expose, being refused one stage earlier than the corpus
assumed it would be.

The second finding is more useful than the headline. Because the judge removes
false fires outright, the embedding threshold no longer has to carry that job —
and dropping it to 0.48 recovers the crosslingual family completely, 6 of 6
against 3 of 6 at the 0.70 the embedding alone would need. Azerbaijani recall
was written up here as "not working"; it works, at a threshold that was
previously unusable because of the noise it let through.

Cost: ~150 judge calls for the sweep, serial, on a CPU 1.5B. In the pipeline the
calls for one turn run concurrently through the shared small-model semaphore,
bounded by `recall.judge_timeout_s` (5 s), and a timeout keeps the block.

## The false-fire figure above does not describe the running system

Measured 2026-08-05. **The 0.00 is an artefact of the harness, not a property
of the judge.** It should be read as an upper bound on the judge's ability, not
as what the pipeline does.

Everything above used each seed's *prompt* on both sides of the pipeline: as the
text embedded to represent a stored block, and as the note shown to the judge.
No block this system writes looks like that. A reasoning block's key is
`build_stimulus(question, answer, reading)` and its text is a think trace of
several thousand characters; a result block's key and text are the answer. So
the sweep measured question-to-question similarity and question-to-question
relevance, and the geometry the store actually has had never been swept at all.

`make_seed_blocks.py` closes that gap: it runs each seed prompt once against the
reasoning model with no middleware and records the think trace and the answer,
so a seed can be represented the way the store really represents it.
`eval_retrieval.py --key-source {prompt,composite,content}` then sweeps the two
settings of `embed_source` on the representation each one actually produces.

At the shipped operating point (threshold 0.48, k=4, judge on), with the judge
shown `block.text` as `_filter_by_relevance` does:

| | prompt (harness) | composite (shipped) | content (proposed) |
|---|---|---|---|
| recall | 0.75 | 0.96 | 0.96 |
| false-fire | **0.00** | **0.64** | **0.45** |
| trap fired | 0/6 | 5/6 | 5/6 |
| distractor fired | 0/6 | 5/6 | 4/6 |
| control fired | 0/5 | 2/5 | 1/5 |

The production evidence agrees with the second and third columns. `grading_traps.md`
records `ocr1-trap` injecting 2,005 tokens of the seed's client-side stack into a
phase-2 question and the answer anchoring on it. If the judge really refused 6 of
6 traps in production, that could not have happened.

### It is not length, and it is not the embed key

`eval_judge_notes.py` isolates it. Holding the pair fixed and varying only what
the judge is shown as the note:

| note shown to the judge | chars (ocr1) | legitimate recall kept | traps leaked |
|---|---|---|---|
| seed prompt (harness) | 208 | 18/18 | **0/6** |
| the answer | 8,463 | 18/18 | **5/6** |
| the think trace (production) | 10,627 | 18/18 | **5/6** |

Answer and trace behave identically despite a 25% length difference, so it is not
the size of the note. It is that a real block *contains the answer material*, and
the shipped prompt asks whether the note "contains information that would change
or improve the answer" — for a phase-1 note against a phase-2 question about the
same codebase, that is honestly yes. The refusal cases are listed afterwards as
exceptions, and on a 1.5B model reading 8,000 characters they are what gets lost.

Legitimate recall is untouched in every variant. This is specifically and only a
failure to say no.

### Rewording the prompt did not fix it

Two candidate rewordings, scored on the same pairs (`--variants`):

| wording | legitimate recall | traps leaked |
|---|---|---|
| shipped | 18/18 | 5/6 |
| task-match ("same task, or different task on the same system?") | **3/18** | 1/6 |
| same-step (stage test first, as the question) | 17/18 | 5/6 |

One collapsed recall, the other changed nothing. Tuning further against six trap
examples would be fitting noise, so the shipped wording stands and the problem is
recorded rather than papered over.

### What this changes

`semantic-mind.md` ranks "embed reasoning blocks from their own text" first, on
the theory that the composite is what makes traps score 0.841 and that fixing the
representation stops them scoring high in the first place. Measured, it does not.
Trap mean top-similarity goes 0.756 → 0.698 under `content`, still far above the
0.48 threshold, and *every* relation drops by roughly the same amount — so the
separation between exact and trap gets slightly **worse** (0.140 → 0.109), not
better. `content` is modestly better on false fires (0.64 → 0.45) on n=11, which
is a real but small effect and not the mechanism the roadmap claimed.

So `embed_source` stays `composite`. Both texts are stored on every block, so the
decision is reversible with a re-embed if a larger corpus says otherwise.

The defect that is real, and larger than the analysis estimated, is the judge's
inability to refuse related-but-wrong material when it reads an actual block.
That is roadmap item 4, not item 1, and it is not a prompt-wording problem.

Reproducing:

```
python make_seed_blocks.py
python eval_retrieval.py --key-source composite --judge --json baseline_composite.json
python eval_retrieval.py --key-source content   --judge --json baseline_content.json
python eval_judge_notes.py --variants
```

Two harness bugs found and fixed while doing this. The embedding cache was keyed
on the *number* of corpus rows, so any experiment that changed the key text
while keeping the row count silently scored the previous run's vectors — it is
keyed on a hash of the text now. And the sweep table printed a `│`, which is
un-encodable in the console's cp1252 and crashed the run whenever output was
redirected to a file.

## Judge throughput

The review's "a large store will never finish a pass" is answerable now that
`judge_pass` carries counters. Measured on a 164-block store, manual pass
(`min_age=0`, so nothing is age-gated out):

```
visited=163  model_calls=1  truncated=1  purged=0  elapsed_s=7.3  stopped_early=false
```

163 blocks in 7.3 s because the gates are cheap and the model is the exception:
75 skipped on type, 45 on size, 42 kept for being recalled 3+ times, and exactly
one was worth a rewrite. A pass costs the model roughly what the store has newly
accumulated, not what it holds. `max_pass_seconds` bounds the case where that
stops being true, and `stopped_early` in the log is how you would know.

## Correction detection

`python eval_correction.py --no-model`, 34 hand-written rows (17 corrections,
17 not), against the 17 shipped patterns:

| metric | value |
|---|---|
| precision | 0.87 |
| recall | 0.76 |
| false-positive rate | 0.12 (2 of 17) |

The two false positives are real and worth naming:

- *"no, keep going with the second option"* — matched by `^\s*(no|nope|nah)[,.!]`.
  A leading "no" is often a redirection, not a complaint.
- *"why doesn't work stealing help here?"* — matched by `(doesn't|does not) work`,
  where "work" is part of a noun phrase.

Left unfixed on purpose. Tuning regexes against 34 rows someone wrote to test
those regexes is how you get a pattern list that scores well on its own test set
and no better in the field. The structural fix went in instead:
`judge.corrected_grace_s` means a pattern-sourced correction stops a block being
recalled but cannot purge it for 24 hours, and never at all if it was ever
recalled. A false positive now hides a memory for a day rather than deleting it.

The four false negatives ("that gave me: package not found", "no such file on my
system", "are you sure? the docs say 10", "it returns 404 when I try that") are
exactly the phrasings the patterns are documented as not reaching, and are the
verifier's job. Score it with the judge server up:

```
python eval_correction.py
```

**Caveat that applies to both numbers:** these rows are hand-written, and 34 of
them. They bound the shape of the problem, not the rate. `--from-chats` mines
real user messages out of a live `chats.db` for labelling, which is the only way
the negative half stops reflecting what someone thought to test.

## End to end: does having the block help?

Measured 4 August 2026. Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M
as the reasoning model, `threshold: 0.48`, `judge_enabled: true`, `k: 4`.

The honest headline: **retrieval works, the cost is now known, and the benefit
is still unmeasured.** A recalled block costs about 1,300 prompt tokens and
several seconds a turn -- that part is solid. Whether it shortens the reasoning
trace this run cannot say: the median moved the wrong way but the confidence
interval spans zero on 9 usable pairs. That is an underpowered result, not a
negative one, and the difference matters. One outright anchoring failure turned
up in four hand-graded traps.

What would settle it: more probes, `max_tokens` high enough that nothing is
censored (11 of 32 rows hit the cap here), and a store the size of a real one
rather than the 6 seeds this used. A model that emits `<think>` inline would
also remove the field-parsing hazard described below.

### A third arm, because two cannot answer it

The shipped two-arm design compares the middleware against a bare
`llama-server`, which varies the memory *and* the ~184 prompt tokens of
`web_search` / `web_fetch` definitions the middleware injects. Those were
measured apart by adding a middleware instance with recall disabled (`k: 0`,
`threshold: 0.99`, its own empty store, port 8011):

| | median delta | 95% CI |
|---|---|---|
| middleware + tool definitions, no memory | +184 prompt tokens, +0.6 s | tight |
| memory on top of that | +1,334 prompt tokens, +6.4 s | [874, 1458], [0.0, 8.9] |

This run never wiped the live store. It ran against a second middleware instance
on its own store, sharing the three llama servers -- the protocol wants an empty
store per repeat, and 1,812 real blocks carrying recall counts, pins and
verification history are not something `/admin/import` can put back (new ids,
verification reset by design, every block landing `truncated`).

### Does memory reach the model

Read off the prompt-token gap against the same probe with recall off, so it is
measured rather than asserted. 35 probes, one repeat:

| relation | fired | median tokens injected |
|---|---|---|
| exact (6) | 6/6 | 1,716 |
| paraphrase (6) | 6/6 | 1,466 |
| crosslingual (6) | **6/6** | 1,464 |
| trap (6) | 5/6 | 1,656 |
| distractor (6) | 3/6 | 1,148 |
| control (5) | 3/5 | 2,081 |

The judge refused 58 of 134 candidates (43%). Crosslingual firing 6 of 6 is the
`retrieval_sweep` claim reproduced on live traffic rather than synthetic
vectors, and it is the strongest result here.

The distractor and control rows are **not** a false-fire measurement and must
not be read as one. The treatment arm writes its own blocks as it goes, so by
the late probes the store held ~84 blocks rather than the 6 seeds, and a fire
there may be legitimate recall of an earlier probe. The shipped harness has the
same property; it wipes between repeats, not between probes. Measuring false
fires end to end needs a store frozen after warm.

### Does it shorten the reasoning

Undetermined, and the run is too small to settle it. Re-measured on a store
wiped and re-warmed to exactly the 6 seeds, so only a seed could be recalled,
`max_tokens` 2048, 16 probes:

| relation | n | median delta think_chars | 95% CI |
|---|---|---|---|
| ALL | 9 | **+395** | [-604, +1012] |
| exact | 2 | +867 | [388, 1345] |
| paraphrase | 2 | +704 | [395, 1012] |
| crosslingual | 2 | -1,906 | [-3612, -200] |
| trap | 3 | +511 | [-604, 889] |

Negative would mean memory made the model think less. The overall interval
spans zero, and the two families memory is supposed to help most -- exact and
paraphrase restatements -- move the wrong way.

Two limits sit on top of that and both cut against reading it as a firm result.
**n is 2 or 3 per family**: a bootstrap CI on two points resamples between two
numbers and the stars it prints mean nothing, which is why only the ALL row is
worth quoting. And **11 of 32 rows hit the 2048-token cap**; censored pairs are
excluded from the length comparison, which is what shrinks n from 16 to 9. What
this run establishes is that the effect is not large and not reliably negative,
not that it is exactly +395.

### The metric that was silently zero

This model's think trace arrives in `message.reasoning_content`, not wrapped in
`<think>` tags inside `content` -- llama.cpp parses it out for any model whose
template declares reasoning. `eval_e2e.py` regexed the tags out of `content`
only, so `think_chars` read 0 in every arm on this stack, which is
indistinguishable from "memory changed nothing" and was in fact the harness not
looking. A first attempt at this run also capped generation at 512 tokens, and
since the trace never finished, `content` came back empty in 102 of 105 rows and
every length metric read as a flat zero. Both are fixed; the middleware itself
always read both fields.

### Trap answers, hand graded

Four traps, read rather than counted, because no script catches a model
anchoring on a block that did not apply:

| trap | verdict |
|---|---|
| `ocr1-trap` | **memory helped.** Asked for a server-side endpoint, it built one, and correctly cited the existing client-side camera work as context. Recall off invented a history that was never given ("since you specified FastAPI and OpenCV"). |
| `ssh1-trap` | **memory helped.** Answered rule 8 (read-only) correctly and listed rule 10 (destructive) beside it as genuine prior context, without confusing the two. |
| `phase3-trap` | **anchoring failure.** The probe asked to *add* OCR properties; the stored seed said *remove* them. Recall off implemented adding. Recall on opened "The goal is to remove OCR-related properties" -- the seed's fact asserted over the question asked. |
| `git-trap` | **neutral to worse.** Only 94 tokens were admitted, so there was effectively no memory; the answer degraded from 2,336 characters of actionable conflict-resolution steps to a 471-character restatement. |

Two helped, one failed the way the family was designed to catch, one was noise.
That is the real shape of the thing: recall buys **continuity**, and it can also
overwrite the question.

### What this says about the design

Not "the thesis is dead" -- nothing here is powered enough to say that. What it
says is that the reasoning-savings claim is **still unproven after being
measured once**, which is different from both "proven" and "refuted", and that
the cost side is no longer in doubt.

One thing the run did show positively, and it was not what it set out to test:
recall supplies continuity a fresh context cannot. The recall-off arm invented a
history it was never given ("since you specified FastAPI and OpenCV"); the
recall-on arm cited the real prior work and answered the new question. That is a
benefit worth naming even though no counter measures it.

Reproducing: `evaluate/results.jsonl` (run 1, three arms, 35 probes) and
`results_run2.jsonl` (run 2, two arms, 16 probes, reasoning trace captured).
Run 1's output-length columns are void for the reason above; its prompt-token
and latency columns are not.
