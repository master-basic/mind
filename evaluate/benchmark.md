Good — this is the question that actually matters, and it's the one that will turn the project from "interesting design" into something people star.

The key move is splitting this into two separate evals, because most people conflate them and end up with numbers that mean nothing:

Retrieval eval — does the right block come back? No LLM generation involved. Deterministic, runs in seconds, sweepable across thresholds. This is where your public question about recall.threshold gets answered.
End-to-end eval — does having the block actually help? Slow, noisy, needs many repeats.

Do the first one first. It's cheap, it produces the chart, and if retrieval is bad there's no point measuring anything downstream.

Let me set up the corpus design and the harness.

Architected end-to-end testing framework with performance metrics and state management
Architected end-to-end testing framework with performance metrics and state management

Logic checks out — with the fake bag-of-words embedder, exact scores 1.00, trap scores high (shares most vocabulary), crosslingual collapses to noise. Exactly the shape you'd predict, which means the harness is measuring what it claims to.

Now the end-to-end half:

Created 2 files, ran 2 commands, read a file
Created 2 files, ran 2 commands, read a file

Start with eval_retrieval.py — it only needs your embedding server on :8082, nothing else running:

python eval_retrieval.py --endpoint http://127.0.0.1:8082

Five seconds, and you get a threshold sweep from 0.30 to 0.94 with recall rate and false-fire rate at each step, plus retrieval_sweep.csv to plot.

The corpus design is the actual work

The scripts are mechanical. What makes an eval honest is the corpus, and the trick is that every family needs adversarial members. Six relation types:

exact — identical restatement. Must recall. If this fails, something is broken.
paraphrase — same problem, different words. Must recall. This is the real target.
crosslingual — same problem in Azerbaijani. Must recall. You already flagged this as not working; now you can quantify how badly.
trap — same vocabulary, different answer. Phase 1 works but phase 2 fails; digest auth vs basic auth. Retrieval firing here is correct. Blindly reusing the old conclusion is the failure. This family is the whole reason to hand-grade.
distractor — high lexical overlap, unrelated problem. Must not fire.
control — nothing to do with anything. Must not fire.

Almost everyone building this measures only "does recall fire." The number nobody publishes is the false-fire rate, and that's the one that makes your post credible. The distractor prompts in the corpus are deliberately written to share surface words with a real family while meaning something completely different — that's what separates a real embedding evaluation from a vibe check.

Twenty-five rows is enough to start. Get to ~60 (10 families × 6) before you publish anything.

Where to get prompts

Write them yourself, from your own work. That's not a fallback — it's the better option:

You can grade correctness, which is the hard part of any eval
Your traffic is infra debugging, so a math benchmark would measure the wrong distribution entirely
You have years of real recurring problems: FortiGate ADVPN, DKIM selectors, nginx failover, rclone mounts, digest auth. The corpus already has six families seeded from those.

If you want an auto-graded objective layer on top, MBPP or HumanEval are the right choice — Python problems with unit tests, so correctness needs no human. Group them into families by technique (dynamic programming, string parsing, recursion) and the reasoning genuinely is reusable across a family. Skip GSM8K; arithmetic reasoning doesn't transfer the way you're claiming yours does.

Do not use a bigger LLM to generate your paraphrases unmonitored. LLM paraphrases share embedding-space structure with the original in ways human rewording doesn't, and you'll measure an inflated recall rate. Write them by hand, or generate then rewrite each one.

Protocol details that decide whether the numbers mean anything

Baseline arm bypasses the middleware entirely — point at :8080 directly. No config flag, no code path that might differ. Two URLs, same model, same seed.

Wipe the store between repeats. The script pauses and waits for you to do it. tmpfs blocks, the SQLite index, and the NVMe snapshot all have to go, or repeat 2 starts warm and your numbers drift.

temperature: 0 and a fixed seed, but expect residual nondeterminism from llama.cpp batching. Hence --repeats 3 and medians.

Paired analysis, not group means. analyse.py pairs each probe against itself across arms. Between-prompt variance is enormous — some questions produce 800 tokens of reasoning, others 12,000 — and comparing group averages across 20 prompts would drown any real effect. The bootstrap CI flags with * only when it excludes zero.

The metrics that matter
Metric	Where	What it tells you
recall rate vs false-fire rate	retrieval	the threshold answer
mean top-sim per relation	retrieval	whether crosslingual is near firing or nowhere close
Δ think_chars, paired	e2e	the headline "saves N% of reasoning" claim
Δ latency	e2e	whether recall overhead eats the savings
trap-family answer quality	hand-graded	anchoring failures

That last row has no script because it can't have one. Read the twelve trap-family answers. If the model recalls the phase-1 block and confidently tells you to check phase-1 proposals when the question was about phase 2, you've found the real limitation of the design — and writing that up honestly will get you more respect on r/LocalLLaMA than any speedup number.

Run the retrieval sweep first, tonight. It's fast, it produces a chart, and "here is the precision/recall curve for semantic recall on paraphrased technical questions, including the false-fire rate" is a far better second post than anything else you could write.

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