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