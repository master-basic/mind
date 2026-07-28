# Throughput: what the memory layer costs per turn

Same prompts, same model, same seed, run straight at `llama-server` and again
through the middleware. The only difference between the arms is the middleware,
so the difference in the numbers is the middleware.

```
python eval_throughput.py --repeats 3 --max-tokens 400
```

## What was measured

| | |
|---|---|
| Date | 28 July 2026 |
| Reasoning | Hermes3.6-35B-A3B-Uncensored-Genesis-V5-APEX-Compact, MoE, experts in system RAM, `n_ctx` 65,536 |
| Judge / reranker | Qwen2.5-1.5B-Instruct-Q4_K_M, CPU only, one slot |
| Embedding | nomic-embed-text-v1.5-Q8_0 |
| Recall config | `threshold: 0.48`, `judge_enabled: true`, `k: 4` |
| Store | 164 blocks |
| Protocol | 4 prompts × 3 repeats × 2 arms, streamed, `temperature: 0`, seed 42, warm KV cache, arms interleaved per prompt |

## Headline

| | direct | through middleware | delta |
|---|---|---|---|
| decode | 56.6 tok/s | 52.8 tok/s | −6.7% |
| time to first token | 334 ms | 2,078 ms | **+1,744 ms** |
| prompt tokens | 28 | 436 | +408 |

Medians across all twelve runs per arm. Per-prompt medians:

| prompt | arm | ttft ms | decode tok/s | prompt tok | out tok | total s |
|---|---|---|---|---|---|---|
| p0 capital of Portugal | direct | 282 | 55.5 | 19 | 153 | 3.0 |
| p0 | middleware | **554** | 48.8 | 427 | 103 | 3.0 |
| p1 write-ahead log | direct | 306 | 56.7 | 23 | 400 | 7.4 |
| p1 | middleware | 2,673 | 52.9 | 431 | 333 | 8.7 |
| p2 train arithmetic | direct | 516 | 57.3 | 49 | 400 | 7.5 |
| p2 | middleware | 2,121 | 54.7 | 457 | 400 | 9.4 |
| p3 merge two sorted lists | direct | 336 | 57.5 | 33 | 400 | 7.3 |
| p3 | middleware | 2,035 | 52.6 | 441 | 400 | 9.8 |

## Where the 1,744 ms goes

The per-prompt table contains a natural experiment. p0 retrieved **no candidates
at all** — nothing in the store was within 0.48 of "name the capital of
Portugal" — so no relevance-judge calls were made. p1–p3 each retrieved 3–4
candidates and paid for a judge call on every one:

| | candidates | judge calls | middleware ttft |
|---|---|---|---|
| p0 | 0 | 0 | **554 ms** |
| p1 | 4 | 4 | 2,673 ms |
| p2 | 3 | 3 | 2,121 ms |
| p3 | 4 | 4 | 2,035 ms |

So the cost splits roughly:

- **~270 ms** — embedding the query, the vector search, and prefilling the
  larger prompt. That is p0's 554 ms against its own direct arm's 282 ms.
- **~1,500–2,200 ms** — the relevance judge, 3–4 calls on a CPU 1.5B model
  serialised two at a time through the shared slot semaphore.

The reranker is the cost. Everything else the memory layer does is rounding
error against one turn of a 35B model.

## Two things the timings alone would have hidden

**The +408 prompt tokens are not memory.** Recall admitted **zero** blocks on
all twelve middleware turns — the judge rejected all 33 candidates. Measured
directly against `llama-server`, the same prompt costs 19 tokens bare and 427
with the `web_search` / `web_fetch` definitions the middleware injects. The
entire prompt-size difference is tool definitions. In this configuration the
memory layer's effect on prompt size was nil.

And the rejections were *correct*: the benchmark prompts are about Portugal,
write-ahead logs, train timetables and Python, while the store holds notes about
this project. Top similarities were 0.49–0.57 — above the 0.48 threshold, which
is exactly the over-retrieval the second stage exists to catch. This is the
`retrieval_sweep` finding reproduced on live traffic: at 0.48 the embedding
admits things it should not, and the judge throws them out.

But the cost is paid whether or not anything survives. Three to four CPU calls
and ~2 s to conclude "nothing here" is the common case for an off-topic
question, and there is no short-circuit — a cheap similarity floor, or skipping
the judge when the best candidate is barely over the line, would recover most of
that. Not implemented.

**The −6.7% decode number should not be called a 6.7% decode tax.** The
middleware is not in the decode path; it re-emits chunks, which is per-token
Python work, but the spread says something else is going on. Direct decode
across twelve runs was 54.0–57.8 tok/s, a tight band. Middleware decode was
42.4–66.3 — and its best run beat *every* direct run. That is not a fixed
overhead, that is contention noise: this model keeps its experts in system RAM,
so decode is partly CPU-bandwidth-bound, and the middleware puts fire-and-forget
work (tagging, correction checking) and the CPU judge on the same cores. Three
repeats cannot separate those. Treat decode as "unchanged, noisily" until
someone runs it with the housekeeping disabled.

## Caveats

- The arms do not generate identical text. Tool definitions in the prompt change
  what the model writes (p0: 153 tokens direct vs 103 through the middleware),
  so end-to-end wall time is not like-for-like. That is why TTFT and decode are
  reported separately rather than a single tok/s.
- Three repeats, one machine, one model, warm cache. Enough to locate a 1.7 s
  cost; not enough to argue about 5%.
- Warm cache favours neither arm here: with a single user message and no
  history, both arms re-prefill their whole prompt anyway.
- This measures cost only. Whether the injected memory *earns* its cost by
  shortening the think trace is `eval_e2e.py`'s question, and is unanswered.

## Reproducing

```bash
python evaluate/eval_throughput.py --repeats 3            # warm, as above
python evaluate/eval_throughput.py --repeats 3 --cold     # clear KV each request
python evaluate/eval_throughput.py --max-tokens 800       # longer generations
```

Raw per-run data lands in `throughput_results.json`.
