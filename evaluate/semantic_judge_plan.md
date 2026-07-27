# Semantic Judge for Recall Relevance

## Problem

Embedding similarity alone cannot distinguish:
- **trap** blocks: same problem, different nuance (e.g. phase 1 vs phase 2) — sim 0.841
- **distractor** blocks: same vocabulary, unrelated domain — sim 0.708

Both fire above threshold and inject irrelevant blocks into the reasoning model's context.

## Design

### Two-stage recall pipeline

```
User message
  → embed → KNN (threshold 0.62, k=4)
  → candidate blocks
  → semantic judge filters candidates
  → filtered blocks → inject into prompt
```

### Semantic judge prompt

```
Question: {user_message}

Memory block (retrieved by similarity):
{block.text}

Does this block directly help answer the question?
Answer YES only if the block contains information that would
change or materially improve the answer. Answer NO if:
- The block is about a different phase/version of the problem
- The block shares vocabulary but addresses a different issue
- The block is too old or specific to apply here

Reply with exactly: {"verdict": "yes"} or {"verdict": "no"}
```

### Config additions

```yaml
recall:
  judge_enabled: true        # enable semantic judge on recall candidates
  judge_threshold: 0.5       # not used yet, binary yes/no
```

### Pipeline changes (pipeline.py)

1. In `recall_blocks()`, after KNN returns candidates, for each candidate:
   - Call `_judge_candidate(user_message, block)` on the 1.5B judge model
   - Skip blocks where verdict is "no"
2. `_judge_candidate()` makes a short HTTP call to `judge_endpoint`:
   - System: "You are a relevance judge. Reply with one JSON object."
   - User: question + block text
   - Parse `{"verdict": "yes"}` or `{"verdict": "no"}`
   - Timeout: 5s per candidate (fast on CPU for short prompts)
   - On error: keep the block (fail-open, don't lose recall)

### Latency

- ~200-400ms per candidate on CPU (1.5B model, short prompt)
- With k=4 candidates, worst case ~1.6s added
- Can parallelize all 4 judge calls with asyncio.gather

## Benchmark plan

### Re-run retrieval eval with judge

1. Modify `eval_retrieval.py` to add `--judge` flag:
   - After KNN returns candidates above threshold, call semantic judge
   - Filter candidates before counting hits/false-fires
   - Report recall/false-fire with and without judge

2. Expected improvement:
   - trap false-fires should drop (judge sees "phase 1" vs "phase 2")
   - distractor false-fires should drop (judge sees unrelated context)
   - exact/paraphrase should stay (judge confirms relevance)
   - crosslingual: no change (judge is monolingual)

3. Run command:
   ```
   python eval_retrieval.py --endpoint http://127.0.0.1:8082 --judge --judge-endpoint http://127.0.0.1:8081
   ```

### Hand-grade trap answers

After e2e eval, read the 6 trap-family answers:
- Does the model confidently recall a block that doesn't apply?
- Does it anchor on phase 1 advice when the question is about phase 2?
- This is the real limitation — write it up honestly for r/LocalLLaMA

## Implementation order

1. Add `recall.judge_enabled` to config.py
2. Add `_judge_candidate()` to pipeline.py
3. Filter `recall_blocks()` through judge
4. Add `--judge` flag to eval_retrieval.py
5. Re-run benchmark, compare with/without judge
6. Hand-grade trap answers from e2e eval

## Files to modify

- `cued_recall/cued_recall/config.py` — add judge_enabled config
- `cued_recall/cued_recall/pipeline.py` — add judge filtering in recall_blocks()
- `evaluate/eval_retrieval.py` — add --judge flag for benchmark
