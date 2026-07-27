#!/usr/bin/env python3
"""
Paired analysis of results.jsonl from eval_e2e.py.

Pairs each probe's baseline run against its treatment run and reports the
median delta with a bootstrap confidence interval. Paired comparison is used
because between-prompt variance dwarfs the effect you are trying to measure --
comparing group means on 20 prompts would show nothing either way.

Usage: python analyse.py results.jsonl
"""

import json
import sys
from collections import defaultdict

import numpy as np


def boot_ci(x, n=10000, seed=0):
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.jsonl"
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    by_key = defaultdict(dict)
    for r in recs:
        by_key[(r["repeat"], r["id"])][r["arm"]] = r

    groups = defaultdict(list)
    for (rep, pid), arms in by_key.items():
        if "baseline" not in arms or "treatment" not in arms:
            continue
        b, t = arms["baseline"], arms["treatment"]
        rel = t["relation"]
        for metric in ("think_chars", "latency_s", "completion_tokens"):
            if b.get(metric) is None or t.get(metric) is None:
                continue
            groups[(rel, metric)].append(t[metric] - b[metric])
            groups[("ALL", metric)].append(t[metric] - b[metric])

    print(f"{'relation':<14} {'metric':<19} {'n':>3} {'median Δ':>10} "
          f"{'95% CI':>22}")
    print("-" * 72)
    for (rel, metric), deltas in sorted(groups.items()):
        d = np.asarray(deltas, dtype=float)
        lo, hi = boot_ci(d)
        med = float(np.median(d))
        flag = "" if (lo <= 0 <= hi) else "  *"
        print(f"{rel:<14} {metric:<19} {len(d):>3} {med:>10.1f} "
              f"{f'[{lo:.1f}, {hi:.1f}]':>22}{flag}")

    print("\n* = bootstrap CI excludes zero. Everything else is noise;")
    print("  report it as noise rather than rounding it into a win.")


if __name__ == "__main__":
    main()
