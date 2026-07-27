#!/usr/bin/env python3
"""
Retrieval-layer evaluation for cued-recall.

Measures ONLY the recall pipeline: embed probe -> cosine similarity against
stored block keys -> top-k above threshold. No generation, no judge model.

Runs in seconds, is fully deterministic, and answers the question:
  "where do I set recall.threshold before I start injecting noise?"

Usage:
  python eval_retrieval.py --endpoint http://127.0.0.1:8082 --corpus corpus.jsonl
  python eval_retrieval.py --fake        # self-test with synthetic vectors
"""

import argparse
import json
import os
import sys
import hashlib
import urllib.request

import numpy as np

CACHE = "embeddings.npz"


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------
def embed_http(texts, endpoint, path, model, timeout=120):
    """Call an OpenAI-compatible or native llama.cpp embedding endpoint."""
    out = []
    for t in texts:
        if path == "/v1/embeddings":
            payload = {"input": t, "model": model}
        else:
            payload = {"content": t}
        req = urllib.request.Request(
            endpoint.rstrip("/") + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        if "data" in data:
            vec = data["data"][0]["embedding"]
        elif "embedding" in data:
            vec = data["embedding"]
        else:
            raise RuntimeError(f"unrecognised embedding response: {list(data)[:5]}")
        # some builds nest one level deeper
        if vec and isinstance(vec[0], list):
            vec = vec[0]
        out.append(np.asarray(vec, dtype=np.float32))
    return np.vstack(out)


def embed_fake(texts):
    """Deterministic pseudo-embeddings for offline self-test only."""
    vecs = []
    for t in texts:
        words = set(t.lower().split())
        v = np.zeros(256, dtype=np.float32)
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
            v[h % 256] += 1.0
        vecs.append(v)
    return np.vstack(vecs)


def normalise(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def evaluate(sims, probes, seeds, threshold, k):
    """
    sims: (n_probes, n_seeds) cosine similarity matrix
    Returns per-relation counts plus the two headline numbers.
    """
    per_rel = {}
    hits = misses = 0
    false_fires = quiet = 0

    for i, p in enumerate(probes):
        row = sims[i]
        order = np.argsort(-row)[:k]
        retrieved = [seeds[j] for j in order if row[j] >= threshold]
        fams = {s["family"] for s in retrieved}
        correct = p["family"] in fams

        rel = p["relation"]
        d = per_rel.setdefault(rel, {"n": 0, "fired": 0, "correct": 0, "top_sim": []})
        d["n"] += 1
        d["fired"] += 1 if retrieved else 0
        d["correct"] += 1 if correct else 0
        d["top_sim"].append(float(row.max()) if len(row) else 0.0)

        if p["should_recall"]:
            hits += 1 if correct else 0
            misses += 0 if correct else 1
        else:
            false_fires += 1 if retrieved else 0
            quiet += 0 if retrieved else 1

    n_should = hits + misses
    n_shouldnt = false_fires + quiet
    return {
        "threshold": threshold,
        "recall_rate": hits / n_should if n_should else float("nan"),
        "false_fire_rate": false_fires / n_shouldnt if n_shouldnt else float("nan"),
        "per_relation": per_rel,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8082")
    ap.add_argument("--path", default="/v1/embeddings",
                    help="/v1/embeddings or /embedding")
    ap.add_argument("--model", default="embed")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--fake", action="store_true", help="offline self-test")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.corpus, encoding="utf-8")
            if l.strip()]
    seeds = [r for r in rows if r["role"] == "seed"]
    probes = [r for r in rows if r["role"] == "probe"]
    if not seeds or not probes:
        sys.exit("corpus needs both seed and probe rows")

    texts = [r["prompt"] for r in seeds + probes]

    if args.fake:
        mat = embed_fake(texts)
    elif not args.no_cache and os.path.exists(CACHE):
        z = np.load(CACHE)
        if z["n"] == len(texts):
            mat = z["mat"]
            print(f"[cache] reusing {CACHE}")
        else:
            mat = embed_http(texts, args.endpoint, args.path, args.model)
            np.savez(CACHE, mat=mat, n=len(texts))
    else:
        mat = embed_http(texts, args.endpoint, args.path, args.model)
        np.savez(CACHE, mat=mat, n=len(texts))

    mat = normalise(mat)
    seed_m = mat[: len(seeds)]
    probe_m = mat[len(seeds):]
    sims = probe_m @ seed_m.T

    print(f"\nseeds={len(seeds)}  probes={len(probes)}  dim={mat.shape[1]}  k={args.k}\n")

    results = []
    print(f"{'thr':>6} {'recall':>8} {'false-fire':>11}   margin")
    print("-" * 46)
    for thr in np.arange(0.30, 0.96, 0.02):
        r = evaluate(sims, probes, seeds, float(thr), args.k)
        results.append(r)
        margin = r["recall_rate"] - r["false_fire_rate"]
        bar = "#" * int(max(margin, 0) * 30)
        print(f"{thr:6.2f} {r['recall_rate']:8.2f} {r['false_fire_rate']:11.2f}   {bar}")

    best = max(results, key=lambda r: r["recall_rate"] - r["false_fire_rate"])
    print(f"\nbest separation at threshold {best['threshold']:.2f}: "
          f"recall={best['recall_rate']:.2f} false-fire={best['false_fire_rate']:.2f}")

    print("\nper-relation top similarity (mean) at that threshold:")
    for rel, d in sorted(best["per_relation"].items()):
        print(f"  {rel:<14} n={d['n']:<3} fired={d['fired']:<3} "
              f"correct={d['correct']:<3} mean_top_sim={np.mean(d['top_sim']):.3f}")

    with open("retrieval_sweep.csv", "w") as f:
        f.write("threshold,recall_rate,false_fire_rate\n")
        for r in results:
            f.write(f"{r['threshold']:.2f},{r['recall_rate']:.4f},"
                    f"{r['false_fire_rate']:.4f}\n")
    print("\nwrote retrieval_sweep.csv")


if __name__ == "__main__":
    main()
