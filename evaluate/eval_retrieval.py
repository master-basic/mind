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
  python eval_retrieval.py --judge       # add the second-stage relevance
                                         # filter and print both columns
"""

import argparse
import json
import os
import sys
import time
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
def evaluate(sims, probes, seeds, threshold, k, keep=None):
    """
    sims: (n_probes, n_seeds) cosine similarity matrix
    keep: optional (n_probes, n_seeds) bool matrix from the semantic judge --
          False means the judge said this seed does not apply to this probe,
          so it is dropped before anything is counted. None = no judge.
    Returns per-relation counts plus the two headline numbers.
    """
    per_rel = {}
    hits = misses = 0
    false_fires = quiet = 0

    for i, p in enumerate(probes):
        row = sims[i]
        order = np.argsort(-row)[:k]
        retrieved = [seeds[j] for j in order
                     if row[j] >= threshold and (keep is None or keep[i][j])]
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


# --------------------------------------------------------------------------
# semantic judge (second stage)
# --------------------------------------------------------------------------
def load_shipped_prompt():
    """Import the prompt the middleware actually uses.

    Deliberately not a copy: a reranker eval that scores its own private
    wording measures nothing about the system. If the package cannot be
    imported, say so instead of quietly falling back.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "cued_recall"))
    try:
        from cued_recall.utils import RELEVANCE_SYSTEM, relevance_prompt
    except ImportError as e:
        sys.exit(f"--judge needs the cued_recall package importable: {e}")
    return RELEVANCE_SYSTEM, relevance_prompt


def judge_pairs(probes, seeds, sims, k, endpoint, timeout=60):
    """Ask the small model about every pair that could ever be retrieved.

    The top-k selection does not depend on the threshold, so the candidate set
    per probe is fixed and the judge can be asked once per pair rather than
    once per pair per threshold -- 33 sweep steps over the same verdicts.
    Returns a (n_probes, n_seeds) bool matrix, True = keep.
    """
    system, prompt_for = load_shipped_prompt()
    keep = np.ones((len(probes), len(seeds)), dtype=bool)
    asked = 0
    for i, p in enumerate(probes):
        for j in np.argsort(-sims[i])[:k]:
            payload = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",
                     "content": prompt_for(p["prompt"], seeds[j]["prompt"])},
                ],
                "temperature": 0,
                "max_tokens": 4,
            }
            req = urllib.request.Request(
                endpoint.rstrip("/") + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = json.loads(r.read())
                text = (data["choices"][0]["message"]["content"] or "").strip().lower()
            except Exception as e:
                print(f"  [judge] pair ({i},{j}) failed: {type(e).__name__}: {e}")
                continue          # fail-open, exactly as the pipeline does
            keep[i][j] = not text.startswith("no")
            asked += 1
    print(f"[judge] {asked} verdicts, {int((~keep).sum())} candidates rejected")
    return keep


def sweep(sims, probes, seeds, k, keep=None):
    return [evaluate(sims, probes, seeds, float(t), k, keep)
            for t in np.arange(0.30, 0.96, 0.02)]


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
    ap.add_argument("--judge", action="store_true",
                    help="also sweep with the semantic judge filtering "
                         "candidates, and print both columns side by side")
    ap.add_argument("--judge-endpoint", default="http://127.0.0.1:8081",
                    help="the small model, i.e. the judge server")
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

    results = sweep(sims, probes, seeds, args.k)
    judged = None
    if args.judge:
        t0 = time.perf_counter()
        keep = judge_pairs(probes, seeds, sims, args.k, args.judge_endpoint)
        elapsed = time.perf_counter() - t0
        n_pairs = min(args.k, len(seeds)) * len(probes)
        print(f"[judge] {elapsed:.1f}s for {n_pairs} pairs "
              f"({elapsed / max(n_pairs, 1) * 1000:.0f} ms each, serial)")
        judged = sweep(sims, probes, seeds, args.k, keep)

    if judged:
        print(f"{'thr':>6} {'recall':>8} {'false-fire':>11} │"
              f"{'recall+J':>9} {'false-fire+J':>13}")
        print("-" * 52)
        for r, j in zip(results, judged):
            print(f"{r['threshold']:6.2f} {r['recall_rate']:8.2f} "
                  f"{r['false_fire_rate']:11.2f} │{j['recall_rate']:9.2f} "
                  f"{j['false_fire_rate']:13.2f}")
    else:
        print(f"{'thr':>6} {'recall':>8} {'false-fire':>11}   margin")
        print("-" * 46)
        for r in results:
            margin = r["recall_rate"] - r["false_fire_rate"]
            bar = "#" * int(max(margin, 0) * 30)
            print(f"{r['threshold']:6.2f} {r['recall_rate']:8.2f} "
                  f"{r['false_fire_rate']:11.2f}   {bar}")

    def report(rows, label):
        best = max(rows, key=lambda r: r["recall_rate"] - r["false_fire_rate"])
        print(f"\n[{label}] best separation at threshold {best['threshold']:.2f}: "
              f"recall={best['recall_rate']:.2f} "
              f"false-fire={best['false_fire_rate']:.2f}")
        print("per-relation at that threshold:")
        for rel, d in sorted(best["per_relation"].items()):
            print(f"  {rel:<14} n={d['n']:<3} fired={d['fired']:<3} "
                  f"correct={d['correct']:<3} "
                  f"mean_top_sim={np.mean(d['top_sim']):.3f}")
        return best

    report(results, "embedding only")
    if judged:
        report(judged, "with judge")
        # The operating point is what the config actually uses, and it is the
        # only number that answers "should judge_enabled be on".
        at = min(results, key=lambda r: abs(r["threshold"] - 0.62))
        at_j = min(judged, key=lambda r: abs(r["threshold"] - 0.62))
        print(f"\nat the shipped threshold 0.62: "
              f"recall {at['recall_rate']:.2f} -> {at_j['recall_rate']:.2f}, "
              f"false-fire {at['false_fire_rate']:.2f} -> "
              f"{at_j['false_fire_rate']:.2f}")

    # --fake numbers describe the harness, not the embedding model. Writing
    # them over the real sweep would leave a file that looks like a result.
    out_csv = "retrieval_sweep_fake.csv" if args.fake else "retrieval_sweep.csv"
    with open(out_csv, "w") as f:
        if judged:
            f.write("threshold,recall_rate,false_fire_rate,"
                    "recall_rate_judged,false_fire_rate_judged\n")
            for r, j in zip(results, judged):
                f.write(f"{r['threshold']:.2f},{r['recall_rate']:.4f},"
                        f"{r['false_fire_rate']:.4f},{j['recall_rate']:.4f},"
                        f"{j['false_fire_rate']:.4f}\n")
        else:
            f.write("threshold,recall_rate,false_fire_rate\n")
            for r in results:
                f.write(f"{r['threshold']:.2f},{r['recall_rate']:.4f},"
                        f"{r['false_fire_rate']:.4f}\n")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
