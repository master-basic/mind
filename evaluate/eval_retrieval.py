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
def load_shipped_utils():
    """The middleware's own text helpers, for the same reason as the prompt.

    build_stimulus and truncate_tokens decide what a block's key text actually
    is. A harness with its own copy would measure a representation the system
    does not use, which is the fault this flag exists to correct.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "cued_recall"))
    try:
        from cued_recall.utils import build_stimulus, truncate_tokens
    except ImportError as e:
        sys.exit(f"--key-source needs the cued_recall package importable: {e}")
    return build_stimulus, truncate_tokens


def load_shipped_fit():
    """The middleware's own embed-size cap, bound to the live server's window.

    A key text the pipeline would have to trim is one this sweep must not
    score whole, or the eval measures a representation production cannot
    produce.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "cued_recall"))
    try:
        from cued_recall.config import Config
        from cued_recall.embed import EmbeddingClient
    except ImportError as e:
        sys.exit(f"--key-source needs the cued_recall package importable: {e}")
    cfg_path = os.path.join(os.path.dirname(here), "cued_recall", "config.yaml")
    cfg = Config(cfg_path)
    client = EmbeddingClient(cfg.embed_endpoint,
                             ctx_tokens=cfg.embed_ctx_tokens,
                             chars_per_token=cfg.chars_per_token,
                             tokens_per_word=cfg.tokens_per_word)
    client.detect_ctx_tokens()
    print(f"[fit] embedder window: {client.ctx_tokens} tokens")
    return client.fit


def truncate_tokens_shipped(text, n):
    _, truncate = load_shipped_utils()
    return truncate(text, n)


def seed_note_texts(seeds, key_source, blocks_path, judge_note="text"):
    """What the judge is shown for each seed.

    Independent of what is indexed. _filter_by_relevance passes block.text and
    never the embed key, so tying the note to --key-source would score the
    judge on a prompt the pipeline never builds -- and the judge's verdict is
    the only thing standing between the trap family and the answer.

    judge_note="question" shows the question the block was written to answer
    instead of the block's words. See eval_judge_notes.py: on this corpus that
    is the difference between refusing 0 of 6 traps and refusing 6 of 6, at no
    cost to legitimate recall.
    """
    if judge_note == "question" or key_source == "prompt":
        return [s["prompt"] for s in seeds]
    by_id = _load_blocks(blocks_path, seeds)
    _, truncate_tokens = load_shipped_utils()
    return [truncate_tokens(by_id[s["id"]].get("reasoning")
                            or by_id[s["id"]].get("answer") or "", 1024)
            for s in seeds]


def _load_blocks(blocks_path, seeds):
    if not os.path.exists(blocks_path):
        sys.exit(f"needs {blocks_path}. Generate it first:\n"
                 f"    python make_seed_blocks.py")
    by_id = {}
    for line in open(blocks_path, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            by_id[row["id"]] = row
    for s in seeds:
        if s["id"] not in by_id:
            sys.exit(f"{blocks_path} has no entry for seed {s['id']}; "
                     f"re-run make_seed_blocks.py")
    return by_id


def seed_key_texts(seeds, key_source, blocks_path):
    """The text that represents each seed as a stored block."""
    if key_source == "prompt":
        return [s["prompt"] for s in seeds]

    build_stimulus, truncate_tokens = load_shipped_utils()
    fit = load_shipped_fit()
    by_id = _load_blocks(blocks_path, seeds)

    out = []
    for s in seeds:
        gen = by_id[s["id"]]
        if key_source == "composite":
            # Exactly what _create_blocks writes onto a reasoning block.
            out.append(build_stimulus(s["prompt"], gen["answer"], ""))
        else:
            # embed_text: the block's own words. A reasoning block holds the
            # think trace; fall back to the answer for a model that emitted
            # none, which is the same fallback embed_source_text makes.
            own = gen.get("reasoning") or gen.get("answer") or ""
            out.append(truncate_tokens(own, 1024))
    # Last, and through the shipped client: the embedder's window is a hard
    # limit, and a key text the pipeline would have to trim is one this sweep
    # must not score whole. Without it the eval measures a representation
    # production cannot produce -- 1,024 words of the ocr1 trace is 2,338
    # tokens against a 2,048-token server.
    return [fit(t) for t in out]


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


def judge_pairs(probes, seeds, sims, k, endpoint, notes=None, timeout=60):
    """Ask the small model about every pair that could ever be retrieved.

    The top-k selection does not depend on the threshold, so the candidate set
    per probe is fixed and the judge can be asked once per pair rather than
    once per pair per threshold -- 33 sweep steps over the same verdicts.
    Returns a (n_probes, n_seeds) bool matrix, True = keep.

    `notes` is what the judge is shown as the archived note. The pipeline
    passes block.text, so under --key-source that is the block's own words;
    the seed's prompt is only the right stand-in for the original
    question-to-question harness.
    """
    system, prompt_for = load_shipped_prompt()
    if notes is None:
        notes = [s["prompt"] for s in seeds]
    keep = np.ones((len(probes), len(seeds)), dtype=bool)
    asked = 0
    for i, p in enumerate(probes):
        for j in np.argsort(-sims[i])[:k]:
            payload = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",
                     "content": prompt_for(p["prompt"], notes[j])},
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
    ap.add_argument("--key-source", default="prompt",
                    choices=("prompt", "composite", "content"),
                    help="what text represents a stored block. 'prompt' is "
                         "this harness's original behaviour and models no "
                         "block type the system actually writes; 'composite' "
                         "and 'content' are config.embed_source's two settings "
                         "and need seed_blocks.jsonl (make_seed_blocks.py)")
    ap.add_argument("--blocks", default="seed_blocks.jsonl",
                    help="generated seed answers/traces, for --key-source")
    ap.add_argument("--json", dest="json_out",
                    help="write the full sweep here, for before/after diffs")
    ap.add_argument("--judge-note", default="text",
                    choices=("text", "question"),
                    help="what the judge is shown as the note: the block's own "
                         "words (production today) or the question it was "
                         "written to answer")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.corpus, encoding="utf-8")
            if l.strip()]
    seeds = [r for r in rows if r["role"] == "seed"]
    probes = [r for r in rows if r["role"] == "probe"]
    if not seeds or not probes:
        sys.exit("corpus needs both seed and probe rows")

    # What text stands in for a stored block.
    #
    # The original harness embedded each seed's *prompt*, which models no block
    # this system writes: a reasoning block's key is build_stimulus(question,
    # answer, reading) and a result block's is its own answer text. So the
    # 0.841 trap figure was measured on question-to-question similarity, and
    # the store's real geometry was never swept at all. --key-source lets the
    # two settings of config.embed_source be compared on the representation
    # they actually produce.
    seed_keys = seed_key_texts(seeds, args.key_source, args.blocks)
    texts = seed_keys + [truncate_tokens_shipped(r["prompt"], 512)
                         for r in probes]

    # Keyed on the text, not the row count. The cache used to accept any file
    # with the same number of rows, so editing a prompt's wording -- or
    # swapping in a different key text for the same corpus, which is exactly
    # what comparing two representations does -- silently scored the old
    # vectors against the new labels.
    key = hashlib.sha256(
        "\x00".join(texts).encode("utf-8")
    ).hexdigest()

    if args.fake:
        mat = embed_fake(texts)
    else:
        cached = None
        if not args.no_cache and os.path.exists(CACHE):
            z = np.load(CACHE, allow_pickle=False)
            # A cache written before the key existed has no way to prove it
            # matches, so it is a miss. Re-embedding 41 texts costs seconds;
            # scoring the wrong vectors costs a wrong conclusion.
            if "key" in z.files and str(z["key"]) == key:
                cached = z["mat"]
                print(f"[cache] reusing {CACHE}")
            else:
                print(f"[cache] {CACHE} is for different text, re-embedding")
        if cached is None:
            mat = embed_http(texts, args.endpoint, args.path, args.model)
            np.savez(CACHE, mat=mat, n=len(texts), key=key)
        else:
            mat = cached

    mat = normalise(mat)
    seed_m = mat[: len(seeds)]
    probe_m = mat[len(seeds):]
    sims = probe_m @ seed_m.T

    print(f"\nseeds={len(seeds)}  probes={len(probes)}  dim={mat.shape[1]}  k={args.k}\n")

    results = sweep(sims, probes, seeds, args.k)
    judged = None
    if args.judge:
        t0 = time.perf_counter()
        notes = seed_note_texts(seeds, args.key_source, args.blocks,
                                judge_note=args.judge_note)
        keep = judge_pairs(probes, seeds, sims, args.k, args.judge_endpoint,
                           notes=notes)
        elapsed = time.perf_counter() - t0
        n_pairs = min(args.k, len(seeds)) * len(probes)
        print(f"[judge] {elapsed:.1f}s for {n_pairs} pairs "
              f"({elapsed / max(n_pairs, 1) * 1000:.0f} ms each, serial)")
        judged = sweep(sims, probes, seeds, args.k, keep)

    if judged:
        print(f"{'thr':>6} {'recall':>8} {'false-fire':>11} |"
              f"{'recall+J':>9} {'false-fire+J':>13}")
        print("-" * 52)
        for r, j in zip(results, judged):
            print(f"{r['threshold']:6.2f} {r['recall_rate']:8.2f} "
                  f"{r['false_fire_rate']:11.2f} |{j['recall_rate']:9.2f} "
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

    if args.json_out:
        # The whole sweep, per-relation counts included, so a later phase is a
        # diff against a file rather than against somebody's memory of a
        # terminal. mean_top_sim is kept per relation because that -- not the
        # headline recall -- is the number the representation change moves.
        def serialise(rows):
            out = []
            for r in rows:
                per_rel = {
                    rel: {"n": d["n"], "fired": d["fired"],
                          "correct": d["correct"],
                          "mean_top_sim": round(float(np.mean(d["top_sim"])), 4)}
                    for rel, d in sorted(r["per_relation"].items())
                }
                out.append({"threshold": round(r["threshold"], 2),
                            "recall_rate": round(r["recall_rate"], 4),
                            "false_fire_rate": round(r["false_fire_rate"], 4),
                            "per_relation": per_rel})
            return out

        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "corpus": args.corpus,
            "key_source": args.key_source,
            "judge_note": args.judge_note,
            "k": args.k,
            "seeds": len(seeds),
            "probes": len(probes),
            "dim": int(mat.shape[1]),
            "embedding_only": serialise(results),
            "with_judge": serialise(judged) if judged else None,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
