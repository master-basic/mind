#!/usr/bin/env python3
"""
End-to-end A/B for cued-recall.

  arm A (baseline)  : client -> llama-server directly        (no memory)
  arm B (treatment) : client -> cued-recall -> llama-server  (memory warm)

Protocol per repeat:
  1. wipe the block store
  2. WARM: send every seed prompt through the middleware, so blocks exist
  3. TEST: send every probe prompt through both arms
  4. record think-tokens, total tokens, latency, and the raw answer

Grading is deliberately NOT automated. It writes results.jsonl for you to
score by hand -- with ~20 probes that is 30 minutes, and hand grading is the
only thing that catches the failure that matters (anchoring on a recalled
block that did not apply).

Usage:
  python eval_e2e.py --repeats 3
"""

import argparse
import json
import re
import time
import urllib.request

THINK = re.compile(r"<think>(.*?)</think>", re.S)


def chat(base, prompt, model, timeout=600):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 42,
        "stream": False,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    dt = time.perf_counter() - t0

    message = data["choices"][0]["message"]
    text = message.get("content") or ""
    usage = data.get("usage", {})
    # Two places the think trace can be, and only one of them was checked.
    # llama.cpp parses reasoning out of the content for models whose template
    # declares it and returns it as its own field -- Gemma4 and Qwen3.5 both
    # land here -- so `content` arrives with the tags already stripped and this
    # regex found nothing. think_chars then read 0 for every arm, which looks
    # exactly like "memory did not shorten the reasoning" rather than like a
    # harness that was not measuring it. The middleware reads both fields
    # (pipeline.py forward_nonstream); the eval has to as well.
    m = THINK.search(text)
    think = m.group(1) if m else (message.get("reasoning_content") or "")
    answer = THINK.sub("", text).strip()

    return {
        "latency_s": round(dt, 2),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "think_chars": len(think),
        "answer_chars": len(answer),
        "think": think,
        "answer": answer,
    }


def admin(mw, route, method="GET"):
    req = urllib.request.Request(mw.rstrip("/") + route, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--middleware", default="http://127.0.0.1:8000")
    ap.add_argument("--baseline", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="reasoning")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.corpus, encoding="utf-8") if l.strip()]
    seeds = [r for r in rows if r["role"] == "seed"]
    probes = [r for r in rows if r["role"] == "probe"]

    out = open(args.out, "a", encoding="utf-8")

    for rep in range(args.repeats):
        print(f"\n=== repeat {rep + 1}/{args.repeats} ===")
        input("wipe the block store + restart cued-recall, then press Enter... ")

        print(f"warming with {len(seeds)} seed prompts")
        for s in seeds:
            chat(args.middleware, s["prompt"], args.model)
            print(f"  warmed {s['id']}")

        stats = admin(args.middleware, "/admin/stats")
        print(f"  store after warm: {stats}")

        for p in probes:
            for arm, base in (("baseline", args.baseline),
                              ("treatment", args.middleware)):
                r = chat(base, p["prompt"], args.model)
                rec = {
                    "repeat": rep,
                    "id": p["id"],
                    "family": p["family"],
                    "relation": p["relation"],
                    "should_recall": p["should_recall"],
                    "arm": arm,
                    **r,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                print(f"  {p['id']:<12} {arm:<10} "
                      f"{r['latency_s']:>6.1f}s  think={r['think_chars']:>6}")

    out.close()
    print(f"\nwrote {args.out} -- now grade the 'answer' fields by hand")


if __name__ == "__main__":
    main()
