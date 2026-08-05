#!/usr/bin/env python3
"""Generate the answer and think trace each seed prompt would produce.

eval_retrieval.py originally embedded a seed's *prompt* as the text standing
in for a stored block. No block this system writes looks like that: a reasoning
block's key is build_stimulus(question, answer, reading) -- the question plus
the answer it produced -- and a result block's is its own answer text. So the
sweep measured question-to-question similarity, and the geometry the store
actually has was never swept. That is also why the 0.841 trap figure, while
real, does not by itself settle what config.embed_source should be: it was
measured on a representation neither setting uses.

This fills the gap. For each seed it runs one turn against the reasoning model
-- no middleware, no memory, so the answer is what a cold model produces -- and
records the think trace and the answer alongside the prompt. eval_retrieval.py
--key-source composite|content then builds the two candidate block keys from
these, using the middleware's own build_stimulus and truncate_tokens.

    python make_seed_blocks.py                 # generate seed_blocks.jsonl
    python make_seed_blocks.py --show          # print what was generated

Deterministic where the server allows it (temperature 0, fixed seed), but a
regenerated file will not be byte-identical across model or llama.cpp
versions, so it is committed rather than regenerated per run.
"""

import argparse
import json
import re
import sys
import time
import urllib.request

THINK = re.compile(r"<think>(.*?)</think>", re.S)


def chat(base, prompt, model, timeout=900):
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
    # Both places the trace can be. llama.cpp strips <think> out of content for
    # models whose template declares reasoning and returns it as its own field,
    # so checking only the regex reads 0 chars on Gemma4 and Qwen3.5 -- the bug
    # that made eval_e2e.py's think_chars silently useless.
    m = THINK.search(text)
    think = m.group(1).strip() if m else (message.get("reasoning_content") or "")
    answer = THINK.sub("", text).strip()
    return think, answer, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--out", default="seed_blocks.jsonl")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080",
                    help="the reasoning server, NOT the middleware -- these "
                         "turns must not be recalled into or stored")
    ap.add_argument("--model", default="reasoning")
    ap.add_argument("--show", action="store_true",
                    help="print an existing seed_blocks.jsonl and exit")
    args = ap.parse_args()

    if args.show:
        for line in open(args.out, encoding="utf-8"):
            r = json.loads(line)
            print(f"\n=== {r['id']} ({r['family']}) ===")
            print(f"  think : {len(r['reasoning']):>6} chars")
            print(f"  answer: {len(r['answer']):>6} chars")
            print(f"  {r['answer'][:200]}...")
        return 0

    rows = [json.loads(l) for l in open(args.corpus, encoding="utf-8")
            if l.strip()]
    seeds = [r for r in rows if r["role"] == "seed"]
    if not seeds:
        sys.exit("corpus has no seed rows")

    print(f"generating {len(seeds)} seed turns against {args.endpoint}\n")
    out = []
    for s in seeds:
        try:
            think, answer, dt = chat(args.endpoint, s["prompt"], args.model)
        except Exception as e:
            sys.exit(f"[ERROR] {s['id']}: {type(e).__name__}: {e}\n"
                     f"Is the reasoning server up at {args.endpoint}?")
        if not answer and not think:
            # An empty turn would silently become an empty block key, and a
            # zero vector scores 0 against everything -- a result that looks
            # like a finding.
            sys.exit(f"[ERROR] {s['id']} produced neither answer nor trace")
        out.append({
            "id": s["id"],
            "family": s["family"],
            "prompt": s["prompt"],
            "reasoning": think,
            "answer": answer,
        })
        print(f"  {s['id']:<14} {dt:>6.1f}s  "
              f"think={len(think):>6}  answer={len(answer):>6}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out} ({len(out)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
