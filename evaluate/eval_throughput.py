#!/usr/bin/env python3
"""
Throughput A/B for cued-recall: what does the memory layer cost per turn?

  arm A (direct)     : client -> llama-server :8080
  arm B (middleware) : client -> cued-recall :8000 -> llama-server :8080

Same model, same prompts, same seed, same max_tokens. The only difference is
the middleware, so the difference in the numbers is the middleware.

WHY THIS MEASURES THREE THINGS AND NOT ONE
------------------------------------------
"Tokens per second" on its own is the metric that hid this system's worst
regression. A turn that spent 107 s prefilling and 1.4 s generating produced a
perfectly healthy-looking decode rate, because decode had nothing to do with
what went wrong. So this reports them apart:

  ttft_ms      time to the first token. Everything the middleware does before
               the reasoning model is asked -- embedding the query, the vector
               search, the relevance judge -- lands here, and so does prefill.
  decode_tps   completion tokens / (total - ttft). The generation rate proper.
               The middleware is not in this path at all, so a difference here
               is noise or contention, not overhead.
  prompt_tokens what the model was actually asked to read. The middleware makes
               this bigger by injecting recall, which is the cost that buys the
               shorter think trace -- eval_e2e.py is where that trade is judged.

KV CACHE
--------
Default is warm: prompts run in order, and llama.cpp keeps whatever prefix it
can. That is how the system is actually used. --cold clears the KV cache before
every single request, which makes each one pay full prefill; use it when you
want to compare prefill cost rather than steady-state behaviour.

Usage:
  python eval_throughput.py                        # 3 repeats, warm cache
  python eval_throughput.py --repeats 5 --cold
  python eval_throughput.py --max-tokens 600       # longer generations
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# Short, medium and long-thinking prompts. Deliberately not about this project:
# a prompt that matches stored blocks would make arm B recall something and
# arm A not, which is a real effect but belongs in eval_e2e.py, not here.
DEFAULT_PROMPTS = [
    "Name the capital of Portugal. One word.",
    "In two sentences, explain what a write-ahead log is for.",
    "A train leaves at 14:05 and arrives at 17:40, stopping twice for 12 "
    "minutes each. How long was it moving? Show your working.",
    "Write a Python function that merges two sorted lists into one sorted "
    "list, without using sorted(). Include a docstring.",
]


def stream_chat(base, prompt, model, max_tokens, conversation_id=None,
                timeout=900):
    """One streamed completion. Returns timing and token counts.

    Token counts come from the server's own `usage` when it sends one, which
    both arms do when asked via stream_options. Counting SSE chunks instead
    would be counting frames, not tokens, and the two differ whenever a chunk
    carries more than one.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
    }
    if conversation_id:
        body["conversation_id"] = conversation_id

    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage = {}
    text_len = 0

    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                # Reasoning models emit think tokens first; those are tokens
                # the user waited for, so they start the clock.
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
                    text_len += len(piece)
    total = time.perf_counter() - t0

    completion = usage.get("completion_tokens") or chunks
    if ttft is None:                       # nothing ever arrived
        ttft, decode_s = total, 0.0
    else:
        decode_s = max(total - ttft, 1e-6)
    return {
        "ttft_ms": round(ttft * 1000, 1),
        "total_s": round(total, 2),
        "decode_tps": round(completion / decode_s, 1) if completion else 0.0,
        "completion_tokens": completion,
        "prompt_tokens": usage.get("prompt_tokens"),
        "chars": text_len,
        "had_usage": bool(usage),
    }


def clear_kv(middleware):
    try:
        req = urllib.request.Request(middleware.rstrip("/") + "/admin/kv/clear",
                                     data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            json.loads(r.read() or "{}")
    except Exception as e:
        print(f"  [warn] could not clear KV: {type(e).__name__}: {e}")


def med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.median(vals), 1) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="http://127.0.0.1:8080",
                    help="llama-server directly")
    ap.add_argument("--middleware", default="http://127.0.0.1:8000")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--model", default="cued-recall",
                    help="ignored by llama-server, echoed by the middleware")
    ap.add_argument("--cold", action="store_true",
                    help="clear the KV cache before every request")
    ap.add_argument("--prompts", help="file with one prompt per line")
    ap.add_argument("--out", default="throughput_results.json")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]
        if not prompts:
            sys.exit(f"{args.prompts} contains no prompts")

    print(f"prompts={len(prompts)}  repeats={args.repeats}  "
          f"max_tokens={args.max_tokens}  cache={'cold' if args.cold else 'warm'}")
    print(f"  A direct     {args.baseline}")
    print(f"  B middleware {args.middleware}\n")

    runs = []
    for rep in range(args.repeats):
        for pi, prompt in enumerate(prompts):
            # Arms are interleaved per prompt rather than run as two blocks, so
            # that anything drifting over the session (thermals, another
            # process, cache growth) hits both arms about equally instead of
            # landing entirely on whichever ran second.
            for arm, base in (("direct", args.baseline),
                              ("middleware", args.middleware)):
                if args.cold:
                    clear_kv(args.middleware)
                try:
                    r = stream_chat(base, prompt, args.model, args.max_tokens,
                                    conversation_id=f"bench-{pi}-{rep}"
                                    if arm == "middleware" else None)
                except (urllib.error.URLError, OSError, TimeoutError) as e:
                    print(f"  [{arm:<10}] p{pi} rep{rep} FAILED: "
                          f"{type(e).__name__}: {e}")
                    continue
                r.update(arm=arm, prompt_index=pi, repeat=rep)
                runs.append(r)
                print(f"  [{arm:<10}] p{pi} rep{rep}  "
                      f"ttft={r['ttft_ms']:>8.1f}ms  "
                      f"decode={r['decode_tps']:>6.1f} tok/s  "
                      f"prompt={str(r['prompt_tokens']):>6}  "
                      f"out={r['completion_tokens']:>4}  "
                      f"total={r['total_s']:>6.2f}s")

    if not runs:
        sys.exit("every request failed -- is the stack up?")

    print("\n" + "=" * 78)
    print(f"{'prompt':<8}{'arm':<12}{'ttft ms':>10}{'decode tok/s':>14}"
          f"{'prompt tok':>12}{'out tok':>9}{'total s':>9}")
    print("-" * 78)
    summary = []
    for pi in range(len(prompts)):
        for arm in ("direct", "middleware"):
            rows = [r for r in runs if r["arm"] == arm and r["prompt_index"] == pi]
            if not rows:
                continue
            s = {"prompt_index": pi, "arm": arm, "n": len(rows),
                 "ttft_ms": med(rows, "ttft_ms"),
                 "decode_tps": med(rows, "decode_tps"),
                 "prompt_tokens": med(rows, "prompt_tokens"),
                 "completion_tokens": med(rows, "completion_tokens"),
                 "total_s": med(rows, "total_s")}
            summary.append(s)
            print(f"p{pi:<7}{arm:<12}{s['ttft_ms']:>10}{s['decode_tps']:>14}"
                  f"{str(s['prompt_tokens']):>12}{s['completion_tokens']:>9}"
                  f"{s['total_s']:>9}")
    print("=" * 78)

    def overall(arm, key):
        vals = [r[key] for r in runs if r["arm"] == arm and r.get(key) is not None]
        return round(statistics.median(vals), 1) if vals else None

    a_tps, b_tps = overall("direct", "decode_tps"), overall("middleware", "decode_tps")
    a_ttft, b_ttft = overall("direct", "ttft_ms"), overall("middleware", "ttft_ms")
    a_pt, b_pt = overall("direct", "prompt_tokens"), overall("middleware", "prompt_tokens")

    print("\nmedians across every run")
    print(f"  decode      {a_tps} -> {b_tps} tok/s"
          + (f"  ({(b_tps - a_tps) / a_tps * 100:+.1f}%)" if a_tps else ""))
    print(f"  ttft        {a_ttft} -> {b_ttft} ms"
          + (f"  ({b_ttft - a_ttft:+.1f} ms of memory-layer work)"
             if (a_ttft and b_ttft) else ""))
    print(f"  prompt tok  {a_pt} -> {b_pt}"
          + (f"  ({b_pt - a_pt:+.0f} injected)" if (a_pt and b_pt) else ""))
    print("\nDecode is the honest 'tokens per second': the middleware is not in "
          "that path,\nso it should be flat. Any real cost shows up in ttft.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": {"repeats": args.repeats,
                              "max_tokens": args.max_tokens,
                              "cold_cache": args.cold,
                              "baseline": args.baseline,
                              "middleware": args.middleware,
                              "prompts": prompts},
                   "runs": runs, "summary": summary}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
