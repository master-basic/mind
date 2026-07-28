#!/usr/bin/env python3
"""
Correction-detection evaluation for cued-recall.

Answers the question the design has been assuming rather than measuring: how
often does correction detection fire on a message that was NOT a correction?
That direction is the expensive one. A block marked corrected stops being
recalled at once, and a pattern-sourced one becomes purgeable ahead of the
normal age gate, so a false positive costs a memory.

Two detectors, scored separately because they carry different authority:

  patterns  -- utils.matches_correction against config correction_patterns.
               Deterministic, runs on the request path, and is trusted enough
               to shorten a block's life.
  verifier  -- the few-shot yes/no classifier on the small model, for the
               phrasings the patterns cannot safely reach. Needs the judge
               server; skipped with --no-model.

The labelled set is corrections.jsonl: {"message", "answer", "is_correction"}.
Hand-labelled on purpose -- correctness is the whole point and no script can
supply it. --from-chats mines candidate negatives out of a live chats.db so the
negative half reflects real traffic rather than only phrasings someone thought
to write down, which is exactly the bias that made the old five-phrase list
look adequate.

Usage:
  python eval_correction.py
  python eval_correction.py --no-model            # patterns only, no servers
  python eval_correction.py --from-chats r:/cued_recall/store/chats.db
"""

import argparse
import json
import os
import sqlite3
import sys
import urllib.request


def load_shipped():
    """Import the shipped matcher and pattern list, never a copy of them."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "cued_recall"))
    try:
        from cued_recall.config import Config
        from cued_recall.utils import matches_correction
    except ImportError as e:
        sys.exit(f"needs the cued_recall package importable: {e}")
    cfg_path = os.path.join(os.path.dirname(here), "cued_recall", "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(here), "cued_recall",
                                "config.example.yaml")
    cfg = Config(cfg_path)
    return cfg, matches_correction


def verifier_says_yes(answer, message, endpoint, timeout=60):
    """One call to the shipped classifier. None means it failed or hedged."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "cued_recall"))
    from cued_recall.verifier import CorrectionVerifier

    prompt = CorrectionVerifier._prompt(answer, message)
    payload = {
        "messages": [
            {"role": "system", "content": CorrectionVerifier.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": CorrectionVerifier.MAX_TOKENS,
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
        print(f"  [verifier] failed: {type(e).__name__}: {e}")
        return None
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    return None


def score(name, rows, predict):
    """Confusion matrix and the three rates, with the counts behind them."""
    tp = fp = tn = fn = skipped = 0
    false_positives = []
    false_negatives = []
    for r in rows:
        got = predict(r)
        if got is None:
            skipped += 1
            continue
        if r["is_correction"]:
            if got:
                tp += 1
            else:
                fn += 1
                false_negatives.append(r["message"])
        else:
            if got:
                fp += 1
                false_positives.append(r["message"])
            else:
                tn += 1

    n_pos, n_neg = tp + fn, fp + tn
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / n_pos if n_pos else float("nan")
    fpr = fp / n_neg if n_neg else float("nan")

    print(f"\n=== {name} ===")
    print(f"  positives={n_pos}  negatives={n_neg}"
          + (f"  undecided={skipped}" if skipped else ""))
    print(f"  tp={tp}  fp={fp}  tn={tn}  fn={fn}")
    print(f"  precision={precision:.2f}  recall={recall:.2f}  "
          f"false-positive rate={fpr:.2f}")
    # The individual mistakes matter more than the rate at this sample size:
    # with tens of rows, one bad row moves the second decimal place, and a
    # false positive is a deleted memory whichever way the rate rounds.
    for m in false_positives:
        print(f"  FP: {m[:100]}")
    for m in false_negatives:
        print(f"  FN: {m[:100]}")
    return {"name": name, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "false_positive_rate": fpr,
            "n_positives": n_pos, "n_negatives": n_neg, "undecided": skipped}


def mine_chats(db_path, limit=200):
    """Unlabelled user messages from a live transcript store.

    Written out with is_correction: null for a human to fill in. Real traffic
    is where the false positives live: "no problem, carry on" and "I know it
    doesn't exist yet" were both found this way and both killed patterns that
    looked fine in isolation.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT content FROM messages WHERE role='user' "
        "ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [{"message": r[0][:500], "answer": "", "is_correction": None}
            for r in rows if (r[0] or "").strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default="corrections.jsonl")
    ap.add_argument("--judge-endpoint", default="http://127.0.0.1:8081")
    ap.add_argument("--no-model", action="store_true",
                    help="score the patterns only; needs no servers")
    ap.add_argument("--from-chats", metavar="CHATS_DB",
                    help="mine unlabelled candidates from a chats.db and exit")
    ap.add_argument("--out", default="correction_results.json")
    args = ap.parse_args()

    if args.from_chats:
        rows = mine_chats(args.from_chats)
        path = "corrections_candidates.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} unlabelled rows to {path}")
        print("Set is_correction to true/false on each, then append the ones "
              f"you are sure about to {args.labelled}.")
        return

    if not os.path.exists(args.labelled):
        sys.exit(f"no labelled set at {args.labelled} -- start one with "
                 "--from-chats, or hand-write it")
    rows = [json.loads(l) for l in open(args.labelled, encoding="utf-8")
            if l.strip()]
    rows = [r for r in rows if r.get("is_correction") is not None]
    if not rows:
        sys.exit("every row is unlabelled (is_correction: null)")

    cfg, matches_correction = load_shipped()
    print(f"{len(rows)} labelled rows, {len(cfg.correction_patterns)} patterns")

    results = [score("patterns", rows,
                     lambda r: matches_correction(r["message"],
                                                  cfg.correction_patterns))]

    if not args.no_model:
        results.append(score(
            f"verifier ({args.judge_endpoint})", rows,
            lambda r: verifier_says_yes(r.get("answer", ""), r["message"],
                                        args.judge_endpoint),
        ))
        # What the pipeline actually does: the verifier is only consulted for
        # messages no pattern caught, so scoring it alone overstates its share
        # of the work in both directions.
        results.append(score(
            "combined (pattern, else verifier)", rows,
            lambda r: (True
                       if matches_correction(r["message"],
                                             cfg.correction_patterns)
                       else verifier_says_yes(r.get("answer", ""),
                                              r["message"],
                                              args.judge_endpoint)),
        ))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
