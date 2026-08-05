#!/usr/bin/env python3
"""What the relevance judge does when the note is a real block.

eval_retrieval.py originally used each seed's *prompt* as the stand-in for a
stored block, on both sides of the pipeline: as the embed key and as the note
shown to the judge. No block looks like that. A reasoning block's text is a
think trace of several thousand characters; a result block's is the answer.

That distinction turns out to carry the whole measured false-fire figure. With
the seed prompt as the note the judge refuses 6 of 6 traps and false-fire
lands at 0.00 -- the number recorded in config.py and benchmark.md as the
reason judge_enabled defaults on. With a real block as the note it refuses 1 of
6, and false-fire at the same operating point is 0.45-0.64. Legitimate recall
is unaffected either way (18/18 across exact, paraphrase and crosslingual), so
this is specifically a failure to say no.

The production evidence agrees with the second number, not the first:
grading_traps.md records ocr1-trap injecting 2,005 tokens of the seed's
client-side stack into a phase-2 question, and the answer anchoring on it. If
the judge really refused 6 of 6 traps in production, that could not have
happened.

    python eval_judge_notes.py                 # score the shipped prompt
    python eval_judge_notes.py --variants      # and the candidate rewordings

Needs seed_blocks.jsonl (make_seed_blocks.py) and the judge server.
"""

import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cued_recall"))

try:
    from cued_recall.utils import (RELEVANCE_SYSTEM, relevance_prompt,
                                   truncate_tokens)
except ImportError as e:
    sys.exit(f"needs the cued_recall package importable: {e}")


# Candidate rewordings, scored against the same pairs as the shipped one.
#
# The shipped prompt asks whether the note "contains information that would
# change or improve the answer". For a phase-1 note against a phase-2 question
# about the same system that is honestly yes -- it is the same codebase, the
# same stack, the same file. The refusal cases are listed afterwards as
# exceptions, and on a 1.5B model reading several thousand characters of answer
# they are the part that gets lost.
def prompt_task_match(question, note):
    """Ask what the note is *about*, not whether it is related."""
    return (
        f"Question:\n{truncate_tokens(question, 300)}\n\n"
        f"Note from the archive:\n{truncate_tokens(note, 900)}\n\n"
        "Is the note about the same task as the question, or about a "
        "different task that happens to involve the same system?\n"
        "The note may share the same project, files, tools and vocabulary and "
        "still be a different task -- an earlier stage, a later stage, the "
        "reverse operation, or a separate problem in the same codebase. Those "
        "are all different tasks.\n"
        "Answer yes only if a person doing the task in the question would use "
        "this note as it stands.\n"
        "Answer yes or no."
    )


def prompt_same_step(question, note):
    """Put the stage test first, as the question rather than as an exception."""
    return (
        f"Question:\n{truncate_tokens(question, 300)}\n\n"
        f"Note from the archive:\n{truncate_tokens(note, 900)}\n\n"
        "First: does the note describe the same step of the work as the "
        "question, or a different step of the same project?\n"
        "Answer no if it is a different step, a different direction of the "
        "same operation, an earlier or later phase, or a different problem "
        "that shares the same tools.\n"
        "Answer yes only if the note is about the very thing being asked.\n"
        "Answer yes or no."
    )


VARIANTS = {
    "shipped": relevance_prompt,
    "task-match": prompt_task_match,
    "same-step": prompt_same_step,
}

# Which note text stands in for the block.
#
#   reasoning  what _filter_by_relevance passes today: block.text
#   answer     block.text for a result block
#   prompt     the question the block was written to answer. Already stored --
#              build_stimulus puts it first in stimulus_text -- so using it
#              costs nothing. This is also what the original harness used, by
#              accident, and it is the only note that refuses the trap family.
#   gist       the tagger's 40-char label, written on every block and read by
#              nothing (the F11 complaint)
#   both       the question and the content together, so the judge can make the
#              same-task check without losing the ability to match on content
NOTE_SOURCES = ("reasoning", "answer", "prompt", "gist", "both")


def ask(endpoint, system, text, timeout=60):
    payload = {"messages": [{"role": "system", "content": system},
                            {"role": "user", "content": text}],
               "temperature": 0, "max_tokens": 4}
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    reply = (data["choices"][0]["message"]["content"] or "").strip().lower()
    return not reply.startswith("no")          # True = keep, as the pipeline


def score_note_sources(args, seeds, probes, note_for, relations):
    """The shipped prompt against every candidate note, on the same pairs.

    This is the comparison that matters: the wording is not the variable that
    moves the trap family, the note is.
    """
    print(f"prompt: shipped   pairs: same-family only   "
          f"judge: {args.judge_endpoint}\n")
    width = 12
    print(f"{'relation':<14}{'n':>3}  " +
          "".join(s.rjust(width) for s in NOTE_SOURCES))
    print("-" * (19 + width * len(NOTE_SOURCES)))

    kept = defaultdict(dict)
    for rel in relations:
        sel = [p for p in probes if p["relation"] == rel]
        line = f"{rel:<14}{len(sel):>3}  "
        for src in NOTE_SOURCES:
            n = sum(ask(args.judge_endpoint, RELEVANCE_SYSTEM,
                        relevance_prompt(p["prompt"], note_for(p["family"], src)))
                    for p in sel)
            kept[rel][src] = n
            line += f"{n}/{len(sel)}".rjust(width)
        print(line)

    print("\nkept = the judge let the block through.")
    print("  exact/paraphrase/crosslingual: kept is correct")
    print("  trap: kept is the anchoring failure in grading_traps.md\n")
    summary = {}
    for src in NOTE_SOURCES:
        good = sum(kept[r][src] for r in relations[:3])
        leaked = kept["trap"][src]
        summary[src] = {"recall_kept": good, "recall_total": 18,
                        "trap_leaked": leaked, "trap_total": 6}
        flag = "  <-- production today" if src == "reasoning" else ""
        print(f"  {src:<12} recall {good}/18   trap leaked {leaked}/6{flag}")

    print("\nn=6 per relation. A note source that wins here has been shown not")
    print("to lose recall on this corpus, not to generalise.")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"comparison": "note_sources",
                       "per_relation": {r: kept[r] for r in relations},
                       "summary": summary}, f, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--blocks", default="seed_blocks.jsonl")
    ap.add_argument("--judge-endpoint", default="http://127.0.0.1:8081")
    ap.add_argument("--variants", action="store_true",
                    help="also score the candidate rewordings")
    ap.add_argument("--all-notes", action="store_true",
                    help="score the shipped prompt against every note source, "
                         "which is the comparison that isolates the defect")
    ap.add_argument("--note-source", default="reasoning",
                    choices=NOTE_SOURCES,
                    help="what stands in for the block's text")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.corpus, encoding="utf-8")
            if l.strip()]
    seeds = {r["family"]: r for r in rows if r["role"] == "seed"}
    probes = [r for r in rows if r["role"] == "probe"
              and r["family"] in seeds]
    gen = {}
    for line in open(args.blocks, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            gen[r["family"]] = r

    def note_for(family, source=None):
        source = source or args.note_source
        g = gen[family]
        if source == "prompt":
            return seeds[family]["prompt"]
        if source == "gist":
            return g.get("gist") or ""
        if source == "both":
            # The shape the pipeline can build for free: a reasoning block
            # already carries its originating question in stimulus_text and its
            # own words in text.
            return (f"This note was written while answering:\n"
                    f"{seeds[family]['prompt']}\n\n"
                    f"The note says:\n{g.get('reasoning') or g['answer']}")
        return g.get(source) or g["answer"]

    # Only same-family pairs. The question is not "can the judge tell two
    # unrelated things apart" -- cosine already does that -- but "when the
    # vector search hands it the block from the right project, can it tell
    # whether that block is about this task."
    variants = VARIANTS if args.variants else {"shipped": relevance_prompt}
    # Recall relations first, then the one that must be refused.
    RELATIONS = ("exact", "paraphrase", "crosslingual", "trap")

    if args.all_notes:
        return score_note_sources(args, seeds, probes, note_for, RELATIONS)

    print(f"note source: {args.note_source}   "
          f"pairs: same-family only   judge: {args.judge_endpoint}\n")
    width = 16
    print(f"{'relation':<14}{'n':>3}  " +
          "".join(v.rjust(width) for v in variants))
    print("-" * (19 + width * len(variants)))

    kept = defaultdict(dict)
    for rel in RELATIONS:
        sel = [p for p in probes if p["relation"] == rel]
        line = f"{rel:<14}{len(sel):>3}  "
        for vname, vfn in variants.items():
            n = sum(ask(args.judge_endpoint, RELEVANCE_SYSTEM,
                        vfn(p["prompt"], note_for(p["family"])))
                    for p in sel)
            kept[rel][vname] = n
            line += f"{n}/{len(sel)}".rjust(width)
        print(line)

    print("\nkept = the judge let the block through.")
    print("  exact/paraphrase/crosslingual: kept is correct")
    print("  trap: kept is the anchoring failure in grading_traps.md\n")
    summary = {}
    for vname in variants:
        good = sum(kept[r][vname] for r in RELATIONS[:3])
        leaked = kept["trap"][vname]
        summary[vname] = {"recall_kept": good, "recall_total": 18,
                          "trap_leaked": leaked, "trap_total": 6}
        print(f"  {vname:<12} recall {good}/18   trap leaked {leaked}/6")

    print("\nn=6 per relation. A wording that wins here has not been shown to")
    print("generalise; it has been shown not to lose recall on this corpus.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"note_source": args.note_source,
                       "per_relation": {r: kept[r] for r in RELATIONS},
                       "summary": summary}, f, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
