#!/usr/bin/env python3
"""What the next judge pass would purge, and why -- without purging anything.

Utility decay (judge.utility_decay, on since 2026-08-05) replaced the rule that
made a single recall a permanent exemption from age-based purging. That was the
point: the store could only ever grow, and "recalled once, eighteen months ago"
outranked nothing at all. But it also means the first pass after the change
sees blocks it was previously forbidden to touch, and on a store that has been
running for a while that can be a lot of them at once.

So: look first. This runs the real Judge._should_purge against the real store
and prints what it decides, block by block, and changes nothing.

    python report_decay.py                    # summary
    python report_decay.py --list             # every block it would purge
    python report_decay.py --old-rule         # what the previous rule would do

Purging is reversible -- it flips a status and drops a vector, and
/admin/blocks/restore brings it back -- unless judge.purge_deletes_file is on,
which it is not by default. Check that before running a pass on a store you
care about.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cued_recall.config import Config          # noqa: E402
from cued_recall.index import VectorIndex      # noqa: E402
from cued_recall.judge import Judge            # noqa: E402
from cued_recall.store import BlockStore       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--list", action="store_true",
                    help="print every block that would be purged")
    ap.add_argument("--old-rule", action="store_true",
                    help="report against the pre-utility rule instead")
    ap.add_argument("--limit", type=int, default=100000)
    args = ap.parse_args()

    cfg = Config(Path(args.config))
    if args.old_rule:
        cfg.judge.utility_decay = False
    store = BlockStore(Path(cfg.store_path))
    index = VectorIndex(Path(cfg.store_path), dim=cfg.embed_dim)
    index.open()
    judge = Judge(cfg, store, index, wal=None)

    candidates = index.decay_candidates(cfg.judge.purge_age_s, limit=args.limit)
    total_blocks = index.list_meta(limit=1)[1]

    print(f"store         : {cfg.store_path}")
    print(f"blocks        : {total_blocks}")
    print(f"rule          : "
          f"{'pre-2026-08-05 (recall_count == 0)' if args.old_rule else 'utility decay'}")
    if not args.old_rule:
        j = cfg.judge
        print(f"weights       : recall {j.utility_recall_weight}/day-credit, "
              f"uncontested {j.utility_uncontested_weight}, "
              f"floor {j.utility_floor}")
    print(f"candidates    : {len(candidates)}\n")

    now = time.time()
    doomed = []
    reasons = {}
    for bid in candidates:
        block = store.get(bid)
        if block is None:
            continue
        meta = index.get_meta(bid) or {}
        recall_count = meta.get("recall_count") or 0
        uncontested = meta.get("uncontested_recalls") or 0
        last_recalled = meta.get("last_recalled") or 0.0
        verification = meta.get("verification", "unknown")
        source = meta.get("verification_source", "")
        pinned = bool(meta.get("pinned") or block.pinned)
        age = now - block.created_at
        if not judge._should_purge(verification, recall_count, age,
                                   worthless=False, source=source,
                                   pinned=pinned, uncontested=uncontested,
                                   last_recalled=last_recalled):
            continue
        reason = ("corrected" if verification == "corrected"
                  else "never_recalled" if recall_count == 0
                  else "utility_exhausted")
        reasons[reason] = reasons.get(reason, 0) + 1
        doomed.append((bid, reason, recall_count, uncontested,
                       age / 86400.0, block.token_count,
                       judge._utility(recall_count, uncontested, age,
                                      last_recalled)))

    print(f"would purge   : {len(doomed)} blocks, "
          f"{sum(d[5] for d in doomed):,} tokens")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<18} {n}")

    # The number that matters for a store being migrated: blocks the old rule
    # protected and the new one does not.
    ever_recalled = [d for d in doomed if d[2] > 0]
    if ever_recalled and not args.old_rule:
        print(f"\n{len(ever_recalled)} of those were recalled at least once, "
              f"so the previous rule would have kept them forever.")
        print("Reversible: purge flips a status and drops a vector. "
              "/admin/blocks/restore brings them back")
        print("unless judge.purge_deletes_file is on "
              f"(currently {cfg.judge.purge_deletes_file}).")

    if args.list and doomed:
        print(f"\n{'block':<10}{'reason':<19}{'recalls':>8}{'unc':>5}"
              f"{'age_d':>8}{'tokens':>8}{'utility':>9}")
        for bid, reason, rc, unc, age_d, tok, util in sorted(
                doomed, key=lambda d: d[6]):
            print(f"{bid[:8]:<10}{reason:<19}{rc:>8}{unc:>5}"
                  f"{age_d:>8.0f}{tok:>8}{util:>9.1f}")

    index.close()
    print("\nNothing was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
