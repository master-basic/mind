#!/usr/bin/env python3
"""Recount block tokens with the model's tokenizer.

Blocks written before the token-count fix stored `len(text.split())` -- a word
count. Sampled against the real tokenizer it measured low on every block (mean
0.77x, worst 0.52x on code and markdown), understating the admin `tokens`
column by ~42% in aggregate and, because the same field enforces
recall.budget_tokens, letting a 3,000-token recall budget spend closer to
3,900.

This rewrites token_count on every stored block, and mirrors it into index.db
so the admin table and the recall budget agree.

    python backfill_token_counts.py --dry-run     # report, change nothing
    python backfill_token_counts.py               # apply

Safe to re-run: it recomputes from block text rather than scaling the old
value, so a second pass over already-corrected blocks is a no-op.
"""

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cued_recall.config import Config          # noqa: E402
from cued_recall.store import BlockStore       # noqa: E402
from cued_recall.utils import count_tokens     # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml", help="path to config.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("--limit", type=int, help="only process the first N blocks")
    args = ap.parse_args()

    cfg = Config(Path(args.config))
    store = BlockStore(Path(cfg.store_path))
    index_db = Path(cfg.store_path) / "index.db"
    endpoint = cfg.reasoning_endpoint

    # Fail early and loudly: silently falling back to the estimator for every
    # block would write 40%-high numbers across the whole store, which is a
    # worse state than the word counts we are replacing.
    probe = await count_tokens("token count probe", endpoint,
                               cfg.chars_per_token, cfg.tokens_per_word)
    estimated = await count_tokens("token count probe", "http://0.0.0.0:1",
                                   cfg.chars_per_token, cfg.tokens_per_word)
    if probe == estimated:
        print(f"[ERROR] tokenizer at {endpoint} is not answering -- every block "
              f"would fall back to the estimator. Start the reasoning server "
              f"and re-run.", file=sys.stderr)
        return 1

    block_ids = [p.stem for p in sorted(store.blocks_dir.glob("*.msgpack"))]
    if args.limit:
        block_ids = block_ids[:args.limit]
    if not block_ids:
        print(f"No blocks found under {store.blocks_dir}")
        return 0

    print(f"{len(block_ids)} blocks in {store.blocks_dir}")
    print(f"tokenizer: {endpoint}")
    print(f"mode: {'DRY RUN -- nothing will be written' if args.dry_run else 'APPLY'}\n")

    conn = None if args.dry_run else sqlite3.connect(index_db)
    changed = failed = 0
    old_total = new_total = 0
    biggest = []

    try:
        for i, block_id in enumerate(block_ids, 1):
            block = await asyncio.to_thread(store.get, block_id)
            if block is None:
                failed += 1
                continue
            text = block.text or ""
            if not text.strip():
                continue

            old = block.token_count or 0
            new = await count_tokens(text, endpoint,
                                     cfg.chars_per_token, cfg.tokens_per_word)
            old_total += old
            new_total += new
            if new != old:
                changed += 1
                biggest.append((new - old, block_id, old, new))
                if not args.dry_run:
                    block.token_count = new
                    await asyncio.to_thread(store.put, block)
                    # The admin table reads index.db, the recall budget reads
                    # the block file; both have to move or they disagree.
                    conn.execute(
                        "UPDATE blocks SET token_count = ? WHERE block_id = ?",
                        (new, block_id),
                    )

            if i % 50 == 0 or i == len(block_ids):
                print(f"  {i}/{len(block_ids)} processed, {changed} changed")

        if conn:
            conn.commit()
    finally:
        if conn:
            conn.close()

    biggest.sort(reverse=True)
    if biggest:
        print("\nlargest corrections:")
        print(f"  {'block':10s} {'was':>8s} {'now':>8s} {'delta':>8s}")
        for delta, block_id, old, new in biggest[:10]:
            print(f"  {block_id[:8]:10s} {old:8,d} {new:8,d} {delta:+8,d}")

    pct = (100 * (new_total - old_total) / old_total) if old_total else 0
    print(f"\nblocks changed : {changed}/{len(block_ids)}")
    if failed:
        print(f"unreadable     : {failed}")
    print(f"token total    : {old_total:,} -> {new_total:,}  ({pct:+.0f}%)")
    if args.dry_run:
        print("\nDry run -- nothing written. Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
