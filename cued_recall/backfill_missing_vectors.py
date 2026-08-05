#!/usr/bin/env python3
"""Re-embed shelved blocks that have no vector, and drop the empty ones.

A block is embedded once, at creation. `_embed_and_store` logs a failure to the
WAL and moves on, and nothing ever retries -- so a block written while the
embedding server was restarting, busy, or briefly unreachable keeps its
metadata row, keeps its text, shows up in the admin table, and is invisible to
recall forever. Nothing surfaces that: `/admin/stats` counts blocks by status,
and status says `shelved` whether or not a vector exists.

Measured on a 1,812-block store on 4 August 2026: 729 shelved blocks had no
vector. 672 were empty -- no text, no stimulus, token_count 0 -- debris from
the empty-turn bug fixed in "Stop the Gemma/OpenCode turn from going out
empty". The other 57 held real content (26 reasoning, 16 result, 15 reading)
and re-embedded on the first attempt, which is what says the original failure
was transient rather than anything about the text.

    python backfill_missing_vectors.py --dry-run    # report, change nothing
    python backfill_missing_vectors.py              # re-embed
    python backfill_missing_vectors.py --purge-empty # also purge the empties

Empties are purged, not deleted: the same reversible status flip decay uses, so
this script cannot destroy anything. Safe to re-run -- it selects on "no vector
present", so a second pass over a repaired store is a no-op.

The same machinery serves the other job that rewrites vectors, changing
config.embed_source between "composite" (a reasoning block indexed by the
question that produced it) and "content" (indexed by what it says):

    python backfill_missing_vectors.py --fill-embed-text --dry-run
    python backfill_missing_vectors.py --fill-embed-text --all

--fill-embed-text populates embed_text on blocks written before that field
existed, copying from the block's own text; --all then re-embeds every
recallable block rather than only the ones with no vector. Run both before
switching embed_source, or the switch silently falls back to the composite for
every older block. Changing embed_source is a measured decision -- run the
sweep in evaluate/ first; see config.py.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cued_recall.config import Config          # noqa: E402
from cued_recall.embed import EmbeddingClient  # noqa: E402
from cued_recall.index import VectorIndex      # noqa: E402
from cued_recall.store import BlockStore       # noqa: E402
from cued_recall.models import BlockStatus     # noqa: E402
from cued_recall.utils import embed_source_text, truncate_tokens  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("--purge-empty", action="store_true",
                    help="also purge blocks that have no text at all")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--all", action="store_true",
                    help="re-embed every recallable block, not only the ones "
                         "with no vector (use after changing embed_source)")
    ap.add_argument("--fill-embed-text", action="store_true",
                    help="populate embed_text from the block's own text where "
                         "it is empty (blocks written before the field existed)")
    args = ap.parse_args()

    cfg = Config(Path(args.config))
    store = BlockStore(Path(cfg.store_path))
    embed = EmbeddingClient(cfg.embed_endpoint)
    index = VectorIndex(Path(cfg.store_path), dim=cfg.embed_dim)
    index.open()

    # Fail loudly rather than reporting every block as unfixable.
    try:
        await asyncio.to_thread(embed.embed, "vector backfill probe")
    except Exception as e:
        print(f"[ERROR] embedding server at {cfg.embed_endpoint} is not "
              f"answering ({e}). Start it and re-run.", file=sys.stderr)
        return 1

    if args.all:
        # Every block recall can reach. Paged rather than one huge list_meta,
        # so this works on a store far larger than the one it was written for.
        targets = []
        for status in ("shelved", "truncated"):
            offset = 0
            while True:
                items, _ = index.list_meta(status=status, limit=500,
                                           offset=offset)
                if not items:
                    break
                targets.extend(m["block_id"] for m in items)
                offset += len(items)
    else:
        targets = index.blocks_without_vectors()
    if args.limit:
        targets = targets[:args.limit]
    print(f"store          : {cfg.store_path}")
    print(f"embedder       : {cfg.embed_endpoint}")
    print(f"embed_source   : {cfg.embed_source}")
    print(f"selected       : {len(targets)} "
          f"({'all recallable blocks' if args.all else 'missing vectors only'})")
    print(f"mode           : "
          f"{'DRY RUN -- nothing written' if args.dry_run else 'APPLY'}\n")
    if not targets:
        return 0

    repaired = empty = failed = filled = 0
    for i, block_id in enumerate(targets, 1):
        block = await asyncio.to_thread(store.get, block_id)
        if block is None:
            failed += 1
            continue
        # Before choosing the source text, not after: a block written before
        # embed_text existed has nothing in the content channel, so under
        # embed_source: content it would fall back to the composite and the
        # switch would quietly do nothing for the older half of the store.
        if args.fill_embed_text and not (block.embed_text or "").strip():
            if (block.text or "").strip():
                filled += 1
                if not args.dry_run:
                    block.embed_text = truncate_tokens(
                        block.text, cfg.embed_token_limit)
                    await asyncio.to_thread(store.put, block)
        text = embed_source_text(block, cfg.embed_source)[:2000]
        if not text.strip():
            empty += 1
            if args.purge_empty and not args.dry_run:
                block.status = BlockStatus.purged
                await asyncio.to_thread(store.put, block)
                index.update_status(block_id, "purged")
            continue
        if args.dry_run:
            repaired += 1
            continue
        try:
            vec = await asyncio.to_thread(embed.embed, text)
            await asyncio.to_thread(index.upsert_vector, block_id, vec)
            repaired += 1
        except Exception as e:
            failed += 1
            print(f"  ! {block_id[:8]} failed: {e}")
        if i % 50 == 0 or i == len(targets):
            print(f"  {i}/{len(targets)} processed, {repaired} re-embedded")

    print(f"\nre-embedded    : {repaired}")
    if args.fill_embed_text:
        print(f"embed_text set : {filled}")
    print(f"empty          : {empty}"
          f"{' (purged)' if args.purge_empty and not args.dry_run else ''}")
    if failed:
        print(f"failed         : {failed}")
    if args.dry_run:
        print("\nDry run -- nothing written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
