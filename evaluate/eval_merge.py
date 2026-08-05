#!/usr/bin/env python3
"""Decide whether judge.merge_enabled is safe to turn on (Phase 3.1 / F1).

The merge pass is built and off by default; the plan says "default off in the
first PR, on after measured". This is the measurement, and it never touches a
store you care about: it works on a fresh copy of a snapshot.

Protocol per run:

  1. copy <store> (default: snapshots/latest) to a temp dir
  2. run the real judge._merge_pass against the copy -- 0 clusters here
     means the 7-day age gate or the 0.90 similarity gate said "not yet",
     not that the pass was skipped
  3. seed merge_min_cluster near-duplicate blocks (real embeddings, aged
     past merge_min_age_s) and run the pass again
  4. print the blocks_merged / merge_rejected / block_retired_into_merge
     WAL events and the merged block itself
  5. probe recall with a related-but-new question the originals also would
     have matched, and check the merged block fires

Needs the judge (8081), embed (8082) and reasoning (8080) servers, like every
other eval in this directory. Reads config.yaml from the cued_recall package
dir. Nothing outside the temp copy is written.

Usage:
  python eval_merge.py
"""

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.join(os.path.dirname(HERE), "cued_recall")
sys.path.insert(0, PACKAGE)

from cued_recall.config import Config            # noqa: E402
from cued_recall.embed import EmbeddingClient    # noqa: E402
from cued_recall.index import VectorIndex        # noqa: E402
from cued_recall.judge import Judge              # noqa: E402
from cued_recall.models import Block, BlockStatus, BlockType, Verification  # noqa: E402
from cued_recall.pipeline import Pipeline        # noqa: E402
from cued_recall.store import BlockStore         # noqa: E402
from cued_recall.wal import WAL                  # noqa: E402

# The DNS-latency family from update_implement.md §12 -- three ways of saying
# one thing, with two specifics that a merge must keep.
SEED_FAMILIES = {
    "dns_latency": {
        "seeds": [
            "first DNS lookup took 840ms and now takes 60ms after raising "
            "dns.cache_ttl to 300 in /etc/resolv.conf",
            "I raised dns.cache_ttl to 300 in /etc/resolv.conf and the "
            "first lookup dropped from 840ms to 60ms",
            "DNS cache TTL set to 300 in /etc/resolv.conf: first lookup "
            "latency fell from 840ms to 60ms",
        ],
        "probe": ("how can I make DNS lookups faster on this server? "
                  "what config file do I change?"),
    },
    "nginx_timeout": {
        "seeds": [
            "the nginx worker timeout is 60 seconds, raised from 30, "
            "in /etc/nginx.conf",
            "raised the nginx timeout to 60 seconds from 30 in /etc/nginx.conf",
            "nginx.conf: worker timeout is 60 seconds, it was 30 seconds",
        ],
        "probe": ("why is my nginx timing out so fast? can I raise the "
                  "worker timeout?"),
    },
    "python_version": {
        "seeds": [
            "we pinned python to 3.12 because 3.13 broke the torch wheels",
            "the project now uses python 3.12: 3.13 broke torch on this box",
            "python 3.12 instead of 3.13 -- torch wheels do not build on 3.13",
        ],
        "probe": ("what python version is the project pinned to, and why "
                  "did we move off 3.13?"),
    },
}

SEED_AGE_S = 30 * 86400  # well past the default 7-day gate


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def event_report(wal, *events):
    return [e for e in wal.iter_all() if e["event"] in events]


async def pass_on_copy(cfg, embed, store_root, family: list) -> dict:
    """One real pass over a copied store seeded with `family`; returns records."""
    index = VectorIndex(store_root, dim=cfg.embed_dim)
    index.open()
    wal = WAL(store_root / "wal.jsonl")
    wal.open()
    store = BlockStore(store_root)
    judge = Judge(cfg, store, index, wal, embed)

    now = time.time()
    for i, text in enumerate(family):
        bid = f"seed_{i}"
        b = Block(block_id=bid, type=BlockType.reasoning,
                  status=BlockStatus.shelved,
                  conversation_id="seed-conv", turn_index=0,
                  token_count=len(text.split()), text=text,
                  embed_text=text, question_text="how do I configure this?",
                  verification=Verification.unknown,
                  created_at=now - SEED_AGE_S)
        store.put(b)
        index.upsert_block_meta(bid, b.type.value, b.status.value,
                                b.created_at, b.conversation_id,
                                b.turn_index, b.token_count,
                                b.verification.value, 0, 0.0)
        index.upsert_vector(bid, unit(await asyncio.to_thread(
            embed.embed, text)))

    # Capture what the model actually wrote even when the pass refuses it --
    # the dropped text is the evidence, not the counter.
    draft = {}
    real_merge_notes = judge._merge_notes

    async def record_notes(blocks):
        text = await real_merge_notes(blocks)
        draft["text"] = text
        draft["parents"] = sorted(b.block_id for b in blocks)
        return text
    judge._merge_notes = record_notes

    out = await judge._merge_pass(time.time() + 300)

    merged = event_report(wal, "blocks_merged")
    rejected = event_report(wal, "merge_rejected")
    retired = event_report(wal, "block_retired_into_merge")
    abandoned = event_report(wal, "merge_abandoned")

    merged_blocks = []
    for ev in merged:
        b = store.get(ev["block_id"])
        if b is not None:
            merged_blocks.append({
                "block_id": b.block_id,
                "text": b.text,
                "parents": sorted(b.parents),
                "tokens": b.token_count,
            })

    report = {
        "counts": out,
        "events": {
            "blocks_merged": merged,
            "merge_rejected": rejected,
            "block_retired_into_merge": retired,
            "merge_abandoned": abandoned,
        },
        "merged_blocks": merged_blocks,
        "draft": draft,
    }
    report["originals"] = []
    for i in range(len(family)):
        bid = f"seed_{i}"
        b = store.get(bid)
        report["originals"].append({
            "block_id": bid,
            "status": b.status.value if b else None,
            "vector_present": index.get_vector(bid) is not None,
            "file_kept": b is not None and bool(b.text),
        })
    wal.close()
    index.close()
    return report


async def recall_probe(cfg, embed, store_root, merged_id: str, probe: str) -> dict:
    """Does the merged block fire for a related-but-new probe?"""
    index = VectorIndex(store_root, dim=cfg.embed_dim)
    index.open()
    wal = WAL(store_root / "wal.jsonl")
    wal.open()
    pipeline = Pipeline(cfg, BlockStore(store_root), index, embed, wal)
    try:
        results = await pipeline.recall_blocks(probe)
        hits = [b.block_id for b, _sim in results]
        return {
            "probe": probe,
            "recalled": hits,
            "merged_block_recalls": merged_id in hits,
        }
    finally:
        wal.close()
        index.close()


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=os.path.join(
        os.path.dirname(HERE), "snapshots", "latest"))
    ap.add_argument("--config", default=os.path.join(PACKAGE, "config.yaml"))
    args = ap.parse_args()

    src = Path(args.store)
    if not (src / "index.db").exists():
        sys.exit(f"{src} does not look like a store root (no index.db)")

    cfg = Config(args.config)
    embed = EmbeddingClient(cfg.embed_endpoint,
                            ctx_tokens=cfg.embed_ctx_tokens,
                            chars_per_token=cfg.chars_per_token,
                            tokens_per_word=cfg.tokens_per_word)
    embed.detect_ctx_tokens()
    dim = len(embed.embed("dimension probe"))
    cfg.embed_dim = dim

    print(f"source store : {src}")
    print(f"embed dim    : {dim}")
    print(f"judge        : {cfg.judge_endpoint}  embed: {cfg.embed_endpoint}\n")

    # Pass 1: the snapshot as it is. If nothing clusters, that is a real
    # result -- every block in this store is younger than merge_min_age_s.
    root1 = Path(tempfile.mkdtemp(prefix="merge_pass_"))
    shutil.copytree(src, root1, dirs_exist_ok=True)
    try:
        cfg.judge.merge_enabled = True
        r1 = await pass_on_copy(cfg, embed, root1, family=[])
        print("== pass 1: snapshot as-is ==")
        print(f"clusters={r1['counts']['clusters']} "
              f"merged={r1['counts']['merged_blocks']} "
              f"retired={r1['counts']['retired']} "
              f"rejected={len(r1['events']['merge_rejected'])}")
        for ev in r1["events"]["merge_rejected"]:
            print(f"  merge_rejected: {ev['reason']} "
                  f"parents={ev['parents']} lost={ev['lost_specifics']}")
        print()

        # Pass 2..n: the plan's acceptance fixture -- 3 near-duplicates, old,
        # one family at a time so a refusal in one cannot mask a merge in
        # another.
        for name, fam in SEED_FAMILIES.items():
            rootn = Path(tempfile.mkdtemp(prefix="merge_seed_"))
            shutil.copytree(src, rootn, dirs_exist_ok=True)
            r = await pass_on_copy(cfg, embed, rootn, family=fam["seeds"])
            verdict = ("merged" if r["merged_blocks"] else "refused"
                       if r["events"]["merge_rejected"] else "no cluster")
            print(f"== pass 2: family {name!r} -> {verdict} ==")
            print(f"clusters={r['counts']['clusters']} "
                  f"merged={r['counts']['merged_blocks']} "
                  f"retired={r['counts']['retired']} "
                  f"rejected={len(r['events']['merge_rejected'])}")
            for ev in r["events"]["merge_rejected"]:
                print(f"  merge_rejected: {ev['reason']} "
                      f"parents={ev['parents']} lost={ev['lost_specifics']}")
            if r["draft"].get("text"):
                print(f"  model draft: {r['draft']['text']}")
            for m in r["merged_blocks"]:
                print(f"  merged {m['block_id']} ({m['tokens']} tokens) from "
                      f"{m['parents']}: {m['text']}")
                r3 = await recall_probe(cfg, embed, rootn, m["block_id"],
                                        fam["probe"])
                print(f"  recall probe -> {r3['recalled']} "
                      f"(merged fires: {r3['merged_block_recalls']})")
            for o in r["originals"]:
                print(f"  original {o['block_id']}: status={o['status']} "
                      f"vector={'present' if o['vector_present'] else 'dropped'} "
                      f"file={'kept' if o['file_kept'] else 'GONE'}")
            shutil.rmtree(rootn, ignore_errors=True)
            print()
    finally:
        shutil.rmtree(root1, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
