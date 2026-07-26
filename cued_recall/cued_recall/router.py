import asyncio
import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from .index import VectorIndex
from .models import Block, BlockStatus, BlockType, Verification
from .store import BlockStore
from .taxonomy import TAXONOMY, TAXONOMY_GROUPS, TAXONOMY_VERSION, validate_gist, validate_tags
from .wal import WAL


def build_admin_router(index: VectorIndex, store: BlockStore, wal: WAL, judge_run_fn,
                        tps_ring: list, embed=None):
    router = APIRouter(prefix="/admin")

    @router.get("/blocks")
    async def list_blocks(
        status: Optional[str] = None,
        type_: Optional[str] = None,
        conversation_id: Optional[str] = None,
        tag: Optional[str] = None,
        recall_min: Optional[int] = None,
        recall_max: Optional[int] = None,
        older_than_days: Optional[float] = None,
        sort: str = "created_at",
        order: str = "desc",
        limit: int = 100,
        offset: int = 0,
    ):
        older_than = (time.time() - older_than_days * 86400
                      if older_than_days else None)
        items, total = index.list_meta(
            status=status,
            type_=type_,
            conversation_id=conversation_id,
            tag=tag,
            recall_min=recall_min,
            recall_max=recall_max,
            older_than=older_than,
            sort=sort,
            order=order,
            limit=limit,
            offset=offset,
        )
        # Echo the sort that was actually applied, not the raw request: an
        # unknown column falls back to created_at, and reflecting the input
        # would tell the caller a sort happened that did not.
        return {"total": total, "items": items, "limit": limit, "offset": offset,
                "sort": sort if sort in VectorIndex.SORTABLE else "created_at",
                "order": "asc" if str(order).lower() == "asc" else "desc"}

    @router.get("/stats/growth")
    async def stats_growth(days: int = 30):
        return {"days": days, "rows": index.growth_by_day(days=days)}

    @router.get("/stats/distribution")
    async def stats_distribution():
        return {"buckets": index.token_histogram()}

    @router.get("/stats/recall")
    async def stats_recall(top: int = 20):
        return index.recall_effectiveness(top=top)

    @router.get("/stats/budget")
    async def stats_budget(limit: int = 50):
        """Recent per-turn recall budget decisions, newest first."""
        # The budget travels on each event rather than being read from config
        # here: the router has no config handle, and a historical turn should
        # show the budget that was actually in force when it ran.
        events = [e for e in wal.read_all()
                  if e.get("event") == "recall_budget"]
        return {"turns": list(reversed(events[-limit:]))}

    @router.get("/tags")
    async def list_tags():
        return {
            "taxonomy_version": TAXONOMY_VERSION,
            "taxonomy": TAXONOMY,
            "groups": TAXONOMY_GROUPS,
            "counts": index.tag_counts(),
        }

    @router.get("/blocks/{block_id}")
    async def get_block(block_id: str):
        block = store.get(block_id)
        if block is None:
            raise HTTPException(status_code=404, detail="block not found")
        meta = index.get_meta(block_id)
        history = [e for e in wal.read_all() if e.get("block_id") == block_id]
        return {"block": block.to_msgpack(), "meta": meta, "wal_events": history}

    @router.post("/blocks/{block_id}/verify")
    async def verify_block(block_id: str, body: dict):
        verification = body.get("verification")
        if verification not in ("accepted", "corrected"):
            raise HTTPException(status_code=400, detail="invalid verification value")
        index.update_verification(block_id, verification)
        block = store.get(block_id)
        if block:
            block.verification = Verification(verification)
            store.put(block)
        wal.write({
            "event": "admin_verify",
            "block_id": block_id,
            "verification": verification,
            "timestamp": time.time(),
        })
        return {"status": "ok"}

    @router.post("/blocks/delete")
    async def delete_blocks(body: dict):
        block_ids = body.get("block_ids", [])
        if not block_ids:
            raise HTTPException(status_code=400, detail="no block_ids provided")
        deleted = 0
        for bid in block_ids:
            index.delete_meta(bid)
            store.delete_file(bid)
            deleted += 1
        wal.write({
            "event": "admin_delete_blocks",
            "count": deleted,
            "timestamp": time.time(),
        })
        return {"status": "ok", "deleted": deleted}

    @router.post("/judge/run")
    async def run_judge():
        # Manual trigger judges all shelved blocks now (ignore the age gate).
        result = await judge_run_fn(min_age=0)
        return result or {"status": "ok", "processed": 0}

    @router.get("/stats")
    async def stats():
        block_counts = index.stats()
        block_files = len(list(store.blocks_dir.glob("*.msgpack")))
        wal_events = len(wal.read_all())
        return {
            "blocks_by_status_type": block_counts,
            "msgpack_files": block_files,
            "wal_events": wal_events,
        }

    @router.get("/tps")
    async def tps():
        if not tps_ring:
            return {"recent": [], "avg_tps": 0, "count": 0}
        avg = sum(e["tps"] for e in tps_ring) / len(tps_ring)
        return {"recent": tps_ring[-20:], "avg_tps": round(avg, 1), "count": len(tps_ring)}

    @router.get("/export")
    async def export_blocks(
        status: Optional[str] = None,
        type_: Optional[str] = None,
        tag: Optional[str] = None,
        include_full_text: bool = False,
        limit: int = 1000,
    ):
        items, _ = index.list_meta(
            status=status, type_=type_, tag=tag, limit=limit, offset=0
        )
        exported = []
        for m in items:
            block = store.get(m["block_id"])
            if block is None:
                continue
            # Privacy default: raw shelved blocks can contain whatever a
            # client pasted (configs, IPs, credentials). Only ship the actual
            # text when it's already a judge-produced summary (status
            # truncated), or when the caller explicitly opts in.
            text = block.text if (include_full_text or m["status"] == "truncated") else None
            exported.append({
                "block_id": block.block_id,
                "type": m["type"],
                "status": m["status"],
                "verification": m["verification"],
                "token_count": block.token_count,
                "created_at": block.created_at,
                "stimulus_text": block.stimulus_text,
                "gist": m.get("gist", ""),
                "tags": m.get("tags", []),
                "text": text,
            })
        return {
            "taxonomy_version": TAXONOMY_VERSION,
            "count": len(exported),
            "blocks": exported,
        }

    @router.post("/import")
    async def import_blocks(body: dict):
        if embed is None:
            raise HTTPException(status_code=503, detail="embedding client not available")
        incoming = body.get("blocks", [])
        imported = 0
        skipped = 0
        for item in incoming:
            text = item.get("text") or ""
            stimulus = item.get("stimulus_text", "") or ""
            if not text and not stimulus:
                skipped += 1
                continue
            try:
                type_ = BlockType(item.get("type", "reasoning"))
            except ValueError:
                type_ = BlockType.reasoning
            block = Block(
                type=type_,
                status=BlockStatus.truncated if text else BlockStatus.shelved,
                conversation_id="imported",
                turn_index=0,
                token_count=item.get("token_count", 0),
                text=text,
                stimulus_text=stimulus,
                # Verification is not portable: someone else's "accepted" is
                # not evidence for this instance. It has to be earned again
                # through this instance's own conversations.
                verification=Verification.unknown,
                created_at=item.get("created_at", time.time()),
                tags=validate_tags(item.get("tags", [])),
                gist=validate_gist(item.get("gist", "")),
            )
            store.put(block)
            index.upsert_block_meta(
                block.block_id, block.type.value, block.status.value,
                block.created_at, block.conversation_id, block.turn_index,
                block.token_count, block.verification.value, 0, 0.0,
            )
            index.set_tags(block.block_id, block.tags, block.gist)
            # Embeddings aren't portable across instances running different
            # embedding models, so the receiving side always re-embeds locally
            # rather than trusting a vector shipped in the export.
            embed_text = (stimulus or text)[:2000]
            if embed_text:
                try:
                    vec = await asyncio.to_thread(embed.embed, embed_text)
                    await asyncio.to_thread(index.upsert_vector, block.block_id, vec)
                except Exception:
                    pass
            imported += 1
        wal.write({
            "event": "admin_import",
            "imported": imported,
            "skipped": skipped,
            "timestamp": time.time(),
        })
        return {"status": "ok", "imported": imported, "skipped": skipped}

    return router
