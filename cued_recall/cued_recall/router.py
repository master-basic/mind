import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from .index import VectorIndex
from .models import BlockStatus, Verification
from .store import BlockStore
from .wal import WAL


def build_admin_router(index: VectorIndex, store: BlockStore, wal: WAL, judge_run_fn, tps_ring: list):
    router = APIRouter(prefix="/admin")

    @router.get("/blocks")
    async def list_blocks(
        status: Optional[str] = None,
        type_: Optional[str] = None,
        conversation_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ):
        items, total = index.list_meta(
            status=status,
            type_=type_,
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
        )
        return {"total": total, "items": items, "limit": limit, "offset": offset}

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

    return router
