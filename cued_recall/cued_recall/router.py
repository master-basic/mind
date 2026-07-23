import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from .index import VectorIndex
from .models import BlockStatus, Verification
from .store import BlockStore
from .wal import WAL


def build_admin_router(index: VectorIndex, store: BlockStore, wal: WAL, judge_run_fn):
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

    @router.post("/judge/run")
    async def run_judge():
        await judge_run_fn()
        return {"status": "judge pass triggered"}

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

    return router
