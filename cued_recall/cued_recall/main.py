import asyncio
import signal
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .config import Config
from .embed import EmbeddingClient
from .index import VectorIndex
from .judge import Judge
from .models import BlockStatus
from .pipeline import Pipeline
from .router import build_admin_router
from .store import BlockStore
from .wal import WAL
from fastapi.staticfiles import StaticFiles


def create_app(config_path: str = "config.yaml") -> FastAPI:
    cfg = Config(config_path)
    store_path = Path(cfg.store_path)
    snapshot_path = Path(cfg.snapshot_path)

    store_path.mkdir(parents=True, exist_ok=True)
    snapshot_path.mkdir(parents=True, exist_ok=True)

    wal = WAL(store_path / "wal.jsonl")
    wal.open()

    store = BlockStore(store_path)
    embed = EmbeddingClient(cfg.embed_endpoint)

    # The vector index dimension MUST match the embedding model's output
    # (e.g. nomic-embed-text-v1.5 = 768). A mismatch makes every upsert/query
    # raise and the whole chat path 500s. Detect it live from the embed server.
    embed_dim = _detect_embed_dim(embed, cfg)
    index = VectorIndex(store_path, dim=embed_dim)
    index.open()

    pipeline = Pipeline(cfg, store, index, embed, wal)
    judge = Judge(cfg, store, index, wal)

    app = FastAPI(title="Cued Recall Middleware")

    judge_pass_counter = [0]
    judge_running = [False]
    judge_tokens = [0]

    async def run_judge_pass(min_age=None):
        if judge_running[0]:
            return {"status": "already running", "processed": 0}
        judge_running[0] = True
        try:
            judge_pass_counter[0] += 1
            processed = await judge.run_pass(min_age=min_age)
            return {"status": "ok", "processed": processed}
        finally:
            judge_running[0] = False

    def _accumulate_judge_tokens(n: int):
        if n <= 0:
            return
        judge_tokens[0] += n
        if judge_tokens[0] >= cfg.judge.interval_tokens:
            judge_tokens[0] = 0
            asyncio.create_task(run_judge_pass())

    pipeline.token_sink = _accumulate_judge_tokens

    # Track real token usage per model (from response `usage`) so the admin
    # context-usage panel works even when llama-server /metrics is unavailable.
    ctx_usage = {}

    def _record_usage(name, total_tokens):
        try:
            ctx_usage[name] = {"total_tokens": int(total_tokens), "at": time.time()}
        except (TypeError, ValueError):
            pass

    pipeline.usage_sink = lambda total: _record_usage("reasoning", total)
    judge.usage_sink = lambda total: _record_usage("judge", total)
    embed.usage_sink = lambda total: _record_usage("embed", total)

    # T/S (tokens per second) tracking — ring buffer of recent completions
    tps_ring = []  # list of {"tokens": int, "elapsed": float, "tps": float, "at": float}
    TPS_RING_MAX = 50

    def _record_tps(tokens: int, elapsed: float):
        if tokens <= 0 or elapsed <= 0:
            return
        tps = tokens / elapsed
        tps_ring.append({"tokens": tokens, "elapsed": round(elapsed, 3), "tps": round(tps, 1), "at": time.time()})
        if len(tps_ring) > TPS_RING_MAX:
            tps_ring.pop(0)

    pipeline.tps_sink = _record_tps
    judge.tps_sink = _record_tps
    embed.tps_sink = _record_tps

    from fastapi.responses import HTMLResponse
    import pathlib

    static_dir = pathlib.Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)

    @app.get("/health")
    async def health():
        return {"status": "ok", "store": str(store_path)}

    @app.get("/admin")
    async def admin_page():
        html = (static_dir / "admin.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/")
    async def root_page():
        html = (static_dir / "chat.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/chat")
    async def chat_page():
        html = (static_dir / "chat.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/admin/models")
    async def admin_models():
        import httpx
        targets = [
            ("reasoning", cfg.reasoning_endpoint),
            ("judge", cfg.judge_endpoint),
            ("embed", cfg.embed_endpoint),
        ]

        async def probe(name, base):
            base = (base or "").rstrip("/")
            info = {
                "name": name, "endpoint": base, "up": False,
                "model": None, "n_ctx": None,
                "kv_used_ratio": None, "used_tokens": None,
            }
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    try:
                        r = await client.get(base + "/props")
                        if r.status_code == 200:
                            info["up"] = True
                            d = r.json()
                            gs = d.get("default_generation_settings", {}) or {}
                            info["n_ctx"] = gs.get("n_ctx") or d.get("n_ctx")
                            mp = d.get("model_path") or d.get("model") or gs.get("model")
                            if mp:
                                info["model"] = str(mp).replace("\\", "/").split("/")[-1]
                    except Exception:
                        pass
                    try:
                        r = await client.get(base + "/metrics")
                        if r.status_code == 200:
                            info["up"] = True
                            for line in r.text.splitlines():
                                if line.startswith("llamacpp:kv_cache_usage_ratio"):
                                    try:
                                        info["kv_used_ratio"] = float(line.split()[-1])
                                    except ValueError:
                                        pass
                    except Exception:
                        pass
                    if not info["up"]:
                        try:
                            r = await client.get(base + "/health")
                            info["up"] = r.status_code == 200
                        except Exception:
                            pass
            except Exception:
                pass
            # Prefer live /metrics; otherwise show n/a (total_tokens is not
            # a reliable proxy for KV cache usage).
            if info["kv_used_ratio"] is None:
                if info["n_ctx"] is not None:
                    info["used_tokens"] = 0
                    info["kv_used_ratio"] = 0
                    info["source"] = "no-metrics"
            elif info["n_ctx"] is not None:
                info["used_tokens"] = int(info["n_ctx"] * info["kv_used_ratio"])
                info["source"] = "live"
            return info

        models = await asyncio.gather(*[probe(n, u) for n, u in targets])
        return {"models": list(models)}

    admin_router = build_admin_router(index, store, wal, run_judge_pass, tps_ring)
    app.include_router(admin_router)

    snapshot_task = None
    shutdown_event = asyncio.Event()

    @app.on_event("startup")
    async def startup():
        nonlocal snapshot_task

        snapshot = snapshot_path / "latest"
        if snapshot.exists() and not any(store.blocks_dir.iterdir()):
            store.restore(snapshot)
            index.restore(snapshot)

        # Shelve any 'hot' blocks left over from previous sessions so they
        # become recallable. The per-turn logic only shelves a turn when the
        # NEXT turn arrives, so the last turn of a session (and everything
        # from a crashed/killed run) would otherwise stay hot forever and
        # never surface in recall.
        await asyncio.to_thread(_shelve_leftover_hot, store, index, wal)

        async def snapshot_loop():
            interval = cfg.snapshot_interval_min * 60
            while not shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, _take_snapshot, store, index, snapshot_path
                        ),
                        timeout=300,
                    )
                except BaseException:
                    pass
                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(), timeout=interval
                    )
                except asyncio.TimeoutError:
                    pass

        snapshot_task = asyncio.create_task(snapshot_loop())

    @app.on_event("shutdown")
    async def shutdown():
        shutdown_event.set()
        if snapshot_task:
            snapshot_task.cancel()
        _take_snapshot(store, index, snapshot_path)
        wal.close()
        embed.close()
        index.close()

    def _derive_conversation(body: dict):
        import hashlib
        import json as _json
        if "conversation_id" in body:
            conv = str(body["conversation_id"])
        else:
            conv = None
            for m in body.get("messages", []):
                if m.get("role") == "user":
                    c = m.get("content", "")
                    if not isinstance(c, str):
                        c = _json.dumps(c, sort_keys=True)
                    conv = hashlib.sha256(c.encode()).hexdigest()[:16]
                    break
            if conv is None:
                conv = str(uuid.uuid4())
        n_user = sum(1 for m in body.get("messages", [])
                     if m.get("role") == "user")
        turn_index = max(0, n_user - 1)
        return conv, turn_index

    @app.api_route("/v1/chat/completions", methods=["POST"])
    async def chat_completions(request: Request):
        body = await request.json()
        stream = body.get("stream", False)

        conv_id, turn_index = _derive_conversation(body)

        user_message = pipeline.get_last_user_message(body)

        await pipeline.detect_and_apply_correction(
            user_message, conv_id, turn_index
        )

        await pipeline.shelve_previous_turn(conv_id, turn_index)

        result = await pipeline.process_turn(body, conv_id, turn_index)

        if turn_index > 0:
            await pipeline.apply_accepted_verification(conv_id, turn_index)

        new_tokens = 0
        if isinstance(result, dict) and "stream" not in result:
            content = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            new_tokens = len(content.split())
        elif isinstance(result, dict) and result.get("type") == "streaming":
            new_tokens = 0

        _accumulate_judge_tokens(new_tokens)

        if stream:
            stream_gen = result.get("stream")
            if stream_gen:
                return StreamingResponse(stream_gen, media_type="text/event-stream")
            return JSONResponse(content={"error": "stream not available"}, status_code=500)

        return JSONResponse(content=result)

    return app


def _detect_embed_dim(embed: EmbeddingClient, cfg: Config) -> int:
    """Probe the embedding server for its output dimension.

    Retries briefly in case the embed server is still warming up, then falls
    back to the configured embed_dim (default 768 for nomic-embed-text-v1.5).
    """
    fallback = getattr(cfg, "embed_dim", 768) or 768
    for attempt in range(15):
        try:
            vec = embed.embed("dimension probe")
            if vec:
                return len(vec)
        except Exception:
            time.sleep(1)
    return fallback


def _shelve_leftover_hot(store: BlockStore, index: VectorIndex, wal: WAL):
    """Promote leftover 'hot' blocks to 'shelved' at startup.

    Recall only searches shelved/truncated blocks, and the per-turn shelving
    only fires when a later turn arrives in the same conversation. So the last
    turn before shutdown -- and anything from an unclean stop -- must be shelved
    here, or it can never be recalled in a future session.
    """
    items, _ = index.list_meta(status="hot", limit=1_000_000)
    count = 0
    for m in items:
        bid = m.get("block_id")
        if not bid:
            continue
        index.update_status(bid, BlockStatus.shelved.value)
        block = store.get(bid)
        if block:
            block.status = BlockStatus.shelved
            store.put(block)
        count += 1
    if count:
        wal.write({
            "event": "startup_shelve",
            "count": count,
            "timestamp": time.time(),
        })


def _take_snapshot(store: BlockStore, index: VectorIndex, snapshot_path: Path):
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    try:
        store.snapshot(tmp)
        index.snapshot(tmp)
        latest = snapshot_path / "latest"
        if latest.exists():
            _rmtree_win32(latest)
        shutil.move(str(tmp), str(latest))
    finally:
        if tmp.exists():
            _rmtree_win32(tmp)


def _rmtree_win32(path: Path, retries: int = 5, delay: float = 0.3):
    """shutil.rmtree with retries on Windows PermissionError (locked files)."""
    import shutil, time
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


def main():
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = Config(config_path)

    app = create_app(config_path)
    host, port = cfg.listen.split(":")
    uvicorn.run(app, host=host, port=int(port), log_level="info")


if __name__ == "__main__":
    main()
