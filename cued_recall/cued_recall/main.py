import asyncio
import json
import os
import signal
import time
import uuid
from pathlib import Path

import uvicorn
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .chats import ChatStore
from .config import Config
from .embed import EmbeddingClient
from .index import VectorIndex
from .judge import Judge
from .models import BlockStatus
from .pipeline import Pipeline
from .router import build_admin_router
from .store import BlockStore
from .sysinfo import snapshot as system_snapshot
from .tagger import Tagger
from .wal import WAL
from fastapi.staticfiles import StaticFiles


def create_app(config_path: str = "config.yaml") -> FastAPI:
    # run.py exports the moment the whole stack was started; model loading
    # happens before this process exists, so falling back to "now" would
    # under-report uptime by however long the GGUFs took to load.
    try:
        launched_at = float(os.environ["CUED_RECALL_LAUNCHED_AT"])
    except (KeyError, ValueError):
        launched_at = time.time()

    cfg = Config(config_path)
    store_path = Path(cfg.store_path)
    snapshot_path = Path(cfg.snapshot_path)

    store_path.mkdir(parents=True, exist_ok=True)
    snapshot_path.mkdir(parents=True, exist_ok=True)

    wal = WAL(store_path / "wal.jsonl")
    wal.open()

    store = BlockStore(store_path)
    chats = ChatStore(store_path)
    chats.open()
    embed = EmbeddingClient(cfg.embed_endpoint)

    # The vector index dimension MUST match the embedding model's output
    # (e.g. nomic-embed-text-v1.5 = 768). A mismatch makes every upsert/query
    # raise and the whole chat path 500s. Detect it live from the embed server.
    embed_dim = _detect_embed_dim(embed, cfg)
    index = VectorIndex(store_path, dim=embed_dim)
    index.open()

    # run.py derives max_context_tokens from the ctx-size it launches with,
    # but the reasoning server may have been started separately with a
    # smaller window. Trust the live server over the config file.
    _clamp_context_budget(cfg)

    pipeline = Pipeline(cfg, store, index, embed, wal)
    judge = Judge(cfg, store, index, wal)
    tagger = Tagger(cfg, store, index, wal)
    pipeline.tagger = tagger

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

    def _record_usage(name, usage):
        try:
            # Use prompt_tokens as proxy for KV cache occupancy (the cache holds
            # input tokens, not generated ones). Fall back to total_tokens.
            pt = usage.get("prompt_tokens")
            if pt is None:
                total = usage.get("total_tokens")
                if total is None:
                    return
                pt = total
            ctx_usage[name] = {"total_tokens": int(pt), "at": time.time()}
        except (TypeError, ValueError):
            pass

    pipeline.usage_sink = lambda usage: _record_usage("reasoning", usage)
    judge.usage_sink = lambda usage: _record_usage("judge", usage)
    embed.usage_sink = lambda usage: _record_usage("embed", usage)
    tagger.usage_sink = lambda usage: _record_usage("tagger", usage)

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

    def _record_chat(conversation_id, user_text, assistant_text):
        # Never let transcript bookkeeping break a turn that already succeeded.
        try:
            source = "chat" if str(conversation_id).startswith("chat-") else "api"
            chats.record_turn(conversation_id, user_text, assistant_text, source)
        except Exception as e:
            wal.write({
                "event": "chat_record_error",
                "conversation_id": conversation_id,
                "error": f"{type(e).__name__}: {e}",
                "timestamp": time.time(),
            })

    pipeline.chat_sink = _record_chat

    from fastapi.responses import HTMLResponse
    import pathlib

    static_dir = pathlib.Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)

    @app.get("/health")
    async def health():
        return {"status": "ok", "store": str(store_path)}

    @app.get("/v1/models")
    async def list_models():
        # OpenAI-compatible clients (opencode, most others) call this during
        # setup to populate the model picker before ever hitting
        # /v1/chat/completions. Without it, adding this middleware as a
        # custom provider 404s immediately, before you can even chat.
        import httpx
        model_id = "cued-recall"
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(cfg.reasoning_endpoint.rstrip("/") + "/props")
                if r.status_code == 200:
                    d = r.json()
                    gs = d.get("default_generation_settings", {}) or {}
                    mp = d.get("model_path") or d.get("model") or gs.get("model")
                    if mp:
                        model_id = str(mp).replace("\\", "/").split("/")[-1]
        except Exception:
            pass
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "created": 0, "owned_by": "cued-recall"},
            ],
        }

    @app.post("/v1/fetch")
    async def fetch_url(body: dict):
        url = body.get("url", "")
        if not url:
            return JSONResponse(content={"error": "No url provided"}, status_code=400)
        try:
            # Reuse the pipeline fetch (SSRF guard + HTML->text applied there).
            content = await pipeline._fetch_url(url)
            return JSONResponse(content={"url": url, "content": content})
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

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
            # Prefer live /metrics; otherwise fall back to prompt_tokens
            # tracking from the last request (more reliable than total_tokens
            # since prompt_tokens directly maps to KV cache occupancy).
            if info["kv_used_ratio"] is None:
                tracked = ctx_usage.get(name)
                if tracked and info["n_ctx"]:
                    used = min(int(tracked["total_tokens"]), int(info["n_ctx"]))
                    info["used_tokens"] = used
                    info["kv_used_ratio"] = used / info["n_ctx"]
                    info["source"] = "last-request"
            elif info["n_ctx"] is not None:
                info["used_tokens"] = int(info["n_ctx"] * info["kv_used_ratio"])
                info["source"] = "live"
            return info

        models = await asyncio.gather(*[probe(n, u) for n, u in targets])
        return {"models": list(models)}

    @app.get("/chats")
    async def list_chats(limit: int = 100, offset: int = 0,
                         source: Optional[str] = None):
        items, total = await asyncio.to_thread(
            chats.list_conversations, limit, offset, source
        )
        return {"total": total, "items": items, "limit": limit, "offset": offset}

    @app.get("/chats/{conversation_id}")
    async def get_chat(conversation_id: str):
        messages = await asyncio.to_thread(chats.get_messages, conversation_id)
        if not messages:
            raise HTTPException(status_code=404, detail="conversation not found")
        return {"conversation_id": conversation_id, "messages": messages}

    @app.patch("/chats/{conversation_id}")
    async def rename_chat(conversation_id: str, body: dict):
        title = (body or {}).get("title", "")
        if not title:
            raise HTTPException(status_code=400, detail="title required")
        if not await asyncio.to_thread(chats.rename, conversation_id, title):
            raise HTTPException(status_code=404, detail="conversation not found")
        return {"status": "ok"}

    @app.delete("/chats/{conversation_id}")
    async def delete_chat(conversation_id: str):
        # Deletes the transcript only. Memory blocks derived from the
        # conversation survive -- forgetting the chat is not the same as
        # forgetting what was learned in it.
        if not await asyncio.to_thread(chats.delete, conversation_id):
            raise HTTPException(status_code=404, detail="conversation not found")
        return {"status": "ok", "blocks_kept": True}

    @app.get("/admin/system")
    async def admin_system():
        """GPU / CPU telemetry for the host and each server process.

        Kept separate from /admin/models so a probing failure degrades to one
        empty panel instead of taking the context-usage cards down with it.
        Both probes block (subprocess + psutil), hence the thread.
        """
        port_map = {
            "reasoning": _endpoint_port(cfg.reasoning_endpoint),
            "judge": _endpoint_port(cfg.judge_endpoint),
            "embed": _endpoint_port(cfg.embed_endpoint),
            "middleware": _endpoint_port(cfg.listen),
        }
        port_map = {k: v for k, v in port_map.items() if v}
        try:
            data = await asyncio.to_thread(system_snapshot, port_map)
        except Exception as e:
            # Never 500: the admin page polls this every 5 s.
            return {"gpus": [], "processes": {}, "host": {}, "error": str(e)}
        data["launched_at"] = launched_at
        data["uptime_s"] = time.time() - launched_at
        return data

    @app.post("/admin/kv/clear")
    async def clear_kv_cache():
        """Erase the llama-server slot KV caches.

        This drops cached prompt prefixes; it does not return VRAM, since the
        KV buffer is allocated once at model load. Use it to force a clean
        re-prefill when a slot is holding stale or unwanted context.
        """
        import httpx

        targets = [
            ("reasoning", cfg.reasoning_endpoint),
            ("judge", cfg.judge_endpoint),
        ]
        results = {}
        for name, base in targets:
            base = (base or "").rstrip("/")
            entry = {"cleared": 0, "slots": 0, "error": None}
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    r = await client.get(base + "/slots")
                    if r.status_code != 200:
                        entry["error"] = f"/slots returned {r.status_code}"
                        results[name] = entry
                        continue
                    slots = r.json()
                    if not isinstance(slots, list):
                        entry["error"] = "unexpected /slots payload"
                        results[name] = entry
                        continue
                    entry["slots"] = len(slots)
                    for s in slots:
                        sid = s.get("id")
                        if sid is None:
                            continue
                        er = await client.post(f"{base}/slots/{sid}?action=erase")
                        if er.status_code == 200:
                            entry["cleared"] += 1
                        elif er.status_code == 501:
                            # Server started without --slot-save-path.
                            entry["error"] = (
                                "server not started with --slot-save-path "
                                "(restart via run.py to enable)"
                            )
                            break
                        else:
                            entry["error"] = f"slot {sid}: HTTP {er.status_code}"
            except Exception as e:
                entry["error"] = str(e)
            results[name] = entry

        total = sum(v["cleared"] for v in results.values())
        wal.write({
            "event": "admin_kv_clear",
            "cleared_slots": total,
            "detail": {k: v["cleared"] for k, v in results.items()},
            "timestamp": time.time(),
        })
        return {"status": "ok", "cleared_slots": total, "servers": results}

    @app.get("/admin/wedge")
    async def wedge_status():
        """Is a llama server's inference queue blocked?

        A lost slot leaves the process alive and the HTTP threads answering,
        while everything that goes through the inference queue blocks forever.
        /health and /props never touch that queue, so "health fast, slots
        hanging" is the signature. Timeouts are short because this is polled.
        """
        import httpx

        targets = [
            ("reasoning", cfg.reasoning_endpoint),
            ("judge", cfg.judge_endpoint),
        ]
        out = {}
        for name, base in targets:
            base = (base or "").rstrip("/")
            entry = {"health": False, "queue": "unknown", "wedged": False}
            try:
                async with httpx.AsyncClient() as client:
                    h = await client.get(base + "/health", timeout=2)
                    entry["health"] = h.status_code == 200
                    if entry["health"]:
                        try:
                            # Any reply means the queue is moving; a 501 from a
                            # server without slot support still counts.
                            await client.get(base + "/slots", timeout=4)
                            entry["queue"] = "ok"
                        except httpx.HTTPError:
                            entry["queue"] = "blocked"
                            entry["wedged"] = True
            except httpx.HTTPError as e:
                entry["queue"] = "unreachable"
                entry["error"] = str(e) or type(e).__name__
            out[name] = entry
        return {"servers": out,
                "any_wedged": any(v["wedged"] for v in out.values()),
                "restart_supported": bool(os.environ.get("CUED_RECALL_RESTART_FILE"))}

    @app.post("/admin/server/restart")
    async def request_server_restart(name: str = "reasoning"):
        """Ask the launcher to restart a llama server.

        The restart is performed by run.py, not here: it owns the process
        handles, and a restart from this side would orphan a multi-GB process
        when the launcher exits. Clearing the KV cache is not an alternative --
        /admin/kv/clear reaches the server through /slots, which is exactly
        what a wedge blocks.
        """
        path = os.environ.get("CUED_RECALL_RESTART_FILE")
        if not path:
            raise HTTPException(
                status_code=503,
                detail="No launcher supervising this middleware "
                       "(start the stack with run.py to enable restarts).",
            )
        if name not in ("reasoning", "judge", "embed"):
            raise HTTPException(status_code=400, detail=f"unknown server '{name}'")
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(name, encoding="utf-8")
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"could not signal launcher: {e}")
        wal.write({"event": "admin_server_restart", "server": name,
                   "timestamp": time.time()})
        return {"status": "requested", "server": name,
                "note": "The launcher picks this up within ~15s; the model "
                        "then reloads, which takes up to a minute."}

    admin_router = build_admin_router(index, store, wal, run_judge_pass, tps_ring, embed)
    app.include_router(admin_router)

    snapshot_task = None
    hot_sweep_task = None
    shutdown_event = asyncio.Event()

    @app.on_event("startup")
    async def startup():
        nonlocal snapshot_task, hot_sweep_task

        snapshot = snapshot_path / "latest"
        if snapshot.exists() and not any(store.blocks_dir.iterdir()):
            store.restore(snapshot)
            index.restore(snapshot)
            chats.restore(snapshot)

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
                            None, _take_snapshot, store, index, snapshot_path,
                            chats
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

        async def hot_sweep_loop():
            # Poll faster than the timeout itself so an abandoned
            # conversation becomes recallable close to hot_shelve_timeout_s
            # after its last message, not after some much longer cycle.
            check_interval = max(2, min(10, cfg.hot_shelve_timeout_s // 2 or 1))
            while not shutdown_event.is_set():
                try:
                    n = await pipeline.shelve_stale_hot_blocks(cfg.hot_shelve_timeout_s)
                    if n:
                        wal.write({
                            "event": "idle_shelve",
                            "count": n,
                            "timestamp": time.time(),
                        })
                except BaseException:
                    pass
                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(), timeout=check_interval
                    )
                except asyncio.TimeoutError:
                    pass

        hot_sweep_task = asyncio.create_task(hot_sweep_loop())

    @app.on_event("shutdown")
    async def shutdown():
        shutdown_event.set()
        if snapshot_task:
            snapshot_task.cancel()
        if hot_sweep_task:
            hot_sweep_task.cancel()
        _take_snapshot(store, index, snapshot_path, chats)
        wal.close()
        embed.close()
        index.close()
        chats.close()

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


def _clamp_context_budget(cfg: Config):
    """Lower max_context_tokens if it exceeds the reasoning server's n_ctx.

    The prompt and the generated reply share one window, so the budget must
    leave room for the reply as well -- including a reasoning model's think
    trace, which precedes any visible output. Best-effort: if the server can't
    be probed, keep whatever the config says.
    """
    import httpx

    RESERVE_OUTPUT = cfg.context_reserve_tokens
    try:
        r = httpx.get(cfg.reasoning_endpoint.rstrip("/") + "/props", timeout=5)
        if r.status_code != 200:
            return
        d = r.json()
        gs = d.get("default_generation_settings", {}) or {}
        n_ctx = gs.get("n_ctx") or d.get("n_ctx")
    except Exception:
        return
    if not n_ctx:
        return
    ceiling = max(2048, int(n_ctx) - RESERVE_OUTPUT)
    if cfg.max_context_tokens > ceiling:
        print(
            f"[WARN] max_context_tokens {cfg.max_context_tokens} exceeds the "
            f"reasoning server's context ({n_ctx}); clamping to {ceiling}"
        )
        cfg.max_context_tokens = ceiling


def _endpoint_port(value: str) -> int:
    """Port out of an endpoint string.

    Handles both the endpoint form ("http://127.0.0.1:8080") and the listen
    form ("127.0.0.1:8000"). Returns 0 when there's no parseable port, which
    the caller filters out.
    """
    try:
        tail = str(value or "").rsplit(":", 1)[-1]
        return int(tail.split("/")[0])
    except (ValueError, AttributeError):
        return 0


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


def _take_snapshot(store: BlockStore, index: VectorIndex, snapshot_path: Path,
                   chats: 'ChatStore' = None):
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    try:
        store.snapshot(tmp)
        index.snapshot(tmp)
        if chats is not None:
            chats.snapshot(tmp)
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
