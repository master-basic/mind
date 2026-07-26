"""Host GPU/CPU telemetry for the admin panel.

Deliberately standalone: nothing here imports from the pipeline, index, or
store, so a probing failure can never reach the chat path. Every function in
this module is total -- it returns empty/None on failure rather than raising,
because the admin page polls it every 5 seconds and one bad poll must not take
the page down.
"""

import os
import subprocess
import threading
import time
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - surfaced to the UI, not fatal
    psutil = None


# nvidia-smi is fast (~70 ms measured) but the admin page auto-refreshes every
# 5 s and a user mashing Refresh can stack calls on top of that. Serve repeat
# polls inside this window from cache.
_CACHE_TTL_S = 2.0
_cache: dict = {"at": 0.0, "value": None}
_cache_lock = threading.Lock()

# psutil measures cpu_percent as the delta since the LAST call on the same
# Process object. A freshly constructed Process always reports 0.0, so the
# objects have to survive between polls. Keyed by pid.
_proc_cache: Dict[int, "psutil.Process"] = {}

# Windows only: without this, every nvidia-smi call pops a console window --
# once per refresh cycle, forever.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

_GPU_FIELDS = [
    "index", "name", "memory.total", "memory.used",
    "utilization.gpu", "utilization.memory", "temperature.gpu",
]


def _to_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s or s.startswith("["):  # nvidia-smi writes "[N/A]" for unsupported
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def probe_gpus() -> List[dict]:
    """Device-level GPU stats via nvidia-smi. Empty list if unavailable.

    Note this is per-DEVICE, not per-process. Per-process VRAM
    (--query-compute-apps=used_memory) reports [N/A] for GeForce cards under
    the Windows WDDM driver model, which is why the per-model figures in
    probe_processes() are estimates rather than measurements.
    """
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(_GPU_FIELDS),
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        # FileNotFoundError (no nvidia-smi on PATH), timeout, or anything else.
        return []
    if r.returncode != 0:
        return []

    gpus = []
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_GPU_FIELDS):
            continue
        total = _to_int(parts[2])
        used = _to_int(parts[3])
        gpus.append({
            "index": _to_int(parts[0]),
            "name": parts[1],
            "memory_total_mb": total,
            "memory_used_mb": used,
            "memory_used_ratio": (used / total) if (total and used is not None) else None,
            "util_gpu_pct": _to_int(parts[4]),
            "util_mem_pct": _to_int(parts[5]),
            "temperature_c": _to_int(parts[6]),
        })
    return gpus


def _pids_by_port(ports: List[int]) -> Dict[int, int]:
    """Map listening TCP port -> owning pid."""
    wanted = set(ports)
    found: Dict[int, int] = {}
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.status != psutil.CONN_LISTEN or not c.laddr or not c.pid:
                continue
            port = c.laddr.port
            if port in wanted and port not in found:
                found[port] = c.pid
    except (psutil.AccessDenied, psutil.Error, OSError):
        # On some systems enumerating connections needs elevation. Callers
        # degrade to "unknown pid" rather than failing.
        pass
    return found


def _is_llama_server(proc_name: str, cmdline: List[str]) -> bool:
    if "llama" in (proc_name or "").lower():
        return True
    return bool(cmdline) and "llama" in os.path.basename(cmdline[0]).lower()


def _parse_llama_args(cmdline: List[str]) -> dict:
    """Pull placement facts out of a live llama-server command line.

    Read from the running process rather than from config.yaml or run.py's
    defaults: config.yaml carries no `servers:` block, and mirroring run.py's
    SERVER_DEFAULTS here would silently drift the moment either changes. The
    cmdline is what the process is actually doing.

    Only ever call this for a process that passed _is_llama_server: `-m` and
    `-c` collide with Python's own module and command flags, so the middleware
    (`python -m ...`) would otherwise report its module name as a model path.
    """
    out = {"n_gpu_layers": None, "n_ctx": None, "no_kv_offload": False,
           "model_path": None}
    for i, a in enumerate(cmdline):
        nxt = cmdline[i + 1] if i + 1 < len(cmdline) else None
        if a in ("--n-gpu-layers", "-ngl") and nxt is not None:
            out["n_gpu_layers"] = _to_int(nxt)
        elif a in ("--ctx-size", "-c") and nxt is not None:
            out["n_ctx"] = _to_int(nxt)
        elif a == "--no-kv-offload":
            out["no_kv_offload"] = True
        elif a in ("-m", "--model") and nxt is not None:
            out["model_path"] = nxt
    return out


def _file_size(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def probe_processes(port_map: Dict[str, int]) -> Dict[str, dict]:
    """Per-server CPU/RAM/placement, keyed by the names in port_map."""
    if psutil is None:
        return {}

    pids = _pids_by_port(list(port_map.values()))
    result: Dict[str, dict] = {}

    for name, port in port_map.items():
        entry = {
            "port": port, "pid": None, "up": False,
            "rss_bytes": None, "cpu_percent": None, "num_threads": None,
            "uptime_s": None, "device": None, "n_gpu_layers": None,
            "n_ctx": None, "no_kv_offload": None,
            "model_path": None, "weights_bytes": None, "vram_est_bytes": None,
        }
        pid = pids.get(port)
        if pid is None:
            result[name] = entry
            continue

        entry["pid"] = pid
        try:
            proc = _proc_cache.get(pid)
            if proc is None or not proc.is_running():
                proc = psutil.Process(pid)
                _proc_cache[pid] = proc
                # Prime the counter; this first reading is always 0.0.
                proc.cpu_percent(interval=None)

            entry["up"] = True
            entry["rss_bytes"] = proc.memory_info().rss
            entry["cpu_percent"] = round(proc.cpu_percent(interval=None), 1)
            entry["num_threads"] = proc.num_threads()
            entry["uptime_s"] = round(time.time() - proc.create_time(), 1)

            cmdline = proc.cmdline()
            if not _is_llama_server(proc.name(), cmdline):
                result[name] = entry
                continue

            args = _parse_llama_args(cmdline)
            entry.update(args)
            ngl = args["n_gpu_layers"]
            if ngl is not None:
                entry["device"] = "gpu" if ngl > 0 else "cpu"
            weights = _file_size(args["model_path"])
            entry["weights_bytes"] = weights
            # Estimate, not a measurement: the weights are the only VRAM
            # component we can size from outside the process. KV cache bytes
            # would need per-architecture constants that neither /props nor
            # /metrics exposes, so they are deliberately excluded -- the live
            # kv_used_ratio in /admin/models covers KV occupancy honestly.
            if entry["device"] == "gpu" and weights:
                entry["vram_est_bytes"] = weights
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error, OSError):
            # Leave whatever was collected; the card renders with blanks.
            _proc_cache.pop(pid, None)

        result[name] = entry

    # Drop cache entries for processes that have exited so a recycled pid
    # can't inherit a stale cpu_percent baseline.
    live = {e["pid"] for e in result.values() if e["pid"]}
    for dead in [p for p in _proc_cache if p not in live]:
        _proc_cache.pop(dead, None)

    return result


def probe_host() -> dict:
    if psutil is None:
        return {}
    try:
        vm = psutil.virtual_memory()
        return {
            "ram_total_bytes": vm.total,
            "ram_used_bytes": vm.total - vm.available,
            "ram_used_ratio": (vm.total - vm.available) / vm.total if vm.total else None,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count(logical=True),
        }
    except (psutil.Error, OSError):
        return {}


def snapshot(port_map: Dict[str, int]) -> dict:
    """Full telemetry snapshot, cached for _CACHE_TTL_S. Blocking; call in a thread."""
    with _cache_lock:
        now = time.time()
        if _cache["value"] is not None and now - _cache["at"] < _CACHE_TTL_S:
            return _cache["value"]

        value = {
            "gpus": probe_gpus(),
            "processes": probe_processes(port_map),
            "host": probe_host(),
            "psutil_available": psutil is not None,
            "at": now,
        }
        _cache["value"] = value
        _cache["at"] = now
        return value
