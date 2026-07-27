#!/usr/bin/env python3
"""
Cued Recall — one-command launcher.

Downloads models (or copies from cache), mounts 64GB tmpfs, starts 3 llama-server
instances + middleware. Ctrl+C to stop everything.

Usage:
  python run.py                                   # auto everything
  python run.py --download-to ./models            # download models to ./models first
  python run.py --models-cache ./models           # copy existing models to tmpfs
  python run.py --no-tmpfs --storage ./data       # skip tmpfs, use ./data
  python run.py --reasoning-model ./my-model.gguf # use specific model file
  python run.py --dry-run                         # show what would happen
  python run.py --help                            # full options
"""

import os
import sys
import time
import signal
import subprocess
import shutil
import struct
import json
import re
import platform
import argparse
import threading
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
CONFIG_PATH = ROOT / "cued_recall" / "config.yaml"
LOG_DIR = ROOT / "logs"
DEFAULT_TMPFS = Path("/mnt/ramdisk/cued_recall")
TMPFS_SIZE = "64G"

# Selectable reasoning models (menu in run.bat / --reasoning-choice N).
# moe=True -> experts are kept in system RAM (--cpu-moe) so a 17-20 GB MoE
# runs on a 12 GB GPU with only the ~3B active path + KV in VRAM.
REASONING_CATALOG = [
    {
        "num": 1,
        "label": "Qwen3.5-9B (default)                  6.6 GB  dense, full GPU",
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "file": "Qwen3.5-9B-Q5_K_M.gguf",
        "moe": False,
    },
    {
        "num": 2,
        "label": "Qwen3.5-9B ultra-uncensored-heretic   6.5 GB  dense, full GPU",
        "repo": "mradermacher/Qwen3.5-9B-ultra-uncensored-heretic-v2-i1-GGUF",
        "file": "Qwen3.5-9B-ultra-uncensored-heretic-v2.i1-Q5_K_M.gguf",
        "moe": False,
    },
    {
        "num": 3,
        "label": "Qwen3.5-9B abliterated                5.6 GB  dense, full GPU",
        "repo": "Al3xG/Qwen3.5-9B-abliterated-Q4_K_M-GGUF",
        "file": "Qwen3.5-9B-abliterated-Q4_K_M.gguf",
        "moe": False,
    },
    {
        "num": 4,
        "label": "Qwen3.5-35B-A3B Abliterated          19.9 GB  MoE, experts in RAM",
        "repo": "Carlosian/Qwen3.5-35B-A3B-Abliterated-GGUF",
        "file": "Qwen3.5-35B-A3B-Abliterated.Q4_K_S.gguf",
        "moe": True,
    },
    {
        "num": 5,
        "label": "Qwen3.6-35B-A3B unc-heretic MXFP4    20.3 GB  MoE, experts in RAM",
        "repo": "noctrex/Qwen3.6-35B-A3B-uncensored-heretic-MXFP4_MOE-GGUF",
        "file": "Qwen3.6-35B-A3B-uncensored-heretic-MXFP4_MOE.gguf",
        "moe": True,
    },
    {
        "num": 6,
        "label": "Hermes3.6-35B-A3B Unc Genesis V5     17.4 GB  MoE, experts in RAM",
        "repo": "LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V5-GGUF",
        "file": "Hermes3.6-35B-A3B-Uncensored-Genesis-V5-APEX-Compact.gguf",
        "moe": True,
    },
]

MODEL_MANIFEST = [
    {
        "name": "reasoning",
        "repo": "Qwen/Qwen3-8B-GGUF",
        "file": "Qwen3-8B-Q4_K_M.gguf",
    },
    {
        "name": "judge",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "save_as": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    },
    {
        "name": "embed",
        "repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "file": "nomic-embed-text-v1.5.Q8_0.gguf",
        "save_as": "nomic-embed-text-v1.5-Q8_0.gguf",
    },
]

# --metrics exposes /metrics (incl. kv_cache_usage_ratio) for the admin panel.
# --jinja applies the model's tool-calling chat template; without it llama.cpp
# ignores the `tools` parameter and never emits tool_calls (web_search/web_fetch
# would be dead).
# Placement (tuned for a 12 GB GPU):
#   reasoning: weights + 32K KV cache on GPU (no --no-kv-offload) -> fastest.
#   judge:     fully on CPU (-ngl 0) to free VRAM for the reasoning KV.
#   embed:     weights on GPU, KV/context in RAM (--no-kv-offload).
# -np 1 is load-bearing, not a tuning knob. This build defaults to 4 slots with
# a shared KV cache -- the server logs "n_slots = 4, n_ctx_slot = 61440,
# kv_unified = 'true'". Every slot advertises the whole window, but they draw
# from one pool and each retains its cached prefix, so four conversations
# compete for the same 61,440 tokens and the real ceiling for a new request is
# whatever the others have left. It is not a fixed division: a 16,920-token
# prompt succeeded on an empty cache, while 16,001 failed later once other
# slots held context. That dynamic ceiling is invisible from /props and /slots,
# which both report the undivided figure, and it drops as the server runs --
# which is what put the admin context bar in the red and made a small direct
# query succeed while every substantial middleware turn was refused.
#
# One slot takes the whole window deterministically: measured after the change,
# a 60,000-token prompt succeeds where 16,001 had failed. This is a single-user
# stack, the VRAM sizing already assumes one conversation, and it costs nothing.
SERVER_DEFAULTS = {
    "reasoning": {"port": 8080, "extra": ["--ctx-size", "32768", "--n-gpu-layers", "99", "--metrics", "--jinja", "-np", "1"]},
    "judge":     {"port": 8081, "extra": ["--ctx-size", "8192", "--n-gpu-layers", "0", "--metrics", "-np", "1"]},
    # Embeddings need the whole sequence in one micro-batch; the default
    # --ubatch-size (512) makes any input over ~512 tokens 500. Match batch
    # sizes to the context so larger inputs embed instead of erroring.
    "embed":     {"port": 8082, "extra": ["--embedding", "--ctx-size", "8192",
                                          "--batch-size", "8192", "--ubatch-size", "8192",
                                          "--n-gpu-layers", "0", "--no-kv-offload", "--metrics"]},
}


def info(msg):    print(f"[INFO] {msg}")
def warn(msg):    print(f"[WARN] {msg}")
def die(msg):     print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Cued Recall Memory Middleware — zero-config launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    g = p.add_argument_group("Model sources")
    g.add_argument("--download-to", metavar="DIR",
                   help="Download models to this directory first, then copy to tmpfs")
    g.add_argument("--models-cache", metavar="DIR",
                   help="Path to pre-downloaded models (copied to tmpfs instead of re-downloading)")
    g.add_argument("--no-download", action="store_true",
                   help="Skip model download; fail if models not found in tmpfs or --models-cache")

    g = p.add_argument_group("Storage")
    g.add_argument("--no-tmpfs", action="store_true",
                   help="Use local directories instead of tmpfs (always true on Windows)")
    g.add_argument("--storage", metavar="DIR", default=None,
                   help="Root storage path for models+blocks: ramdisk, NVMe, or any "
                        "directory. Sticky -- once used, future runs without --storage "
                        "keep reusing it instead of silently relocating. Default on a "
                        "genuinely fresh config: /mnt/ramdisk/cued_recall, or ./data on Windows")
    g.add_argument("--snapshot", metavar="DIR", default=None,
                   help="Snapshot backup location, independent of --storage. Sticky like "
                        "--storage. Default on a fresh config: ./snapshots next to run.py")

    g = p.add_argument_group("Model overrides")
    g.add_argument("--reasoning-model", metavar="PATH", help="Path to reasoning model GGUF")
    g.add_argument("--judge-model",     metavar="PATH", help="Path to judge model GGUF")
    g.add_argument("--embed-model",     metavar="PATH", help="Path to embedding model GGUF")
    g.add_argument("--reasoning-choice", type=int, metavar="N",
                   help="Pick reasoning model N from the catalog without showing the menu")
    g.add_argument("--model-menu", action="store_true",
                   help="Force the reasoning model selection menu even if a choice is saved")
    g.add_argument("--reasoning-cpu-moe", action="store_true",
                   help="Force MoE expert offload for the reasoning server (auto-detected for A3B/MoE models)")
    g.add_argument("--reasoning-n-cpu-moe", metavar="N|auto", default="auto",
                   help="How many layers keep their experts in system RAM. 'auto' "
                        "(default) spends the VRAM left over after the KV cache on "
                        "expert layers; pass a number to pin it, or the model's "
                        "layer count to keep every expert off the GPU")

    g = p.add_argument_group("Port overrides")
    g.add_argument("--reasoning-ctx", metavar="N|auto", default="auto",
                   help="Reasoning context size. 'auto' (default) sizes it from "
                        "free VRAM and the model's KV cost; pass a number to pin it")
    g.add_argument("--reasoning-port", type=int, default=8080, help="Reasoning model port (default: 8080)")
    g.add_argument("--judge-port",     type=int, default=8081, help="Judge model port (default: 8081)")
    g.add_argument("--embed-port",     type=int, default=8082, help="Embedding model port (default: 8082)")
    g.add_argument("--middleware-port", type=int, default=8000, help="Middleware port (default: 8000)")

    g = p.add_argument_group("Server selection")
    g.add_argument("--skip-reasoning", action="store_true", help="Do not start reasoning server")
    g.add_argument("--skip-judge",     action="store_true", help="Do not start judge server")
    g.add_argument("--skip-embed",     action="store_true", help="Do not start embedding server")

    g = p.add_argument_group("Other")
    g.add_argument("--dry-run", action="store_true", help="Print planned actions without executing")
    g.add_argument("--llama-bin", metavar="PATH", help="Path to llama-server executable")

    return p.parse_args()


def find_llama_server(args):
    names = ["llama-server.exe", "llama-server"]
    if args.llama_bin:
        p = Path(args.llama_bin)
        if p.is_file():
            return str(p.resolve())
        if p.is_dir():
            # Accept a folder: look in it and the usual build subdirectories
            subdirs = [
                p,
                p / "build" / "bin" / "Release",
                p / "build" / "bin",
                p / "bin",
            ]
            for d in subdirs:
                for name in names:
                    cand = d / name
                    if cand.is_file():
                        return str(cand.resolve())
            die(f"No llama-server executable found under directory: {p}")
        die(f"Specified --llama-bin not found: {args.llama_bin}")
    for name in names:
        which = shutil.which(name)
        if which:
            return which
    candidates = [
        ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
        ROOT / "llama.cpp" / "build" / "bin" / "llama-server.exe",
        ROOT / "llama" / "llama-server.exe",
        Path("C:/llama.cpp/build/bin/Release/llama-server.exe"),
        Path("C:/llama/build/bin/Release/llama-server.exe"),
        ROOT / "llama.cpp" / "build" / "bin" / "llama-server",
        Path("/usr/local/bin/llama-server"),
        Path("/usr/bin/llama-server"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def ensure_hf_hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        info("Installing huggingface-hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub", "-q"])
        import huggingface_hub
        return huggingface_hub


SETTINGS_FILE = ROOT / "run_settings.txt"


def is_moe_model(filename: str) -> bool:
    low = (filename or "").lower()
    return "a3b" in low or "moe" in low


def save_settings(**pairs):
    """Merge KEY=VALUE lines into run_settings.txt, keeping everything else.

    The file is shared with run.bat, which reads every key it finds into
    SAVED_<KEY>. Keys not named here are left untouched, so the per-model
    records below survive a launch that used a different model.

    ASCII-only on the way out: run.bat reads with `for /f`, in the console
    codepage rather than UTF-8.
    """
    lines = []
    if SETTINGS_FILE.exists():
        try:
            raw = SETTINGS_FILE.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = SETTINGS_FILE.read_text(encoding="cp1252")
        prefixes = tuple(f"{k}=" for k in pairs)
        lines = [ln for ln in raw.splitlines()
                 if ln.strip() and not ln.startswith(prefixes)]
    for key, value in pairs.items():
        lines.append(f"{key}={value}")
    SETTINGS_FILE.write_text("\n".join(lines) + "\n",
                             encoding="ascii", errors="replace")


def save_reasoning_choice(num: int):
    """Persist REASONING_CHOICE=N so the next launch skips the menu."""
    save_settings(REASONING_CHOICE=num)


def remember_launch(args, models, reasoning_ctx):
    """Record what this launch resolved, for run.bat to show next time.

    run.bat asks "reuse these settings?" before python starts, so all it has to
    go on is run_settings.txt. It stored REASONING_CHOICE=1 and nothing else,
    and only this file knows that 1 means Qwen3.5-9B-Q5_K_M.gguf -- hence three
    blank lines where the model names should be.

    Context size is keyed per model, not stored once. Autosizing gives each
    model its own window (a 9B dense and a 35B MoE do not land in the same
    place), so a single figure would go stale the moment the model changed.
    """
    def name_of(key):
        path = models.get(key)
        return Path(path).name if path else ""

    pairs = {"JUDGE_NAME": name_of("judge"), "EMBED_NAME": name_of("embed")}
    choice = getattr(args, "reasoning_choice", None)
    if choice and not args.reasoning_model:
        pairs[f"MODEL_{choice}_NAME"] = name_of("reasoning")
        pairs[f"MODEL_{choice}_CTX"] = reasoning_ctx
    else:
        # An explicit --reasoning-model has no catalog number to key on.
        pairs["REASONING_NAME"] = name_of("reasoning")
        pairs["REASONING_CTX"] = reasoning_ctx

    try:
        save_settings(**pairs)
    except OSError as e:
        # A launch that otherwise worked must not fail over a convenience file.
        warn(f"Could not record launch details in {SETTINGS_FILE.name}: {e}")


def choose_reasoning_model(args):
    """Return the chosen catalog entry, or None to keep default behavior.

    Order: explicit --reasoning-model path wins (no menu); --reasoning-choice N
    picks silently; otherwise show a numbered menu, remember the answer in
    run_settings.txt, and download later via the normal resolve flow.
    """
    if args.reasoning_model and not args.model_menu:
        return None

    by_num = {e["num"]: e for e in REASONING_CATALOG}

    if args.reasoning_choice and not args.model_menu:
        entry = by_num.get(args.reasoning_choice)
        if entry:
            info(f"Reasoning model (saved choice {entry['num']}): {entry['file']}")
            return entry
        warn(f"--reasoning-choice {args.reasoning_choice} is not in the catalog; showing menu")

    cache_dir = Path(args.models_cache) if args.models_cache else None
    print()
    print("Which reasoning model do you want to use?")
    for e in REASONING_CATALOG:
        have = cache_dir and (cache_dir / e["file"]).exists()
        status = "[downloaded]" if have else "[will download]"
        print(f"  {e['num']}. {e['label']}  {status}")
    print()
    while True:
        try:
            raw = input(f"Enter number [1-{len(REASONING_CATALOG)}] (default 1): ").strip()
        except EOFError:
            raw = ""
        if not raw:
            raw = "1"
        if raw.isdigit() and int(raw) in by_num:
            entry = by_num[int(raw)]
            break
        print("Invalid choice, try again.")
    save_reasoning_choice(entry["num"])
    info(f"Selected: {entry['file']} (remembered in run_settings.txt)")
    return entry


def _load_configured_paths() -> dict:
    """Read store_path/snapshot_path already in config.yaml, if any.

    Used to make storage sticky: a value already sitting in config.yaml
    reflects a deliberate past choice (a ramdisk, an NVMe folder, wherever),
    and should survive a `run.py` invocation that doesn't explicitly ask to
    change it.
    """
    if not CONFIG_PATH.exists():
        return {}
    import yaml
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return {
        "store_path": config.get("store_path"),
        "snapshot_path": config.get("snapshot_path"),
    }


def _path_usable_here(path_str: str) -> bool:
    """Reject a configured path that belongs to a different OS's conventions.

    A repo cloned onto Windows may still carry a Linux-style store_path (or
    vice versa) from whoever last ran it elsewhere. Treating that as "sticky"
    would resolve to nonsense, so fall back to the platform default instead.
    """
    if not path_str:
        return False
    is_windows = platform.system() == "Windows"
    drive, _ = os.path.splitdrive(path_str)
    if is_windows:
        return bool(drive) or path_str.startswith("\\\\")
    return path_str.startswith("/")


def setup_storage(args):
    """Determine and prepare the storage root (tmpfs or local).

    Ramdisk, NVMe, or any other directory all work the same way via
    --storage. Precedence: explicit --storage this run > whatever is already
    configured (sticky, survives re-runs and repo moves) > platform default
    (first-run-only fallback).
    """
    is_windows = platform.system() == "Windows"
    configured = _load_configured_paths()
    prev_store = configured.get("store_path")
    prev_root = Path(prev_store).parent if prev_store else None

    if args.storage:
        storage_root = Path(args.storage)
        if prev_root and prev_root.resolve() != storage_root.resolve():
            warn(f"Storage root changing: {prev_root} -> {storage_root}")
            warn("Memory at the old location is NOT automatically migrated or deleted.")
    elif prev_store and _path_usable_here(prev_store):
        storage_root = prev_root
        info(f"No --storage given; reusing configured storage root: {storage_root}")
    else:
        if prev_store:
            info(f"Configured store_path ({prev_store}) doesn't match this OS; using default")
        storage_root = (ROOT / "data") if (is_windows or args.no_tmpfs) else DEFAULT_TMPFS

    if args.dry_run:
        info(f"Storage root: {storage_root}" + (" (tmpfs)" if not is_windows and not args.no_tmpfs else ""))
        return storage_root

    if is_windows or args.no_tmpfs:
        storage_root.mkdir(parents=True, exist_ok=True)
        info(f"Using local storage: {storage_root}")
    else:
        try:
            storage_root.mkdir(parents=True, exist_ok=True)
            ret = subprocess.run(
                ["mount", "-t", "tmpfs", "-o", f"size={TMPFS_SIZE}", "tmpfs", str(storage_root)],
                capture_output=True, text=True, timeout=10,
            )
            if ret.returncode != 0:
                warn(f"tmpfs mount failed (need root?): {ret.stderr.strip()}")
                warn("Falling back to local directory")
            else:
                info(f"Mounted {TMPFS_SIZE} tmpfs at {storage_root}")
        except Exception as e:
            warn(f"tmpfs setup failed: {e}, using local directory")

    (storage_root / "models").mkdir(parents=True, exist_ok=True)
    (storage_root / "store").mkdir(parents=True, exist_ok=True)
    return storage_root


def resolve_snapshot_path(args):
    """Same sticky precedence as setup_storage, but independent of --storage.

    This is the piece that was missing: snapshot_path used to always reset to
    ROOT/"snapshots" (wherever the repo folder currently is), with no flag to
    control it and no memory of a previous choice -- so moving or re-cloning
    the repo silently orphaned old snapshots.
    """
    configured = _load_configured_paths()
    prev_snapshot = configured.get("snapshot_path")

    if args.snapshot:
        snapshot_path = Path(args.snapshot)
        if prev_snapshot and Path(prev_snapshot).resolve() != snapshot_path.resolve():
            warn(f"Snapshot path changing: {prev_snapshot} -> {snapshot_path}")
            warn("Snapshots at the old location are NOT automatically migrated.")
    elif prev_snapshot and _path_usable_here(prev_snapshot):
        snapshot_path = Path(prev_snapshot)
    else:
        if prev_snapshot:
            info(f"Configured snapshot_path ({prev_snapshot}) doesn't match this OS; using default")
        snapshot_path = ROOT / "snapshots"

    if not args.dry_run:
        snapshot_path.mkdir(parents=True, exist_ok=True)
    return snapshot_path


def resolve_models(args, storage_root, hf_hub):
    """Resolve model paths using a keep-on-disk, run-from-RAM flow.

    Models live permanently in a persistent directory on the hard drive (the
    --models-cache / --download-to location) and are copied into the RAM-disk
    working directory (<storage>/models) for use. A missing persistent copy is
    downloaded once; a RAM-disk wipe on reboot then only triggers a fast local
    copy, never another multi-GB download.

    If no persistent location is given, models are downloaded straight into the
    working directory (the previous behavior).
    """
    work_dir = storage_root / "models"
    if not args.dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)

    # Where models are kept permanently. Falls back to the working dir when the
    # user did not provide a hard-drive location to keep them in.
    keep_root = args.models_cache or args.download_to
    persist_dir = Path(keep_root) if keep_root else work_dir
    same_location = persist_dir.resolve() == work_dir.resolve()
    if keep_root and not args.dry_run:
        persist_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for entry in MODEL_MANIFEST:
        name = entry["name"]
        save_as = entry.get("save_as", entry["file"])

        explicit = getattr(args, f"{name}_model", None)
        if explicit:
            p = Path(explicit)
            if not p.exists():
                die(f"{name} model not found: {p}")
            result[name] = str(p.resolve())
            info(f"Using explicit {name} model: {p}")
            continue

        work_file = work_dir / save_as
        persist_file = persist_dir / save_as

        # Already staged in the working (RAM) dir -> use as-is.
        if work_file.exists():
            result[name] = str(work_file)
            info(f"{name} model ready: {work_file}")
            continue

        # 1) Make sure a persistent copy exists on disk (download once).
        if persist_file.exists():
            info(f"{name} model found on disk: {persist_file}")
        else:
            if args.no_download:
                die(f"{name} model not found at {persist_file} and --no-download "
                    f"is set. Place it there or use --{name}-model PATH")
            if args.dry_run:
                info(f"Would download {name}: {entry['repo']}/{entry['file']} "
                     f"-> {persist_file}")
            else:
                info(f"Downloading {name} model ({save_as}) to {persist_dir}...")
                try:
                    hf_hub.hf_hub_download(
                        repo_id=entry["repo"],
                        filename=entry["file"],
                        local_dir=persist_dir,
                    )
                    downloaded = persist_dir / entry["file"]
                    if downloaded != persist_file and not persist_file.exists():
                        downloaded.rename(persist_file)
                    info(f"Downloaded {name} -> {persist_file}")
                except Exception as e:
                    warn(f"Failed to download {name}: {e}")
                    continue

        # 2) Stage the persistent copy into the RAM-disk working dir.
        if same_location:
            result[name] = str(persist_file)
            continue
        if args.dry_run:
            info(f"Would copy {name}: {persist_file} -> {work_file}")
            continue
        try:
            info(f"Copying {name} to RAM disk: {persist_file} -> {work_file}")
            shutil.copy2(persist_file, work_file)
            result[name] = str(work_file)
        except Exception as e:
            warn(f"Could not copy {name} to work dir, using disk copy: {e}")
            result[name] = str(persist_file)

    return result


# Headroom carved out of the reasoning server's context window.
#   - the reply itself shares the same window as the prompt
#   - tool definitions ride along in the request but are not counted by
#     _fit_messages, and an agentic client's tool set is not small
RESERVE_OUTPUT_TOKENS = 4096
RESERVE_TOOLS_TOKENS = 2048


def reasoning_ctx_size() -> int:
    """The --ctx-size this launcher passes to the reasoning server."""
    extra = SERVER_DEFAULTS["reasoning"]["extra"]
    for i, a in enumerate(extra):
        if a == "--ctx-size" and i + 1 < len(extra):
            try:
                return int(extra[i + 1])
            except ValueError:
                break
    return 32768


def derive_max_context_tokens(ctx=None) -> int:
    ctx = ctx or reasoning_ctx_size()
    return max(4096, ctx - RESERVE_OUTPUT_TOKENS - RESERVE_TOOLS_TOKENS)


# --------------------------------------------------------------------------
# Context autosizing
#
# llama.cpp reserves the whole KV cache at load time, so --ctx-size is a VRAM
# decision, not a quality knob: too high and the model fails to load, too low
# and long conversations get truncated. Both numbers needed to size it right
# are knowable before launch -- how much VRAM is free (nvidia-smi) and how many
# bytes a token of KV costs (the GGUF header) -- so compute it instead of
# shipping one hardcoded value tuned for one card.
# --------------------------------------------------------------------------

# Per GPU process: CUDA context, cuBLAS workspaces, compute buffers. Measured
# at roughly 300-500 MiB for llama.cpp; take the high end.
CUDA_OVERHEAD_MIB = 500
# Left unallocated. This is a desktop GPU: a browser opening a video or a game
# launching can swing VRAM by a gigabyte, and llama.cpp cannot give KV back
# once allocated -- it would be the desktop that fails, or the next model load.
# Percentage-based so it scales with the card rather than being tuned to 12 GB.
VRAM_SAFETY_FRACTION = 0.10
VRAM_SAFETY_MIN_MIB = 1024
# Headroom for transient GPU allocations during inference (attention scores,
# CUDA kernel scratch buffers, tensor parallelism intermediates). These are
# allocated per-inference and freed after, but must fit alongside the model
# weights and pre-allocated KV cache or the CUDA context stalls.
TRANSIENT_BUF_MIB = 512
# Below this a context is too small to be useful; better to fail loudly.
MIN_AUTO_CTX = 8192
# And above this it is too slow to be useful. VRAM stops being the binding
# constraint once an MoE parks its experts in system RAM: nothing then holds
# the sizing back from the model's trained context, and the pipeline fills
# whatever window it is handed with recalled memory, so every turn pays the
# full prefill rather than just having headroom. Measured at ~385 tok/s on a
# 4070 with Hermes3.6-35B-A3B, a full window costs 2.8 minutes of prompt
# processing here against 11.3 minutes at the 262,144 the VRAM math allowed.
# This is a latency ceiling, not a memory one -- pin --reasoning-ctx N to go
# past it deliberately.
MAX_AUTO_CTX = 65536
_GGUF_SIMPLE = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
# ggml_type -> (elements per block, bytes per block). Needed to turn a tensor's
# shape into its size on disk. An unknown type aborts the whole calculation
# rather than silently undercounting: undercounting weights inflates the
# context and costs a failed model load, which is the failure we're avoiding.
_GGML_TYPE_SIZE = {
    0:  (1, 4),      # F32
    1:  (1, 2),      # F16
    2:  (32, 18),    # Q4_0
    3:  (32, 20),    # Q4_1
    6:  (32, 22),    # Q5_0
    7:  (32, 24),    # Q5_1
    8:  (32, 34),    # Q8_0
    9:  (32, 40),    # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    29: (256, 56),   # IQ1_M
    30: (1, 2),      # BF16
    34: (256, 54),   # TQ1_0
    35: (256, 66),   # TQ2_0
    39: (32, 17),    # MXFP4
}


def _read_gguf(path, limit_keys=400, want_tensors=False):
    """Parse the GGUF header. Returns (metadata, tensors), ({}, []) on trouble.

    Only the header is read -- the tensor data (gigabytes) is never touched.
    `tensors` is [(name, n_bytes), ...] and stays empty unless asked for, since
    filling it means walking all several-hundred tensor descriptors.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return {}, []
            struct.unpack("<I", f.read(4))          # version
            n_tensors, = struct.unpack("<Q", f.read(8))
            n_kv, = struct.unpack("<Q", f.read(8))

            def rd_str():
                n, = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "replace")

            def rd_val(t):
                if t == 8:
                    return rd_str()
                if t == 9:  # array
                    et, = struct.unpack("<I", f.read(4))
                    n, = struct.unpack("<Q", f.read(8))
                    return [rd_val(et) for _ in range(n)]
                fmt = _GGUF_SIMPLE.get(t)
                if fmt is None:
                    raise ValueError(f"unknown gguf type {t}")
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

            # The tensor table follows the KV section, so stopping early at
            # limit_keys would leave the cursor mid-header and make every
            # tensor descriptor garbage. Read every key when tensors are wanted.
            n_read = n_kv if want_tensors else min(n_kv, limit_keys)
            md = {}
            for _ in range(n_read):
                k = rd_str()
                t, = struct.unpack("<I", f.read(4))
                md[k] = rd_val(t)

            if not want_tensors:
                return md, []

            tensors = []
            for _ in range(n_tensors):
                name = rd_str()
                n_dims, = struct.unpack("<I", f.read(4))
                dims = struct.unpack("<%dQ" % n_dims, f.read(8 * n_dims))
                ttype, = struct.unpack("<I", f.read(4))
                struct.unpack("<Q", f.read(8))     # offset into the data blob
                block, per_block = _GGML_TYPE_SIZE.get(ttype, (0, 0))
                if not block:
                    raise ValueError(f"unknown ggml type {ttype}")
                n_elem = 1
                for d in dims:
                    n_elem *= d
                tensors.append((name, n_elem // block * per_block))
            return md, tensors
    except (OSError, struct.error, ValueError, UnicodeDecodeError):
        return {}, []


def read_gguf_metadata(path, limit_keys=400) -> dict:
    """Parse the GGUF key/value header. Returns {} on any problem."""
    return _read_gguf(path, limit_keys)[0]


def moe_gpu_weights_mib(path) -> int:
    """VRAM cost of an MoE's weights once --cpu-moe parks the experts in RAM.

    --cpu-moe overrides the per-expert FFN tensors (ffn_{gate,up,down}_exps) to
    the CPU and leaves everything else -- attention, router, shared expert,
    norms, embeddings, output head -- on the GPU. For Hermes3.6-35B-A3B that is
    1.6 GiB of a 16.2 GiB file, so charging the file size (what a dense model
    genuinely costs) overstates VRAM tenfold and collapses the KV budget past
    zero on any consumer card.

    Returns 0 when the tensor table can't be read, so the caller falls back to
    the file size: pessimistic, costing context but never a failed load.
    """
    tensors = _read_gguf(path, want_tensors=True)[1]
    if not tensors:
        return 0
    return int(sum(n for name, n in tensors if "_exps" not in name)
               // (1024 * 1024))


def moe_expert_layer_mib(path) -> list:
    """Expert-tensor cost of each layer, in MiB, indexed by layer number.

    --cpu-moe is all or nothing: it parks every expert tensor in system RAM,
    which is what makes a 16.6 GiB model fit on a 12 GiB card, and then leaves
    whatever VRAM the weights and KV did not claim sitting idle. --n-cpu-moe N
    keeps only the first N layers' experts on the CPU, so sizing N needs the
    per-layer cost rather than the total. They are not uniform -- this model's
    layers run 330 to 560 MiB depending on how each was quantized.

    Returns [] when the tensor table can't be read or the layers aren't
    numbered as expected, which sends the caller back to plain --cpu-moe.
    """
    tensors = _read_gguf(path, want_tensors=True)[1]
    if not tensors:
        return []
    per_layer = {}
    for name, nbytes in tensors:
        if "_exps" not in name:
            continue
        m = re.match(r"blk\.(\d+)\.", name)
        if not m:
            return []
        idx = int(m.group(1))
        per_layer[idx] = per_layer.get(idx, 0) + nbytes
    if not per_layer or set(per_layer) != set(range(max(per_layer) + 1)):
        return []
    return [per_layer[i] / (1024 * 1024) for i in range(max(per_layer) + 1)]


def kv_bytes_per_token(md: dict) -> int:
    """KV cache cost of one token, from the model's own header.

    The naive formula (every layer keeps a KV cache) is wrong for hybrid
    models. Qwen3.5 for instance sets full_attention_interval=4, meaning only
    every 4th layer holds a KV cache at all -- the rest are Gated DeltaNet
    layers carrying a constant-size recurrent state. Ignoring that overstates
    the cost 4x and would hand back a quarter of the context the card can
    actually hold. Validated against measurement: this returns 32 KiB/token
    for Qwen3.5-9B, where loading at 32k/64k/128k measured 1.00/2.04/4.10 GiB.
    """
    arch = md.get("general.architecture")
    if not arch:
        return 0
    g = lambda k, d=None: md.get(f"{arch}.{k}", d)

    n_layer = g("block_count") or 0
    n_kv_heads = g("attention.head_count_kv") or g("attention.head_count") or 0
    embed = g("embedding_length") or 0
    n_heads = g("attention.head_count") or 0
    k_len = g("attention.key_length") or (embed // n_heads if n_heads else 0)
    v_len = g("attention.value_length") or k_len
    if not (n_layer and n_kv_heads and k_len):
        return 0

    # Hybrid attention: only 1 layer in `full_attention_interval` keeps a cache.
    interval = g("full_attention_interval") or 1
    full_layers = max(1, n_layer // interval) if interval > 1 else n_layer

    # llama.cpp defaults to f16 K and V.
    return int(full_layers * n_kv_heads * (k_len + v_len) * 2)


def gpu_free_mib():
    """(free_mib, total_mib, name) for GPU 0, or None when there's no NVIDIA GPU."""
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.total,memory.used,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        total_s, used_s, name = [p.strip() for p in
                                 r.stdout.strip().splitlines()[0].split(",")]
        total, used = int(float(total_s)), int(float(used_s))
        return max(0, total - used), total, name
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def autosize_reasoning_ctx(reasoning_path, embed_path=None, gpu_layers=99,
                           cpu_moe=False, pinned_ctx=None,
                           free_mib=None, ports_busy=None):
    """Plan the reasoning server's VRAM: context size, then the MoE split.

    One pool of memory, so one decision: the KV cache is sized first (it sets
    how long a conversation can get), and whatever is left over is spent on
    expert layers rather than left idle. Pass `pinned_ctx` to fix the window
    and size only the split around it.

    Returns (ctx, n_cpu_moe, notes), where n_cpu_moe is None for "use plain
    --cpu-moe" and notes are the lines to show the user. Falls back to the
    hardcoded default whenever anything can't be determined -- a bad guess
    here costs a failed model load, so every unknown resolves toward the
    known-good value.
    """
    notes = []
    default = pinned_ctx or reasoning_ctx_size()

    gpu = gpu_free_mib()
    if gpu is None:
        return default, None, ["no NVIDIA GPU detected; keeping default ctx"]
    measured_free, total_mib, gpu_name = gpu
    if free_mib is None:
        free_mib = measured_free

    # A previous instance still holding the GPU makes "free VRAM" meaningless:
    # its weights and KV read as used, so the budget comes out negative and we
    # would size down to nothing. Its ports would collide on launch anyway.
    if ports_busy is None:
        ports_busy = port_in_use(SERVER_DEFAULTS["reasoning"]["port"])
    if ports_busy:
        return default, None, [
            "reasoning port already in use -- another instance is holding the "
            f"GPU, so free VRAM can't be read; keeping default ctx {default:,}"
        ]

    if not reasoning_path or not os.path.exists(reasoning_path):
        # Normal on a --dry-run before the first download.
        return default, None, [f"reasoning model not on disk yet; showing default ctx {default:,}"]

    md = read_gguf_metadata(reasoning_path)
    per_token = kv_bytes_per_token(md)
    if not per_token:
        return default, None, ["could not read KV geometry from the GGUF; keeping default ctx"]

    arch = md.get("general.architecture", "?")
    trained_ctx = md.get(f"{arch}.context_length") or default

    # Everything that must fit alongside the KV cache.
    weights_mib = 0
    moe_note = None
    if gpu_layers and gpu_layers > 0:
        try:
            weights_mib = os.path.getsize(reasoning_path) // (1024 * 1024)
        except OSError:
            pass
        if cpu_moe and weights_mib:
            resident = moe_gpu_weights_mib(reasoning_path)
            if resident:
                moe_note = (f"MoE: experts can be offloaded, so only {resident:,} MiB "
                            f"of the {weights_mib:,} MiB file has to be VRAM-resident")
                weights_mib = resident
            else:
                moe_note = ("MoE, but the tensor table is unreadable; "
                            "charging the whole file to VRAM")
    embed_mib = 0
    if embed_path:
        try:
            embed_mib = os.path.getsize(embed_path) // (1024 * 1024)
        except OSError:
            pass

    # Two GPU-resident llama.cpp processes (reasoning + embed), each paying
    # its own CUDA context.
    overhead = CUDA_OVERHEAD_MIB * (2 if embed_path else 1)
    safety = max(VRAM_SAFETY_MIN_MIB, int(total_mib * VRAM_SAFETY_FRACTION))
    budget_mib = free_mib - weights_mib - embed_mib - overhead - safety - TRANSIENT_BUF_MIB

    notes.append(f"{gpu_name}: {free_mib:,} MiB free of {total_mib:,} MiB")
    if moe_note:
        notes.append(moe_note)
    notes.append(
        f"reserving {weights_mib:,} weights + {embed_mib:,} embed "
        f"+ {overhead} CUDA + {safety} safety + {TRANSIENT_BUF_MIB} transient = "
        f"{budget_mib:,} MiB for KV"
    )

    if budget_mib <= 0:
        # Not a context problem: the weights alone don't fit. Say so, because
        # no --ctx-size will rescue this and the default we fall back to will
        # fail to load just the same.
        notes.append(
            f"the weights alone exceed free VRAM by {-budget_mib:,} MiB -- "
            "the model will not fit on the GPU. Free VRAM, or run it on the "
            "CPU with --reasoning-cpu-moe (MoE) or a smaller quant."
        )
        notes.append(f"keeping default ctx {default:,}")
        return default, None, notes

    kib = per_token / 1024
    if pinned_ctx:
        ctx = pinned_ctx
        hit_trained = hit_latency = False
    else:
        ctx = int(budget_mib * 1024 * 1024 // per_token)
        ctx = (ctx // 4096) * 4096                   # llama.cpp likes round numbers
        ctx = min(ctx, int(trained_ctx))             # never exceed what it was trained for
        hit_trained = ctx >= int(trained_ctx)
        hit_latency = ctx > MAX_AUTO_CTX
        ctx = min(ctx, MAX_AUTO_CTX)
    notes.append(
        f"KV costs {kib:.0f} KiB/token "
        f"({max(1, (md.get(arch + '.block_count') or 1) // (md.get(arch + '.full_attention_interval') or 1))}"
        f"/{md.get(arch + '.block_count')} layers hold a cache)"
    )

    if ctx < MIN_AUTO_CTX:
        notes.append(f"computed ctx {ctx:,} below the {MIN_AUTO_CTX:,} floor; "
                     f"keeping default {default:,}")
        return default, None, notes
    if hit_latency:
        notes.append(f"VRAM allowed more; capped at {MAX_AUTO_CTX:,} so a full "
                     f"window stays a few minutes of prefill, not tens")
    elif hit_trained:
        notes.append(f"capped at the model's trained context ({int(trained_ctx):,})")

    n_cpu_moe, moe_notes = _plan_expert_split(
        reasoning_path, cpu_moe, budget_mib, ctx, per_token
    )
    notes.extend(moe_notes)
    return ctx, n_cpu_moe, notes


def _plan_expert_split(reasoning_path, cpu_moe, budget_mib, ctx, per_token):
    """Decide --n-cpu-moe N from the VRAM the KV cache did not claim.

    Capping the context stops short of using the card: once the window is
    chosen, the KV cache has a fixed size and everything left over in the
    budget is idle. Spend it on expert layers, which is what generation is
    actually bandwidth-bound on -- reading them from VRAM instead of across
    PCIe is the difference the spare memory can buy.

    --n-cpu-moe keeps the FIRST N layers on the CPU, so the GPU takes the
    tail; N is the smallest value whose tail still fits. Returns
    (n_cpu_moe, notes), with None meaning "fall back to plain --cpu-moe".
    """
    if not cpu_moe:
        return None, []
    sizes = moe_expert_layer_mib(reasoning_path)
    if not sizes:
        return None, ["could not read per-layer expert sizes; keeping --cpu-moe"]

    kv_mib = ctx * per_token / (1024 * 1024)
    spare = budget_mib - kv_mib
    n_layers = len(sizes)

    # Walk down from "everything on CPU" and stop at the last layer that fits.
    n = n_layers
    on_gpu = 0.0
    while n > 0 and on_gpu + sizes[n - 1] <= spare:
        on_gpu += sizes[n - 1]
        n -= 1

    if n == n_layers:
        return None, [
            f"{spare:,.0f} MiB spare after KV -- not enough for even one "
            f"expert layer ({min(sizes):,.0f} MiB); keeping --cpu-moe"
        ]
    total_experts = sum(sizes)
    note = (f"{spare:,.0f} MiB spare after {kv_mib:,.0f} MiB of KV: "
            f"--n-cpu-moe {n} puts {n_layers - n}/{n_layers} expert layers "
            f"({on_gpu:,.0f} of {total_experts:,.0f} MiB) on the GPU")
    return n, [note]


def set_arg(extra: list, flag: str, value: str) -> list:
    """Replace flag's value in an argv list, appending the pair if absent."""
    out = list(extra)
    for i, a in enumerate(out):
        if a == flag and i + 1 < len(out):
            out[i + 1] = value
            return out
    return out + [flag, value]


def get_arg(extra: list, flag: str, default=None):
    for i, a in enumerate(extra):
        if a == flag and i + 1 < len(extra):
            return extra[i + 1]
    return default


def del_arg(extra: list, flag: str) -> list:
    """Remove a flag and the value that follows it from an argv list."""
    out, skip = [], False
    for a in extra:
        if skip:
            skip = False
            continue
        if a == flag:
            skip = True
            continue
        out.append(a)
    return out


def resolve_reasoning_ctx(args, models):
    """Decide the context size and the MoE expert split; explain both.

    Returns (ctx, n_cpu_moe), where n_cpu_moe is None for plain --cpu-moe.
    """
    default = reasoning_ctx_size()
    raw = str(getattr(args, "reasoning_ctx", "auto") or "auto").strip().lower()

    pinned = None
    if raw not in ("auto", ""):
        try:
            pinned = int(raw)
        except ValueError:
            warn(f"--reasoning-ctx {raw!r} is not a number or 'auto'; using {default:,}")
            pinned = default

    # Size for what the launch path will actually run, which differs from a
    # plain reading of the flags in two ways. --cpu-moe is not -ngl 0: the
    # server still starts with --n-gpu-layers 99 and only the expert tensors
    # move off the GPU, so assuming an empty GPU overestimates the context as
    # badly as charging the whole file underestimates it. And --cpu-moe is
    # applied to any detected MoE, not only when the flag was passed.
    model_path = models.get("reasoning")
    cpu_moe = bool(args.reasoning_cpu_moe or
                   (model_path and is_moe_model(Path(model_path).name)))
    ctx, n_cpu_moe, notes = autosize_reasoning_ctx(
        model_path, models.get("embed"), gpu_layers=99, cpu_moe=cpu_moe,
        pinned_ctx=pinned,
    )

    # An explicit --reasoning-n-cpu-moe overrides the computed split, the same
    # way --reasoning-ctx overrides the computed window.
    raw_moe = str(getattr(args, "reasoning_n_cpu_moe", "auto") or "auto").strip().lower()
    if raw_moe not in ("auto", ""):
        try:
            n_cpu_moe = int(raw_moe)
            notes.append(f"expert split pinned via --reasoning-n-cpu-moe {n_cpu_moe}")
        except ValueError:
            warn(f"--reasoning-n-cpu-moe {raw_moe!r} is not a number or 'auto'; autosizing")

    info(f"Reasoning ctx: {ctx:,} ({'pinned' if pinned else 'auto'})")
    for n in notes:
        print(f"         {n}")
    if not pinned and ctx != default:
        print(f"         pin a different value with --reasoning-ctx N")
    return ctx, n_cpu_moe


def print_launch_plan(server_defs, launched_at):
    """Show what is about to start, before it starts.

    Until now the launcher printed per-model 'ready' lines scattered through
    the download/copy flow and a final banner of ports, so the one question
    worth answering at a glance -- which model, on which device, with how much
    context -- had no single place to look.
    """
    print()
    print("=" * 72)
    print(f"  Launch plan - {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(launched_at))}")
    print("=" * 72)
    print(f"  {'server':<10} {'port':>5}  {'device':<10} {'ctx':>8}  model")
    print("  " + "-" * 68)
    for sd in server_defs:
        extra = sd["extra"]
        ngl = get_arg(extra, "--n-gpu-layers", "0")
        ctx = get_arg(extra, "--ctx-size", "-")
        try:
            ctx = f"{int(ctx):,}"
        except (TypeError, ValueError):
            pass
        if "--cpu-moe" in extra:
            device = "GPU+MoE"
        elif "--n-cpu-moe" in extra:
            device = f"GPU+MoE{get_arg(extra, '--n-cpu-moe', '')}"
        elif str(ngl) == "0":
            device = "CPU"
        elif "--no-kv-offload" in extra:
            device = "GPU/KV-RAM"
        else:
            device = "GPU"
        print(f"  {sd['name']:<10} {sd['port']:>5}  {device:<10} {ctx:>8}  "
              f"{Path(sd['model']).name}")
    print("=" * 72)
    print()


def update_config(storage_root, snapshot_path, args, reasoning_ctx=None):
    import yaml
    # encoding="utf-8" is required: config.yaml holds non-ASCII correction
    # patterns (e.g. Azerbaijani). Without it, Windows' default codepage
    # mangles them, and yaml.dump without allow_unicode re-escapes them.
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["store_path"] = str(storage_root / "store")
    config["snapshot_path"] = str(snapshot_path)
    config["models_dir"] = str(storage_root / "models")
    # Keep the prompt budget tied to the context window actually being served,
    # so the two can't drift apart and overflow.
    ctx = reasoning_ctx or reasoning_ctx_size()
    derived = derive_max_context_tokens(ctx)
    if config.get("max_context_tokens") != derived:
        info(f"max_context_tokens: {derived:,} "
             f"(ctx {ctx:,} - {RESERVE_OUTPUT_TOKENS} reply "
             f"- {RESERVE_TOOLS_TOKENS} tools)")
    config["max_context_tokens"] = derived
    config.setdefault("tokens_per_word", 1.3)
    config["listen"] = f"127.0.0.1:{args.middleware_port}"
    config["reasoning_endpoint"] = f"http://127.0.0.1:{args.reasoning_port}"
    config["judge_endpoint"] = f"http://127.0.0.1:{args.judge_port}"
    config["embed_endpoint"] = f"http://127.0.0.1:{args.embed_port}"
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


# --------------------------------------------------------------------------
# Wedge watchdog
#
# A llama.cpp slot can be lost while the process stays alive: the HTTP threads
# keep answering, but everything routed through the inference queue blocks
# forever at 0% CPU. /health and /props are served without touching the queue,
# so they stay fast -- that split IS the signature, and it is what makes this
# detectable at all.
#
# Clearing the KV cache cannot recover it: /admin/kv/clear reaches the server
# through GET /slots and POST /slots/{id}?action=erase, both queue tasks, so
# the cure blocks on the disease. Restarting the process is the only fix, and
# it belongs here because this is what owns the handles -- a restart from
# anywhere else orphans a multi-GB process when the launcher exits.
# --------------------------------------------------------------------------

# Healthy, /slots answered in 0.03-0.67 s even under load. Ten seconds is a
# wide margin, and three consecutive strikes means ~45 s of a blocked queue
# before anything is killed -- a restart aborts whatever is generating, so the
# bar to act is deliberately high.
WEDGE_SLOTS_TIMEOUT = 10
WEDGE_STRIKES = 3
WATCHDOG_INTERVAL = 15
RESTART_REQUEST = LOG_DIR / "restart-request"


class ServerSupervisor:
    """Watches the llama servers and restarts one that has wedged."""

    def __init__(self, llama_bin, server_defs, processes):
        self.llama_bin = llama_bin
        self.defs = {sd["name"]: sd for sd in server_defs}
        self.processes = processes          # shared list of (name, proc, port)
        self.strikes = {}
        self.lock = threading.Lock()
        # Names mid-restart. The main loop treats any exited child as a reason
        # to bring the whole stack down, and a reload takes tens of seconds --
        # without this, a watchdog restart would look like a crash and shut
        # everything off.
        self.restarting = set()
        self.stop = threading.Event()

    def is_restarting(self, name):
        with self.lock:
            return name in self.restarting

    def _probe(self, port):
        """Return 'ok', 'wedged', or 'down' for one server."""
        import httpx
        try:
            h = httpx.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if h.status_code != 200:
                return "down"
        except Exception:
            return "down"
        try:
            # Any status counts as alive -- a server without slot support
            # answers 501, which is a reply and therefore not a wedge.
            httpx.get(f"http://127.0.0.1:{port}/slots",
                      timeout=WEDGE_SLOTS_TIMEOUT)
            return "ok"
        except Exception:
            return "wedged"

    def restart(self, name, reason):
        sd = self.defs.get(name)
        if sd is None:
            warn(f"cannot restart unknown server '{name}'")
            return False
        with self.lock:
            if name in self.restarting:
                return False
            self.restarting.add(name)
        try:
            warn(f"restarting {name}: {reason}")
            idx = next((i for i, (n, _, _) in enumerate(self.processes)
                        if n == name), None)
            if idx is None:
                return False
            _, proc, port = self.processes[idx]
            if proc.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            old_log = getattr(proc, "_log_file", None)
            if old_log:
                try:
                    old_log.close()
                except Exception:
                    pass
            fresh = start_server(self.llama_bin, name, sd["model"], port, sd["extra"])
            self.processes[idx] = (name, fresh, port)
            ok = wait_for_server(name, port, fresh)
            info(f"{name} {'restarted' if ok else 'did NOT come back'}")
            self.strikes[name] = 0
            return ok
        finally:
            with self.lock:
                self.restarting.discard(name)

    def _check_requests(self):
        """Honour a restart asked for by the admin page."""
        try:
            if not RESTART_REQUEST.exists():
                return
            wanted = RESTART_REQUEST.read_text(encoding="utf-8").strip() or "reasoning"
            RESTART_REQUEST.unlink()
        except OSError:
            return
        for name in [n.strip() for n in wanted.split(",") if n.strip()]:
            self.restart(name, "requested from the admin page")

    def watch(self):
        while not self.stop.wait(WATCHDOG_INTERVAL):
            self._check_requests()
            for name, proc, port in list(self.processes):
                if name == "middleware" or self.is_restarting(name):
                    continue
                if proc.poll() is not None:
                    continue            # a dead child is the main loop's job
                state = self._probe(port)
                if state == "wedged":
                    n = self.strikes.get(name, 0) + 1
                    self.strikes[name] = n
                    warn(f"{name}: /health ok but /slots blocked "
                         f"({n}/{WEDGE_STRIKES})")
                    if n >= WEDGE_STRIKES:
                        self.restart(name, "inference queue blocked")
                else:
                    self.strikes[name] = 0


def wait_for_server(name, port, proc, timeout=120):
    import httpx
    for attempt in range(timeout // 2):
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                info(f"{name} ready (port {port})")
                return True
        except Exception:
            pass
        if proc.poll() is not None:
            warn(f"{name} crashed (exit code {proc.returncode})")
            return False
        time.sleep(2)
    warn(f"{name} not ready after {timeout}s (port {port})")
    return False


def start_server(llama_bin, name, model_path, port, extra):
    args = [llama_bin, "-m", model_path, "--port", str(port), "--host", "127.0.0.1"]
    if extra:
        args.extend(extra)
    # Output goes to a file, not a PIPE. Nothing ever read that pipe, so once
    # the OS buffer filled (tens of KB -- a few hundred requests) the next
    # write by llama-server would block forever and freeze the server with no
    # log to show for it. A file also means there IS a server-side log to read
    # when a slot wedges; the middleware's own log only shows the timeout.
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    log_file = open(log_path, "a", encoding="utf-8", errors="replace")
    log_file.write(f"\n=== {name} started {time.strftime('%Y-%m-%d %H:%M:%S')} "
                   f"on port {port} ===\n")
    log_file.flush()
    info(f"Starting {name} on port {port} (log: {log_path})...")
    proc = subprocess.Popen(
        args,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    # Keep the handle alive for the process's lifetime: if it were collected,
    # the fd would close under a still-running child.
    proc._log_file = log_file
    return proc


def ensure_config_exists():
    """Bootstrap config.yaml from the template on a genuinely fresh clone.

    config.yaml is gitignored (it holds a specific machine's store_path /
    snapshot_path), so a new clone won't have one at all -- only the generic
    config.example.yaml template. Without this, update_config() would crash
    trying to open a file that was never there.
    """
    if CONFIG_PATH.exists():
        return
    example = CONFIG_PATH.parent / "config.example.yaml"
    if not example.exists():
        die(f"config.yaml not found and no template at {example}")
    shutil.copy2(example, CONFIG_PATH)
    info(f"First run: created {CONFIG_PATH} from config.example.yaml")


def main():
    os.chdir(ROOT)
    args = parse_args()

    launched_at = time.time()
    print("=== Cued Recall Memory Middleware ===")
    print(f"    launched {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(launched_at))}")
    print()

    ensure_config_exists()

    llama_bin = find_llama_server(args)
    if not llama_bin:
        die(
            "llama-server not found. Install llama.cpp:\n"
            "  https://github.com/ggerganov/llama.cpp/releases\n"
            "  Or use --llama-bin PATH"
        )
    info(f"llama-server: {llama_bin}")

    # Reasoning model menu: pick from the catalog (or reuse the remembered
    # choice), then let the normal resolve flow download/copy it.
    chosen = choose_reasoning_model(args)
    if chosen:
        MODEL_MANIFEST[0] = {
            "name": "reasoning",
            "repo": chosen["repo"],
            "file": chosen["file"],
        }

    hf_hub = None if args.no_download and not args.models_cache else ensure_hf_hub()

    storage_root = setup_storage(args)
    snapshot_path = resolve_snapshot_path(args)
    models = resolve_models(args, storage_root, hf_hub)

    if not models and not args.dry_run:
        die("No models resolved. Use --download-to, --models-cache, or explicit --*-model paths")

    reasoning_ctx, reasoning_n_cpu_moe = resolve_reasoning_ctx(args, models)

    # Everything run.bat wants to show is settled by here: the models are
    # located and the window is sized.
    if not args.dry_run:
        remember_launch(args, models, reasoning_ctx)

    skip_map = {
        "reasoning": args.skip_reasoning,
        "judge": args.skip_judge,
        "embed": args.skip_embed,
    }

    port_map = {
        "reasoning": args.reasoning_port,
        "judge": args.judge_port,
        "embed": args.embed_port,
    }

    server_defs = []
    for entry in MODEL_MANIFEST:
        name = entry["name"]
        if skip_map.get(name):
            info(f"Skipping {name} server (--skip-{name})")
            continue
        if name not in models:
            warn(f"No model for {name}, skipping server")
            continue
        model_path = models[name]
        defaults = SERVER_DEFAULTS[name]
        extra = list(defaults["extra"])
        if name == "reasoning":
            extra = set_arg(extra, "--ctx-size", str(reasoning_ctx))
        if name == "reasoning" and (args.reasoning_cpu_moe or is_moe_model(Path(model_path).name)):
            # MoE: keep router/attention/KV on GPU and park expert tensors in
            # system RAM so 17-20 GB A3B models run on a 12 GB card. Only the
            # experts that do not fit go to RAM -- see _plan_expert_split.
            if reasoning_n_cpu_moe is None:
                extra.append("--cpu-moe")
                info("Reasoning is MoE: adding --cpu-moe (all experts in system RAM)")
            elif reasoning_n_cpu_moe > 0:
                extra += ["--n-cpu-moe", str(reasoning_n_cpu_moe)]
                info(f"Reasoning is MoE: adding --n-cpu-moe {reasoning_n_cpu_moe} "
                     f"(first {reasoning_n_cpu_moe} layers' experts in system RAM)")
            else:
                info("Reasoning is MoE but every expert fits in VRAM; no offload")
        if name in ("reasoning", "judge"):
            # llama-server rejects every /slots action with 501 unless it was
            # started with --slot-save-path -- including "erase", which writes
            # nothing. The flag only unlocks the endpoint; KV state is written
            # to this directory solely on an explicit action=save, which
            # nothing here issues, so it stays empty. Required for the admin
            # page's Clear KV Cache button.
            slot_dir = storage_root / "slots" / name
            slot_dir.mkdir(parents=True, exist_ok=True)
            extra += ["--slot-save-path", str(slot_dir)]
        server_defs.append({
            "name": name,
            "model": model_path,
            "port": port_map[name],
            "extra": extra,
        })

    if not server_defs and not args.dry_run:
        die("No servers to start")

    print_launch_plan(server_defs, launched_at)

    if args.dry_run:
        info("Dry run complete. No changes made.")
        return

    update_config(storage_root, snapshot_path, args, reasoning_ctx)

    processes = []
    try:
        for sd in server_defs:
            proc = start_server(llama_bin, sd["name"], sd["model"], sd["port"], sd["extra"])
            processes.append((sd["name"], proc, sd["port"]))

        info("Waiting for servers...")
        for i, (name, proc, port) in enumerate(processes):
            if wait_for_server(name, port, proc):
                continue
            # Both autosized numbers are estimates: VRAM can be taken by
            # another app between measuring and loading, and neither the KV
            # formula nor the expert-layer sum carries much slack. Rather than
            # leave the stack half-up, give ground and try once more. The
            # expert split goes first because it is the bigger lever -- it
            # frees gigabytes of expert weight where halving the window frees
            # only the KV -- and giving up both at once beats burning a second
            # restart working out which one was at fault.
            sd = next((s for s in server_defs if s["name"] == name), None)
            if sd is None or name != "reasoning":
                continue
            retry_extra = sd["extra"]
            surrendered = []
            if "--n-cpu-moe" in retry_extra:
                retry_extra = del_arg(retry_extra, "--n-cpu-moe") + ["--cpu-moe"]
                surrendered.append("expert split -> --cpu-moe")
            old_ctx = int(get_arg(retry_extra, "--ctx-size", "0") or 0)
            new_ctx = (old_ctx // 2 // 4096) * 4096
            if old_ctx > MIN_AUTO_CTX and new_ctx >= MIN_AUTO_CTX:
                retry_extra = set_arg(retry_extra, "--ctx-size", str(new_ctx))
                surrendered.append(f"ctx {old_ctx:,} -> {new_ctx:,}")
            else:
                new_ctx = old_ctx
            if not surrendered:
                continue
            warn(f"{name} failed to start; retrying ({', '.join(surrendered)})")
            sd["extra"] = retry_extra
            retry = start_server(llama_bin, name, sd["model"], port, sd["extra"])
            processes[i] = (name, retry, port)
            if wait_for_server(name, port, retry):
                # The prompt budget was written from the ctx that just failed.
                update_config(storage_root, snapshot_path, args, new_ctx)
                info(f"recovered at ctx {new_ctx:,}; "
                     f"pin it with --reasoning-ctx {new_ctx}")

        info("Starting middleware...")
        # The middleware starts seconds-to-minutes after the launcher (model
        # loads come first), so its own process start time is not when the
        # stack came up. Hand it the real figure for the admin page.
        mw_env = {**os.environ, "CUED_RECALL_LAUNCHED_AT": str(launched_at),
                  # How the admin page asks for a restart. Absent when the
                  # middleware is started on its own, and the endpoint says so
                  # rather than pretending it worked.
                  "CUED_RECALL_RESTART_FILE": str(RESTART_REQUEST)}
        middleware = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "cued_recall.main:create_app",
             "--factory", "--host", "127.0.0.1", "--port", str(args.middleware_port),
             "--log-level", "info"],
            cwd=str(ROOT / "cued_recall"),
            env=mw_env,
        )
        processes.append(("middleware", middleware, args.middleware_port))

        print()
        print("=== All systems running ===")
        print(f"  Middleware:     http://127.0.0.1:{args.middleware_port}/v1/chat/completions")
        print(f"  Admin GUI:      http://127.0.0.1:{args.middleware_port}/admin")
        print(f"  Admin stats:    http://127.0.0.1:{args.middleware_port}/admin/stats")
        for name, _, port in processes:
            if name != "middleware":
                print(f"  {name.capitalize():14} http://127.0.0.1:{port}")
        print(f"  Storage:        {storage_root}")
        print(f"  Snapshots:      {snapshot_path}")
        print("  Press Ctrl+C to stop all processes")
        print()

        def shutdown(sig, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        supervisor = ServerSupervisor(llama_bin, server_defs, processes)
        threading.Thread(target=supervisor.watch, daemon=True).start()
        info(f"Wedge watchdog active (probe every {WATCHDOG_INTERVAL}s, "
             f"restart after {WEDGE_STRIKES} blocked probes)")

        while True:
            time.sleep(3)
            # A server being replaced is briefly absent by design; only an
            # unplanned exit should bring the stack down.
            for name, proc, _ in list(processes):
                if supervisor.is_restarting(name):
                    continue
                rc = proc.poll()
                if rc is not None:
                    warn(f"{name} exited with code {rc}")
            if any(proc.poll() is not None and not supervisor.is_restarting(name)
                   for name, proc, _ in list(processes)):
                time.sleep(1)
                # Re-check: the watchdog may have swapped in a live process
                # during that pause, in which case nothing actually died.
                if any(proc.poll() is not None and not supervisor.is_restarting(name)
                       for name, proc, _ in list(processes)):
                    break

    except KeyboardInterrupt:
        info("Shutting down...")
    finally:
        try:
            supervisor.stop.set()
        except NameError:
            pass          # failed before the watchdog was started
        for name, proc, port in reversed(processes):
            if proc.poll() is None:
                info(f"Stopping {name} (port {port})...")
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        info("All processes stopped")


if __name__ == "__main__":
    main()
