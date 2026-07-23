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
import json
import platform
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
CONFIG_PATH = ROOT / "cued_recall" / "config.yaml"
DEFAULT_TMPFS = Path("/mnt/ramdisk/cued_recall")
TMPFS_SIZE = "64G"

MODEL_MANIFEST = [
    {
        "name": "reasoning",
        "repo": "bartowski/Qwen3-8B-GGUF",
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

SERVER_DEFAULTS = {
    "reasoning": {"port": 8080, "extra": ["--ctx-size", "32768", "--n-gpu-layers", "99", "--no-kv-offload"]},
    "judge":     {"port": 8081, "extra": ["--ctx-size", "8192", "--n-gpu-layers", "99"]},
    "embed":     {"port": 8082, "extra": ["--embedding", "--ctx-size", "2048", "--n-gpu-layers", "99"]},
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
                   help="Root storage path (default: /mnt/ramdisk/cued_recall or ./data on Windows)")

    g = p.add_argument_group("Model overrides")
    g.add_argument("--reasoning-model", metavar="PATH", help="Path to reasoning model GGUF")
    g.add_argument("--judge-model",     metavar="PATH", help="Path to judge model GGUF")
    g.add_argument("--embed-model",     metavar="PATH", help="Path to embedding model GGUF")

    g = p.add_argument_group("Port overrides")
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
    if args.llama_bin:
        if Path(args.llama_bin).exists():
            return str(Path(args.llama_bin).resolve())
        die(f"Specified --llama-bin not found: {args.llama_bin}")
    names = ["llama-server.exe", "llama-server"]
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
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
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


def setup_storage(args):
    """Determine and prepare the storage root (tmpfs or local)."""
    is_windows = platform.system() == "Windows"

    if args.storage:
        storage_root = Path(args.storage)
    elif is_windows or args.no_tmpfs:
        storage_root = ROOT / "data"
    else:
        storage_root = DEFAULT_TMPFS

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
    (ROOT / "snapshots").mkdir(parents=True, exist_ok=True)
    return storage_root


def resolve_models(args, storage_root, hf_hub):
    """Resolve model paths. Download or copy as needed."""
    models_dir = storage_root / "models"
    result = {}

    has_explicit = any([
        args.reasoning_model, args.judge_model, args.embed_model,
    ])

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

        dest = models_dir / save_as
        if dest.exists():
            result[name] = str(dest)
            info(f"{name} model already in storage: {dest}")
            continue

        if args.no_download:
            die(f"{name} model not found at {dest} and --no-download is set. "
                f"Place it there or use --{name}-model PATH")

        if args.models_cache:
            cache = Path(args.models_cache) / save_as
            if cache.exists():
                if not args.dry_run:
                    shutil.copy2(cache, dest)
                info(f"Copied {name} from cache: {cache} -> {dest}")
                result[name] = str(dest)
                continue
            warn(f"{name} not found in cache {cache}, will download")

        if args.dry_run:
            info(f"Would download {name} model: {entry['repo']}/{entry['file']} -> {dest}")
            continue

        info(f"Downloading {name} model ({save_as})...")
        try:
            hf_hub.hf_hub_download(
                repo_id=entry["repo"],
                filename=entry["file"],
                local_dir=models_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            downloaded = models_dir / entry["file"]
            if downloaded != dest and not dest.exists():
                downloaded.rename(dest)
            result[name] = str(dest)
            info(f"Downloaded {name} model: {dest}")
        except Exception as e:
            warn(f"Failed to download {name}: {e}")

    return result


def update_config(storage_root):
    import yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config["store_path"] = str(storage_root / "store")
    config["snapshot_path"] = str(ROOT / "snapshots")
    config["models_dir"] = str(storage_root / "models")
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


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
    info(f"Starting {name} on port {port}...")
    return subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def main():
    os.chdir(ROOT)
    args = parse_args()

    print("=== Cued Recall Memory Middleware ===")
    print()

    llama_bin = find_llama_server(args)
    if not llama_bin:
        die(
            "llama-server not found. Install llama.cpp:\n"
            "  https://github.com/ggerganov/llama.cpp/releases\n"
            "  Or use --llama-bin PATH"
        )
    info(f"llama-server: {llama_bin}")

    hf_hub = None if args.no_download and not args.models_cache else ensure_hf_hub()

    storage_root = setup_storage(args)
    models = resolve_models(args, storage_root, hf_hub)

    if args.dry_run:
        print()
        info("Dry run complete. No changes made.")
        return

    if not models:
        die("No models resolved. Use --download-to, --models-cache, or explicit --*-model paths")

    update_config(storage_root)

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
        server_defs.append({
            "name": name,
            "model": model_path,
            "port": port_map[name],
            "extra": defaults["extra"],
        })

    if not server_defs:
        die("No servers to start")

    processes = []
    try:
        for sd in server_defs:
            proc = start_server(llama_bin, sd["name"], sd["model"], sd["port"], sd["extra"])
            processes.append((sd["name"], proc, sd["port"]))

        info("Waiting for servers...")
        for name, proc, port in processes:
            wait_for_server(name, port, proc)

        info("Starting middleware...")
        middleware = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "cued_recall.main:create_app",
             "--factory", "--host", "127.0.0.1", "--port", str(args.middleware_port),
             "--log-level", "info"],
            cwd=str(ROOT / "cued_recall"),
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
        print("  Press Ctrl+C to stop all processes")
        print()

        def shutdown(sig, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        while True:
            time.sleep(3)
            for name, proc, _ in processes:
                rc = proc.poll()
                if rc is not None:
                    warn(f"{name} exited with code {rc}")
            if any(proc.poll() is not None for _, proc, _ in processes):
                time.sleep(1)
                break

    except KeyboardInterrupt:
        info("Shutting down...")
    finally:
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
