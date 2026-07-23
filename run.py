#!/usr/bin/env python3
"""
Cued Recall — one-command launcher.

Zero-config: downloads models, sets up 64GB tmpfs, starts 3 llama-server
instances + middleware. Ctrl+C to stop everything.
"""

import os
import sys
import time
import signal
import subprocess
import shutil
import json
import platform
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
CONFIG_PATH = ROOT / "cued_recall" / "config.yaml"

TMPFS_ROOT = Path("/mnt/ramdisk/cued_recall")
TMPFS_SIZE = "64G"


def info(msg):
    print(f"[INFO] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def die(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def find_llama_server():
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
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "huggingface-hub", "-q"]
        )
        import huggingface_hub
        return huggingface_hub


def setup_tmpfs():
    """Mount 64G tmpfs. On Windows, just create the directory structure."""
    if platform.system() == "Windows":
        info("Windows detected: using local directory instead of tmpfs")
        TMPFS_ROOT.mkdir(parents=True, exist_ok=True)
        return

    if not TMPFS_ROOT.exists():
        info(f"Mounting {TMPFS_SIZE} tmpfs at {TMPFS_ROOT}...")
        TMPFS_ROOT.mkdir(parents=True, exist_ok=True)
        ret = subprocess.run(
            ["mount", "-t", "tmpfs", "-o", f"size={TMPFS_SIZE}", "tmpfs", str(TMPFS_ROOT)],
            capture_output=True, text=True,
        )
        if ret.returncode != 0:
            warn(f"tmpfs mount failed (need root?): {ret.stderr.strip()}")
            warn("Falling back to local directory")
            return
        info(f"tmpfs mounted: {TMPFS_ROOT} ({TMPFS_SIZE})")

    (TMPFS_ROOT / "models").mkdir(parents=True, exist_ok=True)
    (TMPFS_ROOT / "store").mkdir(parents=True, exist_ok=True)
    (TMPFS_ROOT / "snapshots").mkdir(parents=True, exist_ok=True)


def download_models(hf_hub):
    models_dir = TMPFS_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest = [
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

    downloaded = {}
    for entry in manifest:
        repo = entry["repo"]
        filename = entry["file"]
        save_as = entry.get("save_as", filename)
        dest = models_dir / save_as

        if dest.exists():
            info(f"Model {save_as} already in tmpfs, skipping")
            downloaded[entry["name"]] = str(dest)
            continue

        info(f"Downloading {save_as} from {repo}...")
        try:
            hf_hub.hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=models_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            downloaded_file = models_dir / filename
            if downloaded_file != dest and not dest.exists():
                downloaded_file.rename(dest)
            info(f"Downloaded {save_as} to tmpfs")
            downloaded[entry["name"]] = str(dest)
        except Exception as e:
            warn(f"Failed to download {save_as}: {e}")

    return downloaded, models_dir


def update_config(models_dir: Path):
    import yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    config["store_path"] = str(TMPFS_ROOT / "store")
    config["snapshot_path"] = str(TMPFS_ROOT / "snapshots")
    config["models_dir"] = str(models_dir)

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def wait_for_server(name: str, port: int, proc: subprocess.Popen,
                    timeout: int = 120) -> bool:
    import httpx
    for attempt in range(timeout // 2):
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                info(f"{name} on port {port} is ready")
                return True
        except Exception:
            pass
        if proc.poll() is not None:
            warn(f"{name} crashed (exit code {proc.returncode})")
            return False
        time.sleep(2)
    warn(f"{name} on port {port} not ready after {timeout}s")
    return False


def start_server(llama_bin: str, name: str, model_path: str, port: int,
                 extra: list) -> subprocess.Popen:
    args = [llama_bin, "-m", model_path, "--port", str(port), "--host", "127.0.0.1"]
    if extra:
        args.extend(extra)
    info(f"Starting {name} (port {port})...")
    return subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def main():
    os.chdir(ROOT)

    print("=== Cued Recall Memory Middleware ===")
    print()

    llama_bin = find_llama_server()
    if not llama_bin:
        die(
            "llama-server not found. Install llama.cpp:\n"
            "  https://github.com/ggerganov/llama.cpp/releases\n"
            "  Or build from source and place in PATH."
        )
    info(f"llama-server: {llama_bin}")

    hf_hub = ensure_hf_hub()

    setup_tmpfs()
    _, models_dir = download_models(hf_hub)
    update_config(models_dir)

    server_defs = [
        {
            "name": "reasoning",
            "model": models_dir / "Qwen3-8B-Q4_K_M.gguf",
            "port": 8080,
            "extra": ["--ctx-size", "32768", "--n-gpu-layers", "99", "--no-kv-offload"],
        },
        {
            "name": "judge",
            "model": models_dir / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
            "port": 8081,
            "extra": ["--ctx-size", "8192", "--n-gpu-layers", "99"],
        },
        {
            "name": "embed",
            "model": models_dir / "nomic-embed-text-v1.5-Q8_0.gguf",
            "port": 8082,
            "extra": ["--embedding", "--ctx-size", "2048", "--n-gpu-layers", "99"],
        },
    ]

    missing = [s for s in server_defs if not s["model"].exists()]
    if missing:
        for s in missing:
            warn(f"Model not found: {s['model']}")
        die("Missing models. Check download step above.")

    processes = []

    try:
        for sd in server_defs:
            proc = start_server(llama_bin, sd["name"], str(sd["model"]),
                                sd["port"], sd["extra"])
            processes.append((sd["name"], proc, sd["port"]))

        info("Waiting for servers to become ready...")
        for name, proc, port in processes:
            wait_for_server(name, port, proc)

        info("Starting middleware...")
        middleware = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "cued_recall.main:create_app",
             "--factory", "--host", "127.0.0.1", "--port", "8000",
             "--log-level", "info"],
            cwd=str(ROOT / "cued_recall"),
        )
        processes.append(("middleware", middleware, 8000))

        print()
        print("=== All systems running ===")
        print(f"  Middleware:     http://127.0.0.1:8000/v1/chat/completions")
        print(f"  Admin stats:    http://127.0.0.1:8000/admin/stats")
        print(f"  Reasoning:      http://127.0.0.1:8080")
        print(f"  Judge:          http://127.0.0.1:8081")
        print(f"  Embedding:      http://127.0.0.1:8082")
        print(f"  Models/storage: {TMPFS_ROOT}")
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
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        info("All processes stopped")


if __name__ == "__main__":
    main()
