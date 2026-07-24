# Cued Recall Memory Middleware

A memory layer for AI chat. It sits between your chat program (like Open WebUI or SillyTavern) and a local LLM, and gives the AI the ability to remember how it solved similar problems before.

> **Status: Alpha.** Works, but expect rough edges.

---

## What you need (hardware)

| Requirement | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| GPU VRAM | 8 GB | 12 GB+ (RTX 4070 or better) |
| Storage | 20 GB free | SSD |
| OS | Windows 10/11 or Linux | — |

The three AI models need about 7 GB of VRAM together. 12 GB (RTX 4070) is comfortable.

---

## Step-by-step installation

### Step 1: Install Python

> Skip this if you already have Python 3.11 or newer.

1. Go to https://www.python.org/downloads/
2. Click the big yellow **Download Python** button (get 3.11 or newer)
3. Run the downloaded installer
4. **IMPORTANT:** Check the box that says **"Add Python to PATH"** at the bottom of the first screen
5. Click **Install Now**
6. Close the installer when done

**Verify:** Open a terminal (Windows: press `Win+R`, type `cmd`, press Enter) and type:
```
python --version
```
You should see `Python 3.11.x` or higher.

---

### Step 2: Install Git

> Skip if you already have Git. Check by typing `git --version` in a terminal.

1. Go to https://git-scm.com/downloads
2. Download the version for your OS
3. Run the installer — all default settings are fine

---

### Step 3: Get the project

Open a terminal and run:

```bash
git clone https://github.com/master-basic/mind.git
cd mind
```

Or download the ZIP from https://github.com/master-basic/mind and extract it to a folder.

---

### Step 4: Install llama.cpp (the AI engine)

This is the program that actually runs the AI models on your GPU.

1. Go to https://github.com/ggerganov/llama.cpp/releases
2. Find the latest release (not "Pre-release")
3. Download the file for your system:
   - **Windows with NVIDIA GPU:** `llama-b3735-bin-win-cuda-cu12.4.7-x64.zip` (or similar CUDA build)
   - **Windows with AMD GPU:** `llama-b3735-bin-win-vulkan-x64.zip`
   - **Windows with only CPU (slow):** `llama-b3735-bin-win-avx2-x64.zip` (or `avx` if your CPU is older)
   - **Linux:** `llama-b3735-bin-ubuntu-cuda.tar.xz`
4. Extract the ZIP to a folder, for example `C:\llama.cpp`
5. Add that folder to your system PATH:
   - Windows: Search for "Environment Variables" → "Edit environment variables"
   - Click "Environment Variables"
   - Under "System variables", find `Path`, click "Edit"
   - Click "New" and add the path to the folder (e.g. `C:\llama.cpp`)
   - Click OK on all windows

**Verify:** Open a **new** terminal and type:
```
llama-server --version
```
You should see a version number.

---

### Step 5: (Optional but recommended) Create a RAM disk

A RAM disk uses part of your memory as an ultra-fast temporary drive. Models load faster and the middleware runs smoother.

#### Windows
1. Download ImDisk Toolkit (free): https://sourceforge.net/projects/imdisk-toolkit/
2. Run the installer (Windows may ask for administrator permission — click Yes)
3. Close and reopen your terminal
4. In the `mind` folder, double-click `ramdisk_setup.bat`
5. Answer the questions:
   - **Drive letter:** press Enter for `R` (default)
   - **Size:** type `64` (or however many GB you can spare)
   - **Block size:** press Enter for `4096` (default)
   - **Volume label:** press Enter for `CUED_RECALL`
   - Type `y` to confirm
6. The RAM disk is ready at `R:\cued_recall`

#### Linux (automatic)
The launcher does this for you — mounts a 64 GB tmpfs at `/mnt/ramdisk/cued_recall`.

---

### Step 6: Run everything

Now the easy part. The launcher will:
1. Install required Python packages
2. Download the three AI models (about 7 GB total) from Hugging Face
3. Start three AI server instances
4. Start the middleware

#### Windows — just double-click:

```
run.bat
```

Or in a terminal:
```
run.bat
```

The first time, it will ask questions interactively. You can press Enter for all defaults.

#### Faster: if you already have the models downloaded somewhere

```
run.bat --models-cache D:\existing-models
```

#### With the optional RAM disk:

```
run.bat --storage R:\cued_recall
```

#### Linux:
```bash
python run.py
```

---

### Step 7: Wait for startup

The launcher downloads 3 models (one-time, ~7 GB total), then starts the AI servers. This takes:

| Step | Time |
|---|---|
| Installing Python packages | 1–2 minutes |
| Downloading models (first run only) | 5–30 minutes (depends on internet) |
| Starting AI servers | 30–60 seconds |
| **Total first run** | **10–40 minutes** |
| **Subsequent runs** | **1–2 minutes** |

You'll see this when everything is ready:

```
=== All systems running ===
  Middleware:     http://127.0.0.1:8000/v1/chat/completions
  Admin GUI:      http://127.0.0.1:8000/admin
  Reasoning:      http://127.0.0.1:8080
  Judge:          http://127.0.0.1:8081
  Embedding:      http://127.0.0.1:8082
```

---

### Step 8: Connect a chat program

Your chat client (e.g. Open WebUI, SillyTavern, Continue, etc.) connects to the middleware as if it were a standard OpenAI API.

**Settings:**

| Setting | Value |
|---|---|
| API Endpoint | `http://127.0.0.1:8000/v1` |
| API Key | `not-needed` (any value works) |
| Model | `default` (or any string) |

The middleware passes your request to the reasoning AI and records reasoning blocks for future recall.

---

### Step 9: Check the admin panel

Open http://127.0.0.1:8000/admin in your browser.

You'll see:
- **Stats** — how many reasoning blocks are stored, how many file are on disk
- **Blocks table** — every reasoning, result, and reading block with status and verification
- **Run Judge Pass** button — force the judge to review and compress old blocks

The admin page auto-refreshes every 5 seconds.

---

## How to stop

Press **Ctrl+C** in the terminal where the launcher is running. All three AI servers and the middleware shut down cleanly.

On Windows, you can also close the terminal window.

---

## How to restart after reboot

### If you created a RAM disk:
```
ramdisk_setup.bat     # recreate the RAM disk
run.bat               # start everything
```

### Without a RAM disk:
```
run.bat               # start everything
```

A RAM disk is wiped on reboot, so its copy of the models is gone. As long as you gave a **keep folder** (the "Folder on your hard drive to KEEP models" prompt, or `--models-cache DIR`), the models were downloaded there once and are simply **copied back to the RAM disk** on each start — no re-download. If you never set a keep folder, they live in the storage dir and are only re-downloaded when that dir is empty.

---

## Quick reference (launcher options)

| Option | What it does |
|---|---|
| `--storage PATH` | Where to put models and data. Default: `./data` |
| `--models-cache DIR` | Keep models permanently in this hard-drive folder (downloaded here once if missing), then copy them to the RAM disk each run |
| `--no-download` | Fail if models aren't already present |
| `--reasoning-model PATH` | Use a specific reasoning model file |
| `--judge-model PATH` | Use a specific judge model file |
| `--embed-model PATH` | Use a specific embedding model file |
| `--reasoning-port N` | Change reasoning server port (default: 8080) |
| `--judge-port N` | Change judge server port (default: 8081) |
| `--embed-port N` | Change embedding server port (default: 8082) |
| `--middleware-port N` | Change middleware port (default: 8000) |
| `--skip-reasoning` | Don't start the reasoning server |
| `--skip-judge` | Don't start the judge server |
| `--skip-embed` | Don't start the embedding server |
| `--dry-run` | Show what would happen without doing anything |
| `--help` | Show all options |

---

## Files in this project

| File | Purpose |
|---|---|
| `run.bat` | One-click launcher for Windows |
| `run.py` | Orchestrator (starts servers, downloads models) |
| `ramdisk_setup.bat` | Create a RAM disk on Windows |
| `cued_recall/config.yaml` | Middleware settings |
| `cued_recall/cued_recall/main.py` | The middleware server |
| `cued_recall/cued_recall/pipeline.py` | Recall, injection, block creation |
| `cued_recall/cued_recall/judge.py` | Automatic block compression |
| `cued_recall/cued_recall/store.py` | Block storage (msgpack files) |
| `cued_recall/cued_recall/index.py` | Vector search index |
| `cued_recall/cued_recall/embed.py` | Embedding model client |
| `cued_recall/cued_recall/router.py` | Admin API routes |
| `cued_recall/cued_recall/static/admin.html` | Admin web GUI |

---

## Troubleshooting

**"llama-server not found"**
You need to install llama.cpp (Step 4). Download from https://github.com/ggerganov/llama.cpp/releases and add the folder to your PATH.

**"Python 3.11+ is required"**
You have an older Python. Install Python 3.11 or newer from https://www.python.org/downloads/

**"Failed to download model"**
Your internet connection may be slow or the Hugging Face server is busy. Run again — downloads resume automatically.

**"imdisk not found" / RAM disk script says not found**
Install ImDisk Toolkit first (https://sourceforge.net/projects/imdisk-toolkit/), then close and reopen your terminal.

**"CUDA error: out of memory"**
Your GPU doesn't have enough VRAM for all three models at once. Try:
- Use smaller models (edit `run.py` `MODEL_MANIFEST` to use Qwen3-4B instead of 8B)
- Skip the judge server: `--skip-judge` (uses less VRAM)
- Use only CPU for small models (remove `--n-gpu-layers 99` from `run.py` server configs)

**The admin page shows no blocks after chatting**
That's expected if the model didn't emit `<think>` tags. Some models don't. Try asking a math or logic question to trigger reasoning.

**Port 8080/8081/8082 already in use**
Another program is using that port. Use `--reasoning-port 8090 --judge-port 8091 --embed-port 8092` to change ports.

---

## License

MIT
