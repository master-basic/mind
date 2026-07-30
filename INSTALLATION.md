# Installation

Step-by-step setup for Cued Recall Memory Middleware. For what the project is and how it works, see [README.md](README.md).

---

## What you need (hardware)

| Requirement | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| GPU VRAM | 8 GB | 12 GB+ |
| Storage | 20 GB free | SSD |
| OS | Windows 10/11 or Linux | — |

The three AI models need about 7 GB of VRAM together. 12 GB is comfortable.

---

## Step 1: Install Python

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

## Step 2: Install Git

> Skip if you already have Git. Check by typing `git --version` in a terminal.

1. Go to https://git-scm.com/downloads
2. Download the version for your OS
3. Run the installer — all default settings are fine

---

## Step 3: Get the project

Open a terminal and run:

```bash
git clone https://github.com/master-basic/mind.git
cd mind
```

Or download the ZIP from https://github.com/master-basic/mind and extract it to a folder.

---

## Step 4: Install llama.cpp (the AI engine)

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

> You can skip the PATH step and pass `--llama-bin C:\llama.cpp` to the launcher instead.

---

## Step 5: (Optional) Create a RAM disk

A RAM disk uses part of your memory as an ultra-fast temporary drive. Models load faster and the middleware runs smoother. An SSD works fine too — see the note on storage in Step 6.

### Windows
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

### Linux (automatic)
The launcher does this for you — mounts a 64 GB tmpfs at `/mnt/ramdisk/cued_recall`.

---

## Step 6: Run everything

The launcher will:
1. Install required Python packages
2. Download the three AI models (about 7 GB total) from Hugging Face
3. Start three AI server instances
4. Start the middleware

### Windows — just double-click:

```
run.bat
```

The first time, it will ask questions interactively. You can press Enter for all defaults.

### Linux:
```bash
python run.py
```

### Choosing where data lives

Pass `--storage` to put models and memory blocks anywhere — a RAM disk, an NVMe drive, any folder:

```
run.bat --storage S:\AI
```

**The choice is sticky.** Once set, later runs without `--storage` reuse the same location instead of silently relocating. Changing it prints a warning, because memory at the old location is not migrated.

Snapshots are controlled separately with `--snapshot`, and default to a `snapshots/` folder next to the repo so they survive a RAM disk being wiped on reboot.

### If you already have the models downloaded

```
run.bat --models-cache D:\existing-models
```

---

## Step 7: Wait for startup

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
  Storage:        S:\AI
  Snapshots:      D:\Repos\mind\snapshots
```

---

## Step 8: Connect a chat program

Your chat client connects to the middleware as if it were a standard OpenAI API.

| Setting | Value |
|---|---|
| API Endpoint | `http://127.0.0.1:8000/v1` |
| API Key | `not-needed` (any value works) |
| Model | any string |

There is also a built-in chat UI at http://127.0.0.1:8000/ if you don't want a separate client.

### From another PC on your network

By default the middleware only accepts connections from the machine it runs on. To let other computers use it:

1. **Answer the listen-address prompt.** `run.bat` asks `Listen address [...] (blank=127.0.0.1)`. Enter `0.0.0.0`. On Linux, or to skip the prompt, pass it as a flag:

   ```
   python run.py --host 0.0.0.0
   ```

   The answer is remembered in `run_settings.txt` as `HOST=`, so later runs keep it.

2. **Allow the port through Windows Firewall.** Run once, in an Administrator terminal:

   ```
   netsh advfirewall firewall add rule name="Cued Recall 8000" dir=in action=allow protocol=TCP localport=8000
   ```

3. **Use the address from the startup banner.** With `--host 0.0.0.0` it prints this machine's LAN address instead of `127.0.0.1`:

   ```
   === All systems running ===
     Middleware:     http://192.168.1.50:8000/v1/chat/completions
     Admin GUI:      http://192.168.1.50:8000/admin
   ```

   On the other PC, point the chat client's API endpoint at `http://192.168.1.50:8000/v1`, or open that address in a browser for the chat UI.

The three llama.cpp servers (8080–8082) are published to the same address, so the reasoning model itself is reachable from the other machine — not just the middleware in front of it. That is what the `evaluate/` benchmark scripts need, since they talk to those ports directly. They answer with the raw model and no memory layer, so pass `--no-expose-backends` to keep them on loopback and publish only port 8000.

Windows Firewall has to allow every port that was opened, not just 8000; the startup banner prints the exact `netsh` command for the set it published.

> **No password, no encryption.** Anyone who can reach port 8000 can chat with the model, read every stored memory block, and delete blocks from the admin page. Only do this on a home or office network you control. Do not forward the port through a router.

### opencode

opencode needs `npm` set so it knows which SDK adapter to use for a custom provider. A working `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Cued Recall (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "local-model": {
          "name": "Local LLM",
          "tool_call": true,
          "reasoning": true,
          "interleaved": { "field": "reasoning_content" }
        }
      }
    }
  },
  "model": "local/local-model"
}
```

Agentic clients need a model that is genuinely good at tool calling. Abliterated or "uncensored" merges often emit malformed tool calls or loop on the same call — pick a stock or coder-tuned model if tools misbehave.

---

## Step 9: Check the admin panel

Open http://127.0.0.1:8000/admin in your browser.

- **Models** — live context usage per server
- **Stats** — how many blocks are stored
- **Blocks table** — every block with its gist, tags, status, and verification
- **Run Judge Pass** — force a review/compression pass now
- **Clear KV Cache** — drop cached prompt prefixes on the llama.cpp servers
- **Export / Import** — move blocks between machines

The page auto-refreshes every 5 seconds.

---

## How to stop

Press **Ctrl+C** in the terminal where the launcher is running. All three AI servers and the middleware shut down cleanly. On Windows, you can also close the terminal window.

---

## How to restart after reboot

### If you created a RAM disk:
```
ramdisk_setup.bat     # recreate the RAM disk
run.bat               # start everything
```

### Without a RAM disk:
```
run.bat
```

A RAM disk is wiped on reboot, so its copy of the models is gone. As long as you gave a keep folder (`--models-cache DIR`), the models are simply copied back — no re-download.

Memory blocks live in `<storage>/store`. If that is on a RAM disk it is also wiped on reboot; the snapshot directory is what restores it, which is why snapshots default to a persistent location.

---

## Launcher options

| Option | What it does |
|---|---|
| `--storage PATH` | Where models and data live. Sticky across runs |
| `--snapshot PATH` | Where snapshots go. Sticky, independent of `--storage` |
| `--models-cache DIR` | Keep models permanently here, copy to the working dir each run |
| `--llama-bin PATH` | Path to `llama-server` (file or folder) |
| `--no-download` | Fail if models aren't already present |
| `--reasoning-model PATH` | Use a specific reasoning model file |
| `--judge-model PATH` | Use a specific judge model file |
| `--embed-model PATH` | Use a specific embedding model file |
| `--reasoning-choice N` | Pick reasoning model N from the catalog without the menu |
| `--model-menu` | Force the model selection menu |
| `--reasoning-cpu-moe` | Force `--cpu-moe` (auto-detected for MoE models) |
| `--reasoning-port N` | Reasoning server port (default: 8080) |
| `--judge-port N` | Judge server port (default: 8081) |
| `--embed-port N` | Embedding server port (default: 8082) |
| `--middleware-port N` | Middleware port (default: 8000) |
| `--skip-reasoning` / `--skip-judge` / `--skip-embed` | Don't start that server |
| `--no-tmpfs` | Use a local directory instead of tmpfs (Linux) |
| `--dry-run` | Show what would happen without doing anything |
| `--help` | Show all options |

---

## Configuration

Settings live in `cued_recall/config.yaml`. It is gitignored, because it holds machine-specific paths; `config.example.yaml` is the template, copied automatically on first run.

Values the launcher manages for you — `store_path`, `snapshot_path`, `models_dir`, ports, and `max_context_tokens` — are rewritten on every launch. Everything else is yours to edit.

| Setting | Meaning |
|---|---|
| `block_tokens_reasoning` | Max size of a reasoning block before it is split |
| `max_context_tokens` | Prompt budget, derived from the server's `--ctx-size` |
| `tokens_per_word` | Word→token estimate (1.3 for prose, higher for code/CJK) |
| `hot_shelve_timeout_s` | How long before an unanswered turn is shelved anyway |
| `recall.k` | How many blocks to recall per turn |
| `recall.threshold` | Minimum similarity to recall a block (0–1) |
| `recall.budget_tokens` | Max tokens of recalled material injected |
| `judge.interval_tokens` | Tokens between automatic judge passes |
| `judge.min_age_s` | Minimum block age before the judge reviews it |
| `judge.purge_age_s` | Minimum age before a block can be purged |
| `tagger.enabled` | Tag blocks with a gist and tags when shelved |
| `web_search.backend` | `duckduckgo`, `searxng`, `brave`, or `serper` |
| `web_search.searxng_url` | URL of a self-hosted SearXNG instance |
| `web_search.brave_api_key` | Brave Search API key (paid) |
| `correction_patterns` | Phrases that mark the previous answer as wrong |

---

## Troubleshooting

**"llama-server not found"**
Install llama.cpp (Step 4) and add the folder to your PATH, or pass `--llama-bin PATH`.

**"Python 3.11+ is required"**
Install Python 3.11 or newer from https://www.python.org/downloads/

**"Failed to download model"**
Slow connection or a busy Hugging Face server. Run again — downloads resume.

**"imdisk not found"**
Install ImDisk Toolkit, then close and reopen your terminal.

**"CUDA error: out of memory"**
Not enough VRAM for all three models. Try:
- A smaller reasoning model from the launcher menu
- `--skip-judge` (frees VRAM; tagging and compression stop working)
- Lower `--ctx-size` in `run.py`'s `SERVER_DEFAULTS`

**The client 404s when adding the middleware**
It is probably calling `GET /v1/models` during setup. That endpoint exists — check the base URL ends in `/v1`.

**The admin page shows blocks but recall never fires**
Recall only searches `shelved` and `truncated` blocks, and only returns matches above `recall.threshold` (default 0.62). A short factual statement often scores below that against a differently-worded question. Lower the threshold, or check the block's status in the admin table.

**Responses stream but appear empty in an agentic client**
Usually a model that calls tools without producing text. Try a stock or coder-tuned model.

**Port already in use**
`--reasoning-port 8090 --judge-port 8091 --embed-port 8092`

---

## License

MIT
