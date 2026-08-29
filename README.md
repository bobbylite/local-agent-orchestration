**Perfect, the M1 Pro is actually ideal for this.** Let's rebuild it.

---

## Quick setup (uv) — recommended

This project uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies via `pyproject.toml` — the closest equivalent to `npm install`/`npm run` for Python. It replaces the manual venv + pip steps further down; those are kept below for reference/troubleshooting.

**Install uv (one-time, if you don't have it):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install dependencies:**
```bash
cd /path/to/quick-build
uv sync
```
This reads `pyproject.toml`, resolves versions, creates a local `.venv/`, and installs everything into it — one command, no manual `python -m venv` + `pip install` dance.

**Run the tools** (no need to activate the venv first — `uv run` does that for you):
```bash
uv run quick_build.py "async task queue with retry logic"
uv run quick_question.py "what does this project do?"
```

**Add a new dependency:**
```bash
uv add some-package
```
This updates `pyproject.toml` and `uv.lock` and installs it immediately.

**uv only manages the Python side.** You still need, separately:
- [Ollama](https://ollama.ai) installed and running (`ollama serve`), with the local models pulled — see Step 1 and Step 4 below.
- `ANTHROPIC_API_KEY` exported in your shell — see Step 5 below.

---

## Step 1: Install Ollama on macOS

```bash
# Download and install (or use the app: https://ollama.ai/download)
curl -fsSL https://ollama.ai/install.sh | sh
```

Or download the app directly from https://ollama.ai/download/mac

**Start Ollama:**
```bash
ollama serve
```

You should see: `Listening on 127.0.0.1:11434`

---

## Step 2: Create virtual environment

```bash
python3 -m venv ~/agent-env

source ~/agent-env/bin/activate
```

You should see `(agent-env)` in your prompt.

---

## Step 3: Install Python packages

```bash
pip install langchain langchain-anthropic langchain-ollama langgraph rich
```

---

## Step 4: Pull the model

**In a new terminal** (keep Ollama running in first terminal):

```bash
ollama pull qwen2.5-coder:7b-instruct
```

This downloads ~4.5GB. M1 will handle this very well.

**Test it:**
```bash
ollama run qwen2.5-coder:7b-instruct "Write a Python hello world function"
```

---

## Step 5: Set up Claude API key

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

Add to `~/.zshrc` (macOS uses zsh):
```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

---

## Step 6: quick_build.py

This lives in the repo already ([quick_build.py](quick_build.py)) — no need to hand-create it.
It's a two-stage pipeline: Claude Sonnet architects an exact spec, then a local
Qwen2.5-Coder model implements it. Both stages **stream their output live**
(tokens appear as they're generated, with a spinner while each model is still
thinking) so you always see progress instead of a frozen terminal.

Make it executable:
```bash
chmod +x quick_build.py
```

---

## Step 7: Test it

Make sure venv is activated:
```bash
source ~/agent-env/bin/activate
```

Test:
```bash
python quick_build.py "async task queue with retry logic"
```

You should see:
1. Sonnet instructions appearing
2. Qwen writing code (faster than 14B model)
3. File saved

---

## Key differences on M1 Pro:

**Advantages:**
- ✓ Qwen 7B is smaller, faster (5-8 tokens/sec vs 8-12)
- ✓ Metal GPU acceleration (Apple Silicon optimized)
- ✓ Lower power consumption
- ✓ Quieter fan (less thermal stress)

**Same cost:**
- $15-30/month Claude API
- $0 local inference

---

## Quick reference for future sessions

```bash
# Terminal 1: Keep Ollama running
ollama serve

# Terminal 2: Run your tool
source ~/agent-env/bin/activate
python quick_build.py "your request"
```

**Post back once the test works!** Then we can add the alias and README for your Mac setup.

## Custom cli tools 

Custom system tooling examples

### Execute custom python
Where build() is whatever you want the command to be.

```bash
echo 'build() {
  source /Users/bobby/source/agents/quick-build/.venv/bin/activate
  python ~/quick_build.py "$@"
}' >> ~/.zshrc

source ~/.zshrc
```

Test it:
```bash
build "async task queue with retry logic"
```

### Execute Ollama
Where ask() is whatever model you need to ask a question to.

```bash
echo 'ask() {
  ollama run qwen2.5-coder:7b-instruct "$@"
}' >> ~/.zshrc

source ~/.zshrc
```

Test it: 
```bash
ask "write a hello world function"
ask "explain recursion"
ask "what is REST API"
```
---

## Web dashboard

A live view of the agents: the pipeline graph animating stage by stage, streamed
tokens, judge scores, Ollama VRAM residency, and toast notifications for every
transition.

```bash
./launch.sh             # → http://127.0.0.1:8787
```

`launch.sh` is the front door for the whole project. It runs preflight checks
(uv, `uv sync`, Ollama reachable, API key present), frees the port if a previous
dashboard is still holding it, then starts the server:

```bash
./launch.sh                  # start the dashboard (default)
./launch.sh ask "question"   # quick_question.py, after the same preflight
./launch.sh build "a thing"  # quick_build.py
./launch.sh status           # what's running, what's resident in VRAM
./launch.sh stop             # free the dashboard port
```

It only ever kills *its own* dashboard — anything else holding the port is
reported and left alone. Set `DASHBOARD_PORT` to run somewhere else; the server
reads the same variable, so the two cannot drift apart.

`uv run dashboard.py` still works if you'd rather skip the preflight.

**It shows terminal runs too.** Every stage in `quick_question.py` and
`quick_build.py` emits structured events to a shared run log
(`~/.quick-agents/runs.jsonl`, override with `QUICK_AGENTS_HOME`). The dashboard
tails that log, so a `uv run quick_question.py "…"` in another window animates in
an already-open browser tab — tagged `cli` to distinguish it from runs you
dispatch from the page itself.

**Write approvals move to the browser.** When a worker calls `write_file` during
a dashboard-dispatched run, the diff appears in a modal with Allow / Deny instead
of prompting in the terminal. CLI runs keep the existing terminal prompt — the
approval path is chosen by whether an approver is registered, so neither front
end had to change the other's behaviour.

There is **no authentication**; the server binds to loopback only and is meant
for a single developer's machine. Don't expose it.

Files: [dashboard.py](dashboard.py) (SSE hub, Ollama poller, log tailer),
[events.py](events.py) (the event bus), [static/](static/) (the UI).
