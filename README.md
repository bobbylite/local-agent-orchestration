**Perfect, the M1 Pro is actually ideal for this.** Let's rebuild it.

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
Where build() is whatever you want the command to be.

```bash
echo 'build() {
  source ~/agent-env/bin/activate
  python ~/quick_build.py "$@"
}' >> ~/.zshrc

source ~/.zshrc
```

Test it:
```bash
build "async task queue with retry logic"
```