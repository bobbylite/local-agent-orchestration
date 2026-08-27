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
pip install langchain langchain-anthropic langchain-ollama langgraph
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

## Step 6: Create quick_build.py

Create this file in your project directory:

```bash
cat > quick_build.py << 'EOF'
#!/usr/bin/env python3
import sys
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM

sonnet = ChatAnthropic(model="claude-sonnet-5")
qwen = OllamaLLM(model="qwen2.5-coder:7b-instruct", base_url="http://localhost:11434")

def build(request: str):
    """
    1. Ask Sonnet for exact instructions
    2. Give those instructions to Qwen
    3. Qwen writes code
    4. Save to file
    """
    
    print("\n📋 Asking Sonnet for instructions...\n")
    
    # Step 1: Sonnet tells Qwen exactly what to do
    instructions = sonnet.invoke(f"""
You are instructing another AI to write code. Be EXTREMELY specific about:
- Exact function/class names
- Exact parameter names and types
- Exact behavior and edge cases
- Exact imports needed
- Exact output format

Request: {request}

Give your instructions in this format:
INSTRUCTIONS:
[your detailed instructions]

EXAMPLE CODE STRUCTURE:
[show the exact structure/outline]
""")
    
    print(instructions)
    print("\n" + "=" * 60)
    print("💻 Qwen2.5-Coder writing code...\n")
    
    # Step 2: Qwen follows Sonnet's exact instructions
    code = qwen.invoke(f"""Follow these EXACT instructions from an architect:

{instructions}

Now write the complete, production-ready code. Nothing else. Just code.""")
    
    print(code)
    
    # Step 3: Save to file
    filename = request.split()[0].lower() + ".py"
    with open(filename, "w") as f:
        f.write(code)
    
    print("\n" + "=" * 60)
    print(f"✓ Saved to {filename}")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        request = input("What do you want to build? ")
    else:
        request = " ".join(sys.argv[1:])
    
    build(request)
EOF
```

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