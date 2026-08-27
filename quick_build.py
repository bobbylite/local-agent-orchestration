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
