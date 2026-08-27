#!/usr/bin/env python3
"""Quick Build — a two-stage AI pair programmer.

Claude Sonnet plays architect and writes an exact spec; a local
Qwen2.5-Coder model (via Ollama) implements it. Both stages stream live
so you're watching output get rendered, never staring at a blank terminal.
"""

import argparse
import asyncio
import re
import sys
import time
from collections.abc import AsyncIterable
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner

ARCHITECT_MODEL = "claude-sonnet-5"
BUILDER_MODEL = "qwen2.5-coder:7b-instruct"
OLLAMA_URL = "http://localhost:11434"

console = Console()
sonnet = ChatAnthropic(model_name=ARCHITECT_MODEL, timeout=None, stop=None)
qwen = OllamaLLM(model=BUILDER_MODEL, base_url=OLLAMA_URL)


def slugify(request: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", request.lower())
    stem = "_".join(words[:4]) or "build_output"
    return f"{stem}.py"


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip() + "\n"


def extract_text(chunk: object) -> str:
    """Pull plain text out of a stream chunk.

    Ollama chunks carry plain-string content; Anthropic chunks sometimes carry
    a list of content blocks (e.g. [{"type": "text", "text": "..."}]) instead.
    """
    content = chunk if isinstance(chunk, str) else getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
        )
    return ""


async def run_stage(title: str, spinner_text: str, stream: AsyncIterable[object]) -> str:
    """Consume a streaming response, rendering it live as Markdown as tokens arrive."""
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    started = time.monotonic()
    pieces: list[str] = []
    with Live(
        Spinner("dots", text=f"[dim]{spinner_text}…[/dim]"),
        console=console,
        refresh_per_second=12,
        vertical_overflow="visible",
    ) as live:
        async for chunk in stream:
            text = extract_text(chunk)
            if not text:
                continue
            pieces.append(text)
            live.update(Markdown("".join(pieces)))
    elapsed = time.monotonic() - started
    console.print(f"[dim]finished in {elapsed:.1f}s[/dim]\n")
    return "".join(pieces)


async def build(request: str, output: str | None = None) -> None:
    started = time.monotonic()
    console.print(
        Panel.fit(
            f"[bold]{request}[/bold]\n[dim]{ARCHITECT_MODEL} → {BUILDER_MODEL}[/dim]",
            title="🧠 Quick Build",
            border_style="magenta",
        )
    )

    architect_prompt = f"""You are instructing another AI to write code. Be EXTREMELY specific about:
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
[show the exact structure/outline]"""

    try:
        instructions = (
            await run_stage(
                "📋  Architect · Claude Sonnet 5",
                "Thinking through the design",
                sonnet.astream(architect_prompt),
            )
        ).strip()
    except Exception as exc:
        console.print(Panel(f"[red]Claude request failed:[/red] {exc}\n\nCheck that ANTHROPIC_API_KEY is set.", border_style="red"))
        sys.exit(1)

    if not instructions:
        console.print("[red]Architect returned no output — aborting.[/red]")
        sys.exit(1)

    builder_prompt = f"""Follow these EXACT instructions from an architect:

{instructions}

Now write the complete, production-ready code. Output ONLY the code, no prose, no explanation."""

    try:
        raw_code = await run_stage(
            "💻  Builder · Qwen2.5-Coder",
            "Writing code",
            qwen.astream(builder_prompt),
        )
    except Exception as exc:
        console.print(Panel(f"[red]Ollama request failed:[/red] {exc}\n\nIs `ollama serve` running?", border_style="red"))
        sys.exit(1)

    code = strip_code_fence(raw_code)
    if not code.strip():
        console.print("[red]Builder returned no code — aborting.[/red]")
        sys.exit(1)

    path = Path(output or slugify(request))
    path.write_text(code)

    total_elapsed = time.monotonic() - started
    line_count = code.count("\n")
    console.print(
        Panel.fit(
            f"[bold green]✓ Saved[/bold green] [bold]{path}[/bold]  ·  {line_count} lines  ·  {total_elapsed:.1f}s total",
            border_style="green",
        )
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="build",
        description="Two-stage AI pair programmer: Claude architects, Qwen builds.",
    )
    parser.add_argument("request", nargs="*", help="what you want to build")
    parser.add_argument("-o", "--output", help="output filename (default: derived from the request)")
    args = parser.parse_args()

    request = " ".join(args.request).strip() if args.request else console.input("[bold cyan]What do you want to build?[/bold cyan] ").strip()
    if not request:
        console.print("[red]No request given.[/red]")
        sys.exit(1)

    await build(request, output=args.output)


if __name__ == "__main__":
    asyncio.run(main())
