#!/usr/bin/env python3
"""Quick Question — local-first Q&A with a Claude safety net.

A router picks the best local Ollama model for the request. A dedicated
local judge model (Prometheus-2) reviews the answer for up to two rounds.
If the judge still isn't satisfied, Claude Sonnet takes over. Every stage
streams live and file writes always ask for permission first.
"""

import argparse
import asyncio
import contextvars
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule
from rich.status import Status
from rich.syntax import Syntax

WORKER_MODELS = {
    # qwen2.5-coder:7b-instruct is deliberately excluded: on this Ollama build it
    # never emits proper structured tool_calls (bare JSON text instead) and then
    # loops re-emitting the same call after seeing a tool result instead of
    # synthesizing an answer, which breaks the tool-using worker/escalate loop.
    "qwen2.5:7b-instruct": "general and simple reasoning, explanations, writing, and code",
    "qwen2.5:14b-instruct": "more complex reasoning, explanations, writing, and code",
    "llama3.1:latest": "alternate general-purpose model for open-ended questions",
}
DEFAULT_WORKER_MODEL = "qwen2.5:7b-instruct"
ROUTER_MODEL = "qwen2.5:7b-instruct"
JUDGE_MODEL = "prometheus2"  # imported via Modelfile from prometheus-eval/prometheus-7b-v2.0-GGUF
ESCALATION_MODEL = "claude-sonnet-5"
OLLAMA_URL = "http://localhost:11434"
MAX_LOCAL_ATTEMPTS = 2

# Exact absolute-grading prompt from https://github.com/prometheus-eval/prometheus-eval
# (prometheus_eval/prompts.py) — Prometheus-2 was fine-tuned specifically on this format
# and is unreliable with any other judging prompt shape.
JUDGE_SYSTEM_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, objective feedback "
    "based on specific criteria, ensuring each assessment reflects the absolute "
    "standards set for performance."
)
JUDGE_PROMPT_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """
JUDGE_RUBRIC = """[Does the response correctly, completely, and helpfully satisfy the user's request?]
Score 1: The response is irrelevant, incorrect, or does not address the request at all.
Score 2: The response addresses the request but has major errors, omissions, or misunderstandings.
Score 3: The response is partially correct and helpful but has noticeable gaps or inaccuracies.
Score 4: The response correctly and mostly completely addresses the request, with only minor issues.
Score 5: The response fully, accurately, and clearly satisfies the request with no meaningful issues."""
JUDGE_ACCEPT_THRESHOLD = 4

console = Console()

# The spinner belongs to whichever stage is currently running. Tools (called
# deep inside an agent's tool loop) need to pause it before printing or
# prompting. A ContextVar — rather than a plain module global — keeps this
# correct if stages are ever awaited concurrently instead of strictly in
# sequence, since each async task gets its own view of the variable.
_active_status: contextvars.ContextVar[Status | None] = contextvars.ContextVar("_active_status", default=None)


def _pause_spinner() -> None:
    status = _active_status.get()
    if status is not None:
        status.stop()
        _active_status.set(None)


class Verdict(StrEnum):
    ACCEPT = "accept"
    RETRY = "retry"


class StageState(StrEnum):
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"


class FinalSource(StrEnum):
    LOCAL = "local"
    CLAUDE = "claude"


@dataclass(slots=True)
class TrailStep:
    label: str
    state: StageState

    def render(self) -> str:
        glyph = {StageState.OK: "[green]✓ {}[/green]", StageState.FAILED: "[red]✗ {}[/red]", StageState.RUNNING: "[yellow]… {}[/yellow]"}
        return glyph[self.state].format(self.label)


def _lexer_for(path: Path) -> str:
    return {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".json": "json",
        ".md": "markdown", ".sh": "bash", ".yaml": "yaml", ".yml": "yaml",
        ".html": "html", ".css": "css",
    }.get(path.suffix.lower(), "text")


@tool
def read_file(path: str) -> str:
    """Read and return the text contents of a local file."""
    _pause_spinner()
    p = Path(path).expanduser()
    console.print(f"[cyan]🔧 read_file[/cyan] [dim]{p}[/dim]")
    if not p.is_file():
        return f"ERROR: no such file: {p}"
    try:
        return p.read_text()
    except OSError as exc:
        return f"ERROR reading {p}: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """Write text content to a local file. Always asks the user for permission first."""
    _pause_spinner()
    p = Path(path).expanduser()
    console.print(Rule(f"[yellow]✍️  write requested → {p}[/yellow]", style="yellow"))
    console.print(Syntax(content, _lexer_for(p), line_numbers=True, word_wrap=True))
    if not Confirm.ask(f"[bold yellow]Allow writing {len(content)} bytes to {p}?[/bold yellow]", default=False):
        console.print("[red]✗ write denied[/red]\n")
        return "DENIED: the user did not grant permission to write this file."
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    console.print(f"[green]✓ wrote {p}[/green]\n")
    return f"OK: wrote {len(content)} bytes to {p}"


TOOLS = [read_file, write_file]


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


async def run_text_stage(title: str, spinner_text: str, model: BaseChatModel, prompt: str | Sequence[BaseMessage]) -> str:
    """Stream a plain (tool-free) chat completion live, typewriter-style."""
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    started = time.monotonic()
    status = console.status(f"[dim]{spinner_text}…[/dim]", spinner="dots")
    status.start()
    token = _active_status.set(status)
    pieces: list[str] = []
    try:
        async for chunk in model.astream(prompt):
            text = extract_text(chunk)
            if not text:
                continue
            if _active_status.get() is not None:
                _pause_spinner()
            sys.stdout.write(text)
            sys.stdout.flush()
            pieces.append(text)
    finally:
        _pause_spinner()
        _active_status.reset(token)
    print()
    console.print(f"[dim]finished in {time.monotonic() - started:.1f}s[/dim]\n")
    return "".join(pieces)


async def run_agent_stage(title: str, spinner_text: str, model: BaseChatModel, messages: Sequence[BaseMessage]) -> str:
    """Run a tool-using react agent; tools pause the spinner to print/prompt live."""
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    started = time.monotonic()
    status = console.status(f"[dim]{spinner_text}…[/dim]", spinner="dots")
    status.start()
    token = _active_status.set(status)
    agent = create_agent(model, TOOLS)
    try:
        result = await agent.ainvoke({"messages": list(messages)})
    finally:
        _pause_spinner()
        _active_status.reset(token)
    answer = extract_text(result["messages"][-1])
    console.print(answer)
    console.print(f"[dim]finished in {time.monotonic() - started:.1f}s[/dim]\n")
    return answer


ROUTE_RE = re.compile(r"MODEL:\s*([A-Za-z0-9_.:\-]+)", re.IGNORECASE)
# Same pattern prometheus-eval's own parser uses (prometheus_eval/parser.py),
# tolerant of "[RESULT] N", "Score: N", "N out of 5", "N/5", etc.
JUDGE_RESULT_RE = re.compile(
    r"(?:\[RESULT\]|\[SCORE\]|Score:?|score:?|Result:?|\[Result\]:?|score\s+of)"
    r"\s*(?:\(|\[|\s)*(\d+)(?:(?:\)|\]|\s|$)|(?:/\s*5|\s*out\s*of\s*5))?\s*$",
    re.IGNORECASE,
)


def parse_route(text: str) -> str:
    last = None
    for last in ROUTE_RE.finditer(text):
        pass
    if last:
        candidate = last.group(1).strip()
        for tag in WORKER_MODELS:
            if candidate.lower() == tag.lower():
                return tag
    return DEFAULT_WORKER_MODEL


def parse_verdict(text: str) -> tuple[Verdict, str]:
    match = JUDGE_RESULT_RE.search(text.strip())
    if not match:
        # Fail open: an unparseable judge response shouldn't burn a retry round.
        return Verdict.ACCEPT, text.strip()
    score = int(match.group(1))
    feedback = text[: match.start()].strip()
    verdict = Verdict.ACCEPT if score >= JUDGE_ACCEPT_THRESHOLD else Verdict.RETRY
    return verdict, f"(score {score}/5) {feedback}"


def render_trail(trail: list[TrailStep]) -> None:
    console.print("  →  ".join(step.render() for step in trail) + "\n")


class QAState(TypedDict, total=False):
    # total=False because langgraph builds this up incrementally as nodes run;
    # request/worker_model/answer/attempts/verdict are guaranteed present by the
    # time downstream nodes read them, but TypedDict has no way to express that.
    request: str
    worker_model: str
    answer: str
    feedback: str
    attempts: int
    verdict: Verdict
    final_source: FinalSource


async def answer_question(request: str) -> None:
    started = time.monotonic()
    console.print(
        Panel.fit(
            f"[bold]{request}[/bold]\n[dim]router → local worker → {JUDGE_MODEL} judge (×{MAX_LOCAL_ATTEMPTS}) → {ESCALATION_MODEL}[/dim]",
            title="❓ Quick Question",
            border_style="blue",
        )
    )

    trail: list[TrailStep] = []
    router_model = ChatOllama(model=ROUTER_MODEL, base_url=OLLAMA_URL)
    judge_model = ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_URL)
    claude_model = ChatAnthropic(model_name=ESCALATION_MODEL, timeout=None, stop=None)

    async def route_node(state: QAState) -> dict:
        trail.append(TrailStep("Router", StageState.RUNNING))
        options = "\n".join(f"- {tag}: best for {desc}" for tag, desc in WORKER_MODELS.items())
        prompt = f"""Choose the single best local model to answer this request.

Request: {state['request']}

Options:
{options}

In one short sentence, explain your choice. Then on its own final line, output exactly:
MODEL: <tag>"""
        text = await run_text_stage("🧭  Router", "Choosing a worker model", router_model, prompt)
        chosen = parse_route(text)
        trail[-1] = TrailStep("Router", StageState.OK)
        return {"worker_model": chosen}

    async def work_node(state: QAState) -> dict:
        model_tag = state["worker_model"]
        attempts = state.get("attempts", 0) + 1
        trail.append(TrailStep(f"Worker · {model_tag} ({attempts}/{MAX_LOCAL_ATTEMPTS})", StageState.RUNNING))
        system = SystemMessage(
            "You have read_file and write_file tools for the local filesystem. "
            "Only call write_file if the request requires creating or modifying a file; "
            "the user must approve every write. Answer as completely and correctly as you can."
        )
        human_text = state["request"]
        if state.get("feedback"):
            human_text += f"\n\nA reviewer rejected your previous attempt for this reason: {state['feedback']}\nPlease address it."
        worker_model = ChatOllama(model=model_tag, base_url=OLLAMA_URL)
        answer = await run_agent_stage(
            f"💻  Worker · {model_tag}", "Working on it", worker_model, [system, HumanMessage(human_text)]
        )
        trail[-1] = TrailStep(trail[-1].label, StageState.OK)
        return {"answer": answer, "attempts": attempts}

    async def judge_node(state: QAState) -> dict:
        trail.append(TrailStep("Judge", StageState.RUNNING))
        content = JUDGE_PROMPT_TEMPLATE.format(
            instruction=state["request"], response=state["answer"], rubric=JUDGE_RUBRIC
        )
        messages = [SystemMessage(JUDGE_SYSTEM_PROMPT), HumanMessage(content)]
        text = await run_text_stage(f"⚖️  Judge · {JUDGE_MODEL}", "Reviewing the answer", judge_model, messages)
        verdict, feedback = parse_verdict(text)
        trail[-1] = TrailStep("Judge", StageState.OK if verdict is Verdict.ACCEPT else StageState.FAILED)
        render_trail(trail)
        return {"verdict": verdict, "feedback": feedback}

    async def escalate_node(state: QAState) -> dict:
        trail.append(TrailStep("Escalate · Claude Sonnet", StageState.RUNNING))
        system = SystemMessage(
            "You have read_file and write_file tools for the local filesystem. "
            "Only call write_file if the request requires creating or modifying a file; "
            "the user must approve every write. Two local model attempts failed review; answer directly and well."
        )
        human_text = f"{state['request']}\n\n(Local model feedback from the last attempt: {state.get('feedback', 'n/a')})"
        answer = await run_agent_stage(
            f"🚀  Escalate · {ESCALATION_MODEL}", "Claude is taking over", claude_model, [system, HumanMessage(human_text)]
        )
        trail[-1] = TrailStep("Escalate · Claude Sonnet", StageState.OK)
        render_trail(trail)
        return {"answer": answer, "final_source": FinalSource.CLAUDE}

    def after_judge(state: QAState) -> str:
        if state.get("verdict") is Verdict.ACCEPT:
            return "end"
        if state.get("attempts", 0) >= MAX_LOCAL_ATTEMPTS:
            return "escalate"
        return "retry"

    graph = StateGraph(QAState)
    graph.add_node("route", route_node)
    graph.add_node("work", work_node)
    graph.add_node("judge", judge_node)
    graph.add_node("escalate", escalate_node)
    graph.add_edge(START, "route")
    graph.add_edge("route", "work")
    graph.add_edge("work", "judge")
    graph.add_conditional_edges("judge", after_judge, {"end": END, "retry": "work", "escalate": "escalate"})
    graph.add_edge("escalate", END)
    app = graph.compile()

    try:
        result = await app.ainvoke({"request": request})
    except Exception as exc:
        console.print(Panel(f"[red]Pipeline failed:[/red] {exc}\n\nIs `ollama serve` running and is ANTHROPIC_API_KEY set?", border_style="red"))
        sys.exit(1)

    source = result.get("final_source", FinalSource.LOCAL)
    total_elapsed = time.monotonic() - started
    console.print(
        Panel.fit(
            f"[bold green]✓ Answered[/bold green] via [bold]{source}[/bold]  ·  {total_elapsed:.1f}s total",
            border_style="green",
        )
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="question",
        description="Local-first Q&A: router → local worker → local judge (x2) → Claude fallback.",
    )
    parser.add_argument("request", nargs="*", help="what you want to ask")
    args = parser.parse_args()

    request = " ".join(args.request).strip() if args.request else console.input("[bold cyan]What's your question?[/bold cyan] ").strip()
    if not request:
        console.print("[red]No question given.[/red]")
        sys.exit(1)

    await answer_question(request)


if __name__ == "__main__":
    asyncio.run(main())
