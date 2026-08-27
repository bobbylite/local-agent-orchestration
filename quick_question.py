#!/usr/bin/env python3
"""Quick Question — local-first Q&A with a Claude safety net.

A router picks the best local Ollama model for the request. A dedicated
local judge model reviews the answer (LLM-as-judge) for up to two rounds.
If the judge still isn't satisfied, Claude Sonnet takes over. Every stage
streams live and file writes always ask for permission first.
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import TypedDict

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule
from rich.syntax import Syntax

WORKER_MODELS = {
    # qwen2.5-coder:7b-instruct is deliberately excluded: on this Ollama build it
    # never emits proper structured tool_calls (bare JSON text instead) and then
    # loops re-emitting the same call after seeing a tool result instead of
    # synthesizing an answer, which breaks the tool-using worker/escalate loop.
    "qwen2.5:7b-instruct": "general reasoning, explanations, writing, and code",
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
_active_status = None  # the live spinner, paused by tools before they print/prompt


def _pause_spinner() -> None:
    global _active_status
    if _active_status is not None:
        _active_status.stop()
        _active_status = None


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
    except Exception as exc:
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


def extract_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    return getattr(chunk, "content", "") or ""


def run_text_stage(title: str, spinner_text: str, model, prompt: str) -> str:
    """Stream a plain (tool-free) chat completion live, typewriter-style."""
    global _active_status
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    started = time.monotonic()
    status = console.status(f"[dim]{spinner_text}…[/dim]", spinner="dots")
    status.start()
    _active_status = status
    pieces = []
    try:
        for chunk in model.stream(prompt):
            text = extract_text(chunk)
            if not text:
                continue
            if _active_status is not None:
                _pause_spinner()
            sys.stdout.write(text)
            sys.stdout.flush()
            pieces.append(text)
    finally:
        _pause_spinner()
    print()
    console.print(f"[dim]finished in {time.monotonic() - started:.1f}s[/dim]\n")
    return "".join(pieces)


def run_agent_stage(title: str, spinner_text: str, model, messages: list) -> str:
    """Run a tool-using react agent; tools pause the spinner to print/prompt live."""
    global _active_status
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    started = time.monotonic()
    status = console.status(f"[dim]{spinner_text}…[/dim]", spinner="dots")
    status.start()
    _active_status = status
    agent = create_agent(model, TOOLS)
    try:
        result = agent.invoke({"messages": messages})
    finally:
        _pause_spinner()
    answer = result["messages"][-1].content
    if not isinstance(answer, str):
        answer = str(answer)
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


def parse_verdict(text: str) -> tuple[str, str]:
    match = JUDGE_RESULT_RE.search(text.strip())
    if not match:
        # Fail open: an unparseable judge response shouldn't burn a retry round.
        return "accept", text.strip()
    score = int(match.group(1))
    feedback = text[: match.start()].strip()
    verdict = "accept" if score >= JUDGE_ACCEPT_THRESHOLD else "retry"
    return verdict, f"(score {score}/5) {feedback}"


def render_trail(trail: list[tuple[str, str]]) -> None:
    symbols = {"ok": ("[green]✓ {}[/green]"), "fail": ("[red]✗ {}[/red]"), "run": ("[yellow]… {}[/yellow]")}
    console.print("  →  ".join(symbols[state].format(label) for label, state in trail) + "\n")


class QAState(TypedDict, total=False):
    # total=False because langgraph builds this up incrementally as nodes run;
    # request/worker_model/answer/attempts/verdict are guaranteed present by the
    # time downstream nodes read them, but TypedDict has no way to express that.
    request: str
    worker_model: str
    answer: str
    feedback: str
    attempts: int
    verdict: str
    final_source: str


def answer_question(request: str) -> None:
    started = time.monotonic()
    console.print(
        Panel.fit(
            f"[bold]{request}[/bold]\n[dim]router → local worker → {JUDGE_MODEL} judge (×{MAX_LOCAL_ATTEMPTS}) → {ESCALATION_MODEL}[/dim]",
            title="❓ Quick Question",
            border_style="blue",
        )
    )

    trail: list[tuple[str, str]] = []
    router_model = ChatOllama(model=ROUTER_MODEL, base_url=OLLAMA_URL)
    judge_model = ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_URL)
    claude_model = ChatAnthropic(model_name=ESCALATION_MODEL, timeout=None, stop=None)

    def route_node(state: QAState) -> dict:
        trail.append(("Router", "run"))
        options = "\n".join(f"- {tag}: best for {desc}" for tag, desc in WORKER_MODELS.items())
        prompt = f"""Choose the single best local model to answer this request.

Request: {state['request']}

Options:
{options}

In one short sentence, explain your choice. Then on its own final line, output exactly:
MODEL: <tag>"""
        text = run_text_stage("🧭  Router", "Choosing a worker model", router_model, prompt)
        chosen = parse_route(text)
        trail[-1] = ("Router", "ok")
        return {"worker_model": chosen}

    def work_node(state: QAState) -> dict:
        model_tag = state["worker_model"]
        attempts = state.get("attempts", 0) + 1
        trail.append((f"Worker · {model_tag} ({attempts}/{MAX_LOCAL_ATTEMPTS})", "run"))
        system = SystemMessage(
            "You have read_file and write_file tools for the local filesystem. "
            "Only call write_file if the request requires creating or modifying a file; "
            "the user must approve every write. Answer as completely and correctly as you can."
        )
        human_text = state["request"]
        if state.get("feedback"):
            human_text += f"\n\nA reviewer rejected your previous attempt for this reason: {state['feedback']}\nPlease address it."
        worker_model = ChatOllama(model=model_tag, base_url=OLLAMA_URL)
        answer = run_agent_stage(
            f"💻  Worker · {model_tag}", "Working on it", worker_model, [system, HumanMessage(human_text)]
        )
        trail[-1] = (trail[-1][0], "ok")
        return {"answer": answer, "attempts": attempts}

    def judge_node(state: QAState) -> dict:
        trail.append(("Judge", "run"))
        content = JUDGE_PROMPT_TEMPLATE.format(
            instruction=state["request"], response=state["answer"], rubric=JUDGE_RUBRIC
        )
        messages = [SystemMessage(JUDGE_SYSTEM_PROMPT), HumanMessage(content)]
        text = run_text_stage(f"⚖️  Judge · {JUDGE_MODEL}", "Reviewing the answer", judge_model, messages)
        verdict, feedback = parse_verdict(text)
        trail[-1] = ("Judge", "ok" if verdict == "accept" else "fail")
        render_trail(trail)
        return {"verdict": verdict, "feedback": feedback}

    def escalate_node(state: QAState) -> dict:
        trail.append(("Escalate · Claude Sonnet", "run"))
        system = SystemMessage(
            "You have read_file and write_file tools for the local filesystem. "
            "Only call write_file if the request requires creating or modifying a file; "
            "the user must approve every write. Two local model attempts failed review; answer directly and well."
        )
        human_text = f"{state['request']}\n\n(Local model feedback from the last attempt: {state.get('feedback', 'n/a')})"
        answer = run_agent_stage(
            f"🚀  Escalate · {ESCALATION_MODEL}", "Claude is taking over", claude_model, [system, HumanMessage(human_text)]
        )
        trail[-1] = ("Escalate · Claude Sonnet", "ok")
        render_trail(trail)
        return {"answer": answer, "final_source": "claude"}

    def after_judge(state: QAState) -> str:
        if state.get("verdict") == "accept":
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
        result = app.invoke({"request": request})
    except Exception as exc:
        console.print(Panel(f"[red]Pipeline failed:[/red] {exc}\n\nIs `ollama serve` running and is ANTHROPIC_API_KEY set?", border_style="red"))
        sys.exit(1)

    source = result.get("final_source", "local")
    total_elapsed = time.monotonic() - started
    console.print(
        Panel.fit(
            f"[bold green]✓ Answered[/bold green] via [bold]{source}[/bold]  ·  {total_elapsed:.1f}s total",
            border_style="green",
        )
    )


def main() -> None:
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

    answer_question(request)


if __name__ == "__main__":
    main()
