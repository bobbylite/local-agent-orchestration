#!/usr/bin/env python3
"""Quick Question — local-first Q&A with a Claude safety net.

A router picks the best local Ollama model for the request. Every local
worker can read/write files and search the web; if the default 7b worker's
answer is rejected by the local judge (Prometheus-2), the retry is upgraded
to a 14b worker. If the judge still isn't satisfied after two rounds, Claude
Sonnet takes over. Every stage streams live and file writes always ask for
permission first.
"""

import argparse
import asyncio
import contextvars
import html
import re
import sys
import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, TypedDict

import httpx
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule
from rich.spinner import Spinner
from rich.status import Status
from rich.syntax import Syntax

import events

WORKER_MODELS: Final[dict[str, str]] = {
    # qwen2.5-coder:14b-instruct is deliberately excluded: on this Ollama build it
    # never emits proper structured tool_calls (bare JSON text instead) and then
    # loops re-emitting the same call after seeing a tool result instead of
    # synthesizing an answer, which breaks the tool-using worker/escalate loop.
    # The same goes for any other model tag whose `ollama show` capabilities
    # don't include `tools` (e.g. deepseek-coder, codestral, deepseek-coder-v2) —
    # check that before adding a tag here.
    "mistral-nemo:12b": "general and simple reasoning, explanations, writing, and code",
    "qwen2.5:14b-instruct": "more complex reasoning, explanations, writing, and code",
    "hermes3:8b": "alternate general-purpose model for open-ended questions",
}
DEFAULT_WORKER_MODEL: Final[str] = "mistral-nemo:12b"
BUMP_WORKER_MODEL: Final[str] = "qwen2.5:14b-instruct"  # used for the retry after DEFAULT_WORKER_MODEL fails judging
ROUTER_MODEL: Final[str] = "mistral-nemo:12b"
JUDGE_MODEL: Final[str] = "prometheus2"  # imported via Modelfile from prometheus-eval/prometheus-7b-v2.0-GGUF
ESCALATION_MODEL: Final[str] = "claude-sonnet-5"
OLLAMA_URL: Final[str] = "http://localhost:11434"
MAX_LOCAL_ATTEMPTS: Final[int] = 2

# Exact absolute-grading prompt from https://github.com/prometheus-eval/prometheus-eval
# (prometheus_eval/prompts.py) — Prometheus-2 was fine-tuned specifically on this format
# and is unreliable with any other judging prompt shape.
JUDGE_SYSTEM_PROMPT: Final[str] = (
    "You are a fair judge assistant tasked with providing clear, objective feedback "
    "based on specific criteria, ensuring each assessment reflects the absolute "
    "standards set for performance."
)
JUDGE_PROMPT_TEMPLATE: Final[str] = """###Task Description:
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
JUDGE_RUBRIC: Final[str] = """[Does the response correctly, completely, and helpfully satisfy the user's request?]
Score 1: The response is irrelevant, incorrect, or does not address the request at all.
Score 2: The response addresses the request but has major errors, omissions, or misunderstandings.
Score 3: The response is partially correct and helpful but has noticeable gaps or inaccuracies.
Score 4: The response correctly and mostly completely addresses the request, with only minor issues.
Score 5: The response fully, accurately, and clearly satisfies the request with no meaningful issues."""
JUDGE_ACCEPT_THRESHOLD: Final[int] = 4
READ_URL_MAX_CHARS: Final[int] = 8000

console: Final = Console()

# The spinner belongs to whichever stage is currently running. Tools (called
# deep inside an agent's tool loop) need to pause it before printing or
# prompting. A ContextVar — rather than a plain module global — keeps this
# correct if stages are ever awaited concurrently instead of strictly in
# sequence, since each async task gets its own view of the variable.
_active_status: contextvars.ContextVar[Status | Live | None] = contextvars.ContextVar("_active_status", default=None)


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


_LEXER_BY_SUFFIX: Final[dict[str, str]] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".json": "json",
    ".md": "markdown", ".sh": "bash", ".yaml": "yaml", ".yml": "yaml",
    ".html": "html", ".css": "css",
}


def _lexer_for(path: Path) -> str:
    return _LEXER_BY_SUFFIX.get(path.suffix.lower(), "text")


@tool
def read_file(path: str) -> str:
    """Read and return the text contents of a local file."""
    _pause_spinner()
    p = Path(path).expanduser()
    console.print(f"[cyan]🔧 read_file[/cyan] [dim]{p}[/dim]")
    events.emit(events.Kind.TOOL_CALL, tool="read_file", detail=str(p))
    if not p.is_file():
        return f"ERROR: no such file: {p}"
    try:
        return p.read_text()
    except OSError as exc:
        return f"ERROR reading {p}: {exc}"


@tool
async def write_file(path: str, content: str) -> str:
    """Write text content to a local file. Always asks the user for permission first."""
    _pause_spinner()
    p = Path(path).expanduser()
    console.print(Rule(f"[yellow]✍️  write requested → {p}[/yellow]", style="yellow"))
    console.print(Syntax(content, _lexer_for(p), line_numbers=True, word_wrap=True))
    events.emit(events.Kind.TOOL_CALL, tool="write_file", detail=str(p))
    # The dashboard approves from a browser click; with no approver registered
    # (i.e. a plain CLI run) fall back to the terminal prompt.
    if events.has_approver():
        events.emit(
            events.Kind.APPROVAL_REQUESTED,
            path=str(p), content=content, bytes=len(content), lexer=_lexer_for(p),
        )
        allowed = await events.request_approval(path=str(p), bytes=len(content))
        events.emit(events.Kind.APPROVAL_RESOLVED, path=str(p), allowed=allowed)
    else:
        allowed = Confirm.ask(f"[bold yellow]Allow writing {len(content)} bytes to {p}?[/bold yellow]", default=False)
    if not allowed:
        console.print("[red]✗ write denied[/red]\n")
        return "DENIED: the user did not grant permission to write this file."
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    console.print(f"[green]✓ wrote {p}[/green]\n")
    return f"OK: wrote {len(content)} bytes to {p}"


_TAG_RE: Final = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE: Final = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_SEARCH_RESULT_RE: Final = re.compile(
    r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_WEB_USER_AGENT: Final[str] = "Mozilla/5.0 (compatible; quick-question/1.0)"


def _clean_html(fragment: str) -> str:
    return html.unescape(_TAG_RE.sub("", fragment)).strip()


def _resolve_ddg_url(href: str) -> str:
    """DuckDuckGo's HTML results wrap outbound links in a /l/?uddg= redirect; unwrap it."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        uddg = urllib.parse.parse_qs(parsed.query).get("uddg")
        if uddg:
            return uddg[0]
    return href


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web (via DuckDuckGo) and return titles, URLs, and snippets."""
    _pause_spinner()
    console.print(f"[cyan]🔧 web_search[/cyan] [dim]{query}[/dim]")
    events.emit(events.Kind.TOOL_CALL, tool="web_search", detail=query)
    try:
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": _WEB_USER_AGENT},
            timeout=10.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return f"ERROR searching web: {exc}"

    matches = _SEARCH_RESULT_RE.findall(resp.text)[:max_results]
    if not matches:
        return "No results found."
    return "\n\n".join(
        f"{i}. {_clean_html(title)}\n{_resolve_ddg_url(href)}\n{_clean_html(snippet)}"
        for i, (href, title, snippet) in enumerate(matches, start=1)
    )


@tool
async def read_url(url: str, question: str = "") -> str:
    """Fetch a web page and have a separate worker model read it and return a summary.

    Use this on a link found via web_search instead of dumping raw HTML into
    context. Pass `question` to steer the summary toward what you actually need.
    """
    _pause_spinner()
    console.print(f"[cyan]🔧 read_url[/cyan] [dim]{url}[/dim]")
    events.emit(events.Kind.TOOL_CALL, tool="read_url", detail=url)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url, headers={"User-Agent": _WEB_USER_AGENT})
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return f"ERROR fetching {url}: {exc}"

    text = _clean_html(_SCRIPT_STYLE_RE.sub("", resp.text))[:READ_URL_MAX_CHARS]
    if not text:
        return f"ERROR: no readable text found at {url}"

    focus = f" Focus especially on anything relevant to: {question}" if question else ""
    console.print(f"[dim]… reader agent ({DEFAULT_WORKER_MODEL}) is summarizing the page[/dim]")
    reader_model = ChatOllama(model=DEFAULT_WORKER_MODEL, base_url=OLLAMA_URL)
    summary = await reader_model.ainvoke(
        f"Read the following webpage content and write a concise, factual summary of it.{focus}\n\n{text}"
    )
    return f"Summary of {url}:\n{extract_text(summary)}"


TOOLS: Final[list[BaseTool]] = [read_file, write_file, web_search, read_url]
# Currently identical to TOOLS (every worker already has web access) — kept as
# its own name so the model bump can be narrowed back to gating search tools
# behind it without touching work()'s tool-selection logic.
SEARCH_TOOLS: Final[list[BaseTool]] = TOOLS


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


async def run_text_stage(
    title: str, spinner_text: str, model: BaseChatModel, prompt: str | Sequence[BaseMessage], stage: str = ""
) -> str:
    """Stream a plain (tool-free) chat completion live, rendering it as Markdown as tokens arrive."""
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    started = time.monotonic()
    events.set_stage(stage)
    events.emit(events.Kind.STAGE_STARTED, title=title, model=getattr(model, "model", None) or getattr(model, "model_name", ""))
    pieces: list[str] = []
    live = Live(
        Spinner("dots", text=f"[dim]{spinner_text}…[/dim]"),
        console=console,
        refresh_per_second=12,
        vertical_overflow="visible",
    )
    live.start()
    token = _active_status.set(live)
    try:
        async for chunk in model.astream(prompt):
            text = extract_text(chunk)
            if not text:
                continue
            pieces.append(text)
            events.emit_token(text)
            if _active_status.get() is not None:
                live.update(Markdown("".join(pieces)))
    finally:
        _pause_spinner()
        _active_status.reset(token)
    elapsed = time.monotonic() - started
    events.emit(events.Kind.STAGE_FINISHED, title=title, seconds=round(elapsed, 2), chars=sum(map(len, pieces)))
    console.print(f"[dim]finished in {elapsed:.1f}s[/dim]\n")
    return "".join(pieces)


async def run_agent_stage(
    title: str, spinner_text: str, model: BaseChatModel, messages: Sequence[AnyMessage],
    tools: Sequence[BaseTool] = TOOLS, stage: str = "",
) -> str:
    """Run a tool-using react agent; tools pause the spinner to print/prompt live."""
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    started = time.monotonic()
    events.set_stage(stage)
    events.emit(events.Kind.STAGE_STARTED, title=title, model=getattr(model, "model", None) or getattr(model, "model_name", ""))
    status = console.status(f"[dim]{spinner_text}…[/dim]", spinner="dots")
    status.start()
    token = _active_status.set(status)
    agent = create_agent(model, tools)
    try:
        result = await agent.ainvoke({"messages": list(messages)})
    finally:
        _pause_spinner()
        _active_status.reset(token)
    answer = extract_text(result["messages"][-1])
    if answer.strip():
        console.print(Markdown(answer))
        events.emit_token(answer)
    elapsed = time.monotonic() - started
    events.emit(events.Kind.STAGE_FINISHED, title=title, seconds=round(elapsed, 2), chars=len(answer))
    console.print(f"[dim]finished in {elapsed:.1f}s[/dim]\n")
    return answer


_ROUTE_RE: Final = re.compile(r"MODEL:\s*([A-Za-z0-9_.:\-]+)", re.IGNORECASE)
# Same pattern prometheus-eval's own parser uses (prometheus_eval/parser.py),
# tolerant of "[RESULT] N", "Score: N", "N out of 5", "N/5", etc.
_JUDGE_RESULT_RE: Final = re.compile(
    r"(?:\[RESULT\]|\[SCORE\]|Score:?|score:?|Result:?|\[Result\]:?|score\s+of)"
    r"\s*(?:\(|\[|\s)*(\d+)(?:(?:\)|\]|\s|$)|(?:/\s*5|\s*out\s*of\s*5))?\s*$",
    re.IGNORECASE,
)


def parse_route(text: str) -> str:
    last = None
    for last in _ROUTE_RE.finditer(text):
        pass
    if last:
        candidate = last.group(1).strip()
        for tag in WORKER_MODELS:
            if candidate.lower() == tag.lower():
                return tag
    return DEFAULT_WORKER_MODEL


def parse_verdict(text: str) -> tuple[Verdict, str, int | None]:
    """Return the verdict, the feedback to hand back to a retrying worker, and the raw score.

    The score is None when the judge's reply had no parseable result line.
    """
    match = _JUDGE_RESULT_RE.search(text.strip())
    if not match:
        # Fail open: an unparseable judge response shouldn't burn a retry round.
        return Verdict.ACCEPT, text.strip(), None
    score = int(match.group(1))
    feedback = text[: match.start()].strip()
    verdict = Verdict.ACCEPT if score >= JUDGE_ACCEPT_THRESHOLD else Verdict.RETRY
    return verdict, f"(score {score}/5) {feedback}", score


def render_trail(trail: list[TrailStep]) -> None:
    console.print("  →  ".join(step.render() for step in trail) + "\n")


class QAState(TypedDict):
    # Every field ships with a placeholder default in the initial ainvoke() call
    # below, so the dict is genuinely total=True from the start — nodes just
    # overwrite placeholders as the graph progresses, rather than introducing
    # keys that show up partway through.
    request: str
    worker_model: str
    answer: str
    feedback: str
    attempts: int
    verdict: Verdict
    final_source: FinalSource


class RouteUpdate(TypedDict):
    worker_model: str


class WorkUpdate(TypedDict):
    answer: str
    attempts: int
    worker_model: str


class JudgeUpdate(TypedDict):
    verdict: Verdict
    feedback: str


class EscalateUpdate(TypedDict):
    answer: str
    final_source: FinalSource


@dataclass(slots=True)
class QuestionPipeline:
    """Owns the per-question model instances and progress trail for one run."""

    router_model: BaseChatModel
    judge_model: BaseChatModel
    claude_model: BaseChatModel
    history: Sequence[tuple[str, str]] = ()
    trail: list[TrailStep] = field(default_factory=list)

    @classmethod
    def create(cls, history: Sequence[tuple[str, str]] = ()) -> "QuestionPipeline":
        return cls(
            router_model=ChatOllama(model=ROUTER_MODEL, base_url=OLLAMA_URL),
            judge_model=ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_URL),
            claude_model=ChatAnthropic(model_name=ESCALATION_MODEL, timeout=None, stop=None),
            history=history,
        )

    def _context_block(self) -> str:
        """Prior turns of this conversation, formatted as a prompt prefix (empty on turn one)."""
        if not self.history:
            return ""
        turns = "\n\n".join(f"Q: {question}\nA: {answer}" for question, answer in self.history)
        return f"Earlier in this conversation:\n{turns}\n\n"

    async def route(self, state: QAState) -> RouteUpdate:
        self.trail.append(TrailStep("Router", StageState.RUNNING))
        options = "\n".join(f"- {tag}: best for {desc}" for tag, desc in WORKER_MODELS.items())
        prompt = f"""Choose the single best local model to answer this request.

{self._context_block()}Request: {state['request']}

Options:
{options}

In one short sentence, explain your choice. Then on its own final line, output exactly:
MODEL: <tag>"""
        text = await run_text_stage("🧭  Orchestrating", "Choosing a worker model", self.router_model, prompt, stage="route")
        chosen = parse_route(text)
        self.trail[-1] = TrailStep("Router", StageState.OK)
        events.emit(events.Kind.STAGE_FINISHED, stage="route", chose=chosen)
        return {"worker_model": chosen}

    async def work(self, state: QAState) -> WorkUpdate:
        model_tag = state["worker_model"]
        attempts = state["attempts"] + 1
        bumped = attempts > 1 and model_tag == DEFAULT_WORKER_MODEL
        if bumped:
            model_tag = BUMP_WORKER_MODEL
        tools = SEARCH_TOOLS if bumped else TOOLS

        self.trail.append(TrailStep(f"Worker · {model_tag} ({attempts}/{MAX_LOCAL_ATTEMPTS})", StageState.RUNNING))
        system = SystemMessage(
            "You have read_file and write_file tools for the local filesystem"
            + (
                ", a web_search tool for finding pages online, and a read_url tool that fetches "
                "a page and has another worker model summarize it for you. If the request compares "
                "multiple named things (e.g. versions, products, options), call web_search once for "
                "each one individually before writing the comparison — do not rely on memory alone "
                "for any of them, even ones you feel confident about, since your recollection of one "
                "side can be as stale or wrong as the other — then use read_url to confirm details "
                "from the most relevant result"
                if bumped
                else ""
            )
            + ". Only call write_file if the request requires creating or modifying a file; "
            "the user must approve every write. Answer as completely and correctly as you can."
        )
        human_text = f"{self._context_block()}{state['request']}"
        if state["feedback"]:
            human_text += f"\n\nA reviewer rejected your previous attempt for this reason: {state['feedback']}\nPlease address it."
        worker_model = ChatOllama(model=model_tag, base_url=OLLAMA_URL)
        events.emit(events.Kind.STAGE_STARTED, stage="work", model=model_tag, attempt=attempts, bumped=bumped)
        answer = await run_agent_stage(
            f"💻  Worker · {model_tag}", "Working on it", worker_model, [system, HumanMessage(human_text)], tools, stage="work"
        )
        self.trail[-1] = TrailStep(self.trail[-1].label, StageState.OK)
        return {"answer": answer, "attempts": attempts, "worker_model": model_tag}

    async def judge(self, state: QAState) -> JudgeUpdate:
        self.trail.append(TrailStep("Judge", StageState.RUNNING))
        content = JUDGE_PROMPT_TEMPLATE.format(
            instruction=state["request"], response=state["answer"], rubric=JUDGE_RUBRIC
        )
        messages = [SystemMessage(JUDGE_SYSTEM_PROMPT), HumanMessage(content)]
        text = await run_text_stage(f"⚖️  Judge · {JUDGE_MODEL}", "Reviewing the answer", self.judge_model, messages, stage="judge")
        verdict, feedback, score = parse_verdict(text)
        self.trail[-1] = TrailStep("Judge", StageState.OK if verdict is Verdict.ACCEPT else StageState.FAILED)
        events.emit(
            events.Kind.VERDICT, stage="judge", verdict=str(verdict), score=score,
            threshold=JUDGE_ACCEPT_THRESHOLD, feedback=feedback,
        )
        render_trail(self.trail)
        return {"verdict": verdict, "feedback": feedback}

    async def escalate(self, state: QAState) -> EscalateUpdate:
        self.trail.append(TrailStep("Escalate · Claude Sonnet", StageState.RUNNING))
        system = SystemMessage(
            "You have read_file and write_file tools for the local filesystem. "
            "Only call write_file if the request requires creating or modifying a file; "
            "the user must approve every write. Two local model attempts failed review; answer directly and well."
        )
        human_text = f"{self._context_block()}{state['request']}\n\n(Local model feedback from the last attempt: {state['feedback']})"
        answer = await run_agent_stage(
            f"🚀  Escalate · {ESCALATION_MODEL}", "Claude is taking over", self.claude_model,
            [system, HumanMessage(human_text)], stage="escalate",
        )
        self.trail[-1] = TrailStep("Escalate · Claude Sonnet", StageState.OK)
        render_trail(self.trail)
        return {"answer": answer, "final_source": FinalSource.CLAUDE}

    @staticmethod
    def after_judge(state: QAState) -> str:
        if state["verdict"] is Verdict.ACCEPT:
            return "end"
        if state["attempts"] >= MAX_LOCAL_ATTEMPTS:
            return "escalate"
        return "retry"

    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(QAState)
        graph.add_node("route", self.route)
        graph.add_node("work", self.work)
        graph.add_node("judge", self.judge)
        graph.add_node("escalate", self.escalate)
        graph.add_edge(START, "route")
        graph.add_edge("route", "work")
        graph.add_edge("work", "judge")
        graph.add_conditional_edges("judge", self.after_judge, {"end": END, "retry": "work", "escalate": "escalate"})
        graph.add_edge("escalate", END)
        return graph.compile()


async def run_pipeline(request: str, history: Sequence[tuple[str, str]] = ()) -> QAState:
    """Drive one question end to end and return the final state.

    Opens a run on the event bus, emits the closing event either way, and lets
    failures propagate so each front end can present them its own way.
    """
    started = time.monotonic()
    events.new_run(request, tool="quick_question", pipeline=["route", "work", "judge", "escalate"])

    pipeline = QuestionPipeline.create(history)
    app = pipeline.build_graph()
    initial_state: QAState = {
        "request": request,
        "worker_model": "",
        "answer": "",
        "feedback": "",
        "attempts": 0,
        "verdict": Verdict.RETRY,
        "final_source": FinalSource.LOCAL,
    }
    try:
        result: QAState = await app.ainvoke(initial_state)
    except Exception as exc:
        events.emit(events.Kind.RUN_FAILED, error=str(exc))
        raise
    events.emit(
        events.Kind.RUN_FINISHED, source=str(result["final_source"]), seconds=round(time.monotonic() - started, 2),
        worker_model=result["worker_model"], attempts=result["attempts"], answer=result["answer"],
    )
    return result


async def answer_question(request: str, history: Sequence[tuple[str, str]] = ()) -> str:
    started = time.monotonic()
    console.print(
        Panel.fit(
            f"[bold]{request}[/bold]\n[dim]{DEFAULT_WORKER_MODEL} Orchestrator → local worker → {JUDGE_MODEL} judge (×{MAX_LOCAL_ATTEMPTS}) → {ESCALATION_MODEL}[/dim]",
            title="❓ Quick Question",
            border_style="blue",
        )
    )
    try:
        result = await run_pipeline(request, history)
    except Exception as exc:
        console.print(Panel(f"[red]Pipeline failed:[/red] {exc}\n\nIs `ollama serve` running and is ANTHROPIC_API_KEY set?", border_style="red"))
        sys.exit(1)

    console.print(
        Panel.fit(
            f"[bold green]✓ Answered[/bold green] via [bold]{result['final_source']}[/bold]"
            f"  ·  {time.monotonic() - started:.1f}s total",
            border_style="green",
        )
    )
    return result["answer"]


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

    history: list[tuple[str, str]] = []
    while request:
        answer = await answer_question(request, history)
        history.append((request, answer))
        try:
            request = console.input("[bold cyan]Follow-up question[/bold cyan] [dim](blank to quit)[/dim]: ").strip()
        except EOFError:
            break


if __name__ == "__main__":
    asyncio.run(main())
