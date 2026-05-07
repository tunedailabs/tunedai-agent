#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.prompt import Prompt
from rich import box

from .config import get_config, BACKENDS
from .agent import Agent
from .stratum import StratumLoop
from . import tools as T

console = Console()

BANNER = """[bold cyan]TunedAI Agent[/bold cyan] [dim]— reasoning-native coding agent[/dim]
[dim]Type your task. 'exit' to quit. 'tasks' to see task list.[/dim]
"""

ASK_SYSTEM = """\
You are a reasoning-native AI. Before answering, think through the problem carefully.

Output your thinking in this format first:

THINKING
> [what the question is really asking]
> [key considerations or causal factors]
> [what evidence or logic applies]
> [your conclusion and confidence]

Then give your answer clearly and directly.
"""

DEMO_TASK = """\
The payment tests are failing. Find out why and fix the bug.

The test file is: test_payment.py
The payment module is: payment.py

Run the tests with: python3 -m pytest test_payment.py -v
Then read the files, trace the root cause, and fix it.
"""


def render_thinking(thinking: str):
    lines = thinking.strip().splitlines()
    formatted = "\n".join(f"  [dim]>[/dim] {line}" for line in lines if line.strip())
    console.print(Panel(formatted, title="[bold yellow]THINKING[/bold yellow]",
                        border_style="yellow", box=box.SIMPLE_HEAD))


def render_action(tool: str, args: dict):
    args_str = "  " + "\n  ".join(
        f"[dim]{k}:[/dim] [cyan]{str(v)[:120]}[/cyan]" for k, v in args.items()
    )
    console.print(Panel(f"[bold green]►[/bold green] [bold]{tool}[/bold]\n{args_str}",
                        title="[bold green]ACTION[/bold green]",
                        border_style="green", box=box.SIMPLE_HEAD))


def render_result(result_str: str):
    try:
        data = json.loads(result_str)
        out = data.get("stdout", "") or data.get("content", "") or json.dumps(data)
        out = out[:600]
    except Exception:
        out = result_str[:400]
    if out.strip():
        console.print(f"[dim]  ↳ {out.strip()[:300]}[/dim]")


def render_response(text: str):
    console.print()
    console.print(Rule(style="dim"))
    console.print(f"[white]{text}[/white]")
    console.print()


def render_critic(verdict: str, text: str):
    if verdict == "APPROVE":
        console.print(Panel(
            "[bold green]✓ APPROVED[/bold green]\n[dim]Reasoning accepted — no significant issues.[/dim]",
            title="[bold green]CRITIC[/bold green]",
            border_style="green", box=box.SIMPLE_HEAD,
        ))
    else:
        console.print(Panel(
            f"[bold red]✗ CRITIQUE[/bold red]\n[dim]{text.strip()}[/dim]",
            title="[bold red]CRITIC[/bold red]",
            border_style="red", box=box.SIMPLE_HEAD,
        ))


def render_stratum_iteration(iteration: int, max_iter: int):
    console.print(Rule(
        f"[bold magenta]Stratum — Iteration {iteration}/{max_iter}[/bold magenta]",
        style="magenta",
    ))


def render_rungsx(data: dict):
    status = data.get("status", "SKIP")
    claim  = data.get("claim", {})
    cause  = claim.get("proposed_cause", "?")
    effect = claim.get("effect", "?")
    fix    = claim.get("proposed_fix", "?")
    before = data.get("open_paths_before", 0)
    after  = data.get("open_paths_after", 0)
    blocked = data.get("blocked_paths", 0)
    reason = data.get("reason", "") or data.get("skip_reason", "")

    if status == "SKIP":
        console.print(Panel(
            f"[dim]SKIP — {reason}[/dim]",
            title="[bold dim]CAUSAL VERIFIER[/bold dim]",
            border_style="dim", box=box.SIMPLE_HEAD,
        ))
    elif status == "VALID":
        lines = (
            f"[bold green]✓ CAUSAL_VALID[/bold green]\n"
            f"  [dim]Cause:[/dim] [cyan]{cause}[/cyan] [dim]→[/dim] [cyan]{effect}[/cyan]\n"
            f"  [dim]Fix:[/dim]   do([cyan]{fix}[/cyan])\n"
            f"  [dim]Paths before:[/dim] {before}  [dim]blocked:[/dim] {blocked}  "
            f"[dim]residual:[/dim] [green]{after}[/green]"
        )
        console.print(Panel(lines, title="[bold green]CAUSAL VERIFIER[/bold green]",
                            border_style="green", box=box.SIMPLE_HEAD))
    else:
        lines = (
            f"[bold red]✗ CAUSAL_INVALID[/bold red]\n"
            f"  [dim]{reason}[/dim]\n"
            f"  [dim]Residual open paths:[/dim] [red]{after}[/red]"
        )
        console.print(Panel(lines, title="[bold red]CAUSAL VERIFIER[/bold red]",
                            border_style="red", box=box.SIMPLE_HEAD))


def render_stratum_final(data: dict):
    note = data.get("note", "")
    iterations = data["iterations"]
    console.print()
    console.print(Panel(
        f"[bold cyan]Debate complete[/bold cyan] — {iterations} iteration(s)\n"
        + (f"[yellow]{note}[/yellow]" if note else "[green]Approved by critic[/green]"),
        title="[bold cyan]STRATUM FINAL[/bold cyan]",
        border_style="cyan", box=box.DOUBLE_EDGE,
    ))
    console.print()


def display_event(event: dict):
    t = event["type"]
    if t == "thinking":
        render_thinking(event["data"])
    elif t == "action":
        render_action(event["data"]["tool"], event["data"]["args"])
    elif t == "result":
        render_result(event["data"]["result"])
    elif t == "response":
        render_response(event["data"])
    elif t == "critic":
        render_critic(event["data"]["verdict"], event["data"]["text"])
    elif t == "rungsx":
        render_rungsx(event["data"])
    elif t == "stratum_iteration":
        render_stratum_iteration(event["data"]["iteration"], event["data"]["max"])
    elif t == "stratum_final":
        render_stratum_final(event["data"])


def run_ask(question: str | None, backend: str, stratum: bool = False):
    """Free-form Q&A — type any question, see it reason live."""
    import re
    config = get_config(backend)
    api_key = config.api_key or os.getenv("OPENAI_API_KEY") or "none"
    from openai import OpenAI
    client = OpenAI(base_url=config.base_url, api_key=api_key)

    console.print()
    console.print(Panel(
        "[bold cyan]TunedAI — Ask Anything[/bold cyan]\n"
        f"[dim]Model: {config.model}[/dim]",
        border_style="cyan", box=box.DOUBLE_EDGE,
    ))

    if not question:
        console.print()
        try:
            question = Prompt.ask("[bold cyan]Your question[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Cancelled.[/dim]")
            return

    if not question.strip():
        console.print("[dim]No question provided.[/dim]")
        return

    console.print()
    console.print(f"[bold]Q:[/bold] {question.strip()}")
    console.print(Rule("[dim]Thinking...[/dim]", style="dim"))
    console.print()

    if stratum:
        # Run through Stratum Reasoner+Critic loop
        from .stratum import StratumLoop
        loop = StratumLoop(config, max_iterations=2)
        for event in loop.run(question):
            display_event(event)
        return

    # Single-shot streaming with THINKING extraction
    messages = [
        {"role": "system", "content": ASK_SYSTEM},
        {"role": "user",   "content": question.strip()},
    ]

    full_content = ""
    thinking = ""

    with client.chat.completions.create(
        model=config.model,
        messages=messages,
        max_tokens=4096,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            if hasattr(delta, "thinking") and delta.thinking:
                thinking += delta.thinking
            if delta.content:
                full_content += delta.content

    # Extract THINKING block from content if not in native field
    if not thinking:
        match = re.search(r"THINKING\s*\n((?:\s*>.*\n?)+)", full_content, re.MULTILINE)
        if match:
            lines = [l.strip().lstrip(">").strip()
                     for l in match.group(1).splitlines() if l.strip()]
            thinking = "\n".join(lines)
            full_content = (full_content[:match.start()] + full_content[match.end():]).strip()
        else:
            match = re.search(r"<think>(.*?)</think>", full_content, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                full_content = re.sub(r"<think>.*?</think>", "", full_content,
                                      flags=re.DOTALL).strip()

    if thinking:
        render_thinking(thinking)

    if full_content.strip():
        render_response(full_content)

    console.print(Rule("[dim]Done[/dim]", style="dim"))


def run_stratum_demo(backend: str):
    pkg_dir = Path(__file__).parent
    scenario_dir = pkg_dir / "_demo"

    tmpdir = Path(tempfile.mkdtemp(prefix="tunedai-stratum-"))
    shutil.copy(scenario_dir / "payment.py", tmpdir / "payment.py")
    shutil.copy(scenario_dir / "test_payment.py", tmpdir / "test_payment.py")
    (tmpdir / "pytest.ini").write_text("[pytest]\ntestpaths = .\n")
    os.chdir(tmpdir)

    console.print()
    console.print(Panel(
        "[bold magenta]TunedAI Stratum[/bold magenta]\n"
        "[dim]Reasoner + Critic debate loop — errors surface before you see the answer[/dim]",
        border_style="magenta", box=box.DOUBLE_EDGE,
    ))
    console.print()
    console.print(f"[bold]Task:[/bold] {DEMO_TASK.strip()}")
    console.print()

    config = get_config(backend)
    loop = StratumLoop(config, max_iterations=3)

    for event in loop.run(DEMO_TASK):
        display_event(event)

    console.print(Rule("[dim]Stratum complete[/dim]", style="dim"))


def run_demo(backend: str):
    pkg_dir = Path(__file__).parent
    scenario_dir = pkg_dir / "_demo"

    tmpdir = Path(tempfile.mkdtemp(prefix="tunedai-demo-"))
    shutil.copy(scenario_dir / "payment.py", tmpdir / "payment.py")
    shutil.copy(scenario_dir / "test_payment.py", tmpdir / "test_payment.py")
    (tmpdir / "pytest.ini").write_text("[pytest]\ntestpaths = .\n")
    os.chdir(tmpdir)

    console.print()
    console.print(Panel(
        "[bold cyan]TunedAI Terminal Agent[/bold cyan]\n"
        "[dim]Reasoning-native coding agent — shows causal thinking before acting[/dim]",
        border_style="cyan", box=box.DOUBLE_EDGE,
    ))
    console.print()
    console.print(f"[bold]Task:[/bold] {DEMO_TASK.strip()}")
    console.print()
    console.print(Rule("[dim]Agent starting[/dim]", style="dim"))
    console.print()

    config = get_config(backend)
    agent = Agent(config, approval_required=False)

    for event in agent.run(DEMO_TASK):
        display_event(event)

    console.print(Rule("[dim]Done[/dim]", style="dim"))


def run_session(backend: str, approval_required: bool):
    try:
        config = get_config(backend)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    agent = Agent(config, approval_required=approval_required)
    console.print(BANNER)
    console.print(f"[dim]Backend:[/dim] [cyan]{backend}[/cyan]  [dim]Model:[/dim] [cyan]{config.model}[/cyan]")
    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]>[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() in ("exit", "quit", "q"):
            console.print("[dim]Bye.[/dim]")
            break
        if user_input.strip().lower() == "tasks":
            tasks = T.get_tasks()
            if not tasks:
                console.print("[dim]No tasks.[/dim]")
            else:
                for t in tasks:
                    status = "[green]✓[/green]" if t["done"] else "[yellow]○[/yellow]"
                    console.print(f"  {status} [{t['id']}] {t['title']}")
            continue

        for event in agent.run(user_input):
            display_event(event)


def main():
    parser = argparse.ArgumentParser(
        description="TunedAI — reasoning-native coding agent"
    )
    parser.add_argument("--demo", action="store_true", help="Run the built-in coding demo")
    parser.add_argument("--stratum", action="store_true",
                        help="Run with Stratum Reasoner+Critic debate loop")
    parser.add_argument("--ask", nargs="?", const="", metavar="QUESTION",
                        help="Ask any question — see it reason live (no tool calls)")
    parser.add_argument(
        "--backend", default=None, choices=list(BACKENDS.keys()),
        help="Model backend (default: together)",
    )
    parser.add_argument("--no-approval", action="store_true",
                        help="Skip action approval prompts")
    args = parser.parse_args()

    backend = args.backend or "together"

    if args.ask is not None:
        question = args.ask if args.ask else None
        run_ask(question, backend, stratum=args.stratum)
    elif args.stratum:
        run_stratum_demo(backend)
    elif args.demo:
        run_demo(backend)
    else:
        run_session(backend=backend, approval_required=not args.no_approval)


if __name__ == "__main__":
    main()
