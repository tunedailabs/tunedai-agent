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
from . import tools as T

console = Console()

BANNER = """[bold cyan]TunedAI Agent[/bold cyan] [dim]— reasoning-native coding agent[/dim]
[dim]Type your task. 'exit' to quit. 'tasks' to see task list.[/dim]
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
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo")
    parser.add_argument(
        "--backend", default=None, choices=list(BACKENDS.keys()),
        help="Model backend (default: together)",
    )
    parser.add_argument("--no-approval", action="store_true",
                        help="Skip action approval prompts")
    args = parser.parse_args()

    backend = args.backend or "together"

    if args.demo:
        run_demo(backend)
    else:
        run_session(backend=backend, approval_required=not args.no_approval)


if __name__ == "__main__":
    main()
