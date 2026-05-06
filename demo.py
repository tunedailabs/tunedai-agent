#!/usr/bin/env python3
"""
demo.py — Pre-loaded demo session for MOM #13.

Runs the agent against a real failing payment test and shows
the THINKING trace before it touches any code.

Usage:
    python3 demo.py
"""
from __future__ import annotations
import sys
import os

# Make sure we're running from the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich import box

from config import get_config
from agent import Agent

console = Console()

TASK = """\
The payment tests are failing. Find out why and fix the bug.

The test file is: demo/test_payment.py
The payment module is: demo/payment.py

Run the tests with: python3 -m pytest demo/test_payment.py -v
Then read the files, trace the root cause, and fix it.
"""

def display_event(event: dict):
    t = event["type"]
    if t == "thinking":
        lines = event["data"].strip().splitlines()
        formatted = "\n".join(f"  [dim]>[/dim] {l}" for l in lines if l.strip())
        console.print(Panel(
            formatted,
            title="[bold yellow]THINKING[/bold yellow]",
            border_style="yellow",
            box=box.SIMPLE_HEAD,
        ))
    elif t == "action":
        tool = event["data"]["tool"]
        args = event["data"]["args"]
        args_str = "  " + "\n  ".join(
            f"[dim]{k}:[/dim] [cyan]{str(v)[:120]}[/cyan]"
            for k, v in args.items()
        )
        console.print(Panel(
            f"[bold green]►[/bold green] [bold]{tool}[/bold]\n{args_str}",
            title="[bold green]ACTION[/bold green]",
            border_style="green",
            box=box.SIMPLE_HEAD,
        ))
    elif t == "result":
        import json
        try:
            data = json.loads(event["data"]["result"])
            out = data.get("stdout", "") or data.get("content", "") or json.dumps(data)
            out = out[:600]
        except Exception:
            out = event["data"]["result"][:400]
        if out.strip():
            console.print(f"[dim]  ↳ {out.strip()[:300]}[/dim]")
    elif t == "response":
        console.print()
        console.print(Rule(style="dim"))
        console.print(f"[white]{event['data']}[/white]")
        console.print()


def main():
    console.print()
    console.print(Panel(
        "[bold cyan]TunedAI Terminal Agent[/bold cyan]\n"
        "[dim]Reasoning-native coding agent — shows causal thinking before acting[/dim]",
        border_style="cyan",
        box=box.DOUBLE_EDGE,
    ))
    console.print()
    console.print(f"[bold]Task:[/bold] {TASK.strip()}")
    console.print()
    console.print(Rule("[dim]Agent starting[/dim]", style="dim"))
    console.print()

    config = get_config("openai")
    agent = Agent(config, approval_required=False)

    for event in agent.run(TASK, display_fn=None):
        display_event(event)

    console.print(Rule("[dim]Done[/dim]", style="dim"))


if __name__ == "__main__":
    main()
