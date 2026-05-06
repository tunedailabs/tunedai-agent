#!/usr/bin/env python3
"""
main.py — TunedAI Terminal Coding Agent
A reasoning-native coding agent that shows its causal thinking before acting.

Usage:
    python main.py                         # uses default backend (together)
    python main.py --backend openai        # use OpenAI
    python main.py --backend local         # use local Ollama
    python main.py --no-approval           # skip action approval prompts
"""

from __future__ import annotations
import argparse
import json
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Prompt
from rich import box

from config import get_config, BACKENDS
from agent import Agent
import tools as T


console = Console()

BANNER = """[bold cyan]TunedAI Agent[/bold cyan] [dim]— reasoning-native coding agent[/dim]
[dim]Type your task. Type 'exit' to quit. Type 'tasks' to see task list.[/dim]
"""

THINKING_STYLE = "bold yellow"
ACTION_STYLE = "bold green"
RESULT_STYLE = "dim"
RESPONSE_STYLE = "white"
ERROR_STYLE = "bold red"


def render_thinking(thinking: str):
    lines = thinking.strip().splitlines()
    formatted = "\n".join(f"  [dim]>[/dim] {line}" for line in lines if line.strip())
    console.print(
        Panel(
            formatted,
            title="[bold yellow]THINKING[/bold yellow]",
            border_style="yellow",
            box=box.SIMPLE_HEAD,
        )
    )


def render_action(tool: str, args: dict, approved: bool = True):
    args_str = "  " + "\n  ".join(
        f"[dim]{k}:[/dim] [cyan]{str(v)[:120]}[/cyan]" for k, v in args.items()
    )
    status = "[green]►[/green]" if approved else "[red]✗[/red]"
    console.print(
        Panel(
            f"{status} [bold green]{tool}[/bold green]\n{args_str}",
            title="[bold green]ACTION[/bold green]",
            border_style="green",
            box=box.SIMPLE_HEAD,
        )
    )


def render_result(tool: str, result_str: str):
    try:
        data = json.loads(result_str)
        # Truncate long results
        if "content" in data and len(data["content"]) > 800:
            data["content"] = data["content"][:800] + "\n... [truncated]"
        if "stdout" in data and len(data["stdout"]) > 800:
            data["stdout"] = data["stdout"][:800] + "\n... [truncated]"
        display = json.dumps(data, indent=2)
    except Exception:
        display = result_str[:800]

    console.print(f"[dim]  ↳ result: {display[:300]}...[/dim]" if len(display) > 300 else f"[dim]  ↳ {display}[/dim]")


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
        render_result(event["data"]["tool"], event["data"]["result"])
    elif t == "response":
        render_response(event["data"])


def approval_gate(tool: str, args: dict) -> bool:
    """Ask user before destructive/write operations."""
    destructive = {"write_file", "run_shell", "task_done"}
    if tool not in destructive:
        return True
    console.print(f"\n[yellow]Approve:[/yellow] [bold]{tool}[/bold]({json.dumps(args)[:120]})")
    ans = Prompt.ask("  [y/n]", default="y")
    return ans.lower() in ("y", "yes", "")


def run_session(backend: str, approval_required: bool):
    try:
        config = get_config(backend)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    if not config.api_key or config.api_key in ("none", "ollama", "philosopher"):
        pass  # local / no-key backends are fine
    elif not config.api_key:
        console.print(f"[red]No API key set for backend '{backend}'.[/red]")
        console.print(f"[dim]Set the environment variable for this backend.[/dim]")
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

        if user_input.strip().lower() == "history":
            for msg in agent.messages[1:]:  # skip system
                role = msg["role"]
                content = str(msg.get("content", ""))[:200]
                console.print(f"[dim]{role}:[/dim] {content}")
            continue

        # Agentic run — with approval gating if enabled
        pending_approval = None

        for event in agent.run(user_input, display_fn=None):
            if event["type"] == "action" and approval_required:
                tool = event["data"]["tool"]
                args = event["data"]["args"]
                render_action(tool, args, approved=True)

                if not approval_gate(tool, args):
                    # Inject a skipped result so the agent knows
                    agent.messages.append({
                        "role": "tool",
                        "tool_call_id": "skipped",
                        "content": json.dumps({"error": "Action declined by user"}),
                    })
                    console.print("[dim]  ↳ skipped[/dim]")
                    continue
            else:
                display_event(event)


def main():
    parser = argparse.ArgumentParser(
        description="TunedAI Terminal Coding Agent — reasoning-native, model-agnostic"
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=list(BACKENDS.keys()),
        help="Model backend to use (default: TUNEDAI_BACKEND env var or 'together')",
    )
    parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Skip action approval prompts (fully autonomous mode)",
    )
    args = parser.parse_args()

    run_session(
        backend=args.backend or "together",
        approval_required=not args.no_approval,
    )


if __name__ == "__main__":
    main()
