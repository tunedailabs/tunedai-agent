from __future__ import annotations
import os
import subprocess
import json
from pathlib import Path
from typing import Any


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it if it does not exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to list (default: cwd)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return stdout + stderr. Use for tests, grep, git, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "working_dir": {"type": "string", "description": "Optional working directory"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a pattern across files using grep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Directory to search (default: cwd)"},
                    "file_glob": {"type": "string", "description": "e.g. '*.py'"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent git commits. Useful for tracing when a bug was introduced.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of commits (default 10)"},
                    "path": {"type": "string", "description": "Limit to commits touching this file"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_create",
            "description": "Create a named task to track work. Returns task id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_done",
            "description": "Mark a task as completed.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        },
    },
]

_tasks: dict[str, dict] = {}
_task_counter = 0


def read_file(path: str = "", filename: str = "") -> dict[str, Any]:
    path = path or filename
    p = Path(path).expanduser()
    if not p.exists():
        return {"error": f"File not found: {path}"}
    try:
        content = p.read_text(errors="replace")
        return {"path": str(p), "lines": len(content.splitlines()), "content": content}
    except Exception as e:
        return {"error": str(e)}


def write_file(path: str, content: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"written": str(p), "bytes": len(content.encode())}


def list_directory(path: str = ".") -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        return {"error": f"Path not found: {path}"}
    entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
    return {
        "path": str(p),
        "entries": [
            {"name": e.name, "type": "dir" if e.is_dir() else "file", "size": e.stat().st_size if e.is_file() else None}
            for e in entries
        ],
    }


def run_shell(command: str, working_dir: str | None = None) -> dict[str, Any]:
    cwd = Path(working_dir).expanduser() if working_dir else None
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, cwd=cwd, timeout=60
    )
    return {
        "stdout": result.stdout[-8000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else "",
        "returncode": result.returncode,
    }


def search_code(pattern: str, path: str = ".", file_glob: str | None = None) -> dict[str, Any]:
    cmd = f"grep -rn --include='{file_glob or '*'}' {json.dumps(pattern)} {path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    lines = result.stdout.strip().splitlines()
    return {"matches": lines[:100], "total": len(lines)}


def git_log(n: int = 10, path: str | None = None) -> dict[str, Any]:
    cmd = f"git log --oneline -n {n}"
    if path:
        cmd += f" -- {path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return {"error": "Not a git repository or git not available"}
    return {"commits": result.stdout.strip().splitlines()}


def task_create(title: str, description: str = "") -> dict[str, Any]:
    global _task_counter
    _task_counter += 1
    task_id = f"T{_task_counter:03d}"
    _tasks[task_id] = {"id": task_id, "title": title, "description": description, "done": False}
    return {"task_id": task_id, "title": title}


def task_done(task_id: str) -> dict[str, Any]:
    if task_id not in _tasks:
        return {"error": f"Task {task_id} not found"}
    _tasks[task_id]["done"] = True
    return {"task_id": task_id, "status": "done"}


def get_tasks() -> list[dict]:
    return list(_tasks.values())


TOOL_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "run_shell": run_shell,
    "search_code": search_code,
    "git_log": git_log,
    "task_create": task_create,
    "task_done": task_done,
}


def dispatch(name: str, args: dict) -> str:
    if name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = TOOL_MAP[name](**args)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
