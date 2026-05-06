from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    enable_thinking: bool = True


BACKENDS = {
    "together": ModelConfig(
        base_url="https://api.together.xyz/v1",
        api_key=os.getenv("TOGETHER_API_KEY", ""),
        model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        enable_thinking=True,
    ),
    "openai": ModelConfig(
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model="gpt-4o",
        enable_thinking=False,
    ),
    "anthropic": ModelConfig(
        base_url="https://api.anthropic.com/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        model="claude-sonnet-4-6",
        enable_thinking=False,
    ),
    "local": ModelConfig(
        base_url=os.getenv("LOCAL_API_URL", "http://localhost:11434/v1"),
        api_key="ollama",
        model=os.getenv("LOCAL_MODEL", "qwen3:14b"),
        enable_thinking=True,
    ),
}

DEFAULT_BACKEND = os.getenv("TUNEDAI_BACKEND", "together")


def get_config(backend: str | None = None) -> ModelConfig:
    name = backend or DEFAULT_BACKEND
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {list(BACKENDS)}")
    return BACKENDS[name]
