"""
config.py — Model backend configuration. Any OpenAI-compatible endpoint works.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    enable_thinking: bool = True  # Request <think> blocks when supported


BACKENDS = {
    "together": ModelConfig(
        base_url="https://api.together.xyz/v1",
        api_key=os.getenv("TOGETHER_API_KEY", ""),
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        enable_thinking=False,
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
        model=os.getenv("LOCAL_MODEL", "qwen2.5:7b"),
        enable_thinking=False,
    ),
    "philosopher": ModelConfig(
        base_url="https://mark-gentry--qwen3-philosopher-serve.modal.run/v1",
        api_key="philosopher",
        model="philosopher",
        enable_thinking=True,
    ),
}

DEFAULT_BACKEND = os.getenv("TUNEDAI_BACKEND", "anthropic")


def get_config(backend: str | None = None) -> ModelConfig:
    name = backend or DEFAULT_BACKEND
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {list(BACKENDS)}")
    return BACKENDS[name]
