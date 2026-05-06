from __future__ import annotations
import json
import re
from typing import Iterator

from .config import ModelConfig
from . import tools as T


SYSTEM_PROMPT = """\
You are a reasoning-native coding agent with access to tools.

Before your FIRST tool call each turn, output a THINKING block in this exact format:

THINKING
> [what is failing / what is the goal]
> [most likely cause]
> [second most likely cause]
> [which cause you are acting on and why]
> [confidence: X%]

After the THINKING block, immediately call the tool. Do not describe what you will do — just do it.
Keep going until the task is fully complete. When tests pass, say so and stop.
"""


def _parse_thinking_block(text: str) -> tuple[str, str]:
    match = re.search(r"THINKING\s*\n((?:\s*>.*\n?)+)", text, re.MULTILINE)
    if not match:
        return "", text
    lines = [l.strip().lstrip(">").strip() for l in match.group(1).splitlines() if l.strip()]
    thinking = "\n".join(lines)
    clean = (text[:match.start()] + text[match.end():]).strip()
    return thinking, clean


def _parse_think_tags(text: str) -> tuple[str, str]:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not match:
        return "", text
    thinking = match.group(1).strip()
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return thinking, clean


def _openai_tools_to_anthropic(tools: list) -> list:
    result = []
    for t in tools:
        fn = t["function"]
        result.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        })
    return result


class AnthropicAgent:
    def __init__(self, config: ModelConfig):
        import anthropic
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key or None)
        self.messages: list[dict] = []
        self.tools = _openai_tools_to_anthropic(T.TOOL_SCHEMAS)

    def run(self, user_input: str, display_fn=None) -> Iterator[dict]:
        self.messages.append({"role": "user", "content": user_input})
        max_turns = 20

        for _ in range(max_turns):
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=self.messages,
                tools=self.tools,
            )

            text_parts = []
            tool_uses = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            full_text = "\n".join(text_parts).strip()
            thinking, clean_text = _parse_thinking_block(full_text)
            if not thinking:
                thinking, clean_text = _parse_think_tags(full_text)

            if thinking:
                event = {"type": "thinking", "data": thinking}
                if display_fn:
                    display_fn(event)
                yield event

            if clean_text and not tool_uses:
                self.messages.append({"role": "assistant", "content": full_text})
                event = {"type": "response", "data": clean_text}
                if display_fn:
                    display_fn(event)
                yield event
                break

            if clean_text and tool_uses:
                event = {"type": "response", "data": clean_text}
                if display_fn:
                    display_fn(event)
                yield event

            self.messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tu in tool_uses:
                action_event = {"type": "action", "data": {"tool": tu.name, "args": tu.input}}
                if display_fn:
                    display_fn(action_event)
                yield action_event

                result_str = T.dispatch(tu.name, tu.input)

                result_event = {"type": "result", "data": {"tool": tu.name, "result": result_str}}
                if display_fn:
                    display_fn(result_event)
                yield result_event

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_str,
                })

            self.messages.append({"role": "user", "content": tool_results})

            if response.stop_reason == "end_turn" and not tool_uses:
                break


class OpenAIAgent:
    def __init__(self, config: ModelConfig):
        import os
        from openai import OpenAI
        self.config = config
        api_key = config.api_key or os.getenv("OPENAI_API_KEY") or "none"
        self.client = OpenAI(base_url=config.base_url, api_key=api_key)
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _call(self) -> dict:
        kwargs = dict(
            model=self.config.model,
            messages=self.messages,
            tools=T.TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=4096,
        )
        if self.config.enable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": 4096}}

        content = ""
        thinking = ""
        tool_calls_raw = {}

        with self.client.chat.completions.create(**kwargs, stream=True) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue
                if hasattr(delta, "thinking") and delta.thinking:
                    thinking += delta.thinking
                if delta.content:
                    content += delta.content
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_raw:
                            tool_calls_raw[idx] = {"id": tc.id or "", "name": "", "args": ""}
                        if tc.function.name:
                            tool_calls_raw[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_raw[idx]["args"] += tc.function.arguments

        tool_calls = [
            {"id": raw["id"] or f"call_{i}", "type": "function",
             "function": {"name": raw["name"], "arguments": raw["args"]}}
            for i, raw in sorted(tool_calls_raw.items())
        ]

        if not thinking:
            thinking, content = _parse_think_tags(content)
        if not thinking:
            thinking, content = _parse_thinking_block(content)

        return {"thinking": thinking, "content": content, "tool_calls": tool_calls or None}

    def run(self, user_input: str, display_fn=None) -> Iterator[dict]:
        self.messages.append({"role": "user", "content": user_input})
        max_turns = 20

        for _ in range(max_turns):
            response = self._call()
            thinking = response["thinking"]
            content = response["content"]
            tool_calls = response["tool_calls"]

            if thinking:
                event = {"type": "thinking", "data": thinking}
                if display_fn:
                    display_fn(event)
                yield event

            if not tool_calls:
                self.messages.append({"role": "assistant", "content": content})
                event = {"type": "response", "data": content}
                if display_fn:
                    display_fn(event)
                yield event
                break

            if content:
                event = {"type": "response", "data": content}
                if display_fn:
                    display_fn(event)
                yield event

            self.messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": [{"id": tc["id"], "type": "function",
                                "function": {"name": tc["function"]["name"],
                                             "arguments": tc["function"]["arguments"]}}
                               for tc in tool_calls],
            })

            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                action_event = {"type": "action", "data": {"tool": name, "args": args}}
                if display_fn:
                    display_fn(action_event)
                yield action_event

                result_str = T.dispatch(name, args)

                result_event = {"type": "result", "data": {"tool": name, "result": result_str}}
                if display_fn:
                    display_fn(result_event)
                yield result_event

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })


class Agent:
    def __init__(self, config: ModelConfig, approval_required: bool = True):
        self.config = config
        self.approval_required = approval_required
        if "anthropic.com" in (config.base_url or ""):
            self._impl = AnthropicAgent(config)
        else:
            self._impl = OpenAIAgent(config)
        self.messages = getattr(self._impl, "messages", [])

    def run(self, user_input: str, display_fn=None) -> Iterator[dict]:
        yield from self._impl.run(user_input, display_fn=display_fn)
