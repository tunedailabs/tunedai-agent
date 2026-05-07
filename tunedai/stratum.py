"""
stratum.py — Multi-agent debate loop.

Reasoner produces a conclusion. Critic reviews the reasoning trace.
If the Critic finds issues, the Reasoner revises. Loop until agreement
or max_iterations. Human only sees the final approved output + debate log.
"""
from __future__ import annotations
import json
from typing import Iterator
from openai import OpenAI

from .config import ModelConfig
from . import tools as T
from .causal_extractor import extract_causal_claim
from .causal_verifier import verify as causal_verify


REASONER_PROMPT = """\
You are a reasoning-native coding agent with access to tools.

Before your FIRST tool call each turn, output a THINKING block in this exact format:

THINKING
> [what is failing / what is the goal]
> [most likely cause]
> [second most likely cause]
> [which cause you are acting on and why]
> [confidence: X%]

After the THINKING block, immediately call the tool. Do not describe what you will do — just do it.
Keep going until the task is fully complete. When done, summarise your conclusion clearly.
"""

CRITIC_PROMPT = """\
You are a critical reasoning agent. You review another AI agent's reasoning traces and conclusions.

Given a task, the agent's thinking process, and its conclusion, you:
1. Check for logical gaps or unsupported leaps in the reasoning
2. Flag factual claims that appear ungrounded or potentially hallucinated
3. Identify alternative explanations the agent did not consider
4. Assess whether the stated confidence matches the evidence presented

Respond ONLY in this exact format:

VERDICT: APPROVE
Reasoning is sound. No significant issues found.

OR:

VERDICT: CRITIQUE
ISSUES:
> [specific issue with the reasoning]
> [specific issue with the reasoning]
SUGGEST: [what the agent should reconsider or verify]

Be concise and specific. Only critique if there are genuine reasoning problems — do not nitpick style.
"""


def _parse_verdict(text: str) -> tuple[str, str]:
    """Returns ('APPROVE' or 'CRITIQUE', full critic text)."""
    if "VERDICT: APPROVE" in text:
        return "APPROVE", text
    if "VERDICT: CRITIQUE" in text:
        return "CRITIQUE", text
    return "APPROVE", text  # default to approve if unparseable


class StratumLoop:
    """
    Wraps an OpenAI-compatible agent with a Critic agent.
    Runs the Reasoner, then the Critic, iterating until agreement.
    """

    def __init__(self, config: ModelConfig, max_iterations: int = 3):
        import os
        self.config = config
        self.max_iterations = max_iterations
        api_key = config.api_key or os.getenv("OPENAI_API_KEY") or "none"
        self.client = OpenAI(base_url=config.base_url, api_key=api_key)
        self.debate_log: list[dict] = []

    def _call_reasoner(self, messages: list[dict]) -> dict:
        """Single reasoner call — returns thinking, content, tool_calls."""
        content = ""
        thinking = ""
        tool_calls_raw: dict = {}

        with self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=T.TOOL_SCHEMAS,
            tool_choice="auto",
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

        # Parse THINKING block from content if not in native thinking field
        if not thinking:
            import re
            match = re.search(r"THINKING\s*\n((?:\s*>.*\n?)+)", content, re.MULTILINE)
            if match:
                lines = [l.strip().lstrip(">").strip() for l in match.group(1).splitlines() if l.strip()]
                thinking = "\n".join(lines)
                content = (content[:match.start()] + content[match.end():]).strip()
            else:
                match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if match:
                    thinking = match.group(1).strip()
                    import re as re2
                    content = re2.sub(r"<think>.*?</think>", "", content, flags=re2.DOTALL).strip()

        tool_calls = [
            {"id": raw["id"] or f"call_{i}", "type": "function",
             "function": {"name": raw["name"], "arguments": raw["args"]}}
            for i, raw in sorted(tool_calls_raw.items())
        ]

        return {"thinking": thinking, "content": content, "tool_calls": tool_calls or None}

    def _call_critic(self, task: str, thinking_log: str, conclusion: str) -> tuple[str, str]:
        """Ask the Critic to review the reasoning. Returns (verdict, full_text)."""
        critic_messages = [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": (
                f"ORIGINAL TASK:\n{task}\n\n"
                f"AGENT REASONING TRACE:\n{thinking_log}\n\n"
                f"AGENT CONCLUSION:\n{conclusion}"
            )},
        ]

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=critic_messages,
            max_tokens=1024,
            stream=False,
        )

        text = response.choices[0].message.content or ""
        return _parse_verdict(text)

    def run(self, task: str, display_fn=None) -> Iterator[dict]:
        """Run the full Stratum debate loop."""

        reasoner_messages = [
            {"role": "system", "content": REASONER_PROMPT},
            {"role": "user", "content": task},
        ]

        thinking_log: list[str] = []
        conclusion = ""
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # ── Reasoner turn ─────────────────────────────────────────
            iter_event = {
                "type": "stratum_iteration",
                "data": {"iteration": iteration, "max": self.max_iterations}
            }
            if display_fn:
                display_fn(iter_event)
            yield iter_event

            # Run reasoner until it stops calling tools
            turn_thinking = []
            turn_conclusion = ""

            for _ in range(20):  # max tool calls per turn
                response = self._call_reasoner(reasoner_messages)
                thinking = response["thinking"]
                content = response["content"]
                tool_calls = response["tool_calls"]

                if thinking:
                    turn_thinking.append(thinking)
                    event = {"type": "thinking", "data": thinking}
                    if display_fn:
                        display_fn(event)
                    yield event

                if not tool_calls:
                    turn_conclusion = content
                    reasoner_messages.append({"role": "assistant", "content": content})
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

                # Process tool calls
                reasoner_messages.append({
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

                    reasoner_messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str,
                    })

            thinking_log.extend(turn_thinking)
            conclusion = turn_conclusion

            # ── Stratum causal verification ────────────────────────────
            causal_verdict = None
            try:
                claim = extract_causal_claim(
                    thinking="\n".join(turn_thinking),
                    conclusion=conclusion,
                    config=self.config,
                )
                causal_verdict = causal_verify(claim)
                verify_event = {
                    "type": "causal_verify",
                    "data": causal_verdict.to_dict(),
                }
                if display_fn:
                    display_fn(verify_event)
                yield verify_event
            except Exception:
                pass  # verification is non-blocking

            # ── Critic turn ────────────────────────────────────────────
            # Inject causal verdict into Critic context if available
            verify_context = ""
            if causal_verdict and causal_verdict.status != "SKIP":
                verify_context = (
                    f"\n\nSTRATUM CAUSAL VERIFICATION:\n{causal_verdict.summary()}"
                )

            verdict, critic_text = self._call_critic(
                task=task,
                thinking_log="\n\n".join(thinking_log),
                conclusion=conclusion + verify_context,
            )

            self.debate_log.append({
                "iteration": iteration,
                "thinking": "\n".join(turn_thinking),
                "conclusion": conclusion,
                "causal_verdict": causal_verdict.to_dict() if causal_verdict else None,
                "verdict": verdict,
                "critic": critic_text,
            })

            critic_event = {
                "type": "critic",
                "data": {"verdict": verdict, "text": critic_text}
            }
            if display_fn:
                display_fn(critic_event)
            yield critic_event

            if verdict == "APPROVE":
                # Done — emit final approved output
                final_event = {
                    "type": "stratum_final",
                    "data": {
                        "conclusion": conclusion,
                        "iterations": iteration,
                        "debate_log": self.debate_log,
                    }
                }
                if display_fn:
                    display_fn(final_event)
                yield final_event
                return

            # Critique received — inject it and let reasoner revise
            reasoner_messages.append({
                "role": "user",
                "content": (
                    f"A critical review of your reasoning found issues:\n\n{critic_text}\n\n"
                    f"Please reconsider your reasoning and provide a revised conclusion."
                )
            })

        # Max iterations reached — emit final anyway
        final_event = {
            "type": "stratum_final",
            "data": {
                "conclusion": conclusion,
                "iterations": iteration,
                "debate_log": self.debate_log,
                "note": "Max iterations reached — human review recommended",
            }
        }
        if display_fn:
            display_fn(final_event)
        yield final_event
