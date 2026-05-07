"""
causal_extractor.py — Extract causal claims from Reasoner output.

Takes the Reasoner's thinking + conclusion and returns a structured
causal claim: (cause, effect, mechanism, proposed_fix) tuples that
can be fed directly into the causal verifier.

Uses a lightweight LLM call — cheap, fast, no tool calls.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from .config import ModelConfig


EXTRACTOR_PROMPT = """\
You are a causal claim parser. Given an AI agent's reasoning trace and conclusion about a software bug,
extract the causal structure the agent believes is true.

Return ONLY a JSON object with this exact schema — no explanation, no markdown:

{
  "effect": "the failure or symptom being observed (e.g. test_fails, exception_thrown)",
  "proposed_cause": "the root cause the agent identified (e.g. none_input, missing_validation)",
  "mechanism": "the intermediate step connecting cause to effect (e.g. validation_gap, type_error)",
  "proposed_fix": "what the agent intends to change (e.g. add_none_check, fix_return_type)",
  "confounds_mentioned": ["any alternative causes or confounds the agent considered"],
  "confidence": 0.85
}

If the agent did not clearly identify a cause, set proposed_cause to null.
Extract only what the agent explicitly stated — do not infer or add your own analysis.
"""


@dataclass
class CausalClaim:
    effect: str
    proposed_cause: Optional[str]
    mechanism: Optional[str]
    proposed_fix: Optional[str]
    confounds_mentioned: list[str] = field(default_factory=list)
    confidence: float = 0.5
    raw: dict = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return bool(self.effect and self.proposed_cause and self.proposed_fix)

    def to_dict(self) -> dict:
        return {
            "effect": self.effect,
            "proposed_cause": self.proposed_cause,
            "mechanism": self.mechanism,
            "proposed_fix": self.proposed_fix,
            "confounds_mentioned": self.confounds_mentioned,
            "confidence": self.confidence,
        }


def extract_causal_claim(
    thinking: str,
    conclusion: str,
    config: ModelConfig,
) -> CausalClaim:
    """
    Extract a structured causal claim from the Reasoner's output.
    Returns a CausalClaim — check .is_complete before passing to verifier.
    """
    api_key = config.api_key or os.getenv("OPENAI_API_KEY") or "none"
    client = OpenAI(base_url=config.base_url, api_key=api_key)

    user_content = (
        f"AGENT THINKING:\n{thinking.strip()}\n\n"
        f"AGENT CONCLUSION:\n{conclusion.strip()}"
    )

    try:
        resp = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": EXTRACTOR_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=400,
            stream=False,
        )
        raw_text = resp.choices[0].message.content or "{}"

        # Strip markdown fences if present
        import re
        raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text.strip()).rstrip("` \n")

        data = json.loads(raw_text)

        return CausalClaim(
            effect=data.get("effect", "unknown_effect"),
            proposed_cause=data.get("proposed_cause"),
            mechanism=data.get("mechanism"),
            proposed_fix=data.get("proposed_fix"),
            confounds_mentioned=data.get("confounds_mentioned", []),
            confidence=float(data.get("confidence", 0.5)),
            raw=data,
        )

    except Exception as e:
        # Extraction failed — return an incomplete claim so verifier skips gracefully
        return CausalClaim(
            effect="unknown",
            proposed_cause=None,
            mechanism=None,
            proposed_fix=None,
            raw={"error": str(e)},
        )
