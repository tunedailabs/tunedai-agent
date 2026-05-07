"""
causal_verifier.py — Pluggable causal verification interface for Stratum.

Defines the CausalVerdict dataclass and CausalVerifierBase abstract class.
Third parties can implement their own verifier by subclassing CausalVerifierBase.

The Stratum loop calls load_verifier() which returns whichever implementation
is available — enterprise causal engine if installed, NullVerifier otherwise.

Open source users see the interface and the NullVerifier (SKIP on every claim).
Enterprise users receive a verified causal engine implementation separately.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .causal_extractor import CausalClaim


# ── Verdict ──────────────────────────────────────────────────────────────────

@dataclass
class CausalVerdict:
    """
    Result of running a causal verification check on a CausalClaim.

    status:
      "VALID"   — proposed fix blocks all causal paths to the effect
      "INVALID" — fix is incomplete or proposed cause is not connected
      "SKIP"    — verifier unavailable or claim too incomplete to check
    """
    status: str
    claim: CausalClaim

    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    cause_connected: bool = False
    open_paths_before: int = 0
    open_paths_after: int = 0
    blocked_paths: int = 0

    reason: str = ""
    skip_reason: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "VALID"

    def summary(self) -> str:
        if self.status == "SKIP":
            return f"Causal verifier: SKIP — {self.skip_reason}"
        if self.status == "INVALID":
            return (
                f"Causal verifier: INVALID — {self.reason}\n"
                f"  Residual open paths after fix: {self.open_paths_after}"
            )
        return (
            f"Causal verifier: VALID\n"
            f"  Cause '{self.claim.proposed_cause}' → '{self.claim.effect}': connected ✓\n"
            f"  do({self.claim.proposed_fix}): blocks {self.blocked_paths} path(s), "
            f"{self.open_paths_after} residual ✓"
        )

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "nodes": self.nodes,
            "edges": [(p, c) for p, c in self.edges],
            "cause_connected": self.cause_connected,
            "open_paths_before": self.open_paths_before,
            "open_paths_after": self.open_paths_after,
            "blocked_paths": self.blocked_paths,
            "reason": self.reason,
            "skip_reason": self.skip_reason,
            "claim": self.claim.to_dict(),
        }


# ── Base class ────────────────────────────────────────────────────────────────

class CausalVerifierBase(ABC):
    """
    Abstract base class for causal verification engines.

    Implement verify() to plug in any causal reasoning backend.
    The Stratum loop calls this after each Reasoner turn and before
    the Critic — the verdict is injected into the Critic's context.
    """

    @abstractmethod
    def verify(self, claim: CausalClaim) -> CausalVerdict:
        """
        Verify a causal claim extracted from the Reasoner's output.

        Must never raise — return a SKIP verdict on any internal error.
        """
        ...

    def name(self) -> str:
        return self.__class__.__name__


# ── Null verifier (open source default) ──────────────────────────────────────

class NullVerifier(CausalVerifierBase):
    """
    Default verifier when no causal engine is installed.
    Always returns SKIP — Stratum runs without formal verification.
    The Critic still reviews reasoning; it just doesn't see a formal verdict.
    """

    def verify(self, claim: CausalClaim) -> CausalVerdict:
        return CausalVerdict(
            status="SKIP",
            claim=claim,
            skip_reason="No causal engine installed. See tunedai enterprise.",
        )

    def name(self) -> str:
        return "NullVerifier"


# ── Loader ───────────────────────────────────────────────────────────────────

def load_verifier() -> CausalVerifierBase:
    """
    Return the best available causal verifier.

    Tries to import the enterprise causal engine. Falls back to NullVerifier
    if not installed — Stratum continues without formal verification.
    """
    try:
        from tunedai._causal_engine import get_verifier
        return get_verifier()
    except ImportError:
        pass
    return NullVerifier()


def verify(claim: CausalClaim) -> CausalVerdict:
    """Convenience wrapper — load verifier and run in one call."""
    return load_verifier().verify(claim)
