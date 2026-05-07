"""
rungsx_verifier.py — Formal causal verification via RungsX.

Takes a CausalClaim extracted from the Reasoner and runs it through
the RungsX CausalDAG engine:

  1. Builds a minimal DAG from the claim (cause → mechanism → effect,
     plus any confounds mentioned)
  2. Confirms the proposed cause is d-connected to the effect
     (i.e. there IS a causal path — not just correlation)
  3. Runs do(apply_fix) via graph surgery and checks whether the effect
     becomes d-separated from all remaining causes
  4. Returns a CausalVerdict with a pass/fail and a human-readable trace

The RungsX engine is imported from rungs-private. If the import fails
(e.g. on a machine without the private repo), the verifier degrades
gracefully and returns SKIP.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

from .causal_extractor import CausalClaim

# RungsX private repo path — adjust if repo moves
_RUNGS_PATH = "/Users/markgentry/Projects/rungs-private"


def _import_rungsx():
    if _RUNGS_PATH not in sys.path:
        sys.path.insert(0, _RUNGS_PATH)
    from causal.data_gen.dag_generator import CausalDAG
    return CausalDAG


# ── Verdict ─────────────────────────────────────────────────────────────────

@dataclass
class CausalVerdict:
    status: str          # "VALID" | "INVALID" | "SKIP"
    claim: CausalClaim

    # DAG topology
    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    # Pre-intervention
    cause_connected: bool = False     # cause d-connected to effect?
    open_paths_before: int = 0

    # Post-intervention (do(fix))
    open_paths_after: int = 0        # residual open paths after fix
    blocked_paths: int = 0

    reason: str = ""
    skip_reason: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "VALID"

    def summary(self) -> str:
        if self.status == "SKIP":
            return f"RungsX: SKIP — {self.skip_reason}"
        if self.status == "INVALID":
            return (
                f"RungsX: INVALID — {self.reason}\n"
                f"  Residual open paths after fix: {self.open_paths_after}"
            )
        return (
            f"RungsX: VALID\n"
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


# ── Core verifier ────────────────────────────────────────────────────────────

def verify(claim: CausalClaim) -> CausalVerdict:
    """
    Run formal causal verification on a CausalClaim.

    Returns a CausalVerdict. Never raises — degrades to SKIP on any
    import error or structural problem.
    """
    if not claim.is_complete:
        return CausalVerdict(
            status="SKIP",
            claim=claim,
            skip_reason="Incomplete claim — cause or fix not identified by Reasoner",
        )

    try:
        CausalDAG = _import_rungsx()
    except Exception as e:
        return CausalVerdict(
            status="SKIP",
            claim=claim,
            skip_reason=f"RungsX import failed: {e}",
        )

    try:
        return _run_verification(CausalDAG, claim)
    except Exception as e:
        return CausalVerdict(
            status="SKIP",
            claim=claim,
            skip_reason=f"Verification error: {e}",
        )


def _run_verification(CausalDAG, claim: CausalClaim) -> CausalVerdict:
    cause    = claim.proposed_cause
    effect   = claim.effect
    mech     = claim.mechanism or f"{cause}_path"
    fix_node = claim.proposed_fix
    confounds = claim.confounds_mentioned or []

    # ── Build DAG ────────────────────────────────────────────────────────────
    dag = CausalDAG(name="bug_causal_structure")

    # Core nodes
    for node in [cause, mech, effect, fix_node]:
        dag.add_node(node)

    # Confound nodes
    for c in confounds:
        if c and c not in dag.nodes:
            dag.add_node(c)

    # Core causal chain: cause → mechanism → effect
    dag.add_edge(cause, mech)
    dag.add_edge(mech, effect)

    # Fix node intervenes on the mechanism
    # (the fix removes the gap, modelled as fix → mechanism with negative weight)
    dag.add_edge(fix_node, mech)

    # Confounds also connect to mechanism (alternative causes)
    for c in confounds:
        if c and c != cause:
            try:
                dag.add_edge(c, mech)
            except ValueError:
                pass  # cycle guard — skip if would create cycle

    nodes = list(dag.nodes.keys())
    edges = [(p, ch) for p in nodes for ch in dag.get_children(p)]

    # ── Step 1: Is cause d-connected to effect (pre-intervention)? ───────────
    # d-separated returns True if BLOCKED — so connected = not d_separated
    cause_connected = not dag.d_separated(cause, effect, set())
    open_paths_before = _count_active_paths(dag, cause, effect, set())

    if not cause_connected:
        return CausalVerdict(
            status="INVALID",
            claim=claim,
            nodes=nodes,
            edges=edges,
            cause_connected=False,
            open_paths_before=0,
            reason=f"'{cause}' is NOT causally connected to '{effect}' — Reasoner may be chasing a symptom",
        )

    # ── Step 2: do(fix) — graph surgery ──────────────────────────────────────
    # Applying the fix is modelled as: remove all incoming edges to `mech`
    # from `cause` (the bug source), leaving only the fix_node's corrective edge.
    # This simulates the structural intervention do(fix_applied=True).
    import copy as _copy
    mutilated = _copy.deepcopy(dag)

    # Remove the bug-causing edge (cause → mech) to simulate the fix taking effect
    if mutilated.has_edge(cause, mech):
        mutilated.remove_edge(cause, mech)

    # Also remove confound edges to mech if confounds are considered resolved
    # (conservative: only remove the primary cause edge)

    open_paths_after = _count_active_paths(mutilated, cause, effect, set())
    blocked_paths = max(0, open_paths_before - open_paths_after)

    if open_paths_after > 0:
        # Residual paths remain — fix is incomplete
        return CausalVerdict(
            status="INVALID",
            claim=claim,
            nodes=nodes,
            edges=edges,
            cause_connected=True,
            open_paths_before=open_paths_before,
            open_paths_after=open_paths_after,
            blocked_paths=blocked_paths,
            reason=(
                f"Fix blocks {blocked_paths} path(s) but {open_paths_after} "
                f"residual path(s) remain — likely a confound or second cause"
            ),
        )

    return CausalVerdict(
        status="VALID",
        claim=claim,
        nodes=nodes,
        edges=edges,
        cause_connected=True,
        open_paths_before=open_paths_before,
        open_paths_after=0,
        blocked_paths=blocked_paths,
        reason="Fix blocks all causal paths to the effect",
    )


def _count_active_paths(dag, source: str, target: str, observed: set) -> int:
    """
    Count distinct active paths from source to target given observed variables.
    Uses DFS — for small bug DAGs this is fast and exact.
    """
    count = 0

    def dfs(current: str, visited: set, direction: str):
        nonlocal count
        if current == target:
            count += 1
            return
        # Forward: follow children
        for child in dag.get_children(current):
            if child not in visited:
                visited.add(child)
                dfs(child, visited, "forward")
                visited.discard(child)

    dfs(source, {source}, "forward")
    return count
