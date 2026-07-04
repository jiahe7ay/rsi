"""(3) Verifier / reward — objective scoring via the mcpmark checker.

Reward reliability is the whole ballgame. Two rules, both non-negotiable:
  * rebuild/restore the task environment for EVERY rollout (no cross-contamination —
    mcpmark postgres containers get rebuilt, filesystem uses state_manager
    backup/restore);
  * keep the raw checker output in VerifyResult.detail so reward-hacking (agent
    editing the checker, or trivially passing) can be audited later.
"""
from __future__ import annotations

from contextlib import contextmanager

from ouroboros.schema import Trajectory, VerifyResult


@contextmanager
def isolated_env(task: dict):
    """Provision a fresh task environment; tear down / restore on exit."""
    raise NotImplementedError("wrap the mcpmark env manager (rebuild per rollout)")
    yield  # pragma: no cover


def verify(task: dict, trajectory: Trajectory) -> VerifyResult:
    """Score `trajectory` against `task` using the mcpmark checker inside an
    isolated env. Return the objective reward plus raw evidence."""
    raise NotImplementedError("run mcpmark checker inside isolated_env(task)")
