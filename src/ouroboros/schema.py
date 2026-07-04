"""Data contracts for Ouroboros.

Everything downstream (rollout, verify, store, eval, and later SFT/RL) agrees on
these shapes. A trajectory is only meaningful paired with (a) the exact reward the
verifier gave it and (b) enough provenance to reproduce it. Keep this module
dependency-light — it is imported everywhere.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class SamplingConfig:
    """How a rollout was sampled — part of provenance.

    Fields are None when the benchmark runner controls them internally (v1: the
    mcpmark pipeline owns temperature/max-turns; we record what we set, not guesses).
    """
    temperature: Optional[float] = None
    max_turns: Optional[int] = None
    n_samples: int = 1
    top_p: Optional[float] = None
    seed: Optional[int] = None


@dataclass
class RunMeta:
    """Reproducibility stamp. Without this a trajectory cannot be trusted or reproduced."""
    model: str                          # logical model name
    checkpoint: Optional[str]           # ckpt path/hash the rollout ran against
    base_url: str                       # serving endpoint
    mcpmark_rev: Optional[str] = None   # verifier version
    ouroboros_rev: Optional[str] = None
    sampling: Optional[SamplingConfig] = None
    created_at: Optional[str] = None    # ISO ts (stamped by caller; no wall-clock in this module)


@dataclass
class VerifyResult:
    """Objective reward for one trajectory. `passed` is the primary signal;
    `detail` keeps the raw checker output for reward-hacking audits."""
    passed: bool
    reward: float                                        # 0/1 for pass/fail, or graded
    detail: dict[str, Any] = field(default_factory=dict)  # raw checker evidence
    env_rebuilt: bool = False                            # was the env isolated/rebuilt?


@dataclass
class Trajectory:
    """One agent attempt at one task, plus its reward and provenance."""
    task_id: str
    domain: str
    split: str                          # "train" | "eval"
    sample_index: int
    messages: list[dict[str, Any]]      # OpenAI-format conversation (the trajectory)
    meta: RunMeta
    verify: Optional[VerifyResult] = None   # filled in after verification
    api_calls: int = 0
    duration_s: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def trajectory_from_dict(d: dict) -> "Trajectory":
    """Rebuild a Trajectory (with nested dataclasses) from a JSONL record."""
    md = d.get("meta") or {}
    samp = md.get("sampling")
    meta = RunMeta(
        model=md.get("model", ""), checkpoint=md.get("checkpoint"),
        base_url=md.get("base_url", ""), mcpmark_rev=md.get("mcpmark_rev"),
        ouroboros_rev=md.get("ouroboros_rev"),
        sampling=SamplingConfig(**samp) if isinstance(samp, dict) else None,
        created_at=md.get("created_at"),
    )
    v = d.get("verify")
    verify = VerifyResult(**v) if isinstance(v, dict) else None
    return Trajectory(
        task_id=d["task_id"], domain=d.get("domain", ""), split=d.get("split", ""),
        sample_index=int(d.get("sample_index", 0)), messages=d.get("messages") or [],
        meta=meta, verify=verify, api_calls=int(d.get("api_calls", 0)),
        duration_s=float(d.get("duration_s", 0.0)),
    )


@dataclass
class EvalReport:
    """A checkpoint's report card on the held-out split."""
    checkpoint: Optional[str]
    n_tasks: int
    pass_at: dict[int, float]                  # k -> pass@k
    per_domain: dict[str, dict[int, float]]    # domain -> k -> pass@k
    meta: RunMeta

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)
