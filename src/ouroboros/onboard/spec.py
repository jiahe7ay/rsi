"""Data contracts for the onboarding layer.

An onboarding run turns a raw benchmark checkout into a BenchmarkSpec (its "ID
card"), probes whether the environment is ready, elicits whatever is missing,
and smoke-tests one task. These dataclasses are the shared shapes. Values of
`secret` env vars are NEVER stored or printed by this layer.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

SPEC_DIR = Path("data/specs")

# Repo KIND vocabulary — explore classifies into one of these, and onboard ROUTES
# on it: only `benchmark` (and `unknown`, conservatively) takes the rollout/eval
# path; data repos are adapted, not run; agents/libs are noted as reference.
KINDS = ("benchmark", "data-pipeline", "dataset", "agent", "library", "other")


def normalize_kind(raw: str) -> str:
    """Map an LLM's free-text kind (e.g. 'data-synthesis-pipeline') onto the fixed
    vocabulary. Returns 'unknown' if nothing matches (onboard treats unknown as a
    benchmark, the safe default)."""
    k = (raw or "").strip().lower().replace("_", "-").replace(" ", "-")
    if k in KINDS:
        return k
    if any(t in k for t in ("synth", "datagen", "data-gen", "data-generation")) or \
            ("pipeline" in k and "data" in k):
        return "data-pipeline"
    if any(t in k for t in ("dataset", "corpus", "trajector")):
        return "dataset"
    if any(t in k for t in ("benchmark", "eval", "harness", "suite")):
        return "benchmark"
    if any(t in k for t in ("agent", "framework", "runtime", "assistant")):
        return "agent"
    if any(t in k for t in ("library", "sdk", "toolkit", "package")):
        return "library"
    return "unknown"


@dataclass
class EnvRequirement:
    """One credential/config the benchmark needs."""
    key: str
    purpose: str
    example: str = ""
    secret: bool = False
    required: bool = True          # False = only needed for an opt-in domain


@dataclass
class BenchmarkSpec:
    """A benchmark's ID card — what it is and how to run it."""
    name: str
    repo_path: str
    adapter: str                   # adapter that handles this benchmark ("none" if unmatched)
    kind: str = "unknown"
    entrypoint: str = ""           # how a task is launched
    env_file: str = ""             # where secrets/config live (relative to repo)
    required_env: list[EnvRequirement] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)   # dependency hints found
    notes: list[str] = field(default_factory=list)
    # LLM-reported domains: [{name, tasks_dir, required_env:[keys]}]. The LLM
    # supplies STRUCTURE only; real task names are enumerated deterministically
    # (inventory) so they can't be hallucinated.
    domains: list[dict] = field(default_factory=list)
    # For kind in {data-pipeline, dataset}: where the produced data lives, so the
    # data adapter can locate + sample it. {hf_repo, local_glob, format}.
    data_source: dict = field(default_factory=dict)


def save_spec(spec: "BenchmarkSpec", spec_dir: Path = SPEC_DIR) -> Path:
    """Cache an explored spec so later commands (reproduce) skip re-exploring."""
    spec_dir.mkdir(parents=True, exist_ok=True)
    p = spec_dir / f"{spec.name}.json"
    p.write_text(json.dumps(asdict(spec), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_spec(name: str, spec_dir: Path = SPEC_DIR) -> Optional["BenchmarkSpec"]:
    p = spec_dir / f"{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    reqs = [EnvRequirement(**r) for r in d.get("required_env", [])]
    return BenchmarkSpec(
        name=d["name"], repo_path=d["repo_path"], adapter=d.get("adapter", "llm-explored"),
        kind=d.get("kind", "unknown"), entrypoint=d.get("entrypoint", ""),
        env_file=d.get("env_file", ""), required_env=reqs, deps=d.get("deps", []),
        notes=d.get("notes", []), domains=d.get("domains", []),
        data_source=d.get("data_source", {}))


@dataclass
class ProbeItem:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ProbeResult:
    items: list[ProbeItem] = field(default_factory=list)
    missing_env: list[EnvRequirement] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_env and all(i.ok for i in self.items)


@dataclass
class ElicitItem:
    """A structured prompt for something the user must supply."""
    key: str
    purpose: str
    how: str                       # instruction on how to provide it (safely)
    secret: bool = False


@dataclass
class SmokeResult:
    ran: bool                      # did the task execute end-to-end?
    passed: Optional[bool]         # did the verifier pass? (None if it didn't run)
    detail: str
    command: str = ""
