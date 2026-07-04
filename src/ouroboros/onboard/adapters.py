"""Benchmark adapters — the only benchmark-specific code in onboarding.

The framework (steps/orchestrator) is generic; each concrete benchmark gets an
adapter that knows its required env, how to probe its endpoint/services, and how
to smoke-test one task. Add a new benchmark = add an adapter here and register it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from ouroboros.onboard.spec import BenchmarkSpec, EnvRequirement, ProbeItem, SmokeResult


def _run(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)


class BenchmarkAdapter:
    name = "base"
    kind = "unknown"

    def matches(self, repo: Path) -> bool:            # is this repo mine?
        return False

    def enrich(self, spec: BenchmarkSpec) -> None:    # add benchmark-specific spec fields
        ...

    def probe_env(self, spec: BenchmarkSpec, model: str) -> list[ProbeItem]:
        return []

    def smoke(self, spec: BenchmarkSpec, model: str) -> SmokeResult:
        return SmokeResult(ran=False, passed=None, detail="adapter has no smoke")


_REGISTRY: list[BenchmarkAdapter] = []


def register(a: BenchmarkAdapter) -> None:
    _REGISTRY.append(a)


def find_adapter(repo: Path) -> Optional[BenchmarkAdapter]:
    for a in _REGISTRY:
        try:
            if a.matches(repo):
                return a
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# MCPMark — the first real adapter (encodes what onboarding discovered by hand)
# ---------------------------------------------------------------------------
class MCPMarkAdapter(BenchmarkAdapter):
    name = "mcpmark"
    kind = "mcp-agent-benchmark"
    CONDA_ENV = "mcpmark"
    # a minimal, zero-external-secret smoke task (filesystem domain)
    SMOKE = dict(mcp="filesystem", suite="easy", task="file_context/uppercase")

    ENV = [
        EnvRequirement("OPENAI_BASE_URL", "agent 用的 LLM 端点(OpenAI 兼容)", "https://.../v1"),
        EnvRequirement("OPENAI_API_KEY", "该端点的 key", "sk-...", secret=True),
        EnvRequirement("POSTGRES_HOST", "postgres 域用", "localhost", required=False),
        EnvRequirement("POSTGRES_PORT", "postgres 域用", "5432", required=False),
        EnvRequirement("POSTGRES_USERNAME", "postgres 域用", "postgres", required=False),
        EnvRequirement("POSTGRES_PASSWORD", "postgres 域用", "", secret=True, required=False),
    ]

    def matches(self, repo: Path) -> bool:
        return ((repo / "pipeline.py").exists()
                and (repo / "tasks").is_dir()
                and (repo / "src" / "mcp_services").is_dir())

    def enrich(self, spec: BenchmarkSpec) -> None:
        spec.kind = self.kind
        spec.entrypoint = ("python -m pipeline --mcp <domain> --task-suite <suite> "
                           "--tasks <cat/task> --models openai/<model> --k <k>")
        spec.env_file = ".mcp_env"
        spec.required_env = list(self.ENV)
        spec.notes.append("litellm 需 model 名带 openai/ 前缀(裸名报 'LLM Provider NOT provided')")
        spec.notes.append("filesystem 域零外部 secret;postgres 域需本地 pg")

    def probe_env(self, spec: BenchmarkSpec, model: str) -> list[ProbeItem]:
        repo = spec.repo_path
        items: list[ProbeItem] = []
        # endpoint connectivity via GET /models — NEVER prints the key
        ep = (f'cd {repo} && set -a; . ./{spec.env_file} 2>/dev/null; set +a; '
              'curl -sS -m15 "$OPENAI_BASE_URL/models" '
              '-H "Authorization: Bearer $OPENAI_API_KEY" -o /dev/null -w "%{http_code}"')
        try:
            code = _run(ep, timeout=25).stdout.strip()[-3:]
            items.append(ProbeItem("endpoint (GET /models)", code == "200", f"HTTP {code}"))
        except Exception as e:
            items.append(ProbeItem("endpoint (GET /models)", False, f"err {e}"))
        # postgres container (needed only for postgres domain; informational for fs smoke)
        try:
            name = _run("docker ps --format '{{.Names}}' | grep -i postgres | head -1", 15).stdout.strip()
            items.append(ProbeItem("postgres container", bool(name), name or "none running"))
        except Exception:
            items.append(ProbeItem("postgres container", False, "docker unavailable"))
        return items

    def smoke(self, spec: BenchmarkSpec, model: str) -> SmokeResult:
        s, repo = self.SMOKE, spec.repo_path
        cmd = (f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {self.CONDA_ENV} && "
               f"cd {repo} && rm -rf results/ouro_onboard_smoke && "
               f"python -m pipeline --mcp {s['mcp']} --task-suite {s['suite']} "
               f"--tasks {s['task']} --models {model} --k 1 --exp-name ouro_onboard_smoke")
        try:
            p = _run(cmd, timeout=600)
        except subprocess.TimeoutExpired:
            return SmokeResult(ran=False, passed=None, detail="smoke timeout (600s)", command=cmd)
        out = p.stdout + "\n" + p.stderr
        m = re.search(r"Tasks passed:\s*(\d+)/(\d+)", out)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            return SmokeResult(ran=True, passed=(x > 0), detail=f"Tasks passed: {x}/{y}", command=cmd)
        tail = "\n".join(out.strip().splitlines()[-8:])
        return SmokeResult(ran=False, passed=None, detail="未解析到结果;输出尾部:\n" + tail, command=cmd)


register(MCPMarkAdapter())
