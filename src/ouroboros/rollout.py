"""(2) Rollout engine — generate reward-labeled trajectories.

v1 drives mcpmark's OWN pipeline (the validated path: agent execution + isolated
env + verifier, all in one) and then harvests its results directory into our
Trajectory schema. We deliberately do not re-implement the agent loop; a custom
engine (hermes-agent) can slot in later behind the same run() signature.

Rollout and eval MUST share this primitive (only task split + sampling differ)
so there is no train/eval skew.

Known limitation (v1): sampling temperature is controlled inside mcpmark's model
config, not from here — SamplingConfig records only n_samples until we expose it.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from ouroboros import config, splits, store
from ouroboros.schema import RunMeta, SamplingConfig, Trajectory, VerifyResult

# generous ceiling per pipeline invocation (pipeline's own per-task timeout is 3600)
_TASK_TIMEOUT_S = 3900


def _launcher(cfg: dict) -> str:
    """Shell prefix that puts us in the env where mcpmark's deps live."""
    r = cfg.get("rollout", {}) or {}
    return r.get("launcher") or (
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate mcpmark")


# dedicated DB for postgres-domain rollouts, on a non-5432 host port to avoid
# clobbering whatever else runs there. mcpmark reads POSTGRES_HOST/PORT from env.
_PG_CONTAINER = "ouro-postgres"
_PG_IMAGE = "pgvector/pgvector:0.8.0-pg17-bookworm"
_PG_HOST_PORT = 55432


def ensure_postgres(password: str) -> dict:
    """Ensure a clean pgvector DB is up for postgres-domain rollouts; return the
    env overrides (HOST/PORT) that point mcpmark at it. Idempotent."""
    running = subprocess.run(["bash", "-c",
        f"docker ps --format '{{{{.Names}}}}' | grep -qx {_PG_CONTAINER}"]).returncode == 0
    if not running:
        subprocess.run(["bash", "-c", f"docker rm -f {_PG_CONTAINER} 2>/dev/null || true"])
        up = subprocess.run(["bash", "-c",
            f"docker run -d --name {_PG_CONTAINER} -p {_PG_HOST_PORT}:5432 "
            f"-e POSTGRES_DB=postgres -e POSTGRES_USER=postgres "
            f"-e POSTGRES_PASSWORD={password or 'password'} {_PG_IMAGE}"],
            capture_output=True, text=True)
        if up.returncode != 0:
            raise RuntimeError(f"起 postgres 容器失败: {up.stderr[-300:]}")
        for _ in range(30):
            if subprocess.run(["bash", "-c",
                f"docker exec {_PG_CONTAINER} pg_isready -U postgres"],
                    capture_output=True).returncode == 0:
                break
            import time
            time.sleep(1)
    return {"POSTGRES_HOST": "localhost", "POSTGRES_PORT": str(_PG_HOST_PORT),
            "POSTGRES_USERNAME": "postgres",
            "POSTGRES_PASSWORD": password or "password", "POSTGRES_DATABASE": "postgres"}


def rollout_task(repo: Path, launcher: str, domain: str, suite: str, task: str,
                 model: str, n: int, exp: str, env_extra: dict | None = None,
                 patch_text: str | None = None) -> tuple[int, str]:
    """Run ONE task × k samples by driving mcpmark's pipeline directly (local,
    new-version code + layout — matches how we enumerate splits).

    env_extra: per-domain env overrides (postgres DB endpoint). patch_text: when
    set, run via the ouro_patch bootstrap that layers the prompt patch onto the
    agent SYSTEM_PROMPT (mcpmark source untouched). No patch = plain pipeline.
    """
    exports = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in (env_extra or {}).items())
    if patch_text:
        exports += f"export OURO_PROMPT_PATCH={shlex.quote(patch_text)}; "
        shim = shlex.quote(str(Path(__file__).resolve().parent / "improve" / "ouro_patch.py"))
        runner = f"python {shim}"          # layers patch, then hands off to pipeline
    else:
        runner = "python -m pipeline"
    cmd = (f"{launcher} && cd {repo} && {exports}{runner} --mcp {domain} "
           f"--task-suite {suite} --tasks {task} --models {model} --k {n} --exp-name {exp}")
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                       timeout=_TASK_TIMEOUT_S * max(1, n))
    return p.returncode, (p.stdout + p.stderr)[-1500:]


def harvest(repo: Path, exp: str, split: str, model: str, checkpoint: str | None,
            n: int) -> list[Trajectory]:
    """Walk results/<exp>/**/run-N/<cat>__<task>/ and convert to Trajectories.

    meta.json is the ground truth for the reward: execution_result.success is the
    mcpmark verifier's judgement; raw evidence rides along in verify.detail for
    reward-hacking audits.
    """
    base = repo / "results" / exp
    trajs: list[Trajectory] = []
    for meta_path in sorted(base.rglob("meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_dir = meta_path.parent                     # .../run-N/<cat>__<task>
        try:
            sample_index = int(run_dir.parent.name.split("-")[-1]) - 1
        except Exception:
            sample_index = 0
        messages = []
        mp = run_dir / "messages.json"
        if mp.exists():
            try:
                messages = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                messages = []
        exec_res = meta.get("execution_result") or {}
        passed = bool(exec_res.get("success"))
        trajs.append(Trajectory(
            task_id=str(meta.get("task_name", "")).replace("__", "/", 1),
            domain=str(meta.get("mcp", "")),
            split=split,
            sample_index=sample_index,
            messages=messages,
            meta=RunMeta(
                model=model,
                checkpoint=checkpoint,
                base_url=os.environ.get("OPENAI_BASE_URL", ""),
                sampling=SamplingConfig(n_samples=n),   # temp/max_turns: mcpmark-internal (v1)
                created_at=datetime.now().isoformat(timespec="seconds"),
            ),
            verify=VerifyResult(
                passed=passed,
                reward=1.0 if passed else 0.0,
                detail={
                    "error_message": exec_res.get("error_message"),
                    "verification_error": exec_res.get("verification_error"),
                    "token_usage": meta.get("token_usage"),
                    "turn_count": meta.get("turn_count"),
                    "agent_execution_time": meta.get("agent_execution_time"),
                },
                env_rebuilt=True,   # mcpmark rebuilds/restores env per task
            ),
            api_calls=int(meta.get("turn_count") or 0),
            duration_s=float(meta.get("task_execution_time") or 0.0),
        ))
    return trajs


def run_split(model: str, split: str, n: int, domains: list[str] | None, suite: str,
              limit: int | None, checkpoint: str | None, exp_name: str | None,
              config_path, exp_prefix: str, patch_text: str | None = None) -> tuple[list, str]:
    """Shared core for rollout AND eval: run every task in a split (k samples via
    the benchmark), then harvest verified Trajectories. Returns (trajectories, exp).

    The ONLY difference between rollout and eval is the split arg + how the caller
    consumes the result — this guarantees no train/eval skew.
    """
    cfg = config.load(config_path)
    repo = splits.mcpmark_root(cfg)
    launcher = _launcher(cfg)
    tasks = splits.load_split(split, domains=domains, suite=suite, config_path=config_path)
    if limit:
        tasks = tasks[: int(limit)]
    exp = exp_name or f"{exp_prefix}-{split}-{datetime.now().strftime('%m%d-%H%M%S')}"

    print(f"▶ {exp_prefix}: {len(tasks)} task(s) × k={n}  model={model}  "
          f"split={split}/{suite}  exp={exp}")

    # provision managed services once, up front (postgres needs a live DB)
    domain_env: dict[str, dict] = {}
    if any(t["domain"] == "postgres" for t in tasks):
        pw = os.environ.get("POSTGRES_PASSWORD", "password")
        print("  · postgres 域:确保专用 DB 容器就绪 ...", flush=True)
        try:
            domain_env["postgres"] = ensure_postgres(pw)
            print(f"    ✓ {_PG_CONTAINER} @ localhost:{_PG_HOST_PORT}")
        except Exception as e:
            print(f"    ✗ 起 DB 失败: {e} — postgres 任务将跳过")

    for i, t in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {t['domain']}/{t['task']} ...", flush=True)
        if t["domain"] == "postgres" and "postgres" not in domain_env:
            print("      ✗ 无可用 DB — skipped")
            continue
        try:
            rc, tail = rollout_task(repo, launcher, t["domain"], t["suite"], t["task"],
                                    model, n, exp, env_extra=domain_env.get(t["domain"]),
                                    patch_text=patch_text)
        except subprocess.TimeoutExpired:
            print("      ✗ timeout — skipped")
            continue
        if rc != 0:
            last = tail.strip().splitlines()[-1] if tail.strip() else ""
            print(f"      ✗ pipeline rc={rc}: {last[:140]}")

    trajs = harvest(repo, exp, split, model, checkpoint, n)
    return trajs, exp


def run(model: str = "openai/deepseek-v4-pro", split: str = "train", n: int = 1,
        domains: list[str] | None = None, suite: str = "easy",
        limit: int | None = None, checkpoint: str | None = None,
        exp_name: str | None = None, patch: str | None = None, config_path=None) -> dict:
    """Generate + verify + store trajectories for a split into data/rollouts/<exp>.jsonl.

    patch: name/path of a patch file (data/patches/<name>.json) to layer onto the
    agent's prompt for THIS rollout — the prompt-level self-improvement lever."""
    cfg = config.load(config_path)
    patch_text = None
    if patch:
        from ouroboros.improve.patch_gen import load_patch_text
        patch_text = load_patch_text(patch)
        print(f"  · 应用提示词补丁 [{patch}] ({len(patch_text)} 字符)")
    trajs, exp = run_split(model, split, n, domains, suite, limit, checkpoint,
                           exp_name, config_path, exp_prefix="ouro", patch_text=patch_text)
    out_dir = Path((cfg.get("paths", {}) or {}).get("data_dir") or "./data").expanduser() / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{exp}.jsonl"
    passed = 0
    for tr in trajs:
        store.append(tr, out_path)
        passed += 1 if (tr.verify and tr.verify.passed) else 0
    print(f"✓ harvested {len(trajs)} trajectories ({passed} passed) → {out_path}")
    return {"exp": exp, "trajectories": len(trajs), "passed": passed, "path": str(out_path)}
