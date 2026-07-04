"""Auto-verify orchestrator: for ANY benchmark task, turn an agent trajectory into
a reward — with an LLM-generated, human-reviewable rubric and an LLM judge that
EXPLAINS ground truth rather than overriding it.

    task + trajectory (+ authoritative verdict, when available)
        → render the agent's real actions           (trajectory_view)
        → load-or-generate a rubric for the task     (rubric; cached & reviewable)
        → judge: grade compliance + explain verdict  (judge; ground-truth-anchored)
        → VerifyResult(passed=ground truth, reward=ground truth, detail=explanation)

`passed`/`reward` follow the authoritative verifier (mcpmark) whenever it exists;
the LLM contributes the per-criterion rubric, a graded compliance_score (diagnostic
only), an explanation of the verdict, and — the payoff — a failure_locus that turns
mcpmark's silent "no error message" into "the failure is semantic, not format".

`verify_hf` is the improve-facing entry: it re-scores an OLD model's HF trajectories,
keeping mcpmark's verdict authoritative and attaching the explanation improve needs.
"""
from __future__ import annotations

from pathlib import Path

from ouroboros.schema import VerifyResult
from ouroboros.verify.trajectory_view import extract_task, render_actions
from ouroboros.verify.rubric import RUBRIC_DIR, generate_rubric, load_rubric, save_rubric
from ouroboros.verify.judge import judge


def verify_one(messages: list, task_name: str, domain: str, model: str,
               base_url: str, key: str, ground_truth: dict | None = None,
               task_desc: str | None = None, rubric_dir: Path = RUBRIC_DIR,
               regen: bool = False) -> tuple[VerifyResult, dict]:
    """Verify one trajectory. Returns (VerifyResult, rubric_used).

    ground_truth {"passed":bool,"error":str|None}: the authoritative verdict. When
    given, passed/reward follow it and the judge explains it; when None, the judge
    gives a best-effort compliance verdict labeled judge-unverified.
    The rubric is generated once per task and cached (unless regen)."""
    task_desc = task_desc or extract_task(messages)
    rubric = None if regen else load_rubric(task_name, domain, rubric_dir)
    fresh = rubric is None
    if fresh:
        rubric = generate_rubric(task_desc, domain, task_name, model, base_url, key)
        if rubric.get("criteria"):
            save_rubric(rubric, rubric_dir)

    actions = render_actions(messages)
    v = judge(task_desc, actions, rubric, model, base_url, key, ground_truth=ground_truth)
    result = VerifyResult(
        passed=v["passed"], reward=v["reward"],
        detail={
            "authority": v["authority"],               # mcpmark-ground-truth | judge-unverified
            "signal": v["signal"],
            "mcpmark_error": (ground_truth or {}).get("error"),
            "compliance_score": v["compliance_score"],  # rubric-graded, DIAGNOSTIC only
            "explanation": v["explanation"],            # why the verdict happened
            "failure_locus": v["failure_locus"],        # semantic|numeric|selection|omission|format|unknown
            "disagreement": v["disagreement"],          # compliance vs authoritative verdict
            "disagreement_note": v["disagreement_note"],
            "per_criterion": v["per_criterion"],
            "rubric_task": task_name, "rubric_fresh": fresh,
            "judge_model": model,
        },
        env_rebuilt=False,
    )
    return result, rubric


def verify_hf(repo: str, revision: str, model: str, base_url: str, key: str,
              domains: list[str] | None = None, only: str | None = None,
              limit: int | None = None, rubric_dir: Path = RUBRIC_DIR,
              regen: bool = False) -> list[dict]:
    """Pull HF trajectories and auto-verify each, keeping mcpmark's verdict
    authoritative. Returns per-trajectory records
    {task_name, domain, run, mcpmark_success, mcpmark_error, verify:VerifyResult}."""
    from ouroboros.improve.hf_traj import pull
    trajs = pull(repo, revision, domains=domains, only=only, limit=limit)
    out = []
    for t in trajs:
        gt = {"passed": t.success, "error": t.verification_error}
        result, _ = verify_one(t.messages, t.task_name, t.domain, model, base_url, key,
                               ground_truth=gt, rubric_dir=rubric_dir, regen=regen)
        out.append({
            "task_name": t.task_name, "domain": t.domain, "run": t.run,
            "mcpmark_success": t.success, "mcpmark_error": t.verification_error,
            "verify": result,
        })
    return out
