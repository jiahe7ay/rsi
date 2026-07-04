"""(5) Eval harness — pass@k / per-domain report on the held-out split.

The RSI judge. Uses the SAME primitive as rollout (rollout.run_split) but on the
frozen EVAL split, then aggregates k samples per task into pass@k. This is the one
trustworthy answer to "is generation t+1 actually better than t". It must never
touch the train split.

pass@k here = fraction of tasks with >=1 passing sample among k (capability
ceiling). avg@k = mean pass rate per task (stability). Both are reported.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ouroboros import config, rollout
from ouroboros.schema import EvalReport, RunMeta


def _pass_at_k(passes: int, total: int, k: int) -> float:
    """Fraction of "at least 1 of the first k samples passed", estimated simply
    as (any pass among the total samples) when total<=k. With total==k this is
    exactly pass@k; we keep it simple for v1 (no unbiased-estimator combinatorics).
    """
    return 1.0 if passes > 0 else 0.0


def evaluate(checkpoint: str | None = None, model: str = "openai/deepseek-v4-pro",
             k: tuple[int, ...] = (1, 4), domains: list[str] | None = None,
             suite: str = "easy", limit: int | None = None,
             exp_name: str | None = None, config_path=None) -> EvalReport:
    """Roll out over the EVAL split (k samples/task), verify, compute pass@k +
    per-domain, and persist an EvalReport under data/eval/."""
    cfg = config.load(config_path)
    kmax = max(k)
    trajs, exp = rollout.run_split(
        model=model, split="eval", n=kmax, domains=domains, suite=suite,
        limit=limit, checkpoint=checkpoint, exp_name=exp_name,
        config_path=config_path, exp_prefix="ouro-eval")

    # group samples by (domain, task) → pass count / total
    by_task: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for tr in trajs:
        ok = bool(tr.verify and tr.verify.passed)
        by_task[(tr.domain, tr.task_id)].append(ok)

    n_tasks = len(by_task)
    pass_at: dict[int, float] = {}
    per_domain: dict[str, dict[int, float]] = defaultdict(dict)
    dom_tasks: dict[str, list[tuple[int, int]]] = defaultdict(list)  # (passes,total) per task
    all_tasks: list[tuple[int, int]] = []
    for (dom, _task), results in by_task.items():
        passes, total = sum(results), len(results)
        all_tasks.append((passes, total))
        dom_tasks[dom].append((passes, total))

    for kk in sorted(set(k)):
        if all_tasks:
            pass_at[kk] = round(sum(_pass_at_k(p, t, kk) for p, t in all_tasks) / len(all_tasks), 4)
        for dom, lst in dom_tasks.items():
            per_domain[dom][kk] = round(sum(_pass_at_k(p, t, kk) for p, t in lst) / len(lst), 4)
    # avg@k (mean per-task pass rate) reported under key 0 for convenience
    if all_tasks:
        pass_at[0] = round(sum(p / t for p, t in all_tasks if t) / len(all_tasks), 4)
        for dom, lst in dom_tasks.items():
            per_domain[dom][0] = round(sum(p / t for p, t in lst if t) / len(lst), 4)

    report = EvalReport(
        checkpoint=checkpoint, n_tasks=n_tasks, pass_at=pass_at,
        per_domain={d: dict(v) for d, v in per_domain.items()},
        meta=RunMeta(model=model, checkpoint=checkpoint,
                     base_url=__import__("os").environ.get("OPENAI_BASE_URL", ""),
                     created_at=datetime.now().isoformat(timespec="seconds")),
    )
    out_dir = Path((cfg.get("paths", {}) or {}).get("data_dir") or "./data").expanduser() / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{exp}.json").write_text(report.to_json(), encoding="utf-8")

    # human summary
    print(f"\n── EvalReport ({exp}) ──")
    print(f"  tasks: {n_tasks}   samples/task(k): {kmax}")
    print(f"  avg@k : {pass_at.get(0)}")
    for kk in sorted(x for x in pass_at if x > 0):
        print(f"  pass@{kk}: {pass_at[kk]}")
    for dom, v in report.per_domain.items():
        print(f"    {dom:<12} avg@k={v.get(0)}  " +
              "  ".join(f"pass@{kk}={v[kk]}" for kk in sorted(x for x in v if x > 0)))
    print(f"  → {out_dir / (exp + '.json')}")
    return report
