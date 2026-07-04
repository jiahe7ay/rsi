"""Task inventory — after LLM exploration, answer the newcomer's question:
"这个 benchmark 有哪些 domain / 多少 task,我现在能跑哪些?"

Division of labor (anti-hallucination, same philosophy as _task_hints):
  * the LLM supplies STRUCTURE (domain names, tasks_dir, per-domain env keys);
  * deterministic code enumerates the REAL task names from disk and checks env
    readiness by KEY PRESENCE in the env file (values are never read).

The inventory is written to data/benchmarks/<name>.json so `ouro split` (with no
--domains) can freeze exactly the runnable domains.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ouroboros.onboard.spec import BenchmarkSpec
from ouroboros.onboard.steps import _env_keys_with_values

INVENTORY_DIR = Path("data/benchmarks")   # relative to the ouroboros project cwd


def _dirs(p: Path) -> list[Path]:
    try:
        return [x for x in sorted(p.iterdir()) if x.is_dir() and not x.name.startswith((".", "_"))]
    except Exception:
        return []


def _has_task_files(task_dir: Path) -> bool:
    """A real task dir contains files (description/meta/verify), not just empty
    subdirs. Guards against placeholder trees. (rglob follows symlinks.)"""
    try:
        return any(f.is_file() for f in task_dir.rglob("*"))
    except Exception:
        return False


def _leaf_groups(base: Path) -> dict[str, list[str]]:
    """Enumerate task ids grouped by suite when a suite layer exists.

    Layouts handled: <suite>/<category>/<task> (mcpmark) and flat
    <category>/<task> (grouped under "all"). Only real dirs with task files count
    — task NAMES never come from the LLM.
    """
    groups: dict[str, list[str]] = {}
    for suite in _dirs(base):
        leaf = [f"{c.name}/{t.name}" for c in _dirs(suite) for t in _dirs(c)
                if _has_task_files(t)]
        if leaf:
            groups[suite.name] = leaf
    if groups:
        return groups
    leaf = [f"{c.name}/{t.name}" for c in _dirs(base) for t in _dirs(c)
            if _has_task_files(t)]
    return {"all": leaf} if leaf else {}


def build_inventory(spec: BenchmarkSpec) -> dict:
    repo = Path(spec.repo_path)
    set_keys = _env_keys_with_values(repo / spec.env_file) if spec.env_file else set()
    inv = {
        "benchmark": spec.name,
        "repo": str(repo),
        "env_file": spec.env_file,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "domains": [],
    }
    # Dedup domains that resolve to the SAME real path — mcpmark ships symlink
    # aliases (tasks/insforge -> postgres); counting both would double-freeze the
    # same tasks (duplicate eval / reweighted training). Keep the first seen
    # canonical path; record the alias so it's visible, not silently dropped.
    seen_real: dict[str, str] = {}
    for d in spec.domains:
        tdir = (repo / d["tasks_dir"]) if d.get("tasks_dir") else None
        real = None
        if tdir is not None:
            try:
                real = str(tdir.resolve())
            except Exception:
                real = str(tdir)
        alias_of = seen_real.get(real) if real else None
        groups = _leaf_groups(tdir) if (tdir is not None and alias_of is None) else {}
        missing = [k for k in d.get("required_env", []) if k not in set_keys]
        entry = {
            "name": d["name"],
            "tasks_dir": d.get("tasks_dir", ""),
            "suites": {s: len(ts) for s, ts in groups.items()},
            "tasks": groups,
            "required_env": d.get("required_env", []),
            "missing_env": missing,
            "runnable": bool(groups) and not missing and alias_of is None,
        }
        if alias_of is not None:
            entry["alias_of"] = alias_of          # symlink duplicate — not runnable on its own
        elif real is not None:
            seen_real[real] = d["name"]
        inv["domains"].append(entry)
    return inv


def save_inventory(inv: dict) -> Path:
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = INVENTORY_DIR / f"{inv['benchmark']}.json"
    path.write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_inventory(inv: dict) -> None:
    if not inv["domains"]:
        print("  (explore 未产出 domains — 无法列任务清单;可重跑 onboard 或人工看 repo)")
        return
    for d in inv["domains"]:
        suites = " + ".join(f"{s} {n}" for s, n in d["suites"].items()) or "0 tasks"
        if d.get("alias_of"):
            mark, why = "↪ 软链接别名", f"(= {d['alias_of']},不重复计)"
        elif d["runnable"]:
            mark, why = "✅ 可跑", "(所需 env 已配)" if d["required_env"] else "(无额外 secret)"
        elif not d["suites"]:
            mark, why = "⚠ 未找到任务目录", f"(tasks_dir={d['tasks_dir'] or '?'})"
        else:
            mark, why = "❌ 缺 env", ": " + ", ".join(d["missing_env"])
        print(f"  {d['name']:<22} {suites:<28} {mark}{why}")


def latest_inventory() -> dict | None:
    files = sorted(INVENTORY_DIR.glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return json.loads(files[0].read_text(encoding="utf-8"))
