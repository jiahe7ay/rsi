"""LLM-generated grading rubric for a benchmark task.

The rubric is Ouroboros' auto-verify artifact — the counterpart to onboard's spec
and improve's patch. Given ONLY a task description, an LLM distills the concrete,
checkable requirements (output format, must-include / must-exclude content, output
location, numeric precision, naming, …). It is saved to data/rubrics/<domain>__<task>.json
so it is reusable across samples and human-reviewable/editable — a wrong rubric
corrupts reward, so it must be inspectable, never a black box.

Nothing here is benchmark-specific: the LLM reads the task text and invents the
criteria. We only provide the harness + the artifact format.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ouroboros.onboard.llm_explore import _llm

RUBRIC_DIR = Path("data/rubrics")

_SYS = (
    "你是评测标准设计专家。给定一个 agent 任务的描述,提炼出【可据此判定完成质量的、"
    "具体可检查的评分标准】。每条标准必须聚焦任务明确要求、且能从 agent 的动作/产出中"
    "核验的点(例如:输出文件的格式与字段分隔、必须包含或必须排除的内容、输出文件的位置/"
    "命名、数值精度与四舍五入、排序、行结构等)。不要臆造任务没有要求的东西;不要写无法从"
    "轨迹核验的主观项。\n"
    "只输出一个 JSON:{\"criteria\":[{\"id\":\"c1\",\"criterion\":\"一句话可检查的标准\","
    "\"weight\":0~1 的重要度,\"critical\":true/false(不满足则整体判失败)}],"
    "\"notes\":\"评分时判官要特别注意的点\"}。"
    "8 条以内,weight 之和不必为 1(判分时归一化)。"
)


def _all_objs(text: str) -> list[str]:
    """Every balanced {...} span at any nesting depth (a stack of open braces).
    A truncated array still yields its already-closed element objects."""
    stack, spans = [], []
    for i, ch in enumerate(text):
        if ch == '{':
            stack.append(i)
        elif ch == '}' and stack:
            spans.append(text[stack.pop():i + 1])
    return spans


def _json_obj(text: str) -> dict:
    """A dict with 'criteria', else SALVAGE individual criterion objects from a
    truncated/malformed reply (responses get cut mid-array — the earlier elements
    are still complete and usable)."""
    for c in reversed(_all_objs(text)):                 # happy path: full wrapper dict
        try:
            o = json.loads(c)
            if isinstance(o, dict) and "criteria" in o:
                return o
        except Exception:
            continue
    crits = []                                           # salvage: closed element objects
    for blk in _all_objs(text):
        try:
            o = json.loads(blk)
        except Exception:
            continue
        if isinstance(o, dict) and o.get("criterion"):
            crits.append(o)
    return {"criteria": crits} if crits else {}


def generate_rubric(task_desc: str, domain: str, task_name: str,
                    model: str, base_url: str, key: str, tries: int = 3) -> dict:
    """LLM → a checkable rubric for this task. Returns the rubric record (unsaved).

    Reasoning models truncate intermittently, so run up to `tries` and KEEP THE
    MOST COMPLETE rubric (this artifact is cached & reused, so quality matters more
    than latency); stop early once a clearly-complete one (≥5 criteria) appears."""
    best, best_notes, last = [], "", ""
    for _ in range(max(1, tries)):
        out = _llm([{"role": "system", "content": _SYS},
                    {"role": "user", "content": f"任务描述:\n{task_desc[:4000]}"}],
                   base_url, key, model, max_tokens=1800)
        last = out
        parsed = _json_obj(out)
        criteria = []
        for i, c in enumerate(parsed.get("criteria", []) or []):
            crit = str(c.get("criterion", "")).strip()
            if not crit:
                continue
            criteria.append({
                "id": c.get("id") or f"c{i+1}",
                "criterion": crit,
                "weight": float(c.get("weight", 1.0) or 1.0),
                "critical": bool(c.get("critical", False)),
            })
        if len(criteria) > len(best):
            best, best_notes = criteria, str(parsed.get("notes", "")).strip()
        if len(best) >= 5:                # looks complete — no need to burn more calls
            break
    rec = {"task_name": task_name, "domain": domain, "criteria": best,
           "notes": best_notes, "generated_by": model, "task_excerpt": task_desc[:600]}
    if not best:                          # every attempt empty — keep raw for diagnosis
        rec["_raw_last"] = last[:1200]
    return rec


def rubric_path(task_name: str, domain: str, rubric_dir: Path = RUBRIC_DIR) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", f"{domain}__{task_name}")
    return Path(rubric_dir) / f"{safe}.json"


def save_rubric(rubric: dict, rubric_dir: Path = RUBRIC_DIR) -> Path:
    p = rubric_path(rubric["task_name"], rubric["domain"], rubric_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rubric, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_rubric(task_name: str, domain: str, rubric_dir: Path = RUBRIC_DIR) -> dict | None:
    p = rubric_path(task_name, domain, rubric_dir)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None
