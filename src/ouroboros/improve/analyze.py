"""Analyze pulled trajectories: deterministic failure aggregation + an LLM pass
that names the COMMON failure modes and what prompt guidance would fix them.

Deterministic part (aggregate) is cheap and exact. LLM part (summarize_failures)
reads a few of the worst failing trajectories as evidence — it produces the
insight that patch_gen turns into an actual prompt patch.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from ouroboros.onboard.llm_explore import _llm


def _json_obj(text: str) -> dict:
    """Extract the first well-formed dict carrying our expected keys. analyze's
    output isn't the explore action/final shape, so _extract_json won't match it."""
    cands = []
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S)
    if m:
        cands.append(m.group(1))
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                cands.append(text[start:i + 1]); start = None
    for c in reversed(cands):
        try:
            o = json.loads(c)
            if isinstance(o, dict) and ("failure_modes" in o or "summary" in o):
                return o
        except Exception:
            continue
    return {}


def aggregate(trajs: list) -> list[dict]:
    """Per (domain, task): #runs, #fails, sample errors, model. Sorted worst-first."""
    by: dict[tuple, dict] = defaultdict(lambda: {"runs": 0, "fails": 0, "errs": set(), "model": ""})
    for t in trajs:
        s = by[(t.domain, t.task_name)]
        s["runs"] += 1
        s["fails"] += 0 if t.success else 1
        if not t.success and t.verification_error:
            s["errs"].add(str(t.verification_error)[:140])
        s["model"] = t.model or s["model"]
    rows = [{"domain": d, "task": tk, "runs": v["runs"], "fails": v["fails"],
             "fail_rate": round(v["fails"] / v["runs"], 2) if v["runs"] else 0.0,
             "errs": list(v["errs"])[:3], "model": v["model"]}
            for (d, tk), v in by.items()]
    return sorted(rows, key=lambda r: (-r["fails"], -r["fail_rate"], r["domain"]))


def summarize_failures(trajs: list, model: str, base_url: str, key: str,
                       max_tasks: int = 8, msgs_chars: int = 3000) -> dict:
    """LLM reads the worst failing trajectories → common failure modes + fix hints."""
    agg = aggregate(trajs)
    worst = [r for r in agg if r["fails"] > 0][:max_tasks]
    evidence = []
    for r in worst:
        ft = next((t for t in trajs
                   if t.domain == r["domain"] and t.task_name == r["task"] and not t.success), None)
        if not ft:
            continue
        convo = json.dumps(ft.messages, ensure_ascii=False)[:msgs_chars]
        evidence.append(f"[{r['domain']}/{r['task']}] 失败 {r['fails']}/{r['runs']} 次;"
                        f"verifier错误={r['errs']}\n轨迹节选:\n{convo}")
    if not evidence:
        return {"aggregate": agg, "worst": worst, "failure_modes": [],
                "summary": "无失败轨迹可分析。"}

    sysmsg = ("你是 agent 失败分析专家。读下列【失败轨迹】,归纳可用【系统提示词补丁】修正的共性问题。"
              "只输出一个 JSON:{\"failure_modes\":[{\"mode\":\"失败模式\",\"evidence\":\"依据\","
              "\"fix_hint\":\"系统提示词该补充的具体指导\"}],\"summary\":\"一句话总览\"}")
    user = "失败轨迹如下:\n\n" + "\n\n---\n\n".join(evidence)
    out = _llm([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
               base_url, key, model, max_tokens=2500)
    parsed = _json_obj(out)
    return {
        "aggregate": agg,
        "worst": worst,
        "failure_modes": parsed.get("failure_modes", []),
        "summary": parsed.get("summary", out[:400]),
    }
