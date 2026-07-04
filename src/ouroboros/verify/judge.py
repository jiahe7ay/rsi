"""LLM-as-judge — but it EXPLAINS ground truth, it does not override it.

A judge that reads only the transcript tends to rubber-stamp a confident agent
(observed: it scored a ground-truth-FAILED budget task as 7/7 passed). So when an
authoritative verdict exists (mcpmark on HF trajectories / after rollout), that
verdict is the correctness authority and `passed`/`reward` follow it. The judge's
job shifts to attribution:

  1. Grade each rubric criterion (compliance) — strict, evidence-required.
  2. EXPLAIN the authoritative verdict — and when it's FAIL yet every format/structure
     criterion is met, locate the real failure (semantic / numeric / selection / omission).
  3. Flag disagreement between the compliance picture and the verdict — computed
     deterministically, not left to the LLM.

`compliance_score` (weighted fraction of criteria met) is kept as a DIAGNOSTIC
channel only; it is never the training reward when ground truth is present.

Fallback: with no ground truth (a brand-new benchmark, or pre-verification), the
judge gives a best-effort compliance verdict clearly labeled `judge-unverified`.
"""
from __future__ import annotations

import json
import re

from ouroboros.onboard.llm_explore import _llm

# Ground-truth present → the judge is an attribution analyst, not a decider.
_SYS_EXPLAIN = (
    "你是评测归因分析师。这道任务的【正确性判定已由权威评测器给出,是事实,不容推翻】。"
    "你的职责不是重新判对错,而是:\n"
    "1. 逐条核验 agent 的动作/产出是否满足每条评分标准(compliance);met=true 需轨迹中有"
    "明确证据,证据不足一律 false。\n"
    "2. 解释这个权威判定为什么会发生。**尤其当权威判为失败、而格式/结构类标准却都满足时**,"
    "定位失败最可能的性质:semantic(语义/理解错误)、numeric(数值/计算错误)、selection"
    "(选错了对象/集合)、omission(遗漏)、format(格式)、unknown;给出具体假设 + 轨迹依据。\n"
    "3. 若 compliance 画面与权威判定矛盾(如'全满足'却被判失败),点出矛盾在哪。\n"
    "只输出一个 JSON:{\"per_criterion\":[{\"id\":\"c1\",\"met\":true/false,\"evidence\":"
    "\"轨迹依据或'无'\"}],\"verdict_explanation\":\"为什么是这个权威判定(一到两句)\","
    "\"failure_locus\":\"semantic|numeric|selection|omission|format|unknown(通过时留空)\","
    "\"disagreement_note\":\"compliance 与权威判定的矛盾点,无则留空\"}。"
)

# No ground truth → best-effort strict judge (necessary-not-sufficient signal).
_SYS_STANDALONE = (
    "你是严格的评测判官。给你【任务描述】【评分标准】【agent 的真实动作与产出】,逐条判定。"
    "铁律:只有轨迹中有明确证据才判 met=true;证据不足、或需要标准答案才能确定的,一律 false"
    "(宁可漏判不可错判)。你看不到标准答案,只能核验'是否符合任务的明确要求',不要为 agent 的"
    "自信背书。只输出 JSON:{\"per_criterion\":[{\"id\":\"c1\",\"met\":true/false,\"evidence\":"
    "\"依据或'无'\"}],\"verdict_explanation\":\"总体判断理由\",\"failure_locus\":\"\","
    "\"disagreement_note\":\"\"}。"
)


def _all_objs(text: str) -> list[str]:
    """Every balanced {...} span at any depth (stack of open braces); a truncated
    reply still yields its already-closed objects."""
    stack, spans = [], []
    for i, ch in enumerate(text):
        if ch == '{':
            stack.append(i)
        elif ch == '}' and stack:
            spans.append(text[stack.pop():i + 1])
    return spans


def _json_obj(text: str) -> dict:
    """A dict with 'per_criterion', else SALVAGE the per-criterion verdicts (dicts
    carrying a 'met' flag) from a truncated/malformed reply."""
    for c in reversed(_all_objs(text)):                 # happy path: full verdict dict
        try:
            o = json.loads(c)
            if isinstance(o, dict) and "per_criterion" in o:
                return o
        except Exception:
            continue
    items = []                                           # salvage: closed verdict items
    for blk in _all_objs(text):
        try:
            o = json.loads(blk)
        except Exception:
            continue
        if isinstance(o, dict) and "met" in o and "id" in o:
            items.append(o)
    return {"per_criterion": items} if items else {}


def judge(task_desc: str, actions_text: str, rubric: dict, model: str,
          base_url: str, key: str, ground_truth: dict | None = None,
          pass_threshold: float = 1.0) -> dict:
    """Grade + explain one trajectory.

    ground_truth: {"passed": bool, "error": str|None} — the authoritative verdict.
    When present, passed/reward follow it and the judge explains it. When None,
    the judge gives a best-effort compliance verdict labeled `judge-unverified`.

    Returns {passed, reward, per_criterion, compliance_score, authority,
             explanation, failure_locus, disagreement, disagreement_note, signal}.
    """
    criteria = rubric.get("criteria", []) or []
    if not criteria:
        gt = bool((ground_truth or {}).get("passed"))
        return {"passed": gt if ground_truth else False,
                "reward": (1.0 if gt else 0.0) if ground_truth else 0.0,
                "per_criterion": [], "compliance_score": 0.0,
                "authority": "mcpmark-ground-truth" if ground_truth else "judge-unverified",
                "explanation": "空 rubric,无法核验", "failure_locus": "",
                "disagreement": False, "disagreement_note": "", "signal": "empty-rubric"}

    crit_lines = "\n".join(
        f"- [{c['id']}] (w={c['weight']}{',关键' if c.get('critical') else ''}) {c['criterion']}"
        for c in criteria)
    has_gt = ground_truth is not None
    if has_gt:
        gt_pass = bool(ground_truth.get("passed"))
        gt_err = ground_truth.get("error")
        auth = (f"权威评测器判定: {'通过(PASS)' if gt_pass else '失败(FAIL)'}"
                + (f";评测器原始信息: {gt_err}" if gt_err else ";评测器未给出具体原因"))
        sysmsg, head = _SYS_EXPLAIN, auth + "\n\n"
    else:
        sysmsg, head = _SYS_STANDALONE, ""

    user = (f"{head}任务描述:\n{task_desc[:3000]}\n\n评分标准:\n{crit_lines}\n"
            f"{('判官须知: ' + rubric['notes']) if rubric.get('notes') else ''}\n\n"
            f"agent 的真实动作与产出(轨迹):\n{actions_text[:12000]}")
    # Reasoning models emit clean JSON only intermittently — retry until the verdict
    # parses, so an unparsed reply isn't silently misread as "every criterion failed".
    parsed, parse_ok = {}, False
    for _ in range(3):
        out = _llm([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
                   base_url, key, model, max_tokens=2000)
        parsed = _json_obj(out)
        if parsed.get("per_criterion"):
            parse_ok = True
            break

    # Deterministic compliance from the rubric (weight/critical come from the rubric,
    # not whatever the LLM echoed back).
    verdicts = {v.get("id"): v for v in parsed.get("per_criterion", []) if isinstance(v, dict)}
    per, tot_w, got_w, crit_ok = [], 0.0, 0.0, True
    for c in criteria:
        v = verdicts.get(c["id"], {})
        met = bool(v.get("met", False))
        w = float(c.get("weight", 1.0) or 1.0)
        tot_w += w
        got_w += w if met else 0.0
        if c.get("critical") and not met:
            crit_ok = False
        per.append({**c, "met": met, "evidence": str(v.get("evidence", ""))[:400]})
    compliance = round(got_w / tot_w, 4) if tot_w else 0.0

    explanation = str(parsed.get("verdict_explanation", ""))[:600]
    locus = str(parsed.get("failure_locus", "")).strip().lower()
    if not parse_ok:  # judge never returned parseable JSON — don't fake a verdict
        explanation = explanation or "判官三次未返回可解析 JSON,compliance 不可信"
        locus = "parse-failed"

    if has_gt:
        passed = gt_pass
        reward = 1.0 if gt_pass else 0.0                       # authority = ground truth
        # disagreement = compliance picture contradicts the authoritative verdict
        # (only meaningful when the judge actually parsed).
        disagreement = parse_ok and (
            (not gt_pass and compliance >= 0.999) or (gt_pass and compliance < 0.999))
        return {
            "passed": passed, "reward": reward,
            "per_criterion": per, "compliance_score": compliance,
            "authority": "mcpmark-ground-truth",
            "explanation": explanation,
            "failure_locus": "" if gt_pass else (locus or "unknown"),
            "disagreement": disagreement,
            "disagreement_note": str(parsed.get("disagreement_note", ""))[:400],
            "signal": "ground-truth-correctness + rubric-explanation",
        }
    # No authority: best-effort, clearly labeled unverified.
    passed = bool(crit_ok and compliance >= pass_threshold)
    return {
        "passed": passed, "reward": compliance,
        "per_criterion": per, "compliance_score": compliance,
        "authority": "judge-unverified",
        "explanation": explanation, "failure_locus": "",
        "disagreement": False, "disagreement_note": "",
        "signal": "requirement-compliance-unverified",
    }
