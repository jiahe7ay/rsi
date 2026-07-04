"""Turn an analysis (failure modes + fix hints) into a prompt PATCH: a concise
system-prompt addendum, saved to data/patches/<name>.json with full provenance
(what analysis, which HF source it came from). The patch is LAYERED at rollout
time (see ouro_patch) and NEVER edited into the benchmark's own prompts.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ouroboros.improve.analyze import _json_obj  # noqa: F401 (kept for symmetry)
from ouroboros.onboard.llm_explore import _llm

PATCH_DIR = Path("data/patches")


def generate(analysis: dict, model: str, base_url: str, key: str,
             name: str | None = None, source: dict | None = None) -> dict:
    """LLM condenses failure_modes/fix_hints into a clean, general prompt patch;
    persists {patch_text + provenance} under data/patches/. Returns the record."""
    modes = analysis.get("failure_modes", []) or []
    hints = "\n".join(f"- 模式: {m.get('mode','')}\n  修正: {m.get('fix_hint','')}" for m in modes)
    sysmsg = ("你是提示词工程师。把下列失败模式的修正建议,凝练成一段【可直接追加到 agent 系统提示词末尾】"
              "的补丁文本:通用、正向、命令式、面向该类任务的普遍改进,不写死具体任务名/文件名,不超过 12 行。"
              "只输出补丁正文本身,不要任何解释或标题。")
    user = f"分析总览: {analysis.get('summary','')}\n\n失败模式与修正建议:\n{hints or '(无)'}"
    patch_text = _llm([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
                      base_url, key, model, max_tokens=1200).strip()
    rec = {
        "name": name or f"patch-{datetime.now().strftime('%m%d-%H%M%S')}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source or {},                          # HF repo/revision/model it learned from
        "based_on": {"summary": analysis.get("summary", ""),
                     "failure_modes": [m.get("mode") for m in modes]},
        "patch_text": patch_text,
    }
    # NOTE: not persisted here — a patch influences all future rollouts, so it
    # goes through human review (improve.review.review_and_save) before landing
    # in data/patches/. Only an approved patch becomes usable by `rollout --patch`.
    return rec


def load_patch_text(path_or_name: str) -> str:
    """Return patch_text from a patch file (a path, or a name under data/patches/)."""
    p = Path(path_or_name)
    if not p.exists():
        p = PATCH_DIR / (path_or_name if path_or_name.endswith(".json") else f"{path_or_name}.json")
    return json.loads(p.read_text(encoding="utf-8")).get("patch_text", "")
