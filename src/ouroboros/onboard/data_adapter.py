"""Adapt a DATA repo (kind=data-pipeline|dataset) into an Ouroboros data source.

explore() classifies a repo's kind; for a data repo the benchmark flow (probe →
smoke → rollout → eval) does not apply — there are no runnable tasks and no
objective verifier. So instead of running it, we ADAPT it:

    locate the produced dataset (spec.data_source / grep README for a HF link)
        → sample a few REAL rows without downloading shards (HF datasets-server
          /rows API — returns JSON rows)
        → LLM maps the observed fields onto Ouroboros' Trajectory schema, and —
          load-bearing for RSI — reports whether it carries a VERIFIABLE reward
          or only LLM quality scores.

Why the reward distinction matters: a quality-filtered dataset (Toucan) is great
for SFT bootstrap / diversity, but its signal is NOT the objective pass/fail an RL
reward must be. The adapter surfaces that instead of silently treating a quality
score as reward — same principle as auto-verify's "explain, don't override".
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from ouroboros.onboard.llm_explore import _llm

DS_SERVER = "https://datasets-server.huggingface.co"
OUT_DIR = Path("data/datasources")


def _get(url: str, token: str | None = None, tries: int = 6):
    """GET JSON with backoff — the datasets-server 502s while warming a dataset."""
    last = None
    for i in range(tries):
        try:
            hdr = {"Authorization": f"Bearer {token}"} if token else {}
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=45) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(3 * (i + 1))
    raise last


def _json_obj(text: str) -> dict:
    """First balanced {...} that parses and carries a 'field_map' key."""
    cands, depth, start = [], 0, None
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
            if isinstance(o, dict) and "field_map" in o:
                return o
        except Exception:
            continue
    return {}


def locate_data_source(spec, repo: Path) -> dict:
    """Where the produced data lives: spec.data_source first, else grep the repo
    for a huggingface.co/datasets/<owner>/<name> link (robust & cheap)."""
    ds = dict(spec.data_source or {})
    if ds.get("hf_repo"):
        return {**ds, "source": "spec"}
    try:
        out = subprocess.run(
            ["bash", "-c",
             f"grep -rhoE 'huggingface.co/datasets/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+' "
             f"{repo} 2>/dev/null | head -40"],
            capture_output=True, text=True, timeout=25).stdout
    except Exception:
        out = ""
    hits = [ln.split("datasets/", 1)[1].rstrip("/.") for ln in out.splitlines() if "datasets/" in ln]
    if hits:
        best = max(set(hits), key=hits.count)
        return {"hf_repo": best, "format": ds.get("format", ""), "source": "grep-readme"}
    return {"hf_repo": None, "local_glob": ds.get("local_glob", ""),
            "format": ds.get("format", ""), "source": "none"}


def list_configs(hf_repo: str, token: str | None = None) -> list[tuple[str, str]]:
    d = _get(f"{DS_SERVER}/splits?dataset={hf_repo}", token)
    return [(c["config"], c["split"]) for c in d.get("splits", [])]


def sample_rows(hf_repo: str, config: str, split: str, limit: int = 3,
                token: str | None = None) -> list[dict]:
    d = _get(f"{DS_SERVER}/rows?dataset={hf_repo}&config={config}&split={split}"
             f"&offset=0&length={limit}", token)
    return [r.get("row", {}) for r in d.get("rows", [])]


_SCHEMA_HINT = (
    "目标 = Ouroboros Trajectory schema(schema.py):task_id:str、domain:str、"
    "split:'train'|'eval'、sample_index:int、messages:list[dict](OpenAI 对话,含 tool 调用)、"
    "meta(model 等 provenance)、verify:{passed:bool, reward:float}(**客观判分**,可选)。"
)


def map_to_schema(rows: list[dict], model: str, base_url: str, key: str) -> dict:
    """LLM maps the observed dataset fields onto the Trajectory schema and judges
    usability — crucially, whether there is a VERIFIABLE reward or only quality scores."""
    r0 = rows[0] if rows else {}
    digest = []
    for k, v in r0.items():
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        digest.append(f"- {k} ({type(v).__name__}): {s[:200]}")
    sysmsg = (
        "你是数据集适配专家。把观察到的源字段映射到目标 Trajectory schema,并判断可用性。"
        "【关键】目标的 verify 要求【客观 pass/fail 判分】。如果源里只有 LLM 质量评分/打分"
        "(如 *_quality_assessment、score、rating),那它【不是可验证 reward】——只能做 SFT / 增多样性,"
        "不能当 RL reward。据实判断,别把质量分当成 reward。\n"
        "只输出一个 JSON:{\"field_map\":[{\"ouro_field\":\"task_id|domain|messages|...\","
        "\"source_field\":\"源字段名 或 derive 或 missing\",\"note\":\"简短\"}],"
        "\"messages_compatible\":true/false,\"has_verifiable_reward\":true/false,"
        "\"sft_usable\":true/false,\"verdict\":\"一句话结论:这数据集对 Ouroboros 怎么用最合适\"}")
    user = f"{_SCHEMA_HINT}\n\n观察到的源字段(取自真实样本一行):\n" + "\n".join(digest)
    out = _llm([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
               base_url, key, model, max_tokens=1600)
    return _json_obj(out)


def _print_mapping(m: dict) -> None:
    if not m:
        print("  (LLM 未产出可解析的字段映射)")
        return
    print("  字段映射(源 → Trajectory):")
    for fm in m.get("field_map", []):
        print(f"    - {str(fm.get('ouro_field','?')):13} ← {str(fm.get('source_field','?')):24} "
              f"{str(fm.get('note','')):.48}")
    print(f"  messages 兼容: {m.get('messages_compatible')}    SFT 可用: {m.get('sft_usable')}")
    rew = m.get("has_verifiable_reward")
    print(f"  可验证 reward: {rew}" +
          ("" if rew else "  ← 只有质量分,不能当 RL reward;作 SFT 冷启动/多样性"))
    print(f"  结论: {m.get('verdict','')}")


def adapt_data_source(spec, model: str, base_url: str, key: str) -> dict:
    """Route target for kind in {data-pipeline, dataset}: locate → sample → map."""
    repo = Path(spec.repo_path)
    print("\n── ② 分流:数据仓库(非 benchmark)→ 数据适配 " + "─" * 22)
    src = locate_data_source(spec, repo)
    if not src.get("hf_repo"):
        print("  ✗ 未能定位产出数据集(spec.data_source 空,README 无 HF 链接)。")
        if src.get("local_glob"):
            print(f"  提示:产物可能落在本地 {src['local_glob']}(需先跑 pipeline 生成)。")
        return {"stage": "data-adapt", "ok": False, "reason": "no-dataset-located", "kind": spec.kind}

    hf = src["hf_repo"]
    token = os.environ.get("HF_TOKEN")
    print(f"  数据集: {hf}  (定位来源: {src['source']})")
    try:
        cfgs = list_configs(hf, token)
    except Exception as e:
        print(f"  ✗ 查询数据集失败(datasets-server): {type(e).__name__} {e}")
        return {"stage": "data-adapt", "ok": False, "reason": "server-error", "hf_repo": hf}
    if not cfgs:
        print("  ✗ 数据集无可用 config/split。")
        return {"stage": "data-adapt", "ok": False, "reason": "no-config", "hf_repo": hf}
    print(f"  子集/split: {[f'{c}/{s}' for c, s in cfgs][:8]}")
    cfg, split = next(((c, s) for c, s in cfgs if c.lower() in ("sft", "default")), cfgs[0])
    print(f"  取样自: {cfg}/{split}(优先 SFT 就绪子集)")
    try:
        rows = sample_rows(hf, cfg, split, limit=3, token=token)
    except Exception as e:
        print(f"  ✗ 取样失败: {type(e).__name__} {e}")
        return {"stage": "data-adapt", "ok": False, "reason": "sample-error", "hf_repo": hf}
    fields = list(rows[0].keys()) if rows else []
    print(f"  样本字段: {fields}")

    mapping = map_to_schema(rows, model, base_url, key)
    _print_mapping(mapping)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    art = {"name": spec.name, "kind": spec.kind, "hf_repo": hf, "config": cfg, "split": split,
           "sample_fields": fields, "mapping": mapping, "located_via": src["source"]}
    out_path = OUT_DIR / f"{spec.name}.json"
    out_path.write_text(json.dumps(art, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  → 已存数据源适配: {out_path}")
    ok = bool(mapping.get("sft_usable"))
    return {"stage": "data-adapt", "ok": ok, "kind": spec.kind, "hf_repo": hf, "config": cfg,
            "has_verifiable_reward": bool(mapping.get("has_verifiable_reward")), "sft_usable": ok}
