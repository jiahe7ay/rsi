"""Assess what's needed to REPRODUCE a data-pipeline's generation — so the missing
pieces can be ELICITED from the user interactively (agent judges → prompt user).

For kind=data-pipeline, "running" it means generating a small sample of data. The
agent JUDGES the requirements: it reads the entrypoint + pipeline guide, probes
which deps are already importable here, and reports exactly what's missing
(dependency / credential / endpoint / config) and FOR WHICH step — separating the
smallest reproducible step (usually question-gen from LOCAL tool schemas + one LLM)
from the full pipeline (which needs external tool-execution creds like Smithery).

The caller turns the returned `needs` (have=false) into an interactive prompt.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from ouroboros.onboard.llm_explore import _llm, _read_file, _run_command


def _dep_probe(deps: list[str]) -> dict:
    """Which deps are importable in THIS interpreter (have vs missing)."""
    out = {}
    for d in deps:
        r = subprocess.run(
            ["bash", "-c", f"python -c 'import {d}' 2>/dev/null && echo OK || echo NO"],
            capture_output=True, text=True, timeout=25)
        out[d] = r.stdout.strip().endswith("OK")
    return out


def _json_obj(text: str) -> dict:
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
            if isinstance(o, dict) and "needs" in o:
                return o
        except Exception:
            continue
    return {}


_SYS = (
    "你是数据生成管线【复现规划】专家。给你入口脚本源码、管线说明、当前环境依赖可用性、以及已知 env 键。"
    "判断:要复现【产出一小批数据】最小需要什么。\n"
    "关键区分:\n"
    "- 生成'问题/任务'这步通常只读【本地】工具 schema + 调一个 LLM,【不需要】外部工具执行凭证(如 Smithery);\n"
    "- 真实执行工具 / agent completion 那步才需要 Smithery 等外部凭证。\n"
    "对每个 need 给出 have(当前是否已满足):依赖看'当前环境依赖可用性',凭证/endpoint 看已知 env 键是否已在环境里。\n"
    "只输出一个 JSON:{\"target\":\"能产出数据的最小步骤+最小参数\",\"run_command\":\"最小复现命令\","
    "\"workdir\":\"命令应在 repo 内哪个相对目录执行(如 datagen;根目录则 .)\","
    "\"needs\":[{\"key\":\"依赖名/凭证名/endpoint\",\"kind\":\"dependency|credential|endpoint|config\","
    "\"purpose\":\"干嘛\",\"secret\":true/false,\"needed_for\":\"minimal|full\",\"have\":true/false}],"
    "\"blockers\":[\"当前对 minimal 目标最大的阻碍\"],\"ready_to_try\":true/false}")


def assess_reproduce(spec, model: str, base_url: str, key: str) -> dict:
    """Agent judges what's required to reproduce a small sample of the pipeline's data."""
    repo = Path(spec.repo_path)
    entry = spec.entrypoint or "datagen/step1.1_gen_questions.py"
    src = _read_file(repo, {"path": entry, "max_lines": 130})
    calls = _run_command(repo, {"cmd":
        f"grep -nE 'openai|OpenAI|client|completion|chat|api_key|base_url|litellm|"
        f"requests.post|model_config|getenv|environ' {entry} | head -30"})
    guide = ""
    for g in ("datagen/README.MD", "datagen/README.md", "README.md"):
        if (repo / g).is_file():
            guide = _read_file(repo, {"path": g, "max_lines": 70})
            break
    cfg = ""
    if (repo / "datagen/model_configs.json").is_file():
        cfg = _read_file(repo, {"path": "datagen/model_configs.json", "max_lines": 40})
    deps = ["torch", "numpy", "jinja2", "tqdm", "openai", "litellm", "transformers"]
    have = _dep_probe(deps)
    envkeys = [{"key": r.key, "purpose": r.purpose, "secret": r.secret} for r in spec.required_env]

    user = (f"入口脚本: {entry}\n\n=== 源码(前130行)===\n{src}\n\n"
            f"=== LLM 调用相关行(grep)===\n{calls}\n\n"
            f"=== 管线说明 ===\n{guide}\n\n=== model_configs.json ===\n{cfg}\n\n"
            f"=== 当前环境依赖可用性 ===\n{json.dumps(have, ensure_ascii=False)}\n\n"
            f"=== 已知 env 键(explore 得到)===\n{json.dumps(envkeys, ensure_ascii=False)}\n\n"
            f"=== notes ===\n{spec.notes}")
    out = _llm([{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
               base_url, key, model, max_tokens=1900)
    res = _json_obj(out)
    res["_dep_have"] = have          # attach the deterministic probe for the caller
    return res


# ---------------------------------------------------------------------------
# Interactive elicit + real run — the agent judged WHAT is missing; here the
# USER supplies it (terminal prompts, same pattern as the patch approval gate).
# Secrets are collected with hidden input, exported only to the child process,
# and never printed or persisted.
# ---------------------------------------------------------------------------

_DEP_NAME = re.compile(r'^[A-Za-z0-9_.\-\[\]=<>]+$')


def _sh(cmd: str, timeout: int) -> tuple[int, str]:
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)


def _pip_install(dep: str, launcher: str, timeout: int = 1500) -> tuple[bool, str]:
    if not _DEP_NAME.match(dep):
        return False, f"非法依赖名: {dep!r}"
    pre = f"{launcher} && " if launcher else ""
    rc, out = _sh(f"{pre}pip install {dep}", timeout)
    return rc == 0, out[-400:]


def _probe_import(dep: str, launcher: str) -> bool:
    mod = dep.split("[")[0].split("=")[0].split("<")[0].split(">")[0].replace("-", "_")
    pre = f"{launcher} && " if launcher else ""
    try:
        rc, _ = _sh(f"{pre}python -c 'import {mod}'", 60)
        return rc == 0
    except Exception:
        return False


def _ask(prompt: str) -> str:
    """input() that survives EOF (piped/non-interactive stdin) — returns '__eof__'
    so callers treat it as 'no answer' instead of crashing."""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        print("(EOF)")
        return "__eof__"


def elicit_interactive(missing: list[dict], launcher: str, auto: bool = False) -> tuple[dict, list[str]]:
    """Prompt the user for each missing need. Returns (env_extra, unresolved-minimal).
    dependency → offer pip install; credential/endpoint → hidden input (never echoed);
    config → ask the user to confirm it's ready. auto=True installs deps without
    asking but CANNOT fabricate credentials (they stay unresolved)."""
    import getpass
    env_extra: dict[str, str] = {}
    unresolved: list[str] = []
    for n in missing:
        kind = str(n.get("kind", "")).strip()
        key_ = str(n.get("key", "")).strip()
        for_min = n.get("needed_for") != "full"
        scope = "minimal(本次必需)" if for_min else "full(全流程才需要)"
        print(f"\n  ❌ 缺 [{kind}] {key_} — {n.get('purpose','')}\n     范围: {scope}")
        if not for_min:
            print("     → 本次最小复现不需要,跳过。")
            continue
        if kind == "dependency":
            ans = "y" if auto else _ask(f"     安装 {key_} 到当前环境? [Y/n]: ")
            if ans in ("", "y", "yes"):
                print(f"     pip install {key_} ...(可能要几分钟)")
                ok, tail = _pip_install(key_, launcher)
                ok = ok and _probe_import(key_, launcher)
                last = tail.strip().splitlines()[-1][:120] if tail.strip() else ""
                print(f"     {'✓ 已装好并可 import' if ok else '✗ 安装失败: ' + last}")
                if not ok:
                    unresolved.append(key_)
            else:
                unresolved.append(key_)
        elif kind in ("credential", "endpoint"):
            if os.environ.get(key_):
                print("     ✓ 已在当前环境变量中,直接用。")
                continue
            if auto:
                print("     ✗ [--yes] 凭证无法自动补,标记未解决。")
                unresolved.append(key_)
                continue
            try:
                val = getpass.getpass(f"     请输入 {key_} 的值(输入不回显,仅用于本次运行,不落盘): ")
            except Exception:   # EOF / no tty — treat as no answer
                val = ""
            if val.strip():
                env_extra[key_] = val.strip()
                print("     ✓ 已收下(不显示)。")
            else:
                print("     (空输入)")
                unresolved.append(key_)
        else:  # config
            ans = "y" if auto else _ask("     该项就绪了吗? [y/N]: ")
            if ans not in ("y", "yes"):
                unresolved.append(key_)
    return env_extra, unresolved


def _new_files(repo: Path, since: float, limit: int = 20) -> list[Path]:
    out = []
    for p in repo.rglob("*"):
        if p.is_file() and p.stat().st_mtime >= since - 1 \
                and ".git" not in p.parts and "__pycache__" not in p.parts:
            out.append(p)
            if len(out) >= limit:
                break
    return out


def run_reproduce(spec, model: str, base_url: str, key: str, launcher: str = "",
                  auto: bool = False, run_timeout: int = 900) -> dict:
    """assess (agent judges) → elicit (user fills the gaps) → run the minimal
    generation → deterministic success check (new files produced)."""
    import shlex
    import time
    repo = Path(spec.repo_path)
    print(f"▶ reproduce: agent 评估「{spec.name}」的最小复现需求 ...")
    res = assess_reproduce(spec, model, base_url, key)
    if not res.get("run_command"):
        print("  ✗ agent 未产出可执行的复现计划。")
        return {"ok": False, "stage": "assess"}

    print(f"\n  目标   : {res.get('target','')}")
    print(f"  命令   : {res['run_command']}")
    workdir = str(res.get("workdir") or Path(spec.entrypoint).parent or ".")
    print(f"  工作目录: {workdir}")
    needs = res.get("needs", []) or []
    for n in needs:
        mark = "✅" if n.get("have") else "❌"
        print(f"  {mark} [{str(n.get('kind','')):10}] {str(n.get('key',''))[:42]:42} ({n.get('needed_for')})")

    missing = [n for n in needs if not n.get("have")]
    env_extra: dict[str, str] = {}
    if missing:
        env_extra, unresolved = elicit_interactive(missing, launcher, auto=auto)
        if unresolved:
            print(f"\n⏸ 仍缺 {len(unresolved)} 项 minimal 必需: {', '.join(unresolved)} — 补齐后重跑。")
            return {"ok": False, "stage": "elicit", "unresolved": unresolved}

    print("\n  ▶ 实跑最小生成 ...")
    exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in env_extra.items())
    pre = f"{launcher} && " if launcher else ""
    cmd = f"{pre}cd {shlex.quote(str(repo / workdir))} && {exports}{res['run_command']}"
    t0 = time.time()
    try:
        rc, out = _sh(cmd, run_timeout)
    except subprocess.TimeoutExpired:
        print(f"  ✗ 超时({run_timeout}s)。")
        return {"ok": False, "stage": "run", "reason": "timeout"}
    tail = "\n".join(out.strip().splitlines()[-12:])
    print("  --- 输出尾部 ---")
    print("  " + tail.replace("\n", "\n  ")[:1800])

    produced = _new_files(repo, t0)
    ok = rc == 0 and bool(produced)
    if produced:
        print("\n  产出的新文件:")
        for p in produced[:8]:
            rel = p.relative_to(repo)
            note = ""
            if p.suffix in (".jsonl", ".json", ".txt") and p.stat().st_size < 50_000_000:
                try:
                    note = f"  ({sum(1 for _ in open(p, encoding='utf-8', errors='ignore'))} 行, {p.stat().st_size} B)"
                except Exception:
                    pass
            print(f"    + {rel}{note}")
    print(f"\n{'✅ 复现成功:管线真实产出了数据' if ok else '✗ 复现未成功(exit=' + str(rc) + (',无新产出文件' if not produced else '') + ')'}")
    return {"ok": ok, "stage": "run", "exit": rc,
            "produced": [str(p.relative_to(repo)) for p in produced]}
