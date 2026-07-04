"""LLM-agent self-exploration of a benchmark repo (the "heavyweight" analyze).

A minimal ReAct loop: the model gets read-only tools (list_dir / read_file /
run_command) and must explore an UNKNOWN benchmark to produce a BenchmarkSpec —
with no hand-written adapter. Experimental; on failure returns None so the caller
falls back to the adapter path.

Safety: never lets the model read secret files (.mcp_env/.env) or run mutating /
networked commands, so a repo's real credentials never enter the LLM context.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Optional

from ouroboros.onboard.spec import BenchmarkSpec, EnvRequirement, normalize_kind

SYSTEM = """你是 repo onboarding 专家。用工具探索给定 repo,先搞清【它是什么类型】,再搞清怎么用。
每一步【只输出一个 JSON】,不要任何多余文字:
  探索: {"thought":"...","action":"list_dir|read_file|run_command","args":{...}}
  完成: {"thought":"...","final":{"kind":"benchmark|data-pipeline|dataset|agent|library|other","entrypoint":"...","env_file":"...","required_env":[{"key":"...","purpose":"...","secret":true}],"domains":[{"name":"...","tasks_dir":"相对repo的任务目录,如 tasks/filesystem","required_env":["该域独需的env键,无则空"]}],"data_source":{"hf_repo":"org/name(若README指向HF数据集)","local_glob":"若产物落本地的相对路径","format":"parquet|jsonl|json"},"run_hint":"跑一个任务的命令","notes":["..."]}}
【kind 判定——最重要,决定后续怎么处理】:
  benchmark     = 有可跑任务 + 判分器,用来【评测】agent(如 mcpmark)
  data-pipeline = 【生成/合成数据】的管线,通常产出一个数据集(如 Toucan)
  dataset       = 静态数据集(轨迹/任务集合)
  agent         = agent 运行时/框架(本身没有可判分任务)
  library/other = 通用库/其他
【若 kind 是 data-pipeline 或 dataset】重点找【产出的数据在哪】:README 里的 HF 数据集链接(huggingface.co/datasets/...)或本地输出目录,填进 data_source。
【domains 仅 benchmark 需要】按服务/领域组织任务时,把每个域的 name、tasks_dir、该域独需 env 键列全——任务名不用列,系统会枚举。
工具:
  list_dir     args {"path":"."}
  read_file    args {"path":"README.md","max_lines":80}
  run_command  args {"cmd":"grep -n add_argument pipeline.py | head"}
规则:只读探索(ls/find/grep/head/cat/sed -n);禁止装/删/改/联网/长任务;.mcp_env/.env 的真值读不到也【不必读】(env 键名从 README/.env.example/代码推断即可);【信息足够就立刻输出 final,不要反复读同一文件、不要纠结 secret 文件】;最多 %d 步内必须给出 final。"""

_DANGER = re.compile(r'\b(rm|sudo|mv|dd|mkfs|shutdown|reboot|kill|pip|apt|apt-get|npm|yarn|curl|wget|git|chmod|chown|tee)\b|>>|>')
_SECRET = re.compile(r'\.mcp_env|(?<!\.example)\.env(\b|$)')


def _llm(messages, base_url, key, model, max_tokens=3000, retries=4):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {key}",
                                                  "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=150) as r:
                d = json.load(r)
            m = d["choices"][0]["message"]
            return (m.get("content") or m.get("reasoning_content") or "").strip()
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))  # backoff for 502 / transient网关抖动
    raise last


def _extract_json(text):
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
                cands.append(text[start:i + 1])
                start = None
    for c in reversed(cands):
        try:
            o = json.loads(c)
            if isinstance(o, dict) and ("action" in o or "final" in o):
                return o
        except Exception:
            continue
    return None


def _resolve(repo: Path, path_str: str) -> Optional[Path]:
    p = Path(path_str)
    p = (repo / p).resolve() if not p.is_absolute() else p.resolve()
    if p != repo and repo not in p.parents:
        return None  # outside repo
    return p


def _list_dir(repo, args):
    p = _resolve(repo, args.get("path", "."))
    if p is None or not p.exists():
        return "[error] path not found or outside repo"
    entries = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
    return "\n".join(entries[:200])


def _read_file(repo, args):
    path = args.get("path", "")
    if _SECRET.search(path):
        return "[blocked] 拒绝读取 secret 文件(可能含密钥)"
    p = _resolve(repo, path)
    if p is None or not p.is_file():
        return "[error] not a file or outside repo"
    lines = p.read_text(errors="ignore").splitlines()
    n = int(args.get("max_lines", 80))
    # agents page with either arg name; silently ignoring it traps them re-reading page 1
    off = int(args.get("offset", args.get("start_line", 0)) or 0)
    off = max(0, min(off, len(lines)))
    chunk = lines[off:off + n]
    header = f"(lines {off + 1}-{off + len(chunk)} of {len(lines)})\n"
    tail = f"\n... [{len(lines) - off - len(chunk)} more lines, 用 offset={off + len(chunk)} 继续]" \
        if off + len(chunk) < len(lines) else ""
    return header + "\n".join(chunk) + tail


def _run_command(repo, args):
    cmd = args.get("cmd", "")
    if _DANGER.search(cmd) or _SECRET.search(cmd):
        return "[blocked] 命令含禁止操作(mutating/networked)或触及 secret 文件"
    try:
        p = subprocess.run(["bash", "-c", cmd], cwd=str(repo),
                           capture_output=True, text=True, timeout=30)
        return (p.stdout + p.stderr)[:2500] or "[no output]"
    except subprocess.TimeoutExpired:
        return "[error] command timeout (30s)"


def _to_spec(repo: Path, final: dict) -> BenchmarkSpec:
    reqs = []
    for r in final.get("required_env", []) or []:
        if isinstance(r, dict) and r.get("key"):
            reqs.append(EnvRequirement(r["key"], r.get("purpose", ""), secret=bool(r.get("secret"))))
    domains = []
    for d in final.get("domains", []) or []:
        if isinstance(d, dict) and d.get("name"):
            domains.append({
                "name": str(d["name"]).strip(),
                "tasks_dir": str(d.get("tasks_dir") or "").strip(),
                "required_env": [str(k).strip() for k in (d.get("required_env") or []) if str(k).strip()],
            })
    ds_raw = final.get("data_source") or {}
    data_source = {k: str(ds_raw.get(k, "")).strip() for k in ("hf_repo", "local_glob", "format")} \
        if isinstance(ds_raw, dict) else {}
    data_source = {k: v for k, v in data_source.items() if v}   # drop empty fields
    spec = BenchmarkSpec(name=repo.name, repo_path=str(repo), adapter="llm-explored",
                         kind=normalize_kind(final.get("kind", "unknown")),
                         entrypoint=final.get("entrypoint", ""),
                         env_file=final.get("env_file", ""),
                         required_env=reqs,
                         notes=list(final.get("notes", []) or []),
                         domains=domains, data_source=data_source)
    if final.get("run_hint"):
        spec.notes.append("run_hint: " + str(final["run_hint"]))
    return spec


def explore(repo_path, model="deepseek-v4-pro", max_turns=12,
            base_url=None, key=None) -> Optional[BenchmarkSpec]:
    repo = Path(repo_path).expanduser().resolve()
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    key = key or os.environ.get("OPENAI_API_KEY")
    if not base_url or not key:
        print("[explore] 缺 OPENAI_BASE_URL/OPENAI_API_KEY(先 source 一个 .mcp_env)")
        return None
    tools = {"list_dir": _list_dir, "read_file": _read_file, "run_command": _run_command}
    messages = [{"role": "system", "content": SYSTEM % max_turns},
                {"role": "user", "content": f"repo 路径: {repo}\n开始探索(先 list_dir '.')。"}]
    for turn in range(1, max_turns + 1):
        try:
            out = _llm(messages, base_url, key, model)
        except Exception as e:
            print(f"[explore] LLM 调用失败: {e}")
            return None
        obj = _extract_json(out)
        if obj is None:
            print(f"  [turn {turn}] 未产出合法 JSON,提示重试")
            messages.append({"role": "assistant", "content": out[:400]})
            messages.append({"role": "user", "content": "只输出一个合法 JSON(action 或 final),不要多余文字。"})
            continue
        if "final" in obj:
            print(f"  [turn {turn}] FINAL ·· {str(obj.get('thought', ''))[:80]}")
            return _to_spec(repo, obj["final"])
        action, a = obj.get("action", ""), obj.get("args", {}) or {}
        print(f"  [turn {turn}] {action} {json.dumps(a, ensure_ascii=False)[:70]}  ·· {str(obj.get('thought', ''))[:55]}")
        result = tools.get(action, lambda *_: "[error] unknown action")(repo, a)
        messages.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"结果:\n{result}"})
    print(f"[explore] 达到 max_turns={max_turns} 仍未 final")
    return None


def _extract_cmd(text: str) -> str:
    m = re.search(r'<CMD>\s*(.+?)\s*</CMD>', text, re.S)   # preferred: explicit marker
    if not m:
        m = re.search(r'```(?:bash|sh)?\s*(.+?)```', text, re.S)
    if m:
        body = [l.strip() for l in m.group(1).strip().splitlines()
                if l.strip() and not l.strip().startswith("#")]
        if body:
            return " ".join(body)
    for line in reversed(text.splitlines()):
        s = line.strip().strip("`").strip()
        if any(k in s for k in ("python", "pipeline", "conda activate", "bash ", "./run")):
            return s
    return ""  # 提不到就返回空(比把 reasoning 思考文字当命令跑安全)


def _task_hints(repo: Path) -> str:
    """Deterministically list the benchmark's REAL task dirs, fed to the LLM so it
    picks an existing --tasks value instead of hallucinating one (e.g. basic_file_ops)."""
    for d in ("tasks", "benchmarks", "cases", "scenarios", "suites"):
        p = repo / d
        if p.is_dir():
            try:
                out = subprocess.run(["bash", "-c", f"find '{p}' -maxdepth 3 -type d | head -60"],
                                     capture_output=True, text=True, timeout=15).stdout.strip()
            except Exception:
                out = ""
            if out:
                rel = "\n".join(l.replace(str(repo) + "/", "") for l in out.splitlines())
                return "\n真实任务目录(--tasks 只能用其中【真实存在】的,禁止编造 task 名):\n" + rel
    return ""


def smoke_command(spec, model, base_url, key, prev_cmd=None, error=None) -> str:
    """Ask the LLM to produce (or repair) a one-line shell command that runs ONE
    minimal task of this benchmark with the given model. Pure-LLM; self-heals on error.
    Real task names are fed in (see _task_hints) so the model doesn't invent them."""
    from dataclasses import asdict
    keep = {k: v for k, v in asdict(spec).items()
            if k in ("kind", "entrypoint", "env_file", "notes", "required_env")}
    spec_txt = json.dumps(keep, ensure_ascii=False)
    hints = _task_hints(Path(spec.repo_path))
    sysmsg = ("你是 benchmark 运维专家。把【最终要执行的命令放进 <CMD>...</CMD> 标记内】,"
              "标记内只有一行可直接执行的 shell 命令(可含 && / source / conda activate)。标记外可有简短思考。")
    if error:
        user = (f"benchmark spec: {spec_txt}\nrepo: {spec.repo_path}{hints}\n"
                f"上次命令:\n{prev_cmd}\n报错(尾部):\n{error[-900:]}\n"
                f"请【修正】命令,仍用 model={model} 跑单个最简任务;--tasks 只能用上面真实目录里存在的,禁止编造。")
    else:
        user = (f"benchmark spec: {spec_txt}\nrepo: {spec.repo_path}{hints}\n"
                f"给一条命令:在该 repo 根目录、用 model={model}、跑【单个最小任务】做冒烟(不是全量,越小越好);"
                f"--tasks 只能用上面真实目录里存在的 task,禁止编造。如需 conda/source 环境请包含。")
    out = _llm([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
               base_url, key, model, max_tokens=1500)
    return _extract_cmd(out)


# ---------------------------------------------------------------------------
# Smoke as an agent loop — the model diagnoses the MACHINE itself (conda envs,
# which python, how repo scripts activate) instead of us hardcoding hints.
# Only two things stay deterministic: the success judge (Tasks passed Y>0) and
# the safety fences (no sudo/rm, no reading secret file contents).
# ---------------------------------------------------------------------------

SMOKE_SYSTEM = """你是 benchmark 运维 agent。目标:让这个 benchmark 在本机【真实跑通一个最小任务】。
成功判定由系统自动完成(输出含 Tasks passed X/Y 且 Y>0),你不用宣布成功。
每步只输出一个 JSON,不要多余文字:
  诊断: {"thought":"...","action":"list_dir|read_file|run_command","args":{...},"facts":["可选"]}
  试跑: {"thought":"...","action":"run_smoke","args":{"cmd":"一行可执行 shell 命令"},"facts":["可选"]}
【facts 记事本】每步可带 "facts":["新确认的关键事实"](如:哪个解释器存在/哪个目录里有什么环境/缺什么依赖/脚本怎么启动)。
系统会把你记过的全部事实固定在上下文顶部——【工具输出会被淹没,facts 不会】,发现关键信息立刻记下来。
工具:
  list_dir/read_file/run_command — 只读诊断(可查机器环境:which/conda env list/ls ~/miniconda3/envs;可 grep repo 脚本看别人怎么激活环境;pip/git/curl/rm 等被禁)
  run_smoke — 真正执行你的候选命令(timeout 600s),返回输出尾部
提示:命令报错(如 ModuleNotFoundError / command not found)时,先做必要诊断——查清依赖装在哪个环境、正确解释器是什么,不要盲改。
【行动优先】真实报错是最好的诊断:通常 2-4 步诊断后就应该 run_smoke 一次,不要为不确定因素(如 model 名是否注册)反复读文件——直接试,报错会告诉你。禁止重复同一动作。--tasks 只能用真实存在的 task。最多 %d 步。"""

_SMOKE_CMD_DANGER = re.compile(r'\b(sudo|shutdown|reboot|mkfs|rm|chown)\b|\bdd\b')
_SMOKE_CMD_SECRET_READ = re.compile(
    r'\b(cat|head|tail|grep|sed|awk|less|more|strings)\b[^|;&\n]*(\.mcp_env|\.env(?!\.(example|template|sample)))')


def _exec_smoke(repo: Path, cmd: str, timeout: int) -> str:
    if not cmd.strip():
        return "[error] empty cmd"
    if _SMOKE_CMD_DANGER.search(cmd):
        return "[blocked] 命令含破坏性操作(sudo/rm/dd 等),换一种方式"
    if _SMOKE_CMD_SECRET_READ.search(cmd):
        return "[blocked] 禁止读取 secret 文件内容(source 加载可以,cat/grep 不行)"
    try:
        p = subprocess.run(["bash", "-c", cmd], cwd=str(repo),
                           capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr)[-4000:] or "[no output]"
    except subprocess.TimeoutExpired:
        return f"[timeout {timeout}s] 命令超时(任务可能太大/卡住;考虑更小的任务或加超时参数)"


def solve_smoke(spec, model, base_url, key, max_turns=24, run_timeout=600) -> dict:
    """Agent loop: diagnose the machine + iterate run_smoke until a task REALLY runs.
    Returns {ok, cmd, passed?, turns}. Success is judged deterministically (Y>0)."""
    from dataclasses import asdict
    repo = Path(spec.repo_path)
    keep = {k: v for k, v in asdict(spec).items() if k in ("kind", "entrypoint", "env_file", "notes")}
    hints = _task_hints(repo)
    tools = {"list_dir": _list_dir, "read_file": _read_file, "run_command": _run_command}
    base_user = (f"repo: {repo}\nspec: {json.dumps(keep, ensure_ascii=False)}"
                 f"{hints}\n目标:用 model={model} 跑通一个最小任务。开始。")
    messages = [
        {"role": "system", "content": SMOKE_SYSTEM % max_turns},
        {"role": "user", "content": base_user},
    ]
    last_cmd = None
    smoke_attempts = 0
    seen: dict[str, int] = {}          # anti-stall: repeated identical actions
    facts: list[str] = []              # agent's own pinned scratchpad (survives attention decay)
    for turn in range(1, max_turns + 1):
        out = _llm(messages, base_url, key, model)
        obj = _extract_json(out)
        if obj is None:
            print(f"  [turn {turn}] 未产出合法 JSON,提示重试")
            messages += [{"role": "assistant", "content": out[:400]},
                         {"role": "user", "content": "只输出一个合法 JSON(action),不要多余文字。"}]
            continue
        action, a = obj.get("action", ""), obj.get("args", {}) or {}
        # pin agent-recorded facts into the FIRST user message (top-of-context):
        # tool outputs drown in a long loop; the scratchpad does not.
        new_facts = [str(f).strip() for f in (obj.get("facts") or []) if str(f).strip()]
        if new_facts:
            facts.extend(f for f in new_facts if f not in facts)
            messages[1]["content"] = base_user + "\n\n[你已确认的事实(自己记录,勿忘)]\n- " + "\n- ".join(facts)
        print(f"  [turn {turn}] {action} {json.dumps(a, ensure_ascii=False)[:90]}  ·· {str(obj.get('thought', ''))[:55]}"
              + (f"  +facts:{len(new_facts)}" if new_facts else ""))
        if action == "run_smoke":
            smoke_attempts += 1
            last_cmd = str(a.get("cmd", "")).strip()
            result = _exec_smoke(repo, last_cmd, run_timeout)
            m = re.search(r"Tasks passed:\s*(\d+)/(\d+)", result)
            if m and int(m.group(2)) > 0:
                return {"ok": True, "cmd": last_cmd,
                        "passed": f"{m.group(1)}/{m.group(2)}", "turns": turn}
            if m:
                result += "\n[系统判定] 0/0 = 没有任务被执行(task 选择器无匹配),不算跑通。"
            print(f"       ✗ 未跑通: {result.strip().splitlines()[-1][:110]}")
            feed = "\n".join(result.strip().splitlines()[-25:])[:2500]
        else:
            feed = tools.get(action, lambda *_: "[error] unknown action")(repo, a)
            key_ = action + "|" + json.dumps(a, sort_keys=True, ensure_ascii=False)
            seen[key_] = seen.get(key_, 0) + 1
            if seen[key_] > 1:
                feed = (f"[注意] 这是你第 {seen[key_]} 次执行完全相同的动作,内容不会变。"
                        f"换个方向,或直接 run_smoke 用真实报错定位。\n") + feed
        # action-pressure nudge: pure diagnosis for 4+ turns with zero attempts
        if smoke_attempts == 0 and turn >= 4:
            feed += (f"\n[系统] 已 {turn} 步纯诊断、0 次试跑。剩 {max_turns - turn} 步,"
                     "下一步请直接 run_smoke——真实报错比继续读文件信息量大。")
        # reflexion at 1/3 and 2/3 budget: force it to consolidate ITS OWN
        # findings into facts — fixation on one lead (while an earlier tool
        # result already held the answer) is the observed failure mode.
        if turn in (max_turns // 3, (2 * max_turns) // 3):
            feed += (f"\n[系统] 复盘点({turn}/{max_turns})。回看你此前【每个】工具结果,"
                     "把所有已确认的事实用 facts 字段记全(解释器/环境目录内容/依赖/脚本启动方式),"
                     "特别是【已发现但还没利用的线索】——然后选证据最强的路线试跑。")
        messages += [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                     {"role": "user", "content": f"结果:\n{feed}"}]
    return {"ok": False, "cmd": last_cmd, "turns": max_turns}
