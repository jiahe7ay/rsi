"""`ouro agent` — the top-level interactive agent: the USER states a goal in
plain language, and the AGENT decides how to fulfil it by composing the existing
`ouro` commands (and ordinary shell) itself.

This inverts the old flow: instead of a human/Claude planning which subcommand to
run, the agent gets (goal + a live snapshot of `ouro --help` + tools) and drives
a ReAct loop. It asks the user — through a terminal prompt — whenever it needs
clarification, a confirmation, or a credential.

Reused, already-proven harness mechanics (from onboard/llm_explore):
  * facts scratchpad pinned at top-of-context (survives attention decay)
  * one-JSON-per-turn protocol + salvage-tolerant extraction
  * deterministic safety fences (no destructive commands, no reading secret files)
  * errors are fed back verbatim — the agent diagnoses and retries (self-heal)

Secrets: ask_user(secret=true) collects via getpass; the VALUE goes only into the
agent's env dict (exported to child commands) — it is never echoed, never stored,
and never enters the LLM context.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from ouroboros.onboard.llm_explore import _llm, _extract_json, _list_dir, _read_file

SYSTEM = """你是 Ouroboros 顶层执行 agent。用户给你一个需求,你的职责:理解 → 规划 → 用工具一步步实现 → 完成(或明确报告卡点)。
每步【只输出一个 JSON】,不要任何多余文字:
  {"thought":"...","action":"run_command","args":{"cmd":"一行 shell","timeout_s":600},"facts":["可选:新确认的关键事实"]}
  {"thought":"...","action":"ask_user","args":{"prompt":"要问用户的话","secret":false,"env_key":"secret 为 true 时必填:值注入的环境变量名"}}
  {"thought":"...","action":"read_file","args":{"path":"...","max_lines":80,"offset":0}}
  {"thought":"...","action":"list_dir","args":{"path":"."}}
  完成或放弃: {"thought":"...","final":{"done":true/false,"summary":"做了什么/产出在哪/若未完成卡在哪"}}
环境:
- 工作目录: {cwd}(ouro CLI 已可用;conda 环境已激活)
- ouro 子命令速览(要细节就跑 `ouro <cmd> --help`):
{ouro_help}
规则:
- 【ask_user 是向用户拿需求澄清/决策确认/凭证的唯一渠道】。secret:true 时用户输入不回显,值注入 env_key 指定的环境变量,后续命令用 $KEY 引用——你永远看不到真值,也不要试图打印它。
- 【遇到这些情况必须 ask_user,禁止自己闷头决定】:①需要凭证/资源 ②重大取舍(跳过官方步骤、修改共享环境的包版本、降级依赖、改仓库源码前) ③认为卡住无法继续 ④需求有歧义。
- 【未完成用户需求前,禁止用 final 放弃】:卡住了就 ask_user 描述卡点要指示;final(done=false) 只在用户明确同意放弃后使用。
- 修复原则:优先选改动最小、成本最低的路径;装重依赖(>100MB)前先想有没有代码级替代;不要动共享环境已有包的版本(要动先 ask_user)。
- 安全:破坏性命令被系统拦截(sudo/rm/dd/mkfs/shutdown/chmod777 等);读 secret 文件内容被拦(source 加载可以);pip install、git clone 允许。
- 命令失败是最好的诊断信号:读报错 → 修正 → 重试;同一命令连续失败 2 次就换路线或 ask_user。
- 长任务设 timeout_s(上限 1800);产出数据的一步跑完要【验证产物真实存在】再宣布完成。
- facts 记事本:关键事实(路径/配置/已完成的阶段)随时记,系统会钉在上下文顶部不被淹没。
- 每 {max_turns} 步系统会暂停向用户汇报(检查点),用户指示后继续——所以不用担心步数,专注推进。"""

# Destructive/system-mutating commands stay blocked; pip/git are the agent's
# legitimate tools and are allowed (unlike the read-only explore fences).
_DANGER = re.compile(
    r'\b(sudo|shutdown|reboot|mkfs|chown)\b|\bdd\s+if=|\brm\s|chmod\s+777|:\(\)\s*\{')
_SECRET_READ = re.compile(
    r'\b(cat|head|tail|grep|sed|awk|less|more|strings)\b[^|;&\n]*(\.mcp_env|\.env(?!\.(example|template|sample)))')


def _ouro_help() -> str:
    try:
        p = subprocess.run(["bash", "-lc", "ouro --help"], capture_output=True, text=True, timeout=30)
        return (p.stdout + p.stderr)[-2000:]
    except Exception:
        return "(ouro --help 不可用)"


def _exec(cmd: str, cwd: str, env_extra: dict, timeout_s: int) -> str:
    if not cmd.strip():
        return "[error] empty cmd"
    if _DANGER.search(cmd):
        return "[blocked] 命令含破坏性操作(sudo/rm/dd/chmod777 等),换一种方式"
    if _SECRET_READ.search(cmd):
        return "[blocked] 禁止读取 secret 文件内容(source 加载可以,cat/grep 不行)"
    exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in env_extra.items())
    try:
        p = subprocess.run(["bash", "-c", exports + cmd], cwd=cwd,
                           capture_output=True, text=True, timeout=timeout_s)
        out = (p.stdout + p.stderr)[-3500:]
        return f"[exit={p.returncode}]\n{out or '[no output]'}"
    except subprocess.TimeoutExpired:
        return f"[timeout {timeout_s}s] 命令超时;考虑更小的规模或更大的 timeout_s"


def _ask_user(prompt: str, secret: bool, env_key: str, env_extra: dict) -> str:
    """Terminal prompt to the user. A secret's value goes into env_extra ONLY —
    the returned string (what the LLM sees) never contains it."""
    print("\n  ┌─ agent 问你 " + "─" * 46)
    for line in str(prompt).splitlines():
        print(f"  │ {line}")
    print("  └" + "─" * 58)
    if secret:
        import getpass
        key = (env_key or "SECRET").strip() or "SECRET"
        try:
            val = getpass.getpass(f"  [输入不回显,注入 ${key}] > ")
        except Exception:
            val = ""
        if val.strip():
            env_extra[key] = val.strip()
            return f"(用户已提供,值注入环境变量 ${key},对你不可见;在命令中用 ${key} 引用)"
        return "(用户未提供该 secret)"
    try:
        ans = input("  [你的回答] > ")
    except EOFError:
        return "(EOF:用户无输入)"
    return ans.strip() or "(用户回答为空)"


_STOP_WORDS = ("stop", "停", "停止", "结束", "q", "quit", "exit", "放弃")


def _is_stop(ans: str) -> bool:
    a = ans.strip().lower()
    return a in _STOP_WORDS or "(eof" in a


def run_agent(goal: str, model: str, base_url: str, key: str,
              cwd: str | None = None, max_turns: int = 30,
              hard_cap: int = 500) -> dict:
    """The agent NEVER quits on its own while the goal is unmet: every
    `max_turns` steps it CHECKPOINTS — reports progress and asks the user to
    continue/steer/stop; a stuck agent must ask_user for instructions, and
    final(done=false) without user consent is bounced back. Only goal-done,
    a user stop, or the hard safety cap ends the loop."""
    cwd = str(Path(cwd or Path.cwd()).expanduser().resolve())
    repo = Path(cwd)
    sysmsg = SYSTEM.replace("{cwd}", cwd).replace("{ouro_help}", _ouro_help()) \
                   .replace("{max_turns}", str(max_turns))
    base_user = f"用户需求:\n{goal}\n\n开始。"
    messages = [{"role": "system", "content": sysmsg},
                {"role": "user", "content": base_user}]
    env_extra: dict[str, str] = {}
    facts: list[str] = []
    seen: dict[str, int] = {}
    giveup_asked = False   # user already consented once to abandoning?
    turn, next_checkpoint = 0, max_turns
    while turn < hard_cap:
        turn += 1
        try:
            out = _llm(messages, base_url, key, model)
        except Exception as e:
            print(f"[agent] LLM 调用失败: {e}")
            return {"done": False, "summary": f"LLM 调用失败: {e}"}
        obj = _extract_json(out)
        if obj is None:
            print(f"  [turn {turn}] 未产出合法 JSON,提示重试")
            messages += [{"role": "assistant", "content": out[:400]},
                         {"role": "user", "content": "只输出一个合法 JSON(action 或 final)。"}]
            continue

        new_facts = [str(f).strip() for f in (obj.get("facts") or []) if str(f).strip()]
        if new_facts:
            facts.extend(f for f in new_facts if f not in facts)
            messages[1]["content"] = base_user + "\n\n[你已确认的事实(自己记录,勿忘)]\n- " + "\n- ".join(facts)

        if "final" in obj:
            fin = obj["final"] or {}
            done = bool(fin.get("done"))
            summary = str(fin.get("summary", ""))
            if done or giveup_asked:
                print(f"\n  [turn {turn}] FINAL — {'✅ 完成' if done else '⏸ 用户同意结束'}")
                print("  " + summary.replace("\n", "\n  ")[:1500])
                return {"done": done, "summary": summary, "turns": turn}
            # unmet goal + no user consent → bounce the final back as a question
            print(f"\n  [turn {turn}] agent 想放弃 → 转为向用户请示")
            ans = _ask_user("我遇到卡点,当前进展与卡点如下:\n" + summary[:1200] +
                            "\n\n请下指令(输入指示我就继续;输入 stop 才结束)", False, "", env_extra)
            if _is_stop(ans):
                return {"done": False, "summary": summary, "turns": turn}
            giveup_asked = False
            messages += [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                         {"role": "user", "content": f"用户指令:{ans}\n按指令继续,不要停。"}]
            continue

        action, a = obj.get("action", ""), obj.get("args", {}) or {}
        if isinstance(a, str):        # model sometimes emits args as a JSON STRING
            try:
                a = json.loads(a)
            except Exception:
                a = None
        if not isinstance(a, dict):
            messages += [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                         {"role": "user", "content": "结果:\n[error] args 必须是 JSON 对象(不是字符串)。重发这一步。"}]
            print(f"  [turn {turn}] {action} — args 非对象,已让它重发")
            continue
        print(f"  [turn {turn}] {action} {json.dumps(a, ensure_ascii=False)[:90]}  ·· {str(obj.get('thought',''))[:60]}"
              + (f"  +facts:{len(new_facts)}" if new_facts else ""))
        try:
            if action == "run_command":
                t = max(30, min(int(a.get("timeout_s", 600) or 600), 1800))
                feed = _exec(str(a.get("cmd", "")), cwd, env_extra, t)
                k = "cmd|" + str(a.get("cmd", ""))
                seen[k] = seen.get(k, 0) + 1
                if seen[k] > 2:
                    feed = f"[注意] 同一命令已执行 {seen[k]} 次。换路线,或 ask_user。\n" + feed
            elif action == "ask_user":
                feed = _ask_user(str(a.get("prompt", "")), bool(a.get("secret")),
                                 str(a.get("env_key", "")), env_extra)
                if _is_stop(feed):
                    giveup_asked = True   # explicit user stop → allow the next final
            elif action == "read_file":
                feed = _read_file(repo, a)
            elif action == "list_dir":
                feed = _list_dir(repo, a)
            else:
                feed = "[error] unknown action"
        except Exception as e:  # a tool crash must never kill the agent process
            feed = f"[error] 工具执行异常: {type(e).__name__}: {e}"
        messages += [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                     {"role": "user", "content": f"结果:\n{feed}"}]

        # CHECKPOINT — pause and hand the wheel to the user instead of dying
        if turn >= next_checkpoint:
            recent = "\n- ".join(facts[-8:]) if facts else "(无记录)"
            ans = _ask_user(f"[检查点] 已执行 {turn} 步。已确认的关键事实:\n- {recent}\n\n"
                            "继续吗?(回车或输入新指示=继续;stop=结束)", False, "", env_extra)
            if _is_stop(ans):
                return {"done": False, "summary": f"用户在检查点停止;facts: {facts}", "turns": turn}
            next_checkpoint = turn + max_turns
            steer = "" if ans.startswith("(用户回答为空") else f"用户补充指示:{ans}\n"
            messages += [{"role": "user", "content": f"[检查点通过] {steer}继续执行任务。"}]
    print(f"[agent] 达到硬上限 {hard_cap} 步(防失控),强制停止")
    return {"done": False, "summary": f"达到硬上限 {hard_cap} 步", "turns": turn}
