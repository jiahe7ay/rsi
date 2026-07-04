"""onboard() — LLM-explore by DEFAULT (no hand-written adapter on the main path).

Flow: the model self-explores the benchmark (explore) -> we verify readiness with
a generic probe (is the endpoint reachable?) -> we smoke-test with an LLM-generated
command that SELF-HEALS on failure (run -> read error -> LLM fixes -> retry). This
is the full "agent-autonomous onboarding": explore, try, see the error, fix, pass.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from ouroboros.onboard.spec import BenchmarkSpec, ElicitItem, ProbeItem, ProbeResult
from ouroboros.onboard.steps import _env_keys_with_values, elicit


def _hr(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def _print_spec(s: BenchmarkSpec) -> None:
    _hr("① Explore (LLM 自主产出 spec)")
    print(f"  name       : {s.name}")
    print(f"  adapter    : {s.adapter}   (kind: {s.kind})")
    print(f"  entrypoint : {s.entrypoint or '(unknown)'}")
    print(f"  env_file   : {s.env_file or '(none)'}")
    if s.data_source:
        print(f"  data_source: {s.data_source}")
    for n in s.notes:
        print(f"  note       : {n}")
    if s.required_env:
        print("  required env(参考,probe 只硬性看 endpoint):")
        for r in s.required_env:
            tag = "secret" if r.secret else "env"
            print(f"    - {r.key:20} [{tag}]  {r.purpose}")


def _print_probe(p: ProbeResult) -> None:
    _hr("② Probe env")
    for it in p.items:
        print(f"  [{'OK' if it.ok else '--'}] {it.name}: {it.detail}")


def _print_elicit(items: list[ElicitItem]) -> None:
    _hr("③ Elicit (需要你补齐)")
    for e in items:
        lock = "🔐 " if e.secret else "•  "
        print(f"  {lock}{e.key} — {e.purpose}\n       → {e.how}")


def _generic_probe(spec: BenchmarkSpec) -> ProbeResult:
    """Adapter-free probe: env-file existence + endpoint connectivity (never prints key).

    We deliberately do NOT hard-block on per-key missing env: the LLM tends to list
    every provider's key, but a run needs only one working endpoint — so the real
    'can it run?' signal is endpoint reachability, which we test directly.
    """
    repo = Path(spec.repo_path)
    res = ProbeResult()
    if not spec.env_file:
        return res
    p = repo / spec.env_file
    res.items.append(ProbeItem(f"env file {spec.env_file}", p.exists(),
                               f"{len(_env_keys_with_values(p))} key(s) set"))
    if not p.exists():
        return res
    ep = (f'cd {repo} && set -a; . ./{spec.env_file} 2>/dev/null; set +a; '
          'if [ -n "$OPENAI_BASE_URL" ]; then '
          'curl -sS -m15 "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY" '
          '-o /dev/null -w "%{http_code}"; else echo nourl; fi')
    try:
        code = subprocess.run(["bash", "-c", ep], capture_output=True, text=True, timeout=25).stdout.strip()
        res.items.append(ProbeItem("endpoint (GET /models)", code.endswith("200"), f"HTTP {code}"))
    except Exception as e:
        res.items.append(ProbeItem("endpoint (GET /models)", False, str(e)[:60]))
    return res


def onboard(repo_path: str, model: str = "deepseek-v4-pro", do_smoke: bool = True,
            max_repair: int = 2) -> dict:
    from ouroboros.onboard.llm_explore import explore, solve_smoke

    base_url = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY")
    print(f"▶ Onboarding (LLM explore, model={model}): {repo_path}")

    # ① LLM self-exploration -> spec
    spec = explore(repo_path, model=model, max_turns=16)
    if spec is None:
        print("\n✗ LLM 探索未产出 spec。")
        return {"stage": "explore", "ok": False}
    _print_spec(spec)
    from ouroboros.onboard.spec import normalize_kind, save_spec
    save_spec(spec)   # cache for `ouro reproduce` etc. — skip re-exploring

    # ①.5 ROUTE ON KIND — explore's classification decides the pipeline. Only a
    # benchmark (or unknown, conservatively) takes the rollout/eval path; a data
    # repo is ADAPTED (locate produced dataset + sample + map to schema), and an
    # agent/library is just noted. This is what makes explore's kind useful.
    kind = normalize_kind(spec.kind)
    if kind in ("data-pipeline", "dataset"):
        from ouroboros.onboard.data_adapter import adapt_data_source
        r = adapt_data_source(spec, model, base_url, key)
        print(f"\n  ↪ 想复现它的数据生成: ouro reproduce {spec.repo_path}"
              "(agent 判缺什么 → 交互补齐 → 实跑)")
        return r
    if kind in ("agent", "library", "other"):
        _hr("② 分流:非 benchmark")
        label = {"agent": "agent 运行时/框架", "library": "通用库/工具", "other": "其他"}[kind]
        print(f"  kind={kind}:这是{label},没有可判分任务 → 不做 rollout/eval。")
        print("  可作参考或复用其组件(不进入 probe/smoke)。")
        return {"stage": "classified", "ok": True, "kind": kind}
    # benchmark / unknown → the existing runnable-benchmark flow ↓

    # ② generic probe (endpoint reachability is the real signal)
    pr = _generic_probe(spec)
    _print_probe(pr)

    # ③ task inventory — the newcomer's answer to "有哪些 task,我能跑哪些?"
    #    LLM supplied structure; real task names + env readiness are deterministic.
    from ouroboros.onboard.inventory import build_inventory, print_inventory, save_inventory
    _hr("③ 任务清单(哪些能跑)")
    inv = build_inventory(spec)
    print_inventory(inv)
    if inv["domains"]:
        path = save_inventory(inv)
        runnable = [d["name"] for d in inv["domains"] if d["runnable"]]
        print(f"  → 已写 {path};`ouro split`(无参)将只切能跑的 domains: {', '.join(runnable) or '(无)'}")

    needs = elicit(spec, pr)
    if needs:
        _print_elicit(needs)
        print("\n⏸ 缺资源,补齐后重跑。")
        return {"stage": "elicit", "ok": False, "needs": [n.key for n in needs]}

    if not do_smoke:
        print("\n✓ spec 就绪(--no-smoke)。")
        return {"stage": "ready", "ok": True}

    # ④ smoke as an agent loop: the model diagnoses the machine itself
    #    (which python / conda envs / how repo scripts activate) and iterates
    #    run_smoke until a task REALLY runs. Judge stays deterministic (Y>0).
    _hr("④ Smoke (agent 自主诊断 + 试跑;判定=真实跑出 Y>0)")
    if not base_url or not key:
        print("  缺 OPENAI_BASE_URL/KEY(先 `source <repo>/.mcp_env`)。")
        return {"stage": "smoke", "ok": False}
    try:
        res = solve_smoke(spec, model, base_url, key)
    except Exception as e:
        print(f"  ✗ smoke agent 异常: {e}")
        return {"stage": "smoke", "ok": False}
    if res.get("ok"):
        print(f"  ✓ 真实跑通: Tasks passed {res['passed']}(第 {res['turns']} 步)")
        print(f"     $ {res['cmd'][:240]}")
        print("\n🎉 Benchmark onboard 成功(agent 全自主:explore + 自诊断 + 试跑,无写死规则)。")
        return {"stage": "smoke", "ok": True, "cmd": res["cmd"], "passed": res["passed"]}
    print(f"\n✗ smoke agent {res.get('turns')} 步内未跑通(最后命令: {str(res.get('cmd'))[:160]})")
    return {"stage": "smoke", "ok": False}
