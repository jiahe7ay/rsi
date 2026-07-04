"""Ouroboros orchestrator CLI:  ouro <onboard | split | rollout | eval | baseline>."""
from __future__ import annotations

import argparse
import sys


def cmd_onboard(args: argparse.Namespace) -> int:
    """Input a benchmark repo -> analyze, probe env, elicit missing, smoke-test."""
    from ouroboros.onboard import onboard
    r = onboard(args.repo, model=args.model, do_smoke=not args.no_smoke)
    return 0 if r.get("ok") else 1


def cmd_explore(args: argparse.Namespace) -> int:
    """[experimental] LLM agent self-explores an unknown benchmark -> BenchmarkSpec."""
    from ouroboros.onboard.llm_explore import explore
    print(f"▶ LLM 自主探索: {args.repo}  (model={args.model}, max_turns={args.max_turns})")
    spec = explore(args.repo, model=args.model, max_turns=args.max_turns)
    if spec is None:
        print("\n✗ 自主探索未产出 spec → 回退到 adapter 路径(`ouro onboard`)。")
        return 1
    from ouroboros.onboard.orchestrator import _print_spec
    _print_spec(spec)
    print("\n✓ LLM 探索产出 spec(下一步可交给 probe/smoke 验证)。")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    """(1) Freeze train/held-out task splits (idempotent; frozen once)."""
    from ouroboros import splits
    splits.freeze(domains=args.domains.split(",") if args.domains else None,
                  eval_frac=args.eval_frac,
                  suites=tuple(args.suites.split(",")), force=args.force)
    return 0


def cmd_rollout(args: argparse.Namespace) -> int:
    """(2) Generate trajectories on a split via the benchmark pipeline, harvest
    verified results into the store."""
    from ouroboros import rollout
    r = rollout.run(model=args.model, split=args.split, n=args.n,
                    domains=args.domains.split(",") if args.domains else None,
                    suite=args.suite, limit=args.limit,
                    checkpoint=args.checkpoint, exp_name=args.exp_name, patch=args.patch)
    return 0 if r["trajectories"] > 0 else 1


def cmd_improve(args: argparse.Namespace) -> int:
    """Learn from an OLD model's HF trajectories → analyze failures → emit a prompt patch."""
    import os
    from ouroboros.improve import pull, summarize_failures
    from ouroboros.improve.patch_gen import generate
    bu, key = os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_API_KEY")
    print(f"▶ improve: 拉 {args.repo}@{args.revision} 的失败轨迹 ...")
    trajs = pull(args.repo, args.revision,
                 domains=args.domains.split(",") if args.domains else None,
                 only="failed", limit=args.limit)
    print(f"  拉到 {len(trajs)} 条失败轨迹 (traj model={trajs[0].model if trajs else '?'})")
    if not trajs:
        print("  无失败轨迹,无需补丁。")
        return 1
    print("  LLM 归纳失败模式 ...")
    analysis = summarize_failures(trajs, args.model, bu, key, max_tasks=args.max_tasks)
    print("  summary:", analysis["summary"][:200])
    rec = generate(analysis, args.model, bu, key, name=args.name,
                   source={"repo": args.repo, "revision": args.revision,
                           "traj_model": trajs[0].model if trajs else None})
    # human-in-the-loop: scan + diff preview + [a]pprove/[e]dit/[r]eject; only
    # an approved patch is written and usable by rollout --patch.
    from ouroboros.improve.review import review_and_save
    from ouroboros.improve.patch_gen import PATCH_DIR
    from ouroboros import splits, config
    mcpmark_root = splits.mcpmark_root(config.load())
    saved = review_and_save(rec, PATCH_DIR, mcpmark_root=mcpmark_root, auto=args.yes)
    if not saved:
        return 1
    print(f"  用它重造数据:  ouro rollout --patch {rec['name']} --split train --domains ...")
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    """Interactive top-level agent: the user states a goal; the agent composes
    the existing ouro commands (and shell) itself, asking the user when it needs
    clarification or credentials."""
    import os
    from ouroboros.agent import run_agent
    bu, key = os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_API_KEY")
    if not bu or not key:
        print("缺 OPENAI_BASE_URL/OPENAI_API_KEY(先 source 一个 .mcp_env)。")
        return 1
    goal = args.goal
    if not goal:
        try:
            goal = input("你想做什么(自然语言描述需求)> ").strip()
        except EOFError:
            goal = ""
    if not goal:
        print("(空需求,退出)")
        return 1
    print(f"▶ agent 接需求(model={args.model}, max_turns={args.max_turns}): {goal[:120]}")
    r = run_agent(goal, args.model, bu, key, cwd=args.cwd, max_turns=args.max_turns)
    return 0 if r.get("done") else 1


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Reproduce a data pipeline's generation: agent judges what's missing →
    interactive prompts collect it from the user → run the minimal generation."""
    import os
    from pathlib import Path
    from ouroboros import config
    from ouroboros.onboard.spec import load_spec, save_spec
    from ouroboros.onboard.reproduce import run_reproduce
    bu, key = os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_API_KEY")
    if not bu or not key:
        print("缺 OPENAI_BASE_URL/OPENAI_API_KEY(先 source 一个 .mcp_env)。")
        return 1
    repo = Path(args.repo).expanduser().resolve()
    spec = None if args.fresh else load_spec(repo.name)
    if spec is None:
        from ouroboros.onboard.llm_explore import explore
        print(f"(无缓存 spec → 先 LLM explore {repo.name} ...)")
        spec = explore(str(repo), model=args.model)
        if spec is None:
            print("✗ explore 未产出 spec。")
            return 1
        save_spec(spec)
    else:
        print(f"(用缓存 spec: data/specs/{repo.name}.json;--fresh 可重新 explore)")
    launcher = config.load().get("rollout", {}).get("launcher", "")
    r = run_reproduce(spec, args.model, bu, key, launcher=launcher,
                      auto=args.yes, run_timeout=args.timeout)
    return 0 if r.get("ok") else 1


def cmd_verify(args: argparse.Namespace) -> int:
    """LLM auto-verify: re-score HF trajectories keeping the benchmark verdict
    authoritative, attaching a rubric + an explanation of WHY it passed/failed."""
    import os
    from ouroboros.verify import verify_hf
    bu, key = os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_API_KEY")
    print(f"▶ verify: 拉 {args.repo}@{args.revision} 轨迹,LLM 判官解释 mcpmark 的判定 ...")
    recs = verify_hf(args.repo, args.revision, args.model, bu, key,
                     domains=args.domains.split(",") if args.domains else None,
                     only=args.only, limit=args.limit, regen=args.regen)
    if not recs:
        print("  无轨迹。")
        return 1
    print(f"  判了 {len(recs)} 条 (passed/reward=mcpmark 权威;compliance/解释=LLM)\n")
    dis = 0
    for r in recs:
        v = r["verify"]; d = v.detail
        gt = "PASS" if r["mcpmark_success"] else "FAIL"
        locus = f" locus={d['failure_locus']}" if d.get("failure_locus") else ""
        flag = "  ⚠分歧" if d.get("disagreement") else ""
        print(f"  [{r['domain']}/{r['task_name']}] {r['run']}  mcpmark={gt} "
              f"reward={v.reward} compliance={d['compliance_score']}{locus}{flag}")
        if d.get("explanation"):
            print(f"       {d['explanation'][:200]}")
        dis += 1 if d.get("disagreement") else 0
    print(f"\n  分歧 {dis} 条(compliance 全过却被判失败 → 语义/内容错,最该人工看)。")
    print("  rubric 已存 data/rubrics/(可审阅/编辑)。")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """(5) pass@k / per-domain report on the held-out (eval) split."""
    from ouroboros import evalharness
    ks = tuple(int(x) for x in args.k.split(","))
    evalharness.evaluate(checkpoint=args.checkpoint, model=args.model, k=ks,
                         domains=args.domains.split(",") if args.domains else None,
                         suite=args.suite, limit=args.limit, exp_name=args.exp_name)
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """Convenience alias: eval on held-out to establish the starting line."""
    return cmd_eval(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ouro", description="Ouroboros RSI foundation")
    sub = p.add_subparsers(dest="cmd", required=True)

    ob = sub.add_parser("onboard", help="analyze a benchmark repo, probe env, elicit missing, smoke")
    ob.add_argument("repo", help="path to a benchmark checkout (e.g. ~/mcpmark)")
    ob.add_argument("--model", default="deepseek-v4-pro", help="LLM model (explore 直连;smoke 命令由 LLM 生成并自愈)")
    ob.add_argument("--no-smoke", action="store_true", help="stop after explore+probe+elicit")
    ob.set_defaults(func=cmd_onboard)

    ex = sub.add_parser("explore", help="[experimental] LLM agent self-explores an unknown benchmark")
    ex.add_argument("repo", help="path to a benchmark checkout")
    ex.add_argument("--model", default="deepseek-v4-pro", help="model called directly (no openai/ prefix)")
    ex.add_argument("--max-turns", dest="max_turns", type=int, default=12)
    ex.set_defaults(func=cmd_explore)

    s = sub.add_parser("split", help="freeze train/eval task splits")
    s.add_argument("--domains", default=None,
                   help="comma-separated;不传则用最新 onboard inventory 里可跑的 domains")
    s.add_argument("--suites", default="easy,standard", help="comma-separated task suites")
    s.add_argument("--eval-frac", dest="eval_frac", type=float, default=0.2)
    s.add_argument("--force", action="store_true", help="overwrite an existing frozen manifest")
    s.set_defaults(func=cmd_split)

    r = sub.add_parser("rollout", help="generate + verify + store trajectories")
    r.add_argument("--model", default="openai/deepseek-v4-pro",
                   help="model name passed to the benchmark (litellm 需 openai/ 前缀)")
    r.add_argument("--split", default="train", choices=["train", "eval"])
    r.add_argument("--suite", default="easy", help="task suite (easy|standard)")
    r.add_argument("--domains", default=None, help="comma-separated; default = config tasks.domains")
    r.add_argument("-n", type=int, default=1, help="samples per task (pipeline --k)")
    r.add_argument("--limit", type=int, default=None, help="only first N tasks (testing)")
    r.add_argument("--checkpoint", default=None, help="ckpt path/hash for provenance")
    r.add_argument("--exp-name", dest="exp_name", default=None)
    r.add_argument("--patch", default=None,
                   help="提示词补丁名/路径(data/patches/<name>.json),叠加到 agent 提示词")
    r.set_defaults(func=cmd_rollout)

    im = sub.add_parser("improve", help="learn from HF trajectories → generate a prompt patch")
    im.add_argument("--repo", default="taiyi-lab/mcpmark-eval", help="HF dataset repo id")
    im.add_argument("--revision", required=True, help="HF dataset branch/revision")
    im.add_argument("--model", default="deepseek-v4-pro", help="分析/生成补丁用的 LLM(直连 dimcode)")
    im.add_argument("--domains", default=None, help="只分析这些域,逗号分隔")
    im.add_argument("--limit", type=int, default=40, help="拉取失败轨迹上限")
    im.add_argument("--max-tasks", dest="max_tasks", type=int, default=8, help="喂给 LLM 分析的最差任务数")
    im.add_argument("--name", default=None, help="补丁名(默认带时间戳)")
    im.add_argument("--yes", action="store_true",
                    help="跳过交互审批,自动批准(仍跑危险扫描;命中危险则拒绝落盘)")
    im.set_defaults(func=cmd_improve)

    vf = sub.add_parser("verify", help="LLM auto-verify HF trajectories — explains the benchmark verdict")
    vf.add_argument("--repo", default="taiyi-lab/mcpmark-eval", help="HF dataset repo id")
    vf.add_argument("--revision", required=True, help="HF dataset branch/revision")
    vf.add_argument("--model", default="deepseek-v4-pro", help="判官/rubric 用的 LLM(直连 dimcode)")
    vf.add_argument("--domains", default=None, help="只判这些域,逗号分隔")
    vf.add_argument("--only", default=None, choices=["passed", "failed"],
                    help="只判通过/失败的(默认全判)")
    vf.add_argument("--limit", type=int, default=20, help="轨迹上限")
    vf.add_argument("--regen", action="store_true", help="忽略缓存,重新生成 rubric")
    vf.set_defaults(func=cmd_verify)

    ag = sub.add_parser("agent", help="交互 agent:输入框收你的需求,agent 自己组合 ouro 命令实现,缺什么问你")
    ag.add_argument("--goal", default=None, help="需求(不传则进入输入框)")
    ag.add_argument("--model", default="deepseek-v4-pro", help="agent 用的 LLM(直连 dimcode)")
    ag.add_argument("--max-turns", dest="max_turns", type=int, default=30)
    ag.add_argument("--cwd", default=None, help="agent 工作目录(默认当前目录)")
    ag.set_defaults(func=cmd_agent)

    rp = sub.add_parser("reproduce", help="复现数据管线:agent 判缺什么 → 终端交互补齐 → 实跑最小生成")
    rp.add_argument("repo", help="数据管线 repo 路径(如 ~/rsi/Toucan)")
    rp.add_argument("--model", default="deepseek-v4-pro", help="评估用 LLM(直连 dimcode)")
    rp.add_argument("--fresh", action="store_true", help="忽略缓存 spec,重新 explore")
    rp.add_argument("--yes", action="store_true",
                    help="非交互:自动装依赖;凭证无法自动补,缺了会中止")
    rp.add_argument("--timeout", type=int, default=900, help="实跑超时秒数")
    rp.set_defaults(func=cmd_reproduce)

    for name, helptxt in (("eval", "pass@k on the held-out (eval) split"),
                          ("baseline", "measure the starting-line checkpoint (= eval)")):
        sp = sub.add_parser(name, help=helptxt)
        sp.add_argument("--model", default="openai/deepseek-v4-pro")
        sp.add_argument("--checkpoint", default=None, help="ckpt path/hash for provenance")
        sp.add_argument("--k", default="1,4", help="comma-separated k values for pass@k")
        sp.add_argument("--suite", default="easy")
        sp.add_argument("--domains", default=None, help="default = config tasks.domains")
        sp.add_argument("--limit", type=int, default=None, help="only first N eval tasks")
        sp.add_argument("--exp-name", dest="exp_name", default=None)
        sp.set_defaults(func=cmd_eval if name == "eval" else cmd_baseline)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
