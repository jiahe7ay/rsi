# Ouroboros 🐍

Recursive Self-Improvement (RSI) for agentic models. Ouroboros closes the loop:
an agent generates trajectories on verifiable tasks → an objective verifier scores
them → good trajectories train the next model → the stronger model generates better
trajectories. The snake eats its own tail.

**v1 scope: the rollout + eval foundation** (generate → verify → eval), plus the
agent-autonomous tooling around it (onboard / auto-verify / prompt-improve /
reproduce). The training loop (SFT/RL self-bootstrap) comes only after the
foundation is rock-solid.

## Why "foundation first"
The easiest way to fool yourself in RSI is a dirty reward or a polluted eval.
Ouroboros v1 nails down the things every future generation reuses — deterministic
frozen splits, a driven-and-harvested rollout, a reward whose authority is an
objective checker, and a strictly held-out eval — then measures a trustworthy
baseline before any training happens.

## Components (actual state)
| Module | Role |
|---|---|
| `agent.py`       | **top-level entry — `ouro agent`**: you state a goal in plain language (an input prompt opens); the agent reads a live `ouro --help` snapshot and composes the commands below (plus shell) itself. It asks YOU in the terminal whenever it needs clarification, a decision, or a credential (secrets via hidden input, injected into child env only — never into the LLM context). **Never-quit contract**: while the goal is unmet it may not exit — every `--max-turns` steps it checkpoints (progress report + continue/steer/stop), and wanting to give up bounces back to you as a question |
| `onboard/`       | input a repo → LLM **explores** it (ReAct, read-only, no hardcoded adapter), classifies `kind` (benchmark / data-pipeline / dataset / agent / library) and **routes**: benchmark → probe + task inventory + agent smoke; data repo → data adapter (locate produced dataset, sample it, map to Trajectory schema, honestly flag whether it has a verifiable reward); pipeline → `reproduce` |
| `onboard/reproduce.py` | reproduce a data pipeline: agent assesses what's missing (deps / credentials / config, minimal vs full) → **interactive terminal elicit** (secrets via hidden input, never persisted) → run the minimal generation → deterministic new-files check |
| `splits.py`      | ① freeze train / held-out task splits (md5-deterministic, refuses overwrite; manifests in `configs/splits/`) |
| `rollout.py`     | ② drive the benchmark pipeline directly, harvest verified trajectories (auto-provisions postgres; `--patch` layers a prompt patch at runtime — **never** edits the benchmark's own prompts) |
| `verify/`        | ③ LLM auto-verify: per-task rubric (cached, human-reviewable) + a judge that **explains ground truth, never overrides it** — `failure_locus` (semantic/numeric/omission/…) + disagreement flag turn silent verifier failures into analyzable signal |
| `schema.py` / `store.py` | ④ trajectory contracts + store: every trajectory carries reward + provenance |
| `evalharness.py` | ⑤ pass@k / per-domain report on the held-out split (never sees prompt patches) |
| `improve/`       | ⑥ prompt-level self-improvement: pull an old model's HF trajectories → analyze failure modes → generate a prompt **patch** → human approval gate (threat scan + merged-prompt diff + `[a]pprove/[e]dit/[r]eject`) → `rollout --patch` |
| `cli.py`         | `ouro agent \| onboard \| explore \| split \| rollout \| eval \| baseline \| improve \| verify \| reproduce` |

## Reuse vs build
- **Reuse:** mcpmark checker + env (reward authority), hermes-agent ideas (facts scratchpad, verify-on-stop), sglang serving, Megatron-SWIFT (later)
- **Build:** everything in the table above

## Quickstart (real commands)
```bash
# env: any python env with mcpmark's deps (we use the `mcpmark` conda env)
conda activate mcpmark
pip install -e .
set -a; . ~/mcpmark/.mcp_env; set +a         # OPENAI_BASE_URL / OPENAI_API_KEY

# ── the goal-driven way: describe what you want, the agent does the rest ──
ouro agent                                    # opens 你想做什么> ; type e.g.
                                              #   "跑通 Toucan 问题生成+质检,然后生成轨迹"
                                              # it plans, runs commands, and prompts YOU
                                              # for keys/decisions; checkpoints every 40 turns
ouro agent --goal "..." --max-turns 40        # non-interactive goal, same loop

# ── or drive each stage yourself ──
ouro onboard ~/mcpmark                        # LLM explores + classifies + smokes
ouro split                                    # freeze splits (runnable domains only)
ouro rollout --split train --domains filesystem --suite easy -n 1
ouro eval --domains filesystem --k 1

# learn from an old model's failures → patch → regenerate
ouro improve --revision <hf-branch> --domains filesystem --name my-patch
ouro rollout --patch my-patch --split train --domains filesystem

# explain WHY trajectories passed/failed (rubric + failure locus)
ouro verify --revision <hf-branch> --domains filesystem --only failed

# reproduce a data-generation repo (e.g. Toucan): assess → elicit → run
ouro reproduce ~/rsi/Toucan
```

## Configuring the LLM (use any OpenAI-compatible API)
Every LLM call in Ouroboros (`agent` / `onboard` / `explore` / `verify` /
`improve` / `reproduce`) reads exactly two env vars, plus a per-command
`--model` flag:

```bash
export OPENAI_BASE_URL=https://your-endpoint/v1    # must serve /chat/completions (OpenAI-compatible)
export OPENAI_API_KEY=sk-...                       # never printed; passed only as the auth header

ouro agent --model your-model-name                 # e.g. deepseek-v4-pro, qwen3.5-72b, gpt-4o…
ouro verify --revision <br> --model your-model-name
```

Notes:
- Sourcing `~/mcpmark/.mcp_env` in the Quickstart is just a convenient way to set
  those two vars — plain `export`s work identically.
- **Exception — `ouro rollout` / `ouro eval`**: their `--model` is handed to the
  benchmark's own runner (mcpmark → litellm), which needs the `openai/` prefix
  for custom endpoints: `--model openai/your-model-name`. The direct-call
  commands above must NOT have the prefix.
- Endpoints only supporting OpenAI's newer `/v1/responses` API are not required —
  plain `/chat/completions` is enough (reasoning models work; JSON output is
  salvage-parsed against truncation).

## The non-negotiables (RSI failure modes)
1. Reward reliability > everything — the objective checker is the authority;
   the LLM judge only explains it (a transcript-only judge WILL rubber-stamp a
   confident agent — observed, and designed against).
2. Prompt patches are layered at runtime and human-approved; the benchmark's own
   prompts are never edited. Eval never sees patches.
3. Rollout & eval share one primitive (no train/eval skew); splits frozen on day one.
4. Everything reproducible; reward detail + provenance always logged.
5. Quality-scored datasets (e.g. Toucan-1.5M) are SFT/diversity material, **not**
   RL reward.

## Status
Verified end-to-end on the H200 box: filesystem + postgres rollout/eval on mcpmark;
improve loop incl. approval gate; verify on real failed trajectories (semantic vs
omission separation); Toucan onboarded as data-pipeline, its dataset adapted, and
its step1.1 generation reproduced. Training loop: not started (by design).

## Roadmap
foundation (this) → rejection-sampling SFT self-bootstrap (STaR/ReST) → RL (GRPO/DPO) → task+solver co-evolution (auto-curriculum)
