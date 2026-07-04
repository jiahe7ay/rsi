"""Runtime bootstrap: LAYER a prompt patch onto mcpmark's agent SYSTEM_PROMPT,
then run mcpmark's pipeline — WITHOUT editing any mcpmark file.

The patch is a class-attribute append done at process start; mcpmark's source is
untouched (append is in-memory, this run only). Idempotent; no-op if the env var
is unset — so the exact same command with/without a patch is the A/B comparison.

Usage (from rollout, cwd = mcpmark repo):
    OURO_PROMPT_PATCH="<text>" python <this_file> --mcp filesystem --tasks ... --k 1
"""
import os
import runpy
import sys

_TAG = "\n\n[OURO-PATCH — Ouroboros 运行时叠加,未改动 mcpmark 原文件]\n"


def apply_patch() -> int:
    """Append env OURO_PROMPT_PATCH to every known agent SYSTEM_PROMPT. Returns
    how many prompts were patched (0 = nothing to do / classes not found)."""
    patch = os.environ.get("OURO_PROMPT_PATCH", "").strip()
    if not patch:
        return 0
    sys.path.insert(0, os.getcwd())   # mcpmark's `src.*` lives in the cwd we cd'd into
    add = _TAG + patch
    n = 0
    for mod, cls in (("src.agents.mcpmark_agent", "MCPMarkAgent"),
                     ("src.agents.react_agent", "ReactAgent"),
                     ("src.agents.react_agent", "ReActAgent")):
        try:
            m = __import__(mod, fromlist=[cls])
            c = getattr(m, cls, None)
            if c is None:
                continue
            for attr in ("SYSTEM_PROMPT", "DEFAULT_SYSTEM_PROMPT"):
                cur = getattr(c, attr, None)
                if isinstance(cur, str) and add not in cur:
                    setattr(c, attr, cur + add)
                    n += 1
        except Exception:
            continue
    return n


if __name__ == "__main__":
    n = apply_patch()
    sys.stderr.write(f"[ouro_patch] layered patch onto {n} system-prompt(s)\n")
    # hand off to mcpmark's pipeline with the remaining CLI args intact
    sys.argv = ["pipeline"] + sys.argv[1:]
    runpy.run_module("pipeline", run_name="__main__")
