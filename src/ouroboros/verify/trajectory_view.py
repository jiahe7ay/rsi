"""Render an agent trajectory into a compact, judge-readable transcript.

HF trajectories are in OpenAI **Responses** shape:
    {"role":"user","content": "<task>"}                       # the task
    {"type":"function_call","name":..,"arguments":..}          # a tool call
    {"type":"function_call_output","output": "<json string>"}  # its result
    {"role":"assistant","content":[{"text":..,"type":"output_text"}]}  # narration

The judge needs to see *what the agent actually did and produced* — especially the
arguments of write-like calls (the agent's OUTPUT lives there). Tool *results*
(reads) can be huge, so they're truncated harder than call arguments.

This is deliberately tool-name-agnostic: we never special-case `write_file` etc.,
so it works for any benchmark's tools. Both Responses and chat/tool_calls shapes
are handled so the same view feeds offline (HF) and online (rollout) trajectories.
"""
from __future__ import annotations

import json
from typing import Any


def extract_task(messages: list[dict]) -> str:
    """The task statement = the first user turn (fallback: first stringy content)."""
    for m in messages:
        if m.get("role") == "user":
            return _as_text(m.get("content"))
    for m in messages:
        t = _as_text(m.get("content"))
        if t:
            return t
    return ""


def _as_text(content: Any) -> str:
    """Flatten OpenAI content (str | list[{text}] | dict) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        ).strip()
    if isinstance(content, dict):
        return content.get("text", "") or json.dumps(content, ensure_ascii=False)
    return str(content)


def _unwrap_output(output: Any) -> str:
    """mcpmark wraps tool output as {"type":"text","text": "<inner-json-or-text>"};
    the inner may itself be {"content":[{"text":..}]}. Peel a couple of layers so
    the judge reads plain text, but never fail — fall back to the raw string."""
    s = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
    for _ in range(3):
        try:
            o = json.loads(s)
        except Exception:
            break
        if isinstance(o, dict):
            if isinstance(o.get("content"), list):
                parts = [c.get("text", "") if isinstance(c, dict) else str(c)
                         for c in o["content"]]
                s = "\n".join(p for p in parts if p)
                continue
            if "text" in o and isinstance(o["text"], str):
                s = o["text"]
                continue
        break
    return s


def render_actions(messages: list[dict], max_out: int = 600, max_arg: int = 1400,
                   skip_task: bool = True) -> str:
    """Linearize the trajectory into `[ROLE] ...` lines the judge can grade.

    max_arg (tool-call args, where the agent's WRITES live) is kept larger than
    max_out (tool results / reads). The first user turn is dropped when skip_task
    (the task is shown separately by the judge)."""
    lines: list[str] = []
    seen_task = False
    for m in messages:
        typ = m.get("type")
        role = m.get("role")
        if role == "user" and skip_task and not seen_task:
            seen_task = True
            continue
        if typ == "function_call" or ("arguments" in m and "name" in m):
            args = m.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            lines.append(f"[TOOL CALL] {m.get('name','?')}({_clip(args, max_arg)})")
        elif typ == "function_call_output" or ("output" in m and "call_id" in m):
            lines.append(f"[TOOL RESULT] {_clip(_unwrap_output(m.get('output')), max_out)}")
        elif role == "assistant":
            txt = _as_text(m.get("content"))
            if txt:
                lines.append(f"[ASSISTANT] {_clip(txt, max_arg)}")
            for tc in m.get("tool_calls") or []:  # chat-format fallback
                fn = (tc.get("function") or {})
                lines.append(f"[TOOL CALL] {fn.get('name','?')}({_clip(str(fn.get('arguments','')), max_arg)})")
        elif role == "tool":  # chat-format tool result
            lines.append(f"[TOOL RESULT] {_clip(_as_text(m.get('content')), max_out)}")
        elif role == "user":
            lines.append(f"[USER] {_clip(_as_text(m.get('content')), max_out)}")
    return "\n".join(lines)


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f" …[+{len(s)-n} chars]"
