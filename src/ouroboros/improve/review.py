"""Human-in-the-loop approval for prompt patches.

A patch influences ALL future training-data generation, so an LLM-written patch
never goes live unattended. Before a patch is saved as usable, we: (1) threat-scan
it, (2) show a diff preview of the MERGED system prompt the agent will actually
read, (3) ask the user to [a]pprove / [e]dit / [r]eject in the terminal.

Only approve writes data/patches/<name>.json. --yes (auto) skips the prompt for
non-interactive runs but STILL runs the threat scan and refuses to save on a hit
unless explicitly forced.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

# Patch text lands in the agent's system prompt, so scan for the ways that could
# go wrong: overriding rules, exfiltrating secrets, disabling checks, destructive
# or networked commands.
_THREATS = [
    (r'(?i)\b(ignore|disregard|override|forget|bypass)\b.{0,40}\b(previous|above|prior|all|system|instruction|rule|prompt)',
     "试图覆盖/忽略原有指令"),
    (r'(?i)\b(reveal|print|output|show|leak|exfiltrate|send|dump|echo)\b.{0,40}(api[\s_-]?key|token|secret|password|credential|\.env|\.mcp_env)',
     "诱导泄露密钥/secret"),
    (r'(?i)\b(disable|skip|bypass|turn\s*off|ignore)\b.{0,25}(verif|check|validat|safety|guard|test)',
     "试图绕过校验/安全"),
    (r'(?i)(rm\s+-rf|sudo\s|mkfs|dd\s+if=|chmod\s+777|:\(\)\s*\{)',
     "危险破坏性命令"),
    (r'(?i)\b(curl|wget|nc|netcat)\b.{0,60}(http|://|\d+\.\d+\.\d+\.\d+)',
     "疑似联网外传"),
]


def threat_scan(text: str) -> list[str]:
    return [desc for pat, desc in _THREATS if re.search(pat, text)]


def diff_preview(patch_text: str, mcpmark_root=None) -> tuple[str, bool]:
    """Return (merged_prompt_or_note, got_original). Tries to load mcpmark's real
    SYSTEM_PROMPT so the user sees the actual merged text the agent will read."""
    tag = "\n\n[OURO-PATCH — Ouroboros 运行时叠加,未改动 mcpmark 原文件]\n" + patch_text
    if mcpmark_root:
        import sys
        root = str(mcpmark_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from src.agents.mcpmark_agent import MCPMarkAgent
            return MCPMarkAgent.SYSTEM_PROMPT + tag, True
        except Exception as e:
            return f"(无法加载 mcpmark 原 prompt: {type(e).__name__};补丁将追加到系统提示词末尾)\n{tag}", False
    return f"(未提供 mcpmark 路径;补丁将追加到 agent 系统提示词末尾)\n{tag}", False


def _edit(text: str) -> str:
    ed = os.environ.get("EDITOR") or "vi"
    with tempfile.NamedTemporaryFile("w+", suffix=".patch.txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    subprocess.call([ed, path])
    return Path(path).read_text(encoding="utf-8").strip()


def _save(rec: dict, patch_dir: Path) -> str:
    patch_dir.mkdir(parents=True, exist_ok=True)
    out = patch_dir / f"{rec['name']}.json"
    payload = {k: v for k, v in rec.items() if not k.startswith("_")}
    payload["approved"] = True
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 已批准并落盘: {out}")
    return str(out)


def _render(rec: dict, mcpmark_root) -> list[str]:
    text = rec["patch_text"]
    hits = threat_scan(text)
    merged, got = diff_preview(text, mcpmark_root)
    print("\n" + "=" * 66)
    print(f"拟启用补丁: {rec['name']}   source: {rec.get('source', {})}")
    print("── 补丁正文 " + "─" * 40)
    print(text)
    print("── 安全扫描 " + "─" * 40)
    if hits:
        print("  ⚠️  命中可疑模式(请审慎):")
        for h in hits:
            print(f"      - {h}")
    else:
        print("  ✓ 未命中危险模式")
    print(f"── diff 预览(agent 实际会读到的合并系统提示词{'末尾' if got else ''})" + "─" * 12)
    print(merged[-900:] if got else merged)
    print("=" * 66)
    return hits


def review_and_save(rec: dict, patch_dir: Path, mcpmark_root=None,
                    auto: bool = False) -> str | None:
    """Show the patch (scan + diff), then approve/edit/reject. Returns saved path
    or None if rejected. auto=True approves without prompting (but refuses on a
    threat hit — a patch that trips the scanner must be seen by a human)."""
    while True:
        hits = _render(rec, mcpmark_root)
        if auto:
            if hits:
                print("✗ [--yes] 但命中危险模式,拒绝自动落盘;请人工审阅后手动批准。")
                return None
            print("[--yes] 自动批准。")
            return _save(rec, patch_dir)
        try:
            ans = input("批准补丁? [a]pprove / [e]dit / [r]eject: ").strip().lower()
        except EOFError:
            print("\n(非交互输入,未批准。用 --yes 可自动批准无危险补丁。)")
            return None
        if ans in ("a", "approve"):
            return _save(rec, patch_dir)
        if ans in ("r", "reject", ""):
            print("✗ 已拒绝,补丁未落盘。")
            return None
        if ans in ("e", "edit"):
            rec["patch_text"] = _edit(rec["patch_text"])
            print("(已编辑,重新展示 + 扫描)")
            continue
        print("请输入 a / e / r。")
