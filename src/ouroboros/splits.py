"""(1) Task splits — freeze train / held-out sets ONCE.

The single most important anti-cheat in the whole system: the eval split must be
frozen before any rollout and must NEVER enter training data. Freezing writes a
deterministic manifest per (domain, suite) under configs/splits/ so every future
generation reads the exact same held-out set. Overwriting an existing manifest is
refused unless force=True.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ouroboros import config


def mcpmark_root(cfg: dict) -> Path:
    p = (cfg.get("paths", {}) or {}).get("mcpmark") or "~/mcpmark"
    return Path(p).expanduser()


def splits_dir(cfg: dict) -> Path:
    d = (cfg.get("paths", {}) or {}).get("splits_dir") or "./configs/splits"
    return Path(d).expanduser()


def enumerate_tasks(root: Path, domain: str, suite: str) -> list[str]:
    """List real task ids ("category/task") from mcpmark's on-disk layout.

    A task dir must actually contain files — mcpmark ships empty placeholder
    domain trees (e.g. insforge) that would otherwise freeze un-runnable splits.
    """
    base = root / "tasks" / domain / suite
    out: list[str] = []
    if not base.is_dir():
        return out
    for cat in sorted(p for p in base.iterdir() if p.is_dir()):
        for task in sorted(p for p in cat.iterdir() if p.is_dir()):
            if any(f.is_file() for f in task.rglob("*")):
                out.append(f"{cat.name}/{task.name}")
    return out


def freeze(domains: list[str] | None = None, eval_frac: float = 0.2, seed: int = 0,
           suites: tuple[str, ...] = ("easy", "standard"), force: bool = False,
           config_path=None) -> None:
    """Deterministically partition each (domain, suite) into train/eval and write
    the frozen manifest. Ranking = md5(seed:domain:suite:task) so the split is
    reproducible from the manifest alone and independent of enumeration order.

    domains=None → read the RUNNABLE domains from the latest onboard inventory
    (data/benchmarks/*.json), so a newcomer never has to know domain names."""
    if domains is None:
        from ouroboros.onboard.inventory import latest_inventory
        inv = latest_inventory()
        if inv is None:
            raise SystemExit("没有 benchmark inventory — 先跑 `ouro onboard <repo>`,"
                             "或显式传 --domains")
        domains = [d["name"] for d in inv["domains"] if d.get("runnable")]
        skipped = [f"{d['name']}(缺 {', '.join(d['missing_env'])})"
                   for d in inv["domains"] if not d.get("runnable") and d.get("suites")]
        print(f"  domains ← inventory[{inv['benchmark']}] 可跑: {', '.join(domains) or '(无)'}"
              + (f";跳过: {'; '.join(skipped)}" if skipped else ""))
    cfg = config.load(config_path)
    root, sdir = mcpmark_root(cfg), splits_dir(cfg)
    sdir.mkdir(parents=True, exist_ok=True)
    for domain in domains:
        for suite in suites:
            tasks = enumerate_tasks(root, domain, suite)
            if not tasks:
                print(f"  [skip] {domain}/{suite}: no tasks under {root/'tasks'/domain/suite}")
                continue
            manifest = sdir / f"{domain}-{suite}.json"
            if manifest.exists() and not force:
                print(f"  [frozen] {manifest.name} 已存在,拒绝覆盖(护 eval 完整性;重切须 force)")
                continue
            ranked = sorted(tasks, key=lambda t: hashlib.md5(
                f"{seed}:{domain}:{suite}:{t}".encode()).hexdigest())
            n_eval = max(1, round(len(tasks) * eval_frac))
            eval_set = set(ranked[:n_eval])
            data = {
                "domain": domain, "suite": suite, "seed": seed, "eval_frac": eval_frac,
                "n_tasks": len(tasks),
                "train": sorted(t for t in tasks if t not in eval_set),
                "eval": sorted(t for t in tasks if t in eval_set),
            }
            manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [ok] {domain}/{suite}: {len(tasks)} tasks → "
                  f"train {len(data['train'])} / eval {len(data['eval'])} → {manifest}")


def load_split(split: str, domains: list[str] | None = None, suite: str = "easy",
               config_path=None) -> list[dict]:
    """Return [{domain, suite, task}] for split in {'train','eval'} from frozen manifests."""
    if split not in ("train", "eval"):
        raise ValueError(f"split must be train|eval, got {split!r}")
    cfg = config.load(config_path)
    sdir = splits_dir(cfg)
    if domains is None:
        domains = (cfg.get("tasks", {}) or {}).get("domains") or ["filesystem", "postgres"]
    out: list[dict] = []
    for domain in domains:
        manifest = sdir / f"{domain}-{suite}.json"
        if not manifest.exists():
            raise FileNotFoundError(f"split manifest 不存在: {manifest} — 先跑 `ouro split`")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        out.extend({"domain": domain, "suite": suite, "task": t} for t in data.get(split, []))
    return out
