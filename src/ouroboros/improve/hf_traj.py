"""Pull a benchmark's trajectories from an HF dataset (private, token-gated).

Layout observed for taiyi-lab/mcpmark-eval @ eval-mcpmark-260626:
    run{k}/{domain}/{passed|failed}/{category__task}.json
each json: {task_name, success, verification_error, model, messages, ...}

Token resolution: explicit arg > HF_TOKEN env > huggingface_hub cached login.
So once `login(token=...)` has been run on the box, callers pass nothing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class HFTraj:
    task_name: str                 # "category__task"
    domain: str
    run: str                       # run1..runN  → the k-th sample
    success: bool
    model: str
    verification_error: Optional[str]
    messages: list
    src: str                       # path within the repo (provenance)


def _tok(explicit: str | None) -> str | None:
    return explicit or os.environ.get("HF_TOKEN") or None


def pull(repo: str, revision: str, token: str | None = None,
         domains: list[str] | None = None, only: str | None = None,
         limit: int | None = None, repo_type: str = "dataset") -> list[HFTraj]:
    """Download + parse trajectory jsons.

    only ∈ {"passed","failed",None}; domains filters by service; limit caps files
    (for quick tests). Returns list[HFTraj]. Files that fail to parse are skipped.
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    tk = _tok(token)
    files = [f for f in list_repo_files(repo, repo_type=repo_type, revision=revision, token=tk)
             if f.endswith(".json") and f.count("/") >= 3]

    def keep(f: str) -> bool:
        _, dom, status, _ = f.split("/", 3)
        if domains and dom not in domains:
            return False
        if only and status != only:
            return False
        return True

    files = [f for f in files if keep(f)]
    if limit:
        files = files[: int(limit)]

    out: list[HFTraj] = []
    for f in files:
        try:
            p = hf_hub_download(repo, f, repo_type=repo_type, revision=revision, token=tk)
            d = json.loads(open(p, encoding="utf-8").read())
        except Exception:
            continue
        run, dom, _status, fname = f.split("/", 3)
        out.append(HFTraj(
            task_name=d.get("task_name") or fname[:-5],
            domain=dom, run=run,
            success=bool(d.get("success")),
            model=d.get("model", ""),
            verification_error=d.get("verification_error"),
            messages=d.get("messages") or [],
            src=f,
        ))
    return out
