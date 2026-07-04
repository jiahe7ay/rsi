"""Reproducibility helpers: checkpoint hash, git revs, cost accounting.

Every run stamps a RunMeta so results can be reproduced and costs tallied — each
RSI generation is expensive, so always know the token / dollar spend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def checkpoint_hash(path: str | Path) -> Optional[str]:
    """Stable id/hash for a checkpoint dir (for provenance)."""
    raise NotImplementedError


def git_rev(repo: str | Path) -> Optional[str]:
    """Short git rev of a repo (hermes-agent / mcpmark / ouroboros)."""
    raise NotImplementedError
