"""Config loading (configs/default.yaml + per-run overrides)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config; falls back to configs/default.yaml."""
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
