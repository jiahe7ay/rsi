"""(4) Trajectory store — append-only JSONL, reward-labeled, reproducible.

Trajectories land here after verification. Later stages read from here:
  * SFT: filter verify.passed is True, dedupe, (optionally) compress via
    hermes-agent trajectory_compressor to fit the training token budget;
  * RL:  use verify.reward directly.
Provenance (RunMeta) travels with every record so any run is reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ouroboros.schema import Trajectory, trajectory_from_dict


def append(traj: Trajectory, path: str | Path) -> None:
    """Append one trajectory as a JSONL line."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(traj.to_json() + "\n")


def read(path: str | Path) -> Iterator[Trajectory]:
    """Stream trajectories back from a JSONL pool."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield trajectory_from_dict(json.loads(line))
