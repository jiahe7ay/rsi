"""Prompt-level self-improvement: learn from an OLD model's benchmark trajectories
(pulled from HF), diagnose failure modes, and generate a prompt PATCH that is
layered on at rollout time — never editing the benchmark's own prompts.

Pipeline:  HF trajectories → analyze (failure modes) → patch_gen → rollout --patch
"""
from ouroboros.improve.hf_traj import pull, HFTraj
from ouroboros.improve.analyze import aggregate, summarize_failures

__all__ = ["pull", "HFTraj", "aggregate", "summarize_failures"]
