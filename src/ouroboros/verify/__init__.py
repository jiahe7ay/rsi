"""Auto-verify: LLM-generated reward for any benchmark.

Given a benchmark task + an agent trajectory, an LLM generates a checkable rubric
(saved & human-reviewable) and a strict judge scores it → VerifyResult(passed,
reward, detail). This is Ouroboros' controllable reward signal — the LLM implements
the verifier per task; we only provide the harness + the reviewable artifact.

    from ouroboros.verify import verify_one, verify_hf

Pairs with onboard (how to RUN) and improve (how to FIX): auto-verify is how to JUDGE.
"""
from ouroboros.verify.autoverify import verify_one, verify_hf
from ouroboros.verify.rubric import generate_rubric, load_rubric, save_rubric, rubric_path
from ouroboros.verify.judge import judge
from ouroboros.verify.trajectory_view import extract_task, render_actions

__all__ = ["verify_one", "verify_hf", "generate_rubric", "load_rubric", "save_rubric",
           "rubric_path", "judge", "extract_task", "render_actions"]
