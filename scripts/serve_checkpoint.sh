#!/usr/bin/env bash
# Serve a checkpoint with sglang as an OpenAI-compatible endpoint for rollout/eval.
# Fill in per your validated Qwen3.6 serving setup
# (mamba TP needs sglang>=0.5.11, CUDA graph needs CUDA_HOME, GDN needs tilelang, ...).
set -euo pipefail

CKPT="${1:?usage: serve_checkpoint.sh <checkpoint_path> [port]}"
PORT="${2:-30000}"

# TODO: match your validated serving flags (tp=8, CUDA graph, tilelang for GDN).
python -m sglang.launch_server \
  --model-path "$CKPT" \
  --port "$PORT" \
  --tp 8
