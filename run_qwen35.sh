#!/usr/bin/env bash
# Qwen3.5-4B Q4_K_M on llama.cpp + Metal. Polish-layer candidate.
set -euo pipefail
cd "$(dirname "$0")"
exec llama-server -m ./models/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf \
  -ngl 99 --temp 0.7 --top-p 0.95 --top-k 20 --jinja \
  -c 16384 --cache-type-k q4_0 --cache-type-v q4_0 -np 1 \
  --port ${PORT:-8902} --alias qwen3.5-4b
