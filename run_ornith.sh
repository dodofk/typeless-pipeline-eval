#!/usr/bin/env bash
# Ornith-1.5-9B Q4_K_M (Qwen3.5-9B base, agentic-coding RL) on llama.cpp + Metal.
set -euo pipefail
cd "$(dirname "$0")"
exec llama-server -m ./models/ornith-1.5-9b/Ornith-1.5-9B-Q4_K_M.gguf \
  -ngl 99 --temp 0.7 --top-p 0.95 --top-k 20 --jinja \
  -c 16384 --cache-type-k q4_0 --cache-type-v q4_0 -np 1 \
  --port ${PORT:-8901} --alias ornith-1.5-9b
