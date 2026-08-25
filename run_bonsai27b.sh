#!/usr/bin/env bash
# Bonsai-27B 1-bit (binary g128, 1.125 bpw, Qwen3.6-27B base) on llama.cpp + Metal.
#   ./run_bonsai27b.sh          -> llama-server, web UI + OpenAI API at http://localhost:8900
#   ./run_bonsai27b.sh cli "問題"  -> one-shot completion
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${MODEL:-./models/bonsai-27b-1bit/Bonsai-27B-Q1_0.gguf}"
PORT="${PORT:-8900}"
# Sampling per the model card (thinking mode): temp 0.7 / top-p 0.95 / top-k 20
COMMON=(-m "$MODEL" -ngl 99 --temp 0.7 --top-p 0.95 --top-k 20 --jinja)
if [[ "${1:-server}" == "cli" ]]; then
  exec llama-cli "${COMMON[@]}" -c 8192 -no-cnv -p "${2:?need a prompt}" -n 512
else
  exec llama-server "${COMMON[@]}" -c 16384 --port "$PORT" \
       --cache-type-k q4_0 --cache-type-v q4_0 -np 1 --alias bonsai-27b-1bit
fi
