#!/usr/bin/env bash
# Breeze-ASR-25 as an OpenAI-compatible STT API on :8380 (model stays resident).
set -euo pipefail
cd "$(dirname "$0")"
export TRANSCRIBE_LIBRARY="$PWD/transcribe.cpp/build-shared/src/libtranscribe.dylib"
exec .venv-mlx/bin/python breeze_server.py
