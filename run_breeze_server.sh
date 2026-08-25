#!/usr/bin/env bash
# Breeze-ASR-25 as an OpenAI-compatible STT API on :8380 (model stays resident).
set -euo pipefail
cd "$(dirname "$0")"
B="$PWD/transcribe.cpp/build-shared"
# build-shared was compiled with absolute rpaths; if the repo ever moves again the
# baked paths go stale. Point dyld at the real dirs relative to $PWD instead.
export DYLD_LIBRARY_PATH="$B/ggml/src:$B/ggml/src/ggml-metal${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export TRANSCRIBE_LIBRARY="$B/src/libtranscribe.dylib"
exec .venv-mlx/bin/python breeze_server.py
