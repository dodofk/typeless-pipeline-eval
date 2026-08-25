#!/usr/bin/env bash
# local-typeless metric CLI。所有依賴由 uv 管,不用先啟 venv。
exec uv run --project "$(cd "$(dirname "$0")" && pwd)" python -m typeless.cli "$@"
