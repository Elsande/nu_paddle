#!/usr/bin/env bash
# run.sh — peluncur nu-paddle (batch CLI / UI Gradio)
# Pakai:
#   ./run.sh                  -> batch SEMUA dokumen contoh
#   ./run.sh [file1 file2 ...] -> batch dokumen tertentu
#   ./run.sh --ui              -> buka UI Gradio di browser
#   ./run.sh --ui --share      -> UI + link publik sementara
set -euo pipefail
cd "$(dirname "$0")"

PY="$PWD/venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "ERROR: venv tidak ditemukan. Jalankan dulu: python3.12 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

if [[ "${1:-}" == "--ui" || "${1:-}" == "ui" ]]; then
    shift
    exec "$PY" run_batch.py --ui "$@"
fi

exec "$PY" run_batch.py "$@"
