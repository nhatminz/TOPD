#!/usr/bin/env bash
set -euo pipefail
TOPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${TOPD_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
cd "${TOPD_ROOT}"
"${PYTHON_BIN:-python3}" -m pytest -q tests "$@"
"${PYTHON_BIN:-python3}" -m topd.smoke --output "${SMOKE_OUTPUT_DIR:-${TOPD_ROOT}/outputs/smoke}"
