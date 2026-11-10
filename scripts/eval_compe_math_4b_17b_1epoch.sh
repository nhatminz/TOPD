#!/usr/bin/env bash
set -euo pipefail
TOPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${TOPD_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
TOPD_RUN_DIR="${TOPD_RUN_DIR:-${TOPD_ROOT}/outputs/topd_compe_math_4b_17b_seed${TRAIN_SEED:-42}}"
cd "${TOPD_ROOT}"
exec "${PYTHON_BIN:-python3}" -m topd.cli evaluate --run "${TOPD_RUN_DIR}" "$@"
