#!/usr/bin/env bash
set -euo pipefail
TOPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${TOPD_ROOT}:${PYTHONPATH:-}"
cd "${TOPD_ROOT}"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/CMT_run [extra topd.cli compare arguments]" >&2
  exit 2
fi
TOPD_BASELINE_DIR="$1"
shift
TOPD_RUN_DIR="${TOPD_RUN_DIR:-${TOPD_ROOT}/outputs/topd_compe_math_4b_17b_seed${TRAIN_SEED:-42}}"
exec "${PYTHON_BIN:-python3}" -m topd.cli compare --topd "${TOPD_RUN_DIR}" \
  --baseline "${TOPD_BASELINE_DIR}" --output "${TOPD_RUN_DIR}/comparison.json" "$@"
