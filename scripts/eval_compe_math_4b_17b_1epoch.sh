#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"
cd "${TOPD_ROOT}"
resolve_topd_run_paths
exec "${PYTHON_BIN}" -m topd.cli evaluate --run "${TOPD_RUN_OUTPUT}" \
  --output "${TOPD_EVAL_OUTPUT}" "$@"
