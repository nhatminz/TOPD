#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"
cd "${TOPD_ROOT}"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/CMT_run [extra topd.cli compare arguments]" >&2
  exit 2
fi
TOPD_BASELINE_DIR="$1"
shift
resolve_topd_run_paths
exec "${PYTHON_BIN}" -m topd.cli compare --topd "${TOPD_RUN_OUTPUT}" \
  --baseline "${TOPD_BASELINE_DIR}" --output "${TOPD_RUN_OUTPUT}/comparison.json" "$@"
