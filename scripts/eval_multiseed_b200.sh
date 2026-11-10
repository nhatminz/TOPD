#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"
cd "${TOPD_ROOT}"
# Inspect wheel metadata before creating any GPU evaluator. Planning/help stays
# usable offline even if the training environment needs repair.
TOPD_CHECK_DEPENDENCIES=1
for argument in "$@"; do
  case "${argument}" in
    --dry-run|--help|-h) TOPD_CHECK_DEPENDENCIES=0 ;;
  esac
done
if (( TOPD_CHECK_DEPENDENCIES )); then
  "${PYTHON_BIN}" -m topd.dependencies
fi
# Evaluate fixed checkpoints with fresh sampling seeds. No old eval history
# is required. Defaults may be overridden by the same CLI as BellmanOPD.
TOPD_RUN_OUTPUT="$("${PYTHON_BIN}" -m topd.cli paths --key run_output)"
exec "${PYTHON_BIN}" -m b200_experiment.multiseed_evaluation \
  --run-output "${TOPD_RUN_OUTPUT}" --checkpoints final \
  --fresh-seeds --seeds 42 43 44 "$@"
