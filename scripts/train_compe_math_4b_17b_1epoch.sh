#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"
cd "${TOPD_ROOT}"
resolve_topd_run_paths
mkdir -p "$(dirname "${TOPD_LAUNCH_CONFIG}")"
if [[ "${1:-}" == "--dry-run" ]]; then
  exec "${PYTHON_BIN}" -m topd.cli prepare --dry-run --output "${TOPD_LAUNCH_CONFIG}"
fi
if (( $# != 0 )); then
  echo "Usage: $0 [--dry-run]; use documented environment variables for configuration." >&2
  exit 2
fi
"${PYTHON_BIN}" -m topd.cli prepare --output "${TOPD_LAUNCH_CONFIG}"
TOPD_GPU_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
echo "TOPD: ${TOPD_GPU_COUNT} visible GPUs, teacher Qwen3-4B, student Qwen3-1.7B, one complete epoch; training-time evaluation disabled."
if (( TOPD_GPU_COUNT == 1 )); then
  "${PYTHON_BIN}" -m topd.cli train --config "${TOPD_LAUNCH_CONFIG}"
else
  "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${TOPD_GPU_COUNT}" -m topd.cli train --config "${TOPD_LAUNCH_CONFIG}"
fi
