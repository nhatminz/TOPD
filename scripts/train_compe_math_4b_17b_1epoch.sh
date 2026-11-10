#!/usr/bin/env bash
set -euo pipefail
TOPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${TOPD_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${TOPD_ROOT}"
if [[ "${1:-}" == "--dry-run" ]]; then
  exec "${PYTHON_BIN}" -m topd.cli prepare --dry-run --output "${TOPD_ROOT}/outputs/launch_preview.yaml"
fi
if (( $# != 0 )); then
  echo "Usage: $0 [--dry-run]; use documented environment variables for configuration." >&2
  exit 2
fi
TOPD_LAUNCH_CONFIG="$(mktemp "${TMPDIR:-/tmp}/topd-launch.XXXXXXXX.yaml")"
trap 'rm -f "${TOPD_LAUNCH_CONFIG}"' EXIT
PREPARE_ARGS=(--output "${TOPD_LAUNCH_CONFIG}")
if [[ "${DOWNLOAD_MODELS:-0}" == "1" ]]; then
  PREPARE_ARGS+=(--download-models)
fi
"${PYTHON_BIN}" -m topd.cli prepare "${PREPARE_ARGS[@]}"
TOPD_GPU_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
echo "TOPD: ${TOPD_GPU_COUNT} visible GPUs, teacher Qwen3-4B, student Qwen3-1.7B, one complete epoch."
if (( TOPD_GPU_COUNT == 1 )); then
  "${PYTHON_BIN}" -m topd.cli train --config "${TOPD_LAUNCH_CONFIG}"
else
  "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${TOPD_GPU_COUNT}" -m topd.cli train --config "${TOPD_LAUNCH_CONFIG}"
fi
