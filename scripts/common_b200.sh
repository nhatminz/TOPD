#!/usr/bin/env bash
set -euo pipefail

# Same runtime environment and path conventions as the original B200 launchers.
# This copy is self-contained; it never sources or writes the sibling project.
TOPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${TOPD_ROOT}:${PYTHONPATH:-}"
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

resolve_topd_run_paths() {
  TOPD_RUN_OUTPUT="$("${PYTHON_BIN}" -m topd.cli paths --key run_output)"
  TOPD_LAUNCH_CONFIG="$("${PYTHON_BIN}" -m topd.cli paths --key launch_config)"
  TOPD_EVAL_OUTPUT="$("${PYTHON_BIN}" -m topd.cli paths --key eval_output)"
}
