#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"
cd "${TOPD_ROOT}"
# Explicit bounded benchmark only; never invoked by the training launcher.
exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m topd.benchmark "$@"
