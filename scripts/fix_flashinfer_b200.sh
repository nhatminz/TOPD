#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"
cd "${TOPD_ROOT}"
# Explicit repair; train/eval only check dependencies and never install them.
exec "${PYTHON_BIN}" -m topd.dependencies --repair "$@"
