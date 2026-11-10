"""Check FlashInfer wheels without importing CUDA libraries or loading models."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
import shlex
import subprocess
import sys


def flashinfer_versions():
    result = {}
    for package in ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def cubin_install_command(target):
    # Keep the existing FlashInfer/PyTorch/vLLM versions. Only replace cubins.
    return [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
            f"flashinfer-cubin=={target}"]


def check_flashinfer():
    packages = flashinfer_versions()
    python_version = packages["flashinfer-python"]
    if python_version:
        cubin = packages["flashinfer-cubin"]
        if cubin and cubin != python_version:
            raise RuntimeError(
                f"FlashInfer dependency mismatch in {sys.executable}: "
                f"flashinfer-python=={python_version}, flashinfer-cubin=={cubin}. "
                "This prevents vLLM engine initialization. Repair with: "
                + shlex.join(cubin_install_command(python_version)))
        jit_cache = packages["flashinfer-jit-cache"]
        # FlashInfer permits a CUDA build suffix on the optional JIT cache.
        if jit_cache and not jit_cache.startswith(python_version):
            raise RuntimeError(
                f"FlashInfer dependency mismatch in {sys.executable}: "
                f"flashinfer-python=={python_version}, flashinfer-jit-cache=={jit_cache}. "
                f"Install a flashinfer-jit-cache build matching {python_version} and your CUDA version.")
    return {"python": sys.executable, "packages": packages,
            "status": "versions_match" if python_version else "flashinfer_not_installed"}


def repair_cubin(*, dry_run=False):
    packages = flashinfer_versions()
    target = packages["flashinfer-python"]
    if not target:
        raise RuntimeError(
            f"flashinfer-python is not installed in {sys.executable}. "
            "Select the B200 training venv with PYTHON_BIN before repairing.")
    command = cubin_install_command(target)
    needs_repair = packages["flashinfer-cubin"] != target
    report = {"python": sys.executable, "before": packages, "command": command,
              "needs_repair": needs_repair, "dry_run": dry_run}
    print(json.dumps(report, indent=2), flush=True)
    if dry_run:
        return report
    if needs_repair:
        subprocess.run(command, check=True)
    result = check_flashinfer()
    if result["packages"]["flashinfer-cubin"] != target:
        raise RuntimeError("pip completed but this Python still sees different cubins; inspect PYTHONPATH and package locations.")
    # An actual import catches shadowed modules, missing libraries and JIT-cache
    # mismatches too. Do not suppress the upstream version guard during this check.
    environment = dict(os.environ)
    environment.pop("FLASHINFER_DISABLE_VERSION_CHECK", None)
    subprocess.run([sys.executable, "-c",
                    "import flashinfer; print('FlashInfer import OK:', "
                    "flashinfer.__version__, flashinfer.__file__)"],
                   env=environment, check=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair", action="store_true", help="Align cubins with the installed flashinfer-python")
    parser.add_argument("--dry-run", action="store_true", help="Show the repair command without installing/importing")
    args = parser.parse_args(argv)
    if args.dry_run and not args.repair:
        parser.error("--dry-run requires --repair")
    try:
        result = repair_cubin(dry_run=args.dry_run) if args.repair else check_flashinfer()
    except RuntimeError as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
