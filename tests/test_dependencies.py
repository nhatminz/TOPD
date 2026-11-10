import argparse
import json
import os
import subprocess
import sys

import pytest

from topd import dependencies
from topd.configuration import ROOT


def installed(monkeypatch, python="0.6.4", cubin="0.6.6", jit=None):
    packages = {"flashinfer-python": python, "flashinfer-cubin": cubin, "flashinfer-jit-cache": jit}

    def version(package):
        if packages[package] is None:
            raise dependencies.PackageNotFoundError(package)
        return packages[package]

    monkeypatch.setattr(dependencies, "version", version)
    return packages


def test_reported_flashinfer_mismatch_fails_even_if_guard_bypass_is_set(monkeypatch):
    installed(monkeypatch)
    monkeypatch.setenv("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    with pytest.raises(RuntimeError) as error:
        dependencies.check_flashinfer()
    message = str(error.value)
    assert "flashinfer-python==0.6.4" in message and "flashinfer-cubin==0.6.6" in message
    assert "flashinfer-cubin==0.6.4" in message
    assert sys.executable in message and "--no-deps" in message


@pytest.mark.parametrize("python,cubin,jit,status", [
    ("0.6.4", "0.6.4", None, "versions_match"),
    ("0.6.4", None, "0.6.4+cu129", "versions_match"),
    (None, None, None, "flashinfer_not_installed"),
])
def test_supported_flashinfer_metadata_without_cuda_import(monkeypatch, python, cubin, jit, status):
    installed(monkeypatch, python, cubin, jit)
    assert dependencies.check_flashinfer()["status"] == status


def test_incompatible_optional_jit_cache_is_reported(monkeypatch):
    installed(monkeypatch, cubin="0.6.4", jit="0.6.6+cu129")
    with pytest.raises(RuntimeError, match="flashinfer-jit-cache==0.6.6"):
        dependencies.check_flashinfer()


def test_repair_dry_run_cannot_install_or_import_cuda(monkeypatch):
    installed(monkeypatch)
    monkeypatch.setattr(dependencies.subprocess, "run", lambda *a, **kw: pytest.fail("dry run must not execute"))
    report = dependencies.repair_cubin(dry_run=True)
    assert report["command"] == [sys.executable, "-m", "pip", "install", "--no-deps",
                                  "--force-reinstall", "flashinfer-cubin==0.6.4"]


@pytest.mark.parametrize("cubin,expected_installs", [("0.6.6", 1), ("0.6.4", 0)])
def test_repair_only_changes_cubin_then_verifies_real_import(monkeypatch, cubin, expected_installs):
    packages = installed(monkeypatch, cubin=cubin)
    monkeypatch.setenv("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == sys.executable
        if command[1:3] == ["-m", "pip"]:
            assert command[-1] == "flashinfer-cubin==0.6.4" and "--no-deps" in command
            packages["flashinfer-cubin"] = "0.6.4"
        else:
            assert "import flashinfer" in command[2]
            assert "FLASHINFER_DISABLE_VERSION_CHECK" not in kwargs["env"]

    monkeypatch.setattr(dependencies.subprocess, "run", run)
    assert dependencies.repair_cubin()["status"] == "versions_match"
    assert len(calls) == expected_installs + 1


def test_preparation_rejects_mismatch_before_reading_or_hashing_models(monkeypatch, tmp_path):
    import torch
    from topd import cli
    installed(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(cli, "resolve_official_models", lambda *a, **kw: pytest.fail("must fail before model access"))
    with pytest.raises(RuntimeError, match="FlashInfer dependency mismatch"):
        cli.prepare(argparse.Namespace(dry_run=False, output=tmp_path/"config.yaml"))


def test_repair_script_uses_selected_python_and_installed_version(tmp_path):
    for name, version in (("flashinfer-python", "0.6.4"), ("flashinfer-cubin", "0.6.6")):
        metadata = tmp_path/f"{name.replace('-', '_')}-{version}.dist-info"
        metadata.mkdir()
        (metadata/"METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    environment = dict(os.environ, PYTHON_BIN=sys.executable, PYTHON_EXEC="/wrong/python", PYTHONPATH=str(tmp_path))
    result = subprocess.run(["bash", str(ROOT/"scripts/fix_flashinfer_b200.sh"), "--dry-run"],
                            env=environment, check=True, text=True, capture_output=True)
    assert "flashinfer-cubin==0.6.4" in result.stdout
    assert sys.executable in result.stdout and '"dry_run": true' in result.stdout
