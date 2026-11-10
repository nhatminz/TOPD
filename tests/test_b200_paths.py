import shutil
from pathlib import Path

import pytest

import topd.configuration as configuration
from b200_experiment.config import load_config, resolve_runtime_paths


@pytest.fixture(autouse=True)
def clean_launch_environment(monkeypatch):
    for key in (
        "STORAGE_ROOT", "TEACHER_MODEL", "TEACHER_MODEL_PATH", "TEACHER_PATH",
        "STUDENT_MODEL", "STUDENT_MODEL_PATH", "STUDENT_PATH", "TRAIN_DATA", "TRAIN_DATA_PATH",
        "PROMPT_KEY", "TRAIN_PROMPT_KEY", "TRAIN_DATA_SPLIT", "TRAIN_PREFER_SOURCE_PROMPT",
        "BASELINE_CONFIG", "RESUME_FROM_CHECKPOINT", "RESUME", "TRAIN_SEED", "SEED",
        "EXPERIMENT_SEED", "OUTPUT_ROOT", "OUTPUT_DIR", "TOPD_OUTPUT_DIR", "TOPD_RUN_DIR",
        "TOPD_RUN_NAME", "RUN_NAME", "COMPARISON_NAME", "RESULTS_DIR", "TOPD_EVAL_OUTPUT",
        "TOPD_CHECKPOINT", "TRAIN_EVAL_BASE_CACHE_DIR", "VERIFY_OFFICIAL_WEIGHTS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_all_shared_asset_paths_match_b200_reference():
    reference = resolve_runtime_paths(load_config(configuration.ROOT/"configs/qwen3_b200_base.yaml"))
    actual = configuration.launch_config()
    assert actual["paths"] == reference["paths"]
    assert actual["data"] == reference["data"]
    assert actual["evaluation"]["benchmarks"] == reference["evaluation"]["benchmarks"]
    assert actual["training_evaluation"]["base_cache_dir"] == reference["training_evaluation"]["base_cache_dir"]
    assert actual["experiment"]["require_b200"] is True
    assert actual["models"]["teacher_path"] == "/workspace/storage-shared/models/Qwen3-4B"
    assert actual["models"]["student_path"] == "/workspace/storage-shared/models/Qwen3-1.7B"


def test_b200_aliases_and_override_precedence_match_original(monkeypatch):
    values = dict(STORAGE_ROOT="/workspace/b200-storage", TEACHER_MODEL="models/Qwen3-4B",
        TEACHER_MODEL_PATH="models/unused-teacher", STUDENT_MODEL="models/Qwen3-1.7B",
        STUDENT_MODEL_PATH="models/unused-student", TRAIN_DATA="data/train.parquet",
        TRAIN_DATA_PATH="data/unused.parquet", PROMPT_KEY="problem", TRAIN_PROMPT_KEY="unused",
        TRAIN_DATA_SPLIT="null", SEED="43")
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    actual = configuration.launch_config()
    assert actual["models"]["teacher_path"] == "/workspace/b200-storage/models/Qwen3-4B"
    assert actual["models"]["student_path"] == "/workspace/b200-storage/models/Qwen3-1.7B"
    assert actual["data"]["path"] == "/workspace/b200-storage/data/train.parquet"
    assert actual["data"]["prompt_key"] == "problem"
    assert actual["data"]["split"] is None
    assert actual["experiment"]["seed"] == 43


def test_original_path_aliases_accept_absolute_paths(monkeypatch):
    monkeypatch.setenv("TEACHER_PATH", "/workspace/assets/Qwen3-4B")
    monkeypatch.setenv("STUDENT_PATH", "/workspace/assets/Qwen3-1.7B")
    monkeypatch.setenv("TRAIN_DATA_PATH", "/workspace/assets/train.parquet")
    actual = configuration.launch_config()
    assert actual["models"]["teacher_path"] == "/workspace/assets/Qwen3-4B"
    assert actual["models"]["student_path"] == "/workspace/assets/Qwen3-1.7B"
    assert actual["data"]["path"] == "/workspace/assets/train.parquet"


def test_b200_output_and_eval_layout_uses_run_names(monkeypatch):
    monkeypatch.setenv("RUN_NAME", "topd_b200_seed42")
    actual = configuration.b200_run_paths()
    assert actual["run_output"] == str(configuration.ROOT/"outputs/topd_b200_seed42/topd")
    assert actual["eval_output"] == str(configuration.ROOT/"results/topd_b200_seed42/eval/topd")
    assert actual["launch_config"] == str(configuration.ROOT/"results/launch_configs/topd_b200_seed42.yaml")
    assert actual["checkpoint"] == str(Path(actual["run_output"])/"final")


def test_b200_output_root_and_results_dir_overrides(monkeypatch):
    monkeypatch.setenv("RUN_NAME", "experiment42")
    monkeypatch.setenv("OUTPUT_ROOT", "/workspace/output")
    monkeypatch.setenv("RESULTS_DIR", "/workspace/results")
    monkeypatch.setenv("TOPD_CHECKPOINT", "/workspace/output/experiment42/topd/checkpoint-000150")
    actual = configuration.b200_run_paths()
    assert actual["run_output"] == "/workspace/output/experiment42/topd"
    assert actual["eval_output"] == "/workspace/results/eval/topd"
    assert actual["checkpoint"].endswith("checkpoint-000150")


def test_relocated_checkout_never_reuses_workstation_paths(tmp_path, monkeypatch):
    relocated = tmp_path/"b200-checkout/TOPD"
    shutil.copytree(configuration.ROOT/"configs", relocated/"configs")
    monkeypatch.setattr(configuration, "ROOT", relocated)
    actual = configuration.launch_config()
    assert actual["experiment"]["output_dir"] == str(relocated/"outputs/topd_compe_math_4b_17b_seed42/topd")
    assert actual["models"]["student_path"].startswith("/workspace/storage-shared/")


def test_optional_official_verification_is_explicit(monkeypatch):
    assert configuration.launch_config()["topd"]["verify_official_weights"] is False
    monkeypatch.setenv("VERIFY_OFFICIAL_WEIGHTS", "true")
    assert configuration.launch_config()["topd"]["verify_official_weights"] is True
