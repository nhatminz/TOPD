import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from b200_experiment.config import load_config
from topd.cli import data_audit
from topd.configuration import ROOT, resolve_official_models, sha256_file
from topd.smoke import tiny_tokenizer


def model_assets(tmp_path):
    config = load_config(ROOT/"configs/topd_compe_math.yaml")
    info = {}
    for role, hidden, layers in (("student", 2048, 28), ("teacher", 2560, 36)):
        root = tmp_path/role
        root.mkdir()
        (root/"config.json").write_text(json.dumps({"hidden_size": hidden, "num_hidden_layers": layers}))
        (root/"model.safetensors").write_bytes(f"official-{role}".encode())
        config["models"][f"{role}_path"] = str(root)
        info[config["topd"][f"{role}_source"]] = SimpleNamespace(sha=f"{role}-pinned-commit",
            siblings=[SimpleNamespace(rfilename="model.safetensors",
                lfs=SimpleNamespace(sha256=sha256_file(root/"model.safetensors")))])
    return config, info


def test_verify_exact_model_bytes_and_pin_commits(tmp_path):
    config, info = model_assets(tmp_path)
    with patch("huggingface_hub.model_info", side_effect=lambda repo, **kwargs: info[repo]):
        provenance = resolve_official_models(config)
    assert provenance["student"]["source"] == "Qwen/Qwen3-1.7B"
    assert config["topd"]["student_revision"] == "student-pinned-commit"


def test_renamed_base_or_different_initial_weights_rejected(tmp_path):
    config, info = model_assets(tmp_path)
    (Path(config["models"]["student_path"])/"model.safetensors").write_bytes(b"renamed-base-weights")
    with patch("huggingface_hub.model_info", side_effect=lambda repo, **kwargs: info[repo]):
        with pytest.raises(ValueError, match="weights differ"):
            resolve_official_models(config)


def test_data_audit_counts_epoch_and_rejects_question_leakage(tmp_path):
    config = load_config(ROOT/"configs/topd_compe_math.yaml")
    config["data"]["path"] = str(tmp_path/"train.jsonl")
    (tmp_path/"train.jsonl").write_text("\n".join(json.dumps({"problem": f"token{i+3}"}) for i in range(3)))
    evaluation_path = tmp_path/"test.jsonl"
    evaluation_path.write_text(json.dumps({"problem": "token20", "answer": "1"}))
    config["evaluation"]["benchmarks"] = {"Competition-MATH": {"path": str(evaluation_path)}}
    with patch("b200_experiment.models.load_student_tokenizer", return_value=tiny_tokenizer()):
        audit = data_audit(config)
        assert audit["examples"] == 3 and audit["original_rollouts"] == 12
        assert audit["optimizer_updates"] == 1
        evaluation_path.write_text(json.dumps({"problem": "token3", "answer": "1"}))
        with pytest.raises(ValueError, match="overlap"):
            data_audit(config)


def test_original_evaluator_accepts_topd_name(monkeypatch, tmp_path):
    import torch
    from b200_experiment.evaluation import evaluate_suite
    config = load_config(ROOT/"configs/topd_compe_math.yaml")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with patch("b200_experiment.vllm_evaluation.evaluate_vllm_suite", return_value={"model": "TOPD"}):
        result = evaluate_suite("TOPD", tmp_path, config, tmp_path)
    assert result["model"] == "TOPD"
