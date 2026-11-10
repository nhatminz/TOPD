import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from b200_experiment.config import load_config, save_config
from b200_experiment.data import build_response_valid_mask, epoch_batch_indices
from b200_experiment.distributed import (
    DistributedContext, grouped_ppo_minibatch_partition, padded_local_indices,
)
from b200_experiment.opd_core import build_student_topk_opd_reference
from b200_experiment.resume import validate_resume_config
from b200_experiment.scoring import generate_on_policy, score_original_rollout
from b200_experiment.trainer import _opd_train_step
from topd.algorithm import SparseTargets
from topd.cli import parser_selfcheck
from topd.configuration import ROOT, launch_config, validate_config
from topd.engine import TOPDEngine, FutureWindow, generate_near_future
from topd.smoke import run_smoke, smoke_config, tiny_model, tiny_tokenizer


def test_primary_student_identity_and_hyperparameters(monkeypatch):
    for key in ("STUDENT_MODEL_PATH", "TRAIN_SEED", "OUTPUT_DIR", "RESUME_FROM_CHECKPOINT"):
        monkeypatch.delenv(key, raising=False)
    c = launch_config()
    assert c["topd"]["student_source"] == "Qwen/Qwen3-1.7B"
    assert c["topd"]["teacher_source"] == "Qwen/Qwen3-4B"
    assert c["training"]["epochs"] == 1
    assert c["experiment"]["seed"] == 42
    assert c["rollout"]["num_responses"] == 4
    assert c["rollout"]["max_new_tokens"] == 4096
    assert c["training"]["learning_rate"] == 5e-6
    assert c["training"]["ppo_mini_batch_size"] == 64
    assert c["topd"]["future_window"] == 50 and c["selector"]["top_k"] == 16


@pytest.mark.parametrize("section,key,value", [
    ("models", "student_path", "/models/Qwen3-1.7B-Base"),
    ("topd", "student_source", "Qwen/Qwen3-1.7B-Base"),
    ("training", "epochs", 2), ("training", "max_steps", 5),
    ("training", "resume_allow_config_mismatch", True),
    ("topd", "future_window", 16), ("topd", "lambda_topd", float("nan")),
    ("data", "split", "test"), ("data", "path", "/tmp/test.parquet"),
])
def test_primary_invalid_configs_fail(section, key, value):
    c = load_config(ROOT/"configs/topd_compe_math.yaml")
    c[section][key] = value
    with pytest.raises(ValueError):
        validate_config(c)


def test_one_epoch_coverage_uneven_tail_across_ranks():
    seen = []
    for step in range(3):
        global_indices = epoch_batch_indices(9, 4, step, 42)
        real = []
        for rank in range(2):
            local, active = padded_local_indices(global_indices, rank, 2)
            real.extend(i for i, is_active in zip(local, active) if is_active)
        assert real == global_indices
        seen.extend(real)
    assert sorted(seen) == list(range(9))


def test_cmt_group_order_matches_topd():
    ids = ((0, 1, 0, 1), (0, 1, 0, 1))
    for group in range(2):
        for rank in range(2):
            selected, count = grouped_ppo_minibatch_partition(ids, rank, 4, group)
            assert count == 4
            assert all(ids[rank][row] == group for row in selected)


def test_shared_response_mask_eos_and_padding():
    mask = build_response_valid_mask(torch.tensor([[4, 2, 0, 0], [7, 8, 2, 0]]), [2], 0)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, False]]


def test_original_core_scoring_and_evaluator_are_identical_to_audited_source():
    manifest = json.loads((ROOT/"docs/source_manifest.json").read_text())["sha256"]
    for name in ("opd_core.py", "data.py", "scoring.py", "models.py", "evaluation.py",
                 "math_prompts.py", "vllm_rollout.py", "fsdp.py", "distributed.py",
                 "vllm_evaluation.py"):
        relative = f"b200_experiment/{name}"
        content = (ROOT/relative).read_bytes()
        if name == "evaluation.py":
            content = content.replace(b', "IW-OPD", "TOPD")', b', "IW-OPD")')
        assert hashlib.sha256(content).hexdigest() == manifest[relative]


def test_resume_rejects_changed_topd_objective(tmp_path):
    c = load_config(ROOT/"configs/topd_compe_math.yaml")
    save_config(c, tmp_path/"resolved_config.yaml")
    checkpoint = tmp_path/"checkpoint-000001"
    checkpoint.mkdir()
    modified = copy.deepcopy(c)
    modified["topd"]["lambda_topd"] *= 2
    with pytest.raises(ValueError, match="topd"):
        validate_resume_config(checkpoint, modified)


def test_actual_baseline_configuration_owns_shared_settings(tmp_path, monkeypatch):
    reference = load_config(ROOT/"configs/topd_compe_math.yaml")
    reference["experiment"]["method"] = "cmt"
    reference["training_evaluation"]["vllm"]["seed"] = 42
    reference["evaluation"]["vllm"]["seed"] = 43
    reference["rollout"]["seed"] = 17
    saved = tmp_path/"baseline.yaml"
    save_config(reference, saved)
    monkeypatch.setenv("BASELINE_CONFIG", str(saved))
    monkeypatch.delenv("RESUME_FROM_CHECKPOINT", raising=False)
    monkeypatch.delenv("ROLLOUT_SEED", raising=False)
    config = launch_config()
    assert config["training_evaluation"]["vllm"]["seed"] == 42
    assert config["evaluation"]["vllm"]["seed"] == 43
    assert config["rollout"]["seed"] == 17


def test_near_future_generation_starts_at_prefix_and_stops_at_eos():
    from types import SimpleNamespace
    class EOSModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="toy")
            self.seen = []
        def forward(self, input_ids, **kwargs):
            self.seen.append(input_ids.clone())
            logits = torch.zeros(*input_ids.shape, 8)
            logits[..., 2] = 100
            return SimpleNamespace(logits=logits, past_key_values=None)
    model = EOSModel()
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    results = generate_near_future(model, [torch.tensor([4, 5]), torch.tensor([6])],
        pad_id=0, eos_ids=[2], window=50, temperature=0., top_p=1., seeds=[1, 2], distributed=context)
    assert [result.tolist() for result in results] == [[2], [2]]
    assert model.seen[0].tolist() == [[4, 5], [0, 6]]


def test_trajectory_gradients_and_overlapping_windows_add_exactly(tmp_path):
    student = tiny_model()
    teacher = tiny_model(8).requires_grad_(False)
    config = smoke_config(tmp_path)
    config["topd"]["lambda_topd"] = 1.
    ctx = DistributedContext(0, 0, 1, torch.device("cpu"))
    tokenizer = tiny_tokenizer()
    from b200_experiment.vllm_rollout import rollout_batch_from_token_ids
    rollout = rollout_batch_from_token_ids(torch.tensor([[4, 5]]), torch.ones(1, 2), [[6, 7]], 0)
    engine = TOPDEngine(config, student, teacher, tokenizer, ctx)
    window = FutureWindow(torch.tensor([4, 5]), torch.tensor([6, 7]),
        SparseTargets(torch.tensor([8]), torch.ones(2, 1)), 0, 0, 1.)
    def gradients(windows):
        student.zero_grad(set_to_none=True)
        engine.windows, engine.denominator, engine.metrics = windows, len(windows), {}
        engine.backward(rollout, 1)
        return torch.cat([p.grad.flatten() for p in student.parameters()])
    one = gradients([window])
    two = gradients([window, window])
    assert one.norm() > 0
    assert torch.allclose(one, two, atol=2e-6, rtol=2e-5)
    # Different microbatch size preserves the exact mean-of-window-sums scale.
    config["topd"]["loss_micro_batch_size"] = 2
    together = gradients([window, window])
    assert torch.allclose(one, together, atol=2e-6, rtol=2e-5)
    assert all(p.grad is None for p in teacher.parameters())


def test_future_forward_conditions_on_generated_student_prefix(tmp_path):
    config = smoke_config(tmp_path)
    student, teacher, tokenizer = tiny_model(), tiny_model(8).requires_grad_(False), tiny_tokenizer()
    ctx = DistributedContext(0, 0, 1, torch.device("cpu"))
    from b200_experiment.vllm_rollout import rollout_batch_from_token_ids
    rollout = rollout_batch_from_token_ids(torch.tensor([[4, 5]]), torch.ones(1, 2), [[6, 7]], 0)
    engine = TOPDEngine(config, student, teacher, tokenizer, ctx)
    engine.windows = [FutureWindow(torch.tensor([4, 5]), torch.tensor([6, 7]),
        SparseTargets(torch.tensor([8]), torch.ones(2, 1)), 0, 0, .8)]
    engine.denominator, engine.metrics = 1, {}
    seen = []
    handle = student.register_forward_pre_hook(lambda module, args, kwargs:
        seen.append(kwargs["input_ids"].detach().clone()), with_kwargs=True)
    engine.backward(rollout, 1)
    handle.remove()
    assert seen[0].tolist() == [[4, 5, 6]]  # predict 6 after 5, and 7 after generated 6
    assert not seen[0].eq(8).any()  # teacher guidance token is never a student context token


def test_zero_candidates_still_updates_original_opd(tmp_path):
    config = smoke_config(tmp_path)
    config["topd"]["candidate_budget"] = 0
    student, teacher, tokenizer = tiny_model(), tiny_model(8).requires_grad_(False), tiny_tokenizer()
    ctx = DistributedContext(0, 0, 1, torch.device("cpu"))
    rollout = generate_on_policy(student, torch.tensor([[4, 5]]), torch.ones(1, 2),
        max_new_tokens=3, temperature=1., top_p=1., eos_token_ids=[2], pad_token_id=0, seed=42)
    s = score_original_rollout(student, rollout, top_k=4)
    t = score_original_rollout(teacher, rollout, candidate_ids=s.top_k_ids)
    reference = build_student_topk_opd_reference(s.top_k_ids, s.top_k_log_probs,
        t.candidate_log_probs, rollout.valid_mask, top_k=4)
    engine = TOPDEngine(config, student, teacher, tokenizer, ctx, tmp_path)
    optimizer = torch.optim.AdamW(student.parameters(), lr=.001)
    result = _opd_train_step(student, optimizer, rollout, rollout.valid_mask.float(),
        reference, config, ctx.device, ctx, trajectory_engine=engine)
    assert result["optimizer_steps"] == 1
    assert result["topd"]["trajectory_loss"] == 0
    assert result["topd"]["high_loss_candidates"] == 0
    assert result["topd"]["opd_loss"] == result["topd"]["total_loss"]


def test_transport_failure_cannot_silently_fall_back_to_opd(tmp_path):
    config = smoke_config(tmp_path)
    student, teacher, tokenizer = tiny_model(), tiny_model(8).requires_grad_(False), tiny_tokenizer()
    ctx = DistributedContext(0, 0, 1, torch.device("cpu"))
    from b200_experiment.vllm_rollout import rollout_batch_from_token_ids
    from types import SimpleNamespace
    rollout = rollout_batch_from_token_ids(torch.tensor([[4, 5]]), torch.ones(1, 2), [[6]], 0)
    engine = TOPDEngine(config, student, teacher, tokenizer, ctx)
    with patch("topd.engine.solve_transport", side_effect=RuntimeError("forced OT failure")):
        with pytest.raises(RuntimeError, match="forced OT failure"):
            engine.prepare(rollout, [0], SimpleNamespace(advantages=-torch.ones(1, 1, 4)),
                rollout.valid_mask, 1)


def test_math_parser_matches_original_protocol():
    assert parser_selfcheck()["status"] == "passed"


def test_full_smoke_resume_and_checkpoint(tmp_path):
    result = run_smoke(tmp_path)
    assert result["status"] == "passed"
    assert result["mid_rollout_resume_matches_uninterrupted"]
    assert result["epoch_coverage"] == 1.
    assert result["original_rollouts"] == 6
    assert result["optimizer_steps"] == 3


def test_actual_training_orchestration_with_tiny_models(tmp_path, monkeypatch):
    """Exercise run_training itself; replace CUDA hardware/model asset loading only."""
    import b200_experiment.trainer as trainer
    student, teacher, tokenizer = tiny_model(), tiny_model(8).requires_grad_(False), tiny_tokenizer()
    config = smoke_config(tmp_path/"run")
    config["models"]["dtype"] = "bfloat16"  # production config guard; CPU test model stays FP32
    config["models"].update(student_path=str(tmp_path/"student"), teacher_path=str(tmp_path/"teacher"))
    config["selector"].update(top_k=4, score_micro_batch_size=1)
    config["logging"]["tensorboard"]["enabled"] = False
    config["logging"]["cost"]["enabled"] = False
    evaluation_data = tmp_path/"test.jsonl"
    evaluation_data.write_text(json.dumps({"problem": "token20 token21", "answer": "1"})+"\n")
    config["evaluation"]["benchmarks"] = {"Competition-MATH": {"path": str(evaluation_data)}}
    config["data"]["path"] = str(tmp_path/"train.jsonl")
    Path(config["data"]["path"]).write_text("\n".join(json.dumps({"problem": f"token{i+3} token7"}) for i in range(3))+"\n")
    ctx = DistributedContext(0, 0, 1, torch.device("cpu"))
    monkeypatch.setattr(trainer, "initialize_distributed", lambda *args: ctx)
    monkeypatch.setattr(trainer, "load_student_tokenizer", lambda c: tokenizer)
    monkeypatch.setattr(trainer, "load_models", lambda c, device: (student, teacher, tokenizer,
        {"student_path": c["models"]["student_path"], "teacher_path": c["models"]["teacher_path"]}))
    monkeypatch.setattr(trainer, "collect_metadata", lambda *args: {})
    for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    summary = trainer.run_training(config)
    assert summary["method"] == "topd"
    assert summary["topd"]["original_rollouts"] == 6
    assert summary["topd"]["epoch_coverage"] == 1.
    assert (tmp_path/"run/final/config.json").is_file()
