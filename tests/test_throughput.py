import copy
import json
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from b200_experiment.config import load_config, save_config
from b200_experiment.distributed import DistributedContext
from b200_experiment.opd_core import build_student_topk_opd_reference
from b200_experiment.resume import restore_optimizer, _restore_rng_states
from b200_experiment.scoring import score_original_rollout
from b200_experiment.trainer import _opd_train_step, _save_checkpoint
from b200_experiment.vllm_rollout import rollout_batch_from_token_ids
from topd.algorithm import (SparseTargets, batched_forward_kl, forward_kl,
                            cosine_ground_cost, solve_transport, uniform_transport_constraints)
from topd.configuration import ROOT, launch_config
from topd.engine import TOPDEngine, generate_near_future, FutureWindow
from topd.generation import generate_shared_futures
from topd.profiling import UpdateProfiler
from topd.runtime import automatic_batch_size, use_accumulation_no_sync
from topd.smoke import smoke_config, tiny_model, tiny_tokenizer


def context():
    return DistributedContext(0, 0, 1, torch.device("cpu"))


@pytest.mark.parametrize("top_p,temperature", [(1., 1.), (.8, .7), (1., 0.)])
@pytest.mark.parametrize("batch", [1, 3, "auto"])
def test_shared_prefix_k50_preserves_seeded_reference_samples(top_p, temperature, batch, tmp_path):
    model = tiny_model()
    cfg = smoke_config(tmp_path)["topd"]
    cfg.update(generation_batch_size=batch, future_temperature=temperature, future_top_p=top_p)
    prefixes = [torch.tensor([4, 5, 6, 7]), torch.tensor([4, 5]), torch.tensor([9]),
                torch.tensor([4, 5, 6]), torch.tensor([9, 10])]
    pairs = [(0, 2), (0, 0), (1, 0), (0, 1), (1, 1)]
    seeds = [41, 92, 123, 125, 876]
    reference = generate_near_future(model, prefixes, pad_id=0, eos_ids=[2], window=50,
        temperature=temperature, top_p=top_p, seeds=seeds, distributed=context())
    results, metrics = generate_shared_futures(model, prefixes, pairs, seeds,
        settings=cfg, pad_id=0, eos_ids=[2], distributed=context())
    assert [tokens.tolist() for tokens in results] == [tokens.tolist() for tokens in reference]
    assert metrics["prefill_calls"] == 2
    assert metrics["prefix_prefill_tokens"] == 6
    assert metrics["reference_prefill_tokens"] == 12


def test_future_prefix_cache_is_invalidated_after_student_update(tmp_path):
    model = tiny_model()
    cfg = smoke_config(tmp_path)["topd"]
    prefixes, pairs, seeds = [torch.tensor([4, 5])], [(0, 0)], [13]
    kwargs = dict(settings=cfg, pad_id=0, eos_ids=[2], distributed=context())
    first, _ = generate_shared_futures(model, prefixes, pairs, seeds, **kwargs)
    with torch.no_grad():
        model.lm_head.weight.zero_()
    second, _ = generate_shared_futures(model, prefixes, pairs, seeds, **kwargs)
    reference = generate_near_future(model, prefixes, pad_id=0, eos_ids=[2], window=50,
        temperature=cfg["future_temperature"], top_p=cfg["future_top_p"], seeds=seeds, distributed=context())
    assert torch.equal(second[0], reference[0])
    assert not torch.equal(first[0], second[0])


@pytest.mark.parametrize("temperature", [.7, 1., 1.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_batched_kl_loss_and_gradient_equivalence(dtype, temperature):
    torch.manual_seed(12)
    reference = torch.randn(3, 5, 71, dtype=dtype, requires_grad=True)
    optimized = reference.detach().clone().requires_grad_(True)
    targets = [SparseTargets(torch.tensor([3, 9, 12]), torch.softmax(torch.randn(5, 3), -1)),
               SparseTargets(torch.tensor([4]), torch.ones(2, 1)),
               SparseTargets(torch.tensor([3, 4]), torch.softmax(torch.randn(4, 2), -1))]
    expected = sum(forward_kl(reference[row, :target.probabilities.shape[0]], target,
                  temperature=temperature).sum() for row, target in enumerate(targets))
    actual = batched_forward_kl(optimized, targets, temperature=temperature, chunk_size=3)
    expected.backward()
    actual.backward()
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-6)
    tolerance = 4e-3 if dtype == torch.bfloat16 else 3e-7
    assert torch.allclose(reference.grad, optimized.grad, atol=tolerance, rtol=tolerance)
    assert optimized.grad[1, 2:].eq(0).all()


def test_cached_normalized_embeddings_preserve_cost_ot_and_targets(tmp_path):
    student, teacher = tiny_model(), tiny_model(8).requires_grad_(False)
    engine = TOPDEngine(smoke_config(tmp_path), student, teacher, tiny_tokenizer(), context())
    s, t = torch.tensor([4, 7, 4]), torch.tensor([9, 4])
    cache = engine._common_embeddings([s, t])
    optimized = (1-torch.stack([cache[int(token)] for token in s]) @
                    torch.stack([cache[int(token)] for token in t]).T).clamp(0, 2)
    with torch.no_grad():
        expected = cosine_ground_cost(teacher.get_input_embeddings()(s).float(), teacher.get_input_embeddings()(t).float())
    assert torch.equal(optimized, expected)
    assert torch.equal(solve_transport(optimized)[0], solve_transport(expected)[0])
    # Subsequent extraction of known IDs does not touch the embedding table.
    with patch.object(teacher.get_input_embeddings(), "forward", side_effect=AssertionError("cached teacher vector")):
        engine._common_embeddings([s, t])
    assert uniform_transport_constraints(3, 2)[0] is uniform_transport_constraints(3, 2)[0]


def test_full_optimizer_update_matches_reference_path(tmp_path):
    rollout = rollout_batch_from_token_ids(torch.tensor([[4, 5], [8, 9]]), torch.ones(2, 2), [[6, 7, 9, 11], [10, 11]], 0)
    teacher = tiny_model(8).requires_grad_(False)
    outputs, metrics = [], []
    for optimized in (False, True):
        model = tiny_model()
        cfg = smoke_config(tmp_path)
        cfg["topd"].update(candidate_budget=None, candidate_fraction=.75,
            future_backend="shared_prefix" if optimized else "reference", reference_generation_batch_size=2,
            generation_batch_size=3 if optimized else 2, loss_micro_batch_size=3 if optimized else 1,
            batched_kl=optimized, accumulation_no_sync="auto" if optimized else False)
        cfg["training"].update(ppo_mini_batch_size=2)
        s = score_original_rollout(model, rollout, top_k=4)
        t = score_original_rollout(teacher, rollout, candidate_ids=s.top_k_ids)
        reference = build_student_topk_opd_reference(s.top_k_ids, s.top_k_log_probs,
            t.candidate_log_probs, rollout.valid_mask, top_k=4)
        engine = TOPDEngine(cfg, model, teacher, tiny_tokenizer(), context())
        result = _opd_train_step(model, torch.optim.AdamW(model.parameters(), lr=.001),
            rollout, rollout.valid_mask.float(), reference, cfg, context().device, context(), trajectory_engine=engine)
        outputs.append(torch.cat([parameter.detach().flatten() for parameter in model.parameters()]))
        metrics.append(result["topd"])
    assert torch.allclose(outputs[0], outputs[1], atol=3e-7, rtol=3e-6)
    for key in ("high_loss_candidates", "accepted_candidates", "trajectory_loss_tokens", "extra_student_tokens", "extra_teacher_tokens"):
        assert metrics[0][key] == metrics[1][key]
    for key in ("opd_loss", "trajectory_loss", "total_loss", "gradient_norm"):
        assert metrics[0][key] == pytest.approx(metrics[1][key], abs=2e-6, rel=2e-6)


def test_default_and_environment_checkpoint_cadence_is_twenty(monkeypatch):
    monkeypatch.delenv("SAVE_INTERVAL", raising=False)
    assert launch_config()["training"]["save_interval"] == 20
    monkeypatch.setenv("SAVE_INTERVAL", "20")
    assert launch_config()["training"]["save_interval"] == 20


def test_complete_immutable_checkpoints_every_twenty_and_final(tmp_path):
    model, tokenizer = tiny_model(), tiny_tokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    model(torch.tensor([[4, 5]])).logits.sum().backward()
    optimizer.step()
    save_config(smoke_config(tmp_path), tmp_path/"resolved_config.yaml")
    for step in (20, 40, 60, 80):
        checkpoint = _save_checkpoint(model, tokenizer, optimizer, tmp_path, step, False, True, context())
        payload = torch.load(checkpoint/"optimizer.pt", weights_only=True)
        assert payload["step"] == step and payload["optimizer"]["state"]
        assert payload["rng_states"][0]["torch_rng_state"].numel() > 0
        assert "python_rng_state" in payload["rng_states"][0] and "numpy_rng_state" in payload["rng_states"][0]
        assert (checkpoint/"resolved_config.yaml").is_file()
        assert json.loads((checkpoint/"training_state.json").read_text())["optimizer_step"] == step
    _save_checkpoint(model, tokenizer, optimizer, tmp_path, 80, True, True, context())
    assert all((tmp_path/f"checkpoint-{step:06d}/config.json").is_file() for step in (20, 40, 60, 80))
    with pytest.raises(FileExistsError):
        _save_checkpoint(model, tokenizer, optimizer, tmp_path, 20, False, True, context())
    payload = torch.load(tmp_path/"final/optimizer.pt", weights_only=True)
    state = payload["rng_states"][0]
    _restore_rng_states(state, context().device)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    _restore_rng_states(state, context().device)
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def test_cpu_operator_profiler_produces_bounded_trace(tmp_path):
    profiler = UpdateProfiler({"profile_steps": [1]}, 1, 0, tmp_path, torch.device("cpu"))
    torch.randn(3).square()
    result = profiler.finish()
    assert Path(result["operator_trace"]).is_file()
    assert result["communication_gpu_time_sec"] == 0


def test_auto_batches_obey_vram_and_token_limits(monkeypatch):
    model = tiny_model()
    cfg = dict(generation_max_batch_size=64, generation_token_budget=2048)
    monkeypatch.setattr("topd.runtime.available_workspace", lambda *a, **kw: 10**12)
    large = automatic_batch_size(model, torch.device("cpu"), prefix_length=128, future_length=50,
        settings=cfg, phase="generation")
    assert large == 2048//178
    monkeypatch.setattr("topd.runtime.available_workspace", lambda *a, **kw: 0)
    assert automatic_batch_size(model, torch.device("cpu"), prefix_length=128, future_length=50,
        settings=cfg, phase="generation") == 1
    assert automatic_batch_size(model, torch.device("cpu"), prefix_length=128, future_length=50,
        settings=cfg, phase="generation", requested=3) == 3


def test_disabled_no_sync_still_participates_in_rank_consensus():
    from types import SimpleNamespace
    calls = []
    ctx = SimpleNamespace(device=torch.device("cpu"), any=lambda value: calls.append(value) or value)
    assert not use_accumulation_no_sync(tiny_model(), ctx, {"accumulation_no_sync": False})
    assert calls == [True]


def test_generation_oom_retries_all_requests_with_original_seeds(tmp_path, monkeypatch):
    from topd import generation
    cfg = smoke_config(tmp_path)["topd"]
    cfg["generation_batch_size"] = 4
    prefixes, pairs, seeds = [torch.tensor([4, 5])]*4, [(0, 0)]*4, [7, 8, 9, 10]
    original = generation.decode_cached
    calls = []

    def limited(*args, **kwargs):
        calls.append(kwargs["seeds"])
        if len(kwargs["seeds"]) > 2:
            raise torch.cuda.OutOfMemoryError("synthetic batch limit")
        return original(*args, **kwargs)

    monkeypatch.setattr(generation, "decode_cached", limited)
    result, metrics = generate_shared_futures(tiny_model(), prefixes, pairs, seeds,
        settings=cfg, pad_id=0, eos_ids=[2], distributed=context())
    assert calls == [seeds, seeds[:2], seeds[2:]]
    assert metrics["oom_retries"] == 1
    expected = generate_near_future(tiny_model(), prefixes, pad_id=0, eos_ids=[2], window=50,
        temperature=1., top_p=1., seeds=seeds, distributed=context())
    assert [tokens.tolist() for tokens in result] == [tokens.tolist() for tokens in expected]


def test_benchmark_report_includes_all_phases_and_excludes_warmup():
    from topd.benchmark import summarize
    row = dict(optimizer_update_wall_time_sec=2., benchmark_amortized_step_time_sec=3.,
        original_student_rollouts=64, original_opd_tokens=100, trajectory_loss_tokens=50,
        peak_gpu_memory_bytes=123, future_generation_time_sec=.5)
    result = summarize([row | {"optimizer_update_wall_time_sec": 99.}, row, row], warmup=1)
    assert result["mean_seconds_per_optimizer_update"] == 2.
    assert result["mean_amortized_seconds_per_step"] == 3.
    assert result["supervision_tokens_per_second"] == 50.
