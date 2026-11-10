"""Real two-process FULL_SHARD lifecycle/gradient checks on CPU/Gloo."""
import copy
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from b200_experiment.distributed import DistributedContext
from b200_experiment.opd_core import build_student_topk_opd_reference
from b200_experiment.scoring import score_original_rollout
from b200_experiment.trainer import _opd_train_step
from b200_experiment.vllm_rollout import rollout_batch_from_token_ids
from topd.engine import TOPDEngine, generate_near_future
from topd.generation import generate_shared_futures
from topd.smoke import tiny_model, tiny_tokenizer, smoke_config


def fsdp_worker(rank, rendezvous, root, bf16=False):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    ctx = DistributedContext(rank, rank, 2, torch.device("cpu"))
    dtype = torch.bfloat16 if bf16 else torch.float32
    wrap = lambda model: FSDP(model.to(dtype), device_id=ctx.device, use_orig_params=True,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}))
    model, teacher = wrap(tiny_model()), wrap(tiny_model(8).requires_grad_(False))
    cfg = smoke_config(Path(root))
    cfg["topd"].update(generation_batch_size="auto", loss_micro_batch_size=2,
        accumulation_no_sync=True, candidate_budget=None, candidate_fraction=.5)
    # Uneven decode calls are safe only if all nested FSDP forwards are bypassed.
    prefixes = [torch.tensor([4, 5]), torch.tensor([4, 5, 6])] if rank == 0 else []
    pairs, seeds = ([(0, 0), (0, 1)], [12, 24]) if rank == 0 else ([], [])
    generated, _ = generate_shared_futures(model, prefixes, pairs, seeds,
        settings=cfg["topd"], pad_id=0, eos_ids=[2], distributed=ctx)
    assert all("forward" not in wrapper.__dict__ for wrapper in model.modules() if isinstance(wrapper, FSDP))
    if rank == 0:
        reference = generate_near_future(tiny_model().to(dtype), prefixes, pad_id=0, eos_ids=[2], window=50,
            temperature=1., top_p=1., seeds=seeds, distributed=DistributedContext(0, 0, 1, ctx.device))
        assert [tokens.tolist() for tokens in generated] == [tokens.tolist() for tokens in reference]
    dist.barrier()
    rollout = rollout_batch_from_token_ids(torch.tensor([[4, 5]]), torch.ones(1, 2), [[6, 7, 9, 10]], 0)
    s = score_original_rollout(model, rollout, top_k=4)
    t = score_original_rollout(teacher, rollout, candidate_ids=s.top_k_ids)
    reference = build_student_topk_opd_reference(s.top_k_ids, s.top_k_log_probs, t.candidate_log_probs,
        rollout.valid_mask, top_k=4)
    engine = TOPDEngine(cfg, model, teacher, tiny_tokenizer(), ctx)
    class CaptureOptimizer(torch.optim.AdamW):
        def step(self, *args, **kwargs):
            with FSDP.summon_full_params(model, writeback=False, with_grads=True):
                self.gradient = torch.cat([parameter.grad.detach().float().flatten() for parameter in model.parameters()]).clone()
            return super().step(*args, **kwargs)
    optimizer_options = dict(lr=5e-6, betas=(.9, .95), weight_decay=0.)
    optimizer = CaptureOptimizer(model.parameters(), **optimizer_options)
    result = _opd_train_step(model, optimizer,
        rollout, rollout.valid_mask.float(), reference, cfg, ctx.device, ctx,
        objective_valid_mask=rollout.valid_mask & (rank == 0),
        trajectory_active_mask=torch.tensor([rank == 0]), trajectory_group_ids=[0], trajectory_engine=engine)
    assert result["topd"]["original_student_rollouts"] == 1
    assert result["topd"]["high_loss_candidates"] == 2
    assert result["topd"]["accumulation_no_sync"]
    # Gather the updated model only after gradients have been synchronized.
    with FSDP.summon_full_params(model, writeback=False):
        parameters = torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).clone()
    all_parameters = [torch.empty_like(parameters) for _ in range(2)]
    dist.all_gather(all_parameters, parameters)
    assert torch.equal(all_parameters[0], all_parameters[1])
    if rank == 0:
        baseline_model, baseline_teacher = tiny_model().to(dtype), tiny_model(8).to(dtype).requires_grad_(False)
        baseline_cfg = copy.deepcopy(cfg)
        baseline_cfg["topd"].update(future_backend="reference", reference_generation_batch_size=4,
            loss_micro_batch_size=1, batched_kl=False, accumulation_no_sync=False)
        baseline_ctx = DistributedContext(0, 0, 1, ctx.device)
        baseline_engine = TOPDEngine(baseline_cfg, baseline_model, baseline_teacher, tiny_tokenizer(), baseline_ctx)
        class SerialOptimizer(torch.optim.AdamW):
            def step(self, *args, **kwargs):
                self.gradient = torch.cat([parameter.grad.detach().float().flatten() for parameter in baseline_model.parameters()]).clone()
                return super().step(*args, **kwargs)
        serial_optimizer = SerialOptimizer(baseline_model.parameters(), **optimizer_options)
        serial = _opd_train_step(baseline_model, serial_optimizer,
            rollout, rollout.valid_mask.float(), reference, baseline_cfg, ctx.device, baseline_ctx,
            trajectory_engine=baseline_engine)
        expected = torch.cat([parameter.detach().flatten() for parameter in baseline_model.parameters()])
        gradient_error = (optimizer.gradient-serial_optimizer.gradient).norm()/serial_optimizer.gradient.norm()
        print(f'BF16={bf16} relative_gradient_error={float(gradient_error)} weight_error={float((parameters-expected).abs().max())} loss_error={abs(serial["topd"]["total_loss"]-result["topd"]["total_loss"])}', flush=True)
        assert gradient_error < (1e-2 if bf16 else 2e-6)
        assert torch.allclose(parameters, expected, atol=8e-6 if bf16 else 3e-7,
                              rtol=1e-3 if bf16 else 3e-6), (parameters-expected).abs().max()
        assert abs(serial["topd"]["total_loss"]-result["topd"]["total_loss"]) < (3e-3 if bf16 else 2e-6)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("bf16", [False, True])
def test_two_rank_fsdp_shared_inference_and_accumulated_update(tmp_path, bf16):
    mp.spawn(fsdp_worker, args=(str(tmp_path/"rendezvous"), str(tmp_path), bf16), nprocs=2, join=True)
