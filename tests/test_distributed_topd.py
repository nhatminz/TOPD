import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from b200_experiment.distributed import DistributedContext
from b200_experiment.vllm_rollout import rollout_batch_from_token_ids
from b200_experiment.opd_core import build_student_topk_opd_reference
from b200_experiment.scoring import score_original_rollout
from b200_experiment.trainer import _opd_train_step
from topd.algorithm import SparseTargets
from topd.engine import FutureWindow, TOPDEngine
from topd.smoke import smoke_config, tiny_model, tiny_tokenizer


def distributed_worker(rank, rendezvous, result_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    ctx = DistributedContext(rank, rank, 2, torch.device("cpu"))
    model = DDP(tiny_model(), broadcast_buffers=False)
    teacher, tokenizer = tiny_model(8).requires_grad_(False), tiny_tokenizer()
    config = smoke_config(Path(result_path).parent)
    config["topd"].update(lambda_topd=.3, loss_micro_batch_size=1)
    windows = [FutureWindow(torch.tensor([4, 5]), torch.tensor([6, 7]),
        SparseTargets(torch.tensor([8, 9]), torch.tensor([[.4, .6], [.7, .3]])), 0, 0, .8),
        FutureWindow(torch.tensor([4, 5, 6]), torch.tensor([10]),
        SparseTargets(torch.tensor([9]), torch.ones(1, 1)), 0, 1, .9)]
    rollout = rollout_batch_from_token_ids(torch.tensor([[4, 5]]), torch.ones(1, 2), [[6, 7]], 0)
    engine = TOPDEngine(config, model, teacher, tokenizer, ctx)
    engine.windows = windows if rank == 0 else []
    engine.denominator, engine.metrics = 2, {}
    result = engine.backward(rollout, 1)
    gradient = torch.cat([p.grad.flatten() for p in model.parameters()])
    gathered = [torch.zeros_like(gradient) for _ in range(2)]
    dist.all_gather(gathered, gradient)
    if rank == 0:
        reference_model = tiny_model()
        reference = TOPDEngine(config, reference_model, teacher, tokenizer,
            DistributedContext(0, 0, 1, torch.device("cpu")))
        reference.windows, reference.denominator, reference.metrics = windows, 2, {}
        single = reference.backward(rollout, 1)
        expected = torch.cat([p.grad.flatten() for p in reference_model.parameters()])
        error = max(float((g-expected).abs().max()) for g in gathered)
        assert torch.allclose(gathered[0], expected, rtol=2e-5, atol=2e-6)
        assert torch.equal(gathered[0], gathered[1])
        assert abs(result["trajectory_loss"]-single["trajectory_loss"]) < 1e-6
        Path(result_path).write_text(json.dumps({"status": "passed", "world_size": 2,
            "unequal_windows_by_rank": [2, 0], "maximum_gradient_error": error}))
    dist.barrier()
    # Also exercise a one-prompt dataset tail: rank 1 has only a masked filler.
    # Both OPD backward and near-future generation must stay collective-safe.
    student = DDP(tiny_model(), broadcast_buffers=False)
    s = score_original_rollout(student, rollout, top_k=4)
    t = score_original_rollout(teacher, rollout, candidate_ids=s.top_k_ids)
    reference = build_student_topk_opd_reference(s.top_k_ids, s.top_k_log_probs,
        t.candidate_log_probs, rollout.valid_mask, top_k=4)
    engine = TOPDEngine(config, student, teacher, tokenizer, ctx)
    result = _opd_train_step(student, torch.optim.AdamW(student.parameters(), lr=.001),
        rollout, rollout.valid_mask.float(), reference, config, ctx.device, ctx,
        objective_valid_mask=rollout.valid_mask & (rank == 0),
        trajectory_active_mask=torch.tensor([rank == 0]), trajectory_group_ids=[0],
        trajectory_engine=engine)
    assert result["optimizer_steps"] == 1
    assert result["topd"]["original_student_rollouts"] == 1
    assert result["topd"]["accepted_candidates"] == 1
    params = torch.cat([p.detach().flatten() for p in student.parameters()])
    gathered_params = [torch.zeros_like(params) for _ in range(2)]
    dist.all_gather(gathered_params, params)
    assert torch.equal(gathered_params[0], gathered_params[1])
    if rank == 0:
        payload = json.loads(Path(result_path).read_text())
        payload["one_real_rank_tail_passed"] = True
        Path(result_path).write_text(json.dumps(payload))
    dist.barrier()
    dist.destroy_process_group()


def test_two_process_unequal_window_gradient_normalization(tmp_path):
    mp.spawn(distributed_worker, args=(str(tmp_path/"rendezvous"), str(tmp_path/"result.json")),
             nprocs=2, join=True)
    result = json.loads((tmp_path/"result.json").read_text())
    assert result["status"] == "passed"
    assert result["maximum_gradient_error"] < 2e-6
    assert result["one_real_rank_tail_passed"]
