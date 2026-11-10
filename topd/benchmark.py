"""Explicitly requested, bounded B200 benchmark using real MATH main rollouts.

Does not create a scientific training run or claim one-epoch coverage. Models,
data, optimizer, original rollout/score/grouping and TOPD objective match train.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import torch

from .configuration import launch_config


def summarize(rows, *, warmup=1):
    measured = rows[warmup:]
    if not measured:
        raise ValueError("No measured steps remain after warmup")
    update = [row["optimizer_update_wall_time_sec"] for row in measured]
    end_to_end = [row["benchmark_amortized_step_time_sec"] for row in measured]
    phases = ("future_generation_time_sec", "ot_time_sec", "embedding_extraction_time_sec",
              "trajectory_update_time_sec", "original_opd_forward_time_sec", "original_opd_backward_time_sec",
              "optimizer_time_sec", "gradient_clip_time_sec", "benchmark_main_rollout_time_per_step_sec",
              "benchmark_scoring_time_per_step_sec")
    return {"measured_steps": len(measured), "warmup_steps": warmup,
        "mean_seconds_per_optimizer_update": statistics.mean(update),
        "mean_amortized_seconds_per_step": statistics.mean(end_to_end),
        "median_amortized_seconds_per_step": statistics.median(end_to_end),
        "original_trajectories_per_second": sum(row["original_student_rollouts"] for row in measured)/sum(end_to_end),
        "supervision_tokens_per_second": sum(row["original_opd_tokens"]+row["trajectory_loss_tokens"] for row in measured)/sum(end_to_end),
        "peak_gpu_allocated_bytes": max(row["peak_gpu_memory_bytes"] for row in measured),
        "peak_gpu_reserved_bytes": max(row.get("peak_gpu_reserved_bytes", 0) for row in measured),
        "mean_phase_seconds": {phase: statistics.mean(row.get(phase, 0.) for row in measured) for phase in phases},
        "communication_gpu_kernel_seconds": [row.get("communication_gpu_time_sec") for row in measured],
        "scope": "bounded real-data benchmark; no checkpoint/evaluation cost, not a completed training epoch"}


def run(args):
    from b200_experiment.data import (read_records, filter_overlong_prompt_records, epoch_batch_indices,
        expand_prompt_batch, tokenize_prompts)
    from b200_experiment.distributed import initialize_distributed, contiguous_partition, unique_free_port
    from b200_experiment.fsdp import wrap_fsdp_model
    from b200_experiment.models import load_models, load_student_tokenizer, generation_stop_token_ids
    from b200_experiment.opd_core import build_student_topk_opd_reference
    from b200_experiment.scoring import score_original_rollout, cuda_sync
    from b200_experiment.trainer import seed_everything, _make_optimizer, _opd_train_step, _timed
    from b200_experiment.vllm_rollout import VLLMRolloutEngine
    from b200_experiment.config import save_config
    from .dependencies import check_flashinfer
    from .engine import TOPDEngine

    if not torch.cuda.is_available():
        raise RuntimeError("This performance benchmark requires 2 visible NVIDIA B200 GPUs")
    if torch.cuda.device_count() != 2 or any("B200" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Benchmark requires exactly 2 NVIDIA B200 GPUs; no workstation fallback")
    check_flashinfer()
    config = launch_config()
    if config["training"].get("resume_from_checkpoint"):
        raise ValueError("Benchmark always starts from the original student initialization")
    ctx = initialize_distributed("nccl")
    device = ctx.device
    if ctx.world_size != 2:
        raise ValueError("Launch with scripts/benchmark_b200.sh (torchrun, 2 ranks)")
    output = Path(args.output).expanduser().resolve()/args.mode
    error = None
    if ctx.is_main:
        if output.exists():
            error = f"Benchmark output already exists: {output}"
        else:
            output.mkdir(parents=True)
    error = ctx.broadcast_object(error)
    if error:
        raise FileExistsError(error)
    ctx.barrier()
    if args.mode == "reference":
        config["topd"].update(future_backend="reference", reference_generation_batch_size=4,
            generation_batch_size=4, loss_micro_batch_size=1, batched_kl=False,
            accumulation_no_sync=False, ot_workers=1)
    if args.profile:
        config["topd"]["profile_steps"] = [args.warmup+1]
    config["experiment"]["output_dir"] = str(output)
    if ctx.is_main:
        save_config(config, output/"resolved_config.yaml")
    seed_everything(config["experiment"]["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = load_student_tokenizer(config)
    records, _ = read_records(config["data"]["path"], config["data"].get("split"))
    records, _ = filter_overlong_prompt_records(records, tokenizer, config["data"])
    server = VLLMRolloutEngine(config, output, local_rank=ctx.local_rank,
        world_size=ctx.world_size, port=unique_free_port(ctx))
    rows = []
    try:
        server.start()
        student, teacher, tokenizer, _ = load_models(config, device)
        student = wrap_fsdp_model(student, config, ctx, role="student")
        teacher = wrap_fsdp_model(teacher, config, ctx, role="teacher")
        optimizer, _ = _make_optimizer(student.parameters(), config["training"])
        engine = TOPDEngine(config, student, teacher, tokenizer, ctx, output)
        target_steps = args.warmup+args.steps
        rollout_index = completed = 0
        while completed < target_steps:
            prompt_indices = epoch_batch_indices(len(records), config["rollout"]["batch_size"],
                rollout_index, config["experiment"]["seed"])
            begin, end = contiguous_partition(len(prompt_indices), ctx.rank, ctx.world_size)
            local_indices = prompt_indices[begin:end]
            if not local_indices:
                raise ValueError("Benchmark needs at least one real prompt per GPU")
            encoded, _ = tokenize_prompts([records[index] for index in local_indices], tokenizer, config["data"], device)
            encoded, dataset_indices, response_indices = expand_prompt_batch(encoded, local_indices, config["rollout"]["num_responses"])
            rollout, generation_time = _timed(device, server.generate, student,
                encoded["input_ids"], encoded["attention_mask"],
                max_new_tokens=config["rollout"]["max_new_tokens"], temperature=config["rollout"]["temperature"],
                top_p=config["rollout"]["top_p"], eos_token_ids=generation_stop_token_ids(tokenizer, config),
                pad_token_id=tokenizer.pad_token_id, seed=config["rollout"]["seed"],
                sample_seed_offset=(rollout_index*config["rollout"]["batch_size"]+begin)*config["rollout"]["num_responses"])
            cuda_sync(device)
            began = time.perf_counter()
            score_kwargs = dict(micro_batch_size=config["selector"]["score_micro_batch_size"],
                trim_padding=config["selector"]["trim_padding"], length_bucketed=config["selector"]["length_bucketed_scoring"],
                temperature=config["rollout"]["temperature"], retain_response_logits=False)
            student_scores = score_original_rollout(student, rollout, top_k=config["selector"]["top_k"], **score_kwargs)
            teacher_scores = score_original_rollout(teacher, rollout, candidate_ids=student_scores.top_k_ids, **score_kwargs)
            reference = build_student_topk_opd_reference(student_scores.top_k_ids, student_scores.top_k_log_probs,
                teacher_scores.candidate_log_probs, rollout.valid_mask, top_k=config["selector"]["top_k"])
            cuda_sync(device)
            scoring_time = ctx.max_float(time.perf_counter()-began)
            generation_time = ctx.max_float(generation_time)
            group_count = (len(prompt_indices)*config["rollout"]["num_responses"]+config["training"]["ppo_mini_batch_size"]-1)//config["training"]["ppo_mini_batch_size"]
            engine.dataset_indices, engine.response_indices = dataset_indices, response_indices
            result = _opd_train_step(student, optimizer, rollout, rollout.valid_mask.float(), reference,
                config, device, ctx, trajectory_group_ids=response_indices, trajectory_engine=engine,
                optimizer_step_start=completed, max_optimizer_steps=target_steps-completed)
            for metric in result["minibatches"]:
                row = copy.deepcopy(metric["topd"])
                row["benchmark_main_rollout_time_per_step_sec"] = generation_time/group_count
                row["benchmark_scoring_time_per_step_sec"] = scoring_time/group_count
                row["benchmark_amortized_step_time_sec"] = row["optimizer_update_wall_time_sec"]+(generation_time+scoring_time)/group_count
                rows.append(row)
            completed += result["optimizer_steps"]
            rollout_index += 1
            del student_scores, teacher_scores, reference, rollout
        report = {"mode": args.mode, "gpu": [torch.cuda.get_device_name(i) for i in range(2)],
                  "summary": summarize(rows, warmup=args.warmup), "steps": rows,
                  "reference_note": "reference generation/backward under current infrastructure; the reported old 900 s/step is user-observed, not remeasured here"}
        if ctx.is_main:
            (output/"benchmark.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(report["summary"], indent=2), flush=True)
        return report
    finally:
        server.close()
        ctx.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("optimized", "reference"), default="optimized")
    parser.add_argument("--steps", type=int, default=3, help="Measured optimizer updates; max 8")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile", action="store_true", help="Trace the first measured update; includes profiler overhead")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 8 or not 0 <= args.warmup <= 4:
        parser.error("Bounded benchmark requires steps in [1,8], warmup in [0,4]")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
