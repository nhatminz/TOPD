from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from b200_experiment.distributed import unwrap_model
from b200_experiment.fsdp import is_fsdp_model
from b200_experiment.models import generation_stop_token_ids
from b200_experiment.scoring import (
    _sample_top_p, cuda_sync, position_ids_from_mask, rollout_forward_inputs,
    select_rollout_logits, supports_response_only_logits,
)
from b200_experiment.vllm_rollout import rollout_batch_from_token_ids

from .algorithm import (
    SparseTargets, cosine_ground_cost, divergence_mask, extract_prefix,
    forward_kl, select_candidates, solve_transport, trajectory_denominator,
    transport_targets,
)


def pad_prefixes(prefixes, pad_id, device):
    width = max(p.numel() for p in prefixes)
    ids = torch.full((len(prefixes), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, prefix in enumerate(prefixes):
        ids[row, -prefix.numel():] = prefix.to(device)
        mask[row, -prefix.numel():] = 1
    return ids, mask


@torch.no_grad()
def generate_near_future(model, prefixes, *, pad_id, eos_ids, window, temperature,
                         top_p, seeds, distributed):
    """Batched HF generation through FSDP.forward with synchronized EOS exits.

    No full-model materialization and no persistent continuation cache across
    updates. A per-request RNG avoids altering main-rollout random streams.
    """
    device = distributed.device
    ids, attention = pad_prefixes(prefixes, pad_id, device)
    prompt_width = ids.shape[1]
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    model.eval()
    kwargs = dict(input_ids=ids, attention_mask=attention,
                  position_ids=position_ids_from_mask(attention),
                  use_cache=True, return_dict=True)
    if supports_response_only_logits(model):
        kwargs["logits_to_keep"] = 1
    output = model(**kwargs)
    cache, next_logits = output.past_key_values, output.logits[:, -1]
    active = torch.ones(len(prefixes), dtype=torch.bool, device=device)
    results = [[] for _ in prefixes]
    eos = torch.tensor(eos_ids, device=device, dtype=torch.long)
    for index in range(window):
        # Padding is a batching sentinel, never a future target. If padding
        # shares the EOS ID, EOS remains sampleable and is included once.
        next_logits = next_logits.float().clone()
        if pad_id not in eos_ids:
            next_logits[:, pad_id] = -torch.inf
        if temperature == 0:
            token = next_logits.argmax(-1)
        else:
            p = torch.softmax(next_logits.float()/temperature, -1)
            token = torch.stack([_sample_top_p(p[r:r+1], top_p, generators[r]).squeeze(0)
                                 for r in range(len(prefixes))])
        token = torch.where(active, token, torch.full_like(token, pad_id))
        for row in range(len(prefixes)):
            if active[row]:
                results[row].append(int(token[row]))
        attention = torch.cat((attention, active.long()[:, None]), 1)
        active &= ~token[:, None].eq(eos[None, :]).any(-1)
        if index + 1 == window or not distributed.any(bool(active.any())):
            break
        kwargs = dict(input_ids=token[:, None], attention_mask=attention,
                      position_ids=(attention.sum(-1, keepdim=True)-1).clamp_min(0),
                      cache_position=torch.tensor([prompt_width+index], device=device),
                      past_key_values=cache, use_cache=True, return_dict=True)
        if supports_response_only_logits(model):
            kwargs["logits_to_keep"] = 1
        output = model(**kwargs)
        cache, next_logits = output.past_key_values, output.logits[:, -1]
    del output, cache
    return [torch.tensor(tokens, dtype=torch.long) for tokens in results]


@dataclass
class FutureWindow:
    prefix: torch.Tensor
    student_tokens: torch.Tensor
    target: SparseTargets
    row: int
    token_index: int
    distance: float


def quantiles(values):
    if not values:
        return {"count": 0}
    x = torch.tensor(values, dtype=torch.float64)
    return {"count": len(values), "mean": float(x.mean()),
            **{f"q{round(q*100):02d}": float(torch.quantile(x, q))
               for q in (0., .25, .5, .75, .9, .95, 1.)}}


class TOPDEngine:
    def __init__(self, config, student, teacher, tokenizer, distributed, output_dir=None):
        self.config, self.cfg = config, config["topd"]
        self.student, self.teacher, self.tokenizer = student, teacher, tokenizer
        self.distributed = distributed
        self.output_dir = Path(output_dir) if output_dir else None
        self.windows = []
        self.metrics = {}

    def _common_embeddings(self, token_sequences):
        """Teacher input table for BOTH trajectories, with root-only FSDP gather.

        Only embedding/root parameters are gathered, not all decoder blocks.
        The table is never cached, trained, decoded or re-tokenized.
        """
        lengths = [len(ids) for ids in token_sequences]
        ids = torch.cat(token_sequences).to(self.distributed.device)
        context = (FSDP.summon_full_params(self.teacher, recurse=False, writeback=False)
                   if is_fsdp_model(self.teacher) else nullcontext())
        with torch.no_grad(), context:
            embedding = unwrap_model(self.teacher).get_input_embeddings()
            vectors = embedding(ids.to(embedding.weight.device)).detach().float().cpu()
        return list(vectors.split(lengths))

    def prepare(self, rollout, local_indices, reference, valid_mask, step):
        """Fresh teacher AND pre-update student continuations for this PPO group."""
        d, cfg = self.distributed, self.cfg
        self.windows = []
        start = time.perf_counter()
        cuda_sync(d.device)
        local_loss = -reference.advantages.detach().sum(-1)
        group_mask = torch.zeros_like(valid_mask)
        group_mask[local_indices] = valid_mask[local_indices]
        pairs, thresholds = select_candidates(local_loss, group_mask,
            fraction=cfg["candidate_fraction"], budget=cfg.get("candidate_budget"),
            min_loss=cfg.get("candidate_min_loss"))
        batch = int(cfg["generation_batch_size"])
        rounds = math.ceil(d.max_int(len(pairs))/batch)
        window_pairs, diagnostics = [], []
        generation_seconds = ot_seconds = 0.
        physical_requests = physical_student_tokens = physical_teacher_tokens = 0
        student_lengths, teacher_lengths = [], []
        for chunk in range(rounds):
            real = pairs[chunk*batch:(chunk+1)*batch]
            # Every rank executes matching FSDP collectives. Filler requests
            # never enter selection, metrics, targets or the objective.
            count = max(1, len(real))
            prefixes = [extract_prefix(rollout, row, index) for row, index in real]
            if not prefixes:
                prefixes = [rollout.input_ids[0, :rollout.prompt_width][
                    rollout.attention_mask[0, :rollout.prompt_width].bool()].clone()]
            limit = getattr(unwrap_model(self.student).config, "max_position_embeddings", None)
            if limit and any(p.numel()+cfg["future_window"] > limit for p in prefixes):
                raise ValueError("TOPD prefix + 50-token future exceeds student context")
            # Stable request seeds, disjoint teacher/student streams. Main
            # rollout seed stays exactly as in the comparison experiment.
            seeds = [int(cfg["future_seed"]) + step*10000019 + d.rank*100003
                     + chunk*batch + j for j in range(count)]
            cuda_sync(d.device)
            generated_at = time.perf_counter()
            kwargs = dict(pad_id=self.tokenizer.pad_token_id,
                eos_ids=generation_stop_token_ids(self.tokenizer, self.config),
                window=cfg["future_window"], temperature=cfg["future_temperature"],
                top_p=cfg["future_top_p"], distributed=d)
            teacher = generate_near_future(self.teacher, prefixes,
                seeds=[seed+1000000000007 for seed in seeds], **kwargs)
            student = generate_near_future(self.student, prefixes, seeds=seeds, **kwargs)
            physical_requests += len(prefixes)
            physical_student_tokens += sum(map(len, student))
            physical_teacher_tokens += sum(map(len, teacher))
            cuda_sync(d.device)
            generation_seconds += time.perf_counter()-generated_at
            ot_at = time.perf_counter()
            vectors = self._common_embeddings(student+teacher)
            for j, (row, index) in enumerate(real):
                if not len(student[j]) or not len(teacher[j]):
                    raise RuntimeError("Empty near-future continuation")
                cost = cosine_ground_cost(vectors[j], vectors[count+j])
                plan, distance = solve_transport(cost, solver=cfg["ot_solver"],
                    epsilon=cfg["sinkhorn_epsilon"],
                    max_iterations=cfg["ot_max_iterations"], tolerance=cfg["ot_tolerance"])
                target = transport_targets(plan, teacher[j])
                window_pairs.append(FutureWindow(prefixes[j].detach().cpu(), student[j],
                    target, row, index, distance))
                student_lengths.append(len(student[j]))
                teacher_lengths.append(len(teacher[j]))
                diagnostics.append({"row": row, "response_token_index": index,
                    "local_loss": float(local_loss[row, index]), "ot_distance": distance,
                    "prefix_sha256": hashlib.sha256(prefixes[j].cpu().numpy().tobytes()).hexdigest(),
                    "student_future": student[j].tolist(), "teacher_future": teacher[j].tolist(),
                    "ot_row_error": float((plan.sum(1)-1/len(student[j])).abs().max()),
                    "ot_column_error": float((plan.sum(0)-1/len(teacher[j])).abs().max())})
            ot_seconds += time.perf_counter()-ot_at
        gathered_distances = d.all_gather_objects([w.distance for w in window_pairs])
        distances = [v for rank in gathered_distances for v in rank]
        accepted, cutoff = divergence_mask(distances, mode=cfg["divergence_mode"],
            quantile=cfg["ot_quantile"], threshold=cfg["ot_threshold"],
            min_distance=cfg["min_ot_distance"])
        offset = sum(map(len, gathered_distances[:d.rank]))
        local_accepted = accepted[offset:offset+len(window_pairs)].tolist()
        self.windows = [w for w, keep in zip(window_pairs, local_accepted) if keep]
        for diagnostic, keep in zip(diagnostics, local_accepted):
            diagnostic["accepted"] = keep
        total_windows = d.sum_int(len(self.windows))
        total_tokens = d.sum_int(sum(len(w.student_tokens) for w in self.windows))
        original_tokens = d.sum_int(int(group_mask.sum()))
        self.denominator = trajectory_denominator(cfg["trajectory_normalization"],
            total_windows, total_tokens, original_tokens)
        all_thresholds = [v for rank in d.all_gather_objects(thresholds) for v in rank]
        all_losses = [v for rank in d.all_gather_objects(
            [x["local_loss"] for x in diagnostics]) for v in rank]
        self.metrics = dict(optimizer_step=step, algorithm="TOPD",
            original_opd_tokens=original_tokens,
            original_student_rollouts=d.sum_int(len(local_indices)),
            high_loss_candidates=len(distances), accepted_candidates=total_windows,
            rejected_candidates=len(distances)-total_windows,
            candidate_fraction=len(distances)/max(original_tokens, 1),
            accepted_fraction=total_windows/max(len(distances), 1),
            selected_token_fraction=total_windows/max(original_tokens, 1),
            candidate_loss=quantiles(all_losses), candidate_thresholds=quantiles(all_thresholds),
            ot_distance=quantiles(distances), ot_cutoff=cutoff,
            extra_teacher_continuations=d.sum_int(len(teacher_lengths)),
            extra_student_continuations=d.sum_int(len(student_lengths)),
            extra_teacher_tokens=d.sum_int(sum(teacher_lengths)),
            extra_student_tokens=d.sum_int(sum(student_lengths)),
            extra_teacher_continuations_physical=d.sum_int(physical_requests),
            extra_student_continuations_physical=d.sum_int(physical_requests),
            extra_generated_tokens_physical=d.sum_int(physical_student_tokens+physical_teacher_tokens),
            student_window_lengths=quantiles([v for rank in d.all_gather_objects(student_lengths) for v in rank]),
            teacher_window_lengths=quantiles([v for rank in d.all_gather_objects(teacher_lengths) for v in rank]),
            trajectory_loss_tokens=total_tokens,
            trajectory_normalization=cfg["trajectory_normalization"],
            trajectory_denominator=self.denominator, lambda_topd=cfg["lambda_topd"],
            effective_per_token_gradient_factor=cfg["lambda_topd"]/self.denominator,
            future_generation_time_sec=d.max_float(generation_seconds),
            ot_time_sec=d.max_float(ot_seconds),
            prepare_time_sec=d.max_float(time.perf_counter()-start),
            # Costs of dummy collective requests are real, though excluded
            # from scientific candidate/rollout counts.
            collective_filler_requests=d.sum_int(sum(max(1,min(batch,max(0,len(pairs)-i*batch)))-min(batch,max(0,len(pairs)-i*batch)) for i in range(rounds))))
        if hasattr(self, "dataset_indices"):
            self.metrics["consumed_example_responses"] = [pair
                for rank in d.all_gather_objects([
                    [self.dataset_indices[row], self.response_indices[row]]
                    for row in local_indices]) for pair in rank]
        self.diagnostics = diagnostics
        self.student.train()
        unwrap_model(self.student).config.use_cache = False
        return self.metrics

    def backward(self, rollout, step):
        """Accumulate forward-KL gradients before the shared clip/optimizer step."""
        d, cfg = self.distributed, self.cfg
        batch = int(cfg["loss_micro_batch_size"])
        rounds = math.ceil(d.max_int(len(self.windows))/batch)
        numerator, began = 0., time.perf_counter()
        for chunk in range(rounds):
            windows = self.windows[chunk*batch:(chunk+1)*batch]
            active_count = len(windows)
            if not windows:
                prefix = rollout.input_ids[0, :rollout.prompt_width][
                    rollout.attention_mask[0, :rollout.prompt_width].bool()].detach().cpu()
                pad = self.tokenizer.pad_token_id
                windows = [FutureWindow(prefix, torch.tensor([pad]),
                    SparseTargets(torch.tensor([pad]), torch.ones(1, 1)), -1, -1, 0.)]
            prefix_ids, mask = pad_prefixes([w.prefix for w in windows], self.tokenizer.pad_token_id, d.device)
            future = rollout_batch_from_token_ids(prefix_ids, mask,
                [w.student_tokens.tolist() for w in windows], self.tokenizer.pad_token_id)
            width = future.response_ids.shape[1]
            kwargs, selection = rollout_forward_inputs(self.student, future,
                torch.arange(len(windows), device=d.device), width)
            output = self.student(**kwargs)
            logits = select_rollout_logits(output.logits, selection)
            # FP32 reduction is bounded to one short window at a time. Prompt
            # logits never enter the target; targets/plans are detached.
            loss_sum = logits.sum()*0 if not active_count else sum(
                forward_kl(logits[j, :len(w.student_tokens)], w.target,
                    temperature=cfg["trajectory_temperature"]).sum()
                for j, w in enumerate(windows))
            if not loss_sum.requires_grad:
                raise RuntimeError("Student trajectory KL lost its gradient graph")
            scaled = loss_sum * (cfg["lambda_topd"]*d.world_size/self.denominator)
            scaled.backward()
            numerator += float(loss_sum.detach())
            del output, logits, loss_sum, scaled, future
        cuda_sync(d.device)
        self.metrics.update(trajectory_loss=d.sum_float(numerator)/self.denominator,
            trajectory_update_time_sec=d.max_float(time.perf_counter()-began))
        self.metrics["weighted_trajectory_loss"] = self.metrics["trajectory_loss"]*cfg["lambda_topd"]
        return self.metrics

    def record(self, step, opd_loss, gradient_norm):
        d = self.distributed
        self.metrics.update(optimizer_step=step, opd_loss=opd_loss,
            total_loss=opd_loss+self.metrics["weighted_trajectory_loss"],
            gradient_norm=float(gradient_norm),
            peak_gpu_memory_bytes=d.max_int(torch.cuda.max_memory_allocated(d.device))
                if d.device.type == "cuda" else 0)
        self.metrics["student_update_time_sec"] = (self.metrics.get("opd_update_time_sec", 0.)
            + self.metrics["trajectory_update_time_sec"])
        self.metrics["supervision_tokens_per_second"] = (
            self.metrics["original_opd_tokens"]+self.metrics["trajectory_loss_tokens"]
        )/max(self.metrics.get("optimizer_update_wall_time_sec", 0.), 1e-12)
        self.metrics["additional_generated_tokens_per_second"] = (
            self.metrics["extra_generated_tokens_physical"]
        )/max(self.metrics["future_generation_time_sec"], 1e-12)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if d.is_main:
                # Replace rows after resuming an older checkpoint; no duplicate
                # metric entries survive replay of uncommitted optimizer steps.
                from b200_experiment.trainer import _upsert_jsonl_row
                _upsert_jsonl_row(self.output_dir/"topd_metrics.jsonl",
                    {"step": step, **self.metrics}, ("step",))
            if self.cfg.get("log_candidate_details", False):
                path = self.output_dir/"candidate_details"/f"step-{step:06d}-rank-{d.rank:05d}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self.diagnostics, allow_nan=False)+"\n")
        self.windows = []  # future supervision is never reused after an update
        return self.metrics
