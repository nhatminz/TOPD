"""Explicit reproduction choices for Section 3 of arXiv:2606.00305v2.

All transport matrices use STUDENT rows and TEACHER columns. No operation in
this module changes the common OPD objective.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def select_candidates(losses, valid_mask, *, fraction=0.0225, budget=None,
                      min_loss=None):
    """Stable, per-response top-loss budget; ceil(fraction * valid_count).

    Returns (row, response index) pairs. Invalid positions never affect ranks
    or thresholds. A zero fraction/budget intentionally disables candidates.
    """
    if losses.ndim != 2 or losses.shape != valid_mask.shape:
        raise ValueError("Candidate loss/mask must share [batch,time] shape")
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("candidate_fraction must be in [0,1]")
    if budget is not None and (int(budget) != budget or budget < 0):
        raise ValueError("candidate_budget must be a nonnegative integer or null")
    result, thresholds = [], []
    for row in range(losses.shape[0]):
        indices = valid_mask[row].bool().nonzero().flatten()
        values = losses[row, indices].detach().float()
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite local OPD candidate loss")
        count = math.ceil(fraction * indices.numel()) if budget is None else int(budget)
        if min_loss is not None:
            eligible = values >= float(min_loss)
            indices, values = indices[eligible], values[eligible]
        order = torch.argsort(values, descending=True, stable=True)[:count]
        if order.numel():
            thresholds.append(float(values[order[-1]]))
            result.extend((row, int(indices[index])) for index in order.tolist())
    return result, thresholds


def extract_prefix(rollout, row: int, token_index: int) -> torch.Tensor:
    """Remove left padding, include only response tokens strictly before t."""
    if rollout.actor_positions is not None:
        raise ValueError("TOPD math baseline expects single-turn trajectories")
    if not 0 <= token_index < rollout.response_ids.shape[1]:
        raise IndexError("Candidate response index out of bounds")
    if not bool(rollout.valid_mask[row, token_index]):
        raise ValueError("Candidate must be a valid response token")
    if not bool(rollout.valid_mask[row, :token_index].all()):
        raise ValueError("Response mask must be contiguous before candidate")
    end = rollout.prompt_width + token_index
    ids = rollout.input_ids[row, :end]
    mask = rollout.attention_mask[row, :end].bool()
    prefix = ids[mask].detach().clone()
    if prefix.numel() == 0:
        raise ValueError("A future window needs a nonempty causal prefix")
    return prefix


def cosine_ground_cost(student_vectors, teacher_vectors):
    if (student_vectors.ndim != 2 or teacher_vectors.ndim != 2
            or student_vectors.shape[1] != teacher_vectors.shape[1]
            or min(student_vectors.shape[0], teacher_vectors.shape[0]) == 0):
        raise ValueError("Expected nonempty shared-space [position,embedding] tensors")
    s, t = student_vectors.detach().double(), teacher_vectors.detach().double()
    if not torch.isfinite(s).all() or not torch.isfinite(t).all():
        raise FloatingPointError("Nonfinite common-space embeddings")
    if (s.norm(dim=-1) <= 1e-12).any() or (t.norm(dim=-1) <= 1e-12).any():
        raise ValueError("Cosine ground cost is undefined for zero embeddings")
    return (1 - F.normalize(s, dim=-1) @ F.normalize(t, dim=-1).T).clamp(0, 2)


def solve_transport(cost, *, solver="exact", epsilon=0.05,
                    max_iterations=10000, tolerance=1e-7):
    """Uniform balanced OT. Exact assignment if square, HiGHS LP otherwise.

    Sinkhorn is an explicit entropic approximation and must meet the same
    marginal checks. Solver failure always raises; never fall back to OPD.
    """
    c = cost.detach().cpu().double()
    if c.ndim != 2 or min(c.shape) < 1:
        raise ValueError("OT requires a nonempty 2-D cost matrix")
    if not torch.isfinite(c).all() or (c < 0).any():
        raise FloatingPointError("OT cost must be finite and nonnegative")
    n, m = c.shape
    a, b = torch.full((n,), 1/n, dtype=torch.float64), torch.full((m,), 1/m, dtype=torch.float64)
    if solver == "exact":
        from scipy.optimize import linear_sum_assignment, linprog
        from scipy.sparse import coo_matrix
        if n == m:
            rows, cols = linear_sum_assignment(c.numpy())
            plan = torch.zeros_like(c)
            plan[rows, cols] = 1/n
        else:
            cells = np.arange(n*m)
            constraint = coo_matrix((np.ones(2*n*m),
                (np.concatenate((np.repeat(np.arange(n), m), n + np.tile(np.arange(m), n))),
                 np.concatenate((cells, cells)))), shape=(n+m, n*m)).tocsr()
            answer = linprog(c.numpy().reshape(-1), A_eq=constraint,
                b_eq=np.r_[a.numpy(), b.numpy()], bounds=(0, None), method="highs",
                options={"maxiter": int(max_iterations), "primal_feasibility_tolerance": max(1e-10, tolerance)})
            if not answer.success:
                raise RuntimeError(f"Exact OT failed: {answer.message}")
            plan = torch.from_numpy(answer.x.reshape(n, m))
    elif solver == "sinkhorn":
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Sinkhorn epsilon must be positive")
        kernel = -c/epsilon
        u, v = torch.zeros_like(a), torch.zeros_like(b)
        for iteration in range(int(max_iterations)):
            u = a.log() - torch.logsumexp(kernel + v[None, :], dim=1)
            v = b.log() - torch.logsumexp(kernel + u[:, None], dim=0)
            if iteration % 10 == 0:
                plan = (kernel + u[:, None] + v[None, :]).exp()
                if (plan.sum(1)-a).abs().max() <= tolerance:
                    break
        plan = (kernel + u[:, None] + v[None, :]).exp()
    else:
        raise ValueError(f"Unknown OT solver: {solver}")
    if (not torch.isfinite(plan).all() or (plan < -tolerance).any()
            or not torch.allclose(plan.sum(1), a, atol=tolerance, rtol=0)
            or not torch.allclose(plan.sum(0), b, atol=tolerance, rtol=0)):
        raise RuntimeError("OT solution violates finite/nonnegative/marginal constraints")
    distance = float((plan*c).sum())
    if not math.isfinite(distance):
        raise FloatingPointError("Nonfinite OT distance")
    return plan.detach(), distance


@dataclass
class SparseTargets:
    token_ids: torch.Tensor  # unique teacher IDs, [U]
    probabilities: torch.Tensor  # [student positions,U], conditional row mass


def transport_targets(plan, teacher_ids):
    p = plan.detach().double()
    ids = teacher_ids.detach().to(device=p.device, dtype=torch.long)
    if p.ndim != 2 or ids.ndim != 1 or p.shape[1] != ids.numel():
        raise ValueError("Expected student-by-teacher plan and teacher token IDs")
    if not torch.isfinite(p).all() or (p < 0).any() or (p.sum(1) <= 0).any():
        raise ValueError("Transport plan must have positive finite row mass")
    unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    target = p.new_zeros((p.shape[0], unique.numel()))
    target.scatter_add_(1, inverse.expand(p.shape[0], -1), p/p.sum(1, keepdim=True))
    if not torch.allclose(target.sum(1), torch.ones(p.shape[0], dtype=p.dtype, device=p.device), atol=1e-7):
        raise AssertionError("OT conditional targets must sum to one")
    return SparseTargets(unique.detach(), target.float().detach())


def forward_kl(logits, target: SparseTargets, mask=None, *, temperature=1.0):
    """KL(target || student), including correct entropy for repeated token IDs."""
    if logits.ndim != 2 or logits.shape[0] != target.probabilities.shape[0]:
        raise ValueError("Future logits and target positions must align")
    if temperature <= 0:
        raise ValueError("Trajectory temperature must be positive")
    ids = target.token_ids.to(logits.device)
    probabilities = target.probabilities.to(logits.device).detach()
    scaled = logits.float()/temperature
    log_p = scaled.index_select(-1, ids) - torch.logsumexp(scaled, -1, keepdim=True)
    entropy = torch.special.xlogy(probabilities, probabilities).sum(-1)
    loss = entropy - (probabilities*log_p).sum(-1)
    if mask is not None:
        loss = torch.where(mask.bool(), loss, torch.zeros_like(loss))
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Nonfinite forward trajectory KL")
    return loss


def divergence_mask(distances, *, mode="percentile", quantile=0.3,
                    threshold=0.0, min_distance=1e-8):
    d = torch.as_tensor(distances, dtype=torch.float64)
    if not torch.isfinite(d).all():
        raise FloatingPointError("Nonfinite candidate OT distances")
    if mode not in {"percentile", "absolute"} or not 0 <= quantile <= 1:
        raise ValueError("Invalid divergence selection configuration")
    cutoff = float(torch.quantile(d, quantile)) if mode == "percentile" and d.numel() else float(threshold)
    # Include positive ties at the percentile boundary; never accept zero OT.
    return (d >= cutoff) & (d > min_distance), cutoff


def trajectory_denominator(normalization, window_count, token_count, original_token_count):
    if normalization == "window_mean_sum":
        return max(window_count, 1)
    if normalization == "token_mean":
        return max(token_count, 1)
    if normalization == "sum":
        return 1
    if normalization == "original_token_mean":
        return max(original_token_count, 1)
    raise ValueError(f"Unknown trajectory normalization: {normalization}")
