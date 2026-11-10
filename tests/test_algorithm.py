import math

import pytest
import torch

from topd.algorithm import (
    SparseTargets, cosine_ground_cost, divergence_mask, extract_prefix, forward_kl,
    select_candidates, solve_transport, trajectory_denominator, transport_targets,
)
from b200_experiment.scoring import RolloutBatch


def test_candidate_selection_valid_tokens_and_stable_ties():
    values = torch.tensor([[999., 3., 7., 7., float("nan")]])
    mask = torch.tensor([[False, True, True, True, False]])
    pairs, thresholds = select_candidates(values, mask, fraction=.5)
    assert pairs == [(0, 2), (0, 3)]
    assert thresholds == [7.]
    assert select_candidates(values, mask, budget=0)[0] == []
    assert select_candidates(values, mask, budget=1, min_loss=8)[0] == []


def test_candidate_selection_rejects_nonfinite_valid_loss():
    with pytest.raises(FloatingPointError):
        select_candidates(torch.tensor([[float("nan")]]), torch.ones(1, 1, dtype=torch.bool))


def sample_rollout():
    return RolloutBatch(torch.tensor([[0, 10, 11, 20, 21, 2, 0]]),
        torch.tensor([[0, 1, 1, 1, 1, 1, 0]]), torch.tensor([[20, 21, 2, 0]]),
        torch.tensor([[True, True, True, False]]), torch.zeros(1, 4), 3)


@pytest.mark.parametrize("index,expected", [(0, [10, 11]), (1, [10, 11, 20]), (2, [10, 11, 20, 21])])
def test_prefix_extraction_window_starts_at_candidate(index, expected):
    assert extract_prefix(sample_rollout(), 0, index).tolist() == expected


def test_prefix_excludes_padding_and_invalid_positions():
    with pytest.raises(ValueError, match="valid response"):
        extract_prefix(sample_rollout(), 0, 3)


def test_manual_ot_assignment_orientation_and_targets():
    # Student vectors e1,e2; teacher vectors e2,e1. Cross assignment cost=0.
    s, t = torch.eye(2), torch.eye(2).flip(0)
    cost = cosine_ground_cost(s, t)
    plan, distance = solve_transport(cost)
    assert torch.allclose(cost, torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64))
    assert torch.allclose(plan, torch.tensor([[0., .5], [.5, 0.]], dtype=torch.float64))
    assert distance == 0.
    target = transport_targets(plan, torch.tensor([9, 7]))
    assert target.token_ids.tolist() == [7, 9]
    assert torch.equal(target.probabilities, torch.eye(2))


@pytest.mark.parametrize("solver", ["exact", "sinkhorn"])
@pytest.mark.parametrize("shape", [(3, 3), (2, 3), (1, 4), (5, 2)])
def test_ot_uniform_marginals_variable_lengths(solver, shape):
    torch.manual_seed(1)
    plan, distance = solve_transport(torch.rand(shape), solver=solver)
    assert torch.all(plan >= 0) and math.isfinite(distance)
    assert torch.allclose(plan.sum(1), torch.full((shape[0],), 1/shape[0], dtype=torch.float64), atol=1e-7)
    assert torch.allclose(plan.sum(0), torch.full((shape[1],), 1/shape[1], dtype=torch.float64), atol=1e-7)


def test_duplicate_teacher_ids_are_aggregated_before_entropy():
    plan = torch.full((2, 3), 1/6)
    target = transport_targets(plan, torch.tensor([4, 4, 6]))
    assert target.token_ids.tolist() == [4, 6]
    assert torch.allclose(target.probabilities, torch.tensor([[2/3, 1/3]]).expand(2, -1))
    assert torch.allclose(target.probabilities.sum(-1), torch.ones(2))


def test_forward_kl_direction_value_and_gradient():
    logits = torch.tensor([[.2, -.7, 1.1]], requires_grad=True)
    target = SparseTargets(torch.tensor([0, 2]), torch.tensor([[.25, .75]], requires_grad=True))
    loss = forward_kl(logits, target).sum()
    dense = torch.tensor([[.25, 0., .75]])
    expected = torch.special.xlogy(dense, dense).sum() - (dense*logits.log_softmax(-1)).sum()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.allclose(logits.grad, logits.detach().softmax(-1)-dense, atol=1e-7)
    assert target.probabilities.grad is None


def test_masked_future_positions_receive_zero_gradient():
    logits = torch.zeros(2, 3, requires_grad=True)
    target = SparseTargets(torch.tensor([1]), torch.ones(2, 1))
    loss = forward_kl(logits, target, torch.tensor([True, False])).sum()
    loss.backward()
    assert logits.grad[1].eq(0).all()
    assert logits.grad[0].abs().sum() > 0


def test_targets_detached_from_plan():
    plan = torch.eye(2, requires_grad=True)/2
    assert not transport_targets(plan, torch.tensor([1, 2])).probabilities.requires_grad


def test_high_loss_and_high_ot_required():
    keep, threshold = divergence_mask([0., .1, .8, 1.], quantile=.5)
    assert keep.tolist() == [False, False, True, True]
    assert threshold == pytest.approx(.45)
    assert not divergence_mask([0., 0.])[0].any()
    assert divergence_mask([])[0].numel() == 0


def test_window_sum_normalization_keeps_lambda_meaning():
    assert trajectory_denominator("window_mean_sum", 2, 100, 2000) == 2
    assert trajectory_denominator("token_mean", 2, 100, 2000) == 100
    assert trajectory_denominator("sum", 2, 100, 2000) == 1
    assert trajectory_denominator("original_token_mean", 2, 100, 2000) == 2000


def test_invalid_ot_inputs_raise_without_fallback():
    for cost in (torch.tensor([[float("nan")]]), torch.tensor([[-1.]])):
        with pytest.raises(FloatingPointError):
            solve_transport(cost)
    with pytest.raises(ValueError):
        cosine_ground_cost(torch.zeros(1, 2), torch.ones(1, 2))
    with pytest.raises(ValueError):
        transport_targets(torch.zeros(1, 2), torch.tensor([1, 2]))


def test_sinkhorn_failure_is_observable():
    with pytest.raises(RuntimeError, match="marginal"):
        solve_transport(torch.tensor([[0., .5, 1.], [1., .4, 0.]]),
            solver="sinkhorn", epsilon=.0001, max_iterations=1)
