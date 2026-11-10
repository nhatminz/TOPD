"""Persist the unconsumed main rollout inside the atomic HF checkpoint."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from b200_experiment.scoring import BaseScores, RolloutBatch


def tensor_tree(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {key: tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tensor_tree(item, device) for item in value)
    return value


def pending_rollout(rollout, student_scores, teacher_scores, index, context):
    return dict(version=1, rollout_index=index, world_size=context.world_size,
        rank=context.rank, rollout=tensor_tree(asdict(rollout), "cpu"),
        student_scores=tensor_tree(asdict(student_scores), "cpu"),
        teacher_scores=tensor_tree(asdict(teacher_scores), "cpu"))


def load_pending_rollout(checkpoint, index, context):
    path = Path(checkpoint)/f"topd_rollout_rank-{context.rank:05d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Mid-rollout resume requires frozen original rollout: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["version"] != 1 or payload["rollout_index"] != index
            or payload["world_size"] != context.world_size or payload["rank"] != context.rank):
        raise ValueError("Resume rollout index/rank/world size differs from saved checkpoint")
    return (RolloutBatch(**tensor_tree(payload["rollout"], context.device)),
            BaseScores(**tensor_tree(payload["student_scores"], context.device)),
            BaseScores(**tensor_tree(payload["teacher_scores"], context.device)))
