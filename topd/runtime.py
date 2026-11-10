"""Memory planning and collective-free, current-weight inference sessions."""
from contextlib import contextmanager
import math
import time

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from b200_experiment.distributed import unwrap_model
from b200_experiment.fsdp import is_fsdp_model
from b200_experiment.scoring import cuda_sync


@contextmanager
def full_inference_model(model, device, metrics=None):
    """Gather once, bypass *all* FSDP forwards, restore wrappers before reshard.

    Inference only: no gradient graph may escape this context. A plain root
    unwrap is insufficient because decoder blocks remain FSDP wrappers.
    Every rank enters once per role/update, then generates independently.
    """
    cuda_sync(device)
    started = time.perf_counter()
    if not is_fsdp_model(model):
        if metrics is not None:
            metrics["weight_gather_time_sec"] = 0.
        yield unwrap_model(model)
        return
    wrappers = [module for module in model.modules() if isinstance(module, FSDP)]
    sentinel = object()
    originals = []
    with torch.no_grad(), FSDP.summon_full_params(model, recurse=True, writeback=False):
        for wrapper in wrappers:
            originals.append((wrapper, wrapper.__dict__.get("forward", sentinel)))
            wrapper.forward = wrapper._fsdp_wrapped_module.forward
        cuda_sync(device)
        if metrics is not None:
            metrics["weight_gather_time_sec"] = time.perf_counter() - started
        try:
            yield unwrap_model(model)
        finally:
            # Do this before summon_full_params exits and traverses the tree.
            for wrapper, forward in reversed(originals):
                if forward is sentinel:
                    del wrapper.forward
                else:
                    wrapper.forward = forward
    cuda_sync(device)


def available_workspace(device, *, headroom_gib=8., fraction=.65):
    if device.type != "cuda":
        return 2**30
    free, _ = torch.cuda.mem_get_info(device)
    # Reserved-but-unused blocks belong to this process and are reusable.
    reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return max(0, int((free + reusable - headroom_gib*2**30)*fraction))


def automatic_batch_size(model, device, *, prefix_length, future_length, settings,
                         phase, requested="auto"):
    """Conservative token/VRAM bound; scientific batch/normalization never change."""
    if requested != "auto":
        return max(1, int(requested))
    config = unwrap_model(model).config
    hidden = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    vocab = int(config.vocab_size)
    heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", None) or hidden//config.num_attention_heads)
    element = next(model.parameters()).element_size()
    length = max(1, int(prefix_length)+int(future_length))
    if phase == "generation":
        # KV + concatenation/copy workspace, attention and short LM-head buffers.
        per_row = 3*2*layers*heads*head_dim*length*element + 12*vocab
        token_budget = int(settings.get("generation_token_budget", 131072))
        cap = int(settings.get("generation_max_batch_size", 64))
    else:
        # Checkpointed prefix activations plus layer workspace and short logits.
        # Calibrated conservatively, not an assertion of a universal OOM bound.
        intermediate = int(getattr(config, "intermediate_size", 4*hidden))
        per_row = length*element*(3*layers*hidden + 8*intermediate + 16*hidden)
        per_row += future_length*vocab*element*3
        token_budget = int(settings.get("loss_token_budget", 65536))
        cap = int(settings.get("loss_max_batch_size", 32))
    memory = available_workspace(device, headroom_gib=float(settings.get("memory_headroom_gib", 8.)),
                                 fraction=float(settings.get("workspace_fraction", .65)))
    return max(1, min(cap, token_budget//length, memory//max(per_row, 1)))


def use_accumulation_no_sync(model, distributed, settings):
    requested = settings.get("accumulation_no_sync", "auto")
    enabled = requested is True or requested == "auto"
    if enabled and distributed.device.type == "cuda":
        # FULL_SHARD.no_sync retains unsharded gradients; leave ample room.
        schema = getattr(model, "_b200_hf_parameter_shapes", None)
        count = (sum(math.prod(shape) for shape in schema.values())
                 if schema else sum(parameter.numel() for parameter in model.parameters()))
        enabled = available_workspace(distributed.device, fraction=.5) > count*8
    any_disabled = distributed.any(not enabled)
    return enabled and not any_disabled
