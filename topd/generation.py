"""Current-weight HF generation with shared causal-prefix KV and fixed RNG streams.

No vLLM sampler substitution: torch.multinomial and per-request generators are
the same as the reference path. Only causal prefill/decode scheduling changes.
"""
from collections import defaultdict
import gc
import time

import torch
from transformers.cache_utils import DynamicCache

from b200_experiment.scoring import _sample_top_p, cuda_sync
from .runtime import automatic_batch_size, full_inference_model, available_workspace


def fork_prefix_cache(cache, lengths):
    """Left-pad independent causal cache slices; never mutate the source cache."""
    width = max(lengths)
    layers = []
    sources = ([source.to_legacy_cache() for source in cache] if isinstance(cache, list)
               else [cache.to_legacy_cache()]*len(lengths))
    for layer in range(len(sources[0])):
        keys, values = [], []
        for source, length in zip(sources, lengths):
            key, value = source[layer]
            keys.append(torch.nn.functional.pad(key[0, :, :length], (0, 0, width-length, 0)))
            values.append(torch.nn.functional.pad(value[0, :, :length], (0, 0, width-length, 0)))
        layers.append((torch.stack(keys), torch.stack(values)))
    return DynamicCache.from_legacy_cache(tuple(layers))


@torch.no_grad()
def decode_cached(model, cache, logits, lengths, *, pad_id, eos_ids, window,
                  temperature, top_p, seeds, device):
    width, count = max(lengths), len(lengths)
    positions = torch.arange(width, device=device)
    attention = positions[None, :] >= (width-torch.tensor(lengths, device=device))[:, None]
    active = torch.ones(count, device=device, dtype=torch.bool)
    eos = torch.tensor(eos_ids, device=device, dtype=torch.long)
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    tokens, masks = [], []
    for index in range(window):
        scores = logits.float().clone()
        if pad_id not in eos_ids:
            scores[:, pad_id] = -torch.inf
        if temperature == 0:
            token = scores.argmax(-1)
        else:
            probabilities = torch.softmax(scores/temperature, -1)
            token = torch.stack([_sample_top_p(probabilities[row:row+1], top_p, generators[row]).squeeze(0)
                                 for row in range(count)])
        token = torch.where(active, token, torch.full_like(token, pad_id))
        tokens.append(token)
        masks.append(active.clone())
        attention = torch.cat((attention, active[:, None]), 1)
        active = active & ~token[:, None].eq(eos[None, :]).any(-1)
        # No per-row GPU -> Python scalar transfers; one completion check/batch.
        if index+1 == window or not bool(active.any()):
            break
        output = model(input_ids=token[:, None], attention_mask=attention,
            position_ids=(attention.long().sum(-1, keepdim=True)-1).clamp_min(0),
            cache_position=torch.tensor([width+index], device=device),
            past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
        cache, logits = output.past_key_values, output.logits[:, -1]
    token_matrix = torch.stack(tokens, 1).cpu()
    mask_matrix = torch.stack(masks, 1).cpu()
    return [row[mask] for row, mask in zip(token_matrix, mask_matrix)]


@torch.no_grad()
def generate_shared_futures(model, prefixes, pairs, seeds, *, settings, pad_id, eos_ids,
                            distributed):
    """One model materialization/update, one prefill/original response.

    All branches are sampled independently. Cached prefixes belong only to this
    call; student weights are gathered anew after every optimizer update.
    """
    device = distributed.device
    metrics = {"prefix_prefill_tokens": 0, "reference_prefill_tokens": sum(p.numel() for p in prefixes),
               "prefill_calls": 0, "decode_batches": 0, "maximum_batch_size": 0, "oom_retries": 0}
    results = [None]*len(prefixes)
    groups = defaultdict(list)
    for ordinal, (row, _) in enumerate(pairs):
        groups[row].append(ordinal)
    began = time.perf_counter()
    model.eval()
    with full_inference_model(model, device, metrics) as full:
        if str(full.config.model_type) != "qwen3":
            raise ValueError("Shared-prefix futures currently support the required Qwen3 models only")
        pool = {}
        pool_bytes = 0
        pool_limit = int(available_workspace(device)*float(settings.get("prefix_pool_fraction", .25)))

        def flush_pool():
            # Length buckets may combine candidates from different responses.
            # Each branch still selects only its own immutable causal KV slice.
            ordered = sorted(pool, key=lambda ordinal: (-prefixes[ordinal].numel(), ordinal))
            cursor = 0
            retry_cap = int(settings.get("generation_max_batch_size", 64))
            while cursor < len(ordered):
                batch = automatic_batch_size(full, device, prefix_length=prefixes[ordered[cursor]].numel(),
                    future_length=settings["future_window"], settings=settings, phase="generation",
                    requested=settings.get("generation_batch_size", "auto"))
                chosen = ordered[cursor:cursor+min(batch, retry_cap)]
                chosen_lengths = [prefixes[ordinal].numel() for ordinal in chosen]

                def attempt():
                    cache = fork_prefix_cache([pool[ordinal][0] for ordinal in chosen], chosen_lengths)
                    return decode_cached(full, cache, torch.stack([pool[ordinal][1] for ordinal in chosen]),
                        chosen_lengths, pad_id=pad_id, eos_ids=eos_ids,
                        window=settings["future_window"], temperature=settings["future_temperature"],
                        top_p=settings["future_top_p"], seeds=[seeds[ordinal] for ordinal in chosen], device=device)
                try:
                    futures = attempt()
                    out_of_memory = False
                except torch.cuda.OutOfMemoryError:
                    if len(chosen) <= 1:
                        raise
                    retry_cap = max(1, len(chosen)//2)
                    metrics["oom_retries"] += 1
                    out_of_memory = True
                if out_of_memory:
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Reinitialize generators from identical seeds on retry.
                    continue
                for ordinal, future in zip(chosen, futures):
                    results[ordinal] = future
                metrics["decode_batches"] += 1
                metrics["maximum_batch_size"] = max(metrics["maximum_batch_size"], len(chosen))
                cursor += len(chosen)
                del futures
            pool.clear()

        for ordinals in groups.values():
            ordered = sorted(ordinals, key=lambda ordinal: (prefixes[ordinal].numel(), ordinal))
            longest_cpu = prefixes[ordered[-1]].cpu()
            lengths = [prefixes[ordinal].numel() for ordinal in ordered]
            # Enforce causal sharing rather than silently reusing unrelated KV.
            if any(not torch.equal(prefixes[ordinal].cpu(), longest_cpu[:prefixes[ordinal].numel()])
                   for ordinal in ordered):
                raise AssertionError("Future branches must share their original response's causal prefix")
            longest = longest_cpu.to(device)
            text_config = full.config
            estimated_cache_bytes = (2*text_config.num_hidden_layers*text_config.num_key_value_heads
                *text_config.head_dim*longest.numel()*next(full.parameters()).element_size())
            if pool and pool_bytes+estimated_cache_bytes > pool_limit:
                flush_pool()
                pool_bytes = 0
            predictors = torch.tensor([length-1 for length in lengths], device=device)
            output = full(input_ids=longest[None], attention_mask=torch.ones_like(longest[None]),
                position_ids=torch.arange(longest.numel(), device=device)[None],
                use_cache=True, return_dict=True, logits_to_keep=predictors)
            source_cache, initial_logits = output.past_key_values, output.logits[0]
            metrics["prefix_prefill_tokens"] += longest.numel()
            metrics["prefill_calls"] += 1
            del output
            for index, ordinal in enumerate(ordered):
                pool[ordinal] = (source_cache, initial_logits[index])
            pool_bytes += estimated_cache_bytes+initial_logits.numel()*initial_logits.element_size()
            del source_cache, initial_logits
        if pool:
            flush_pool()
    cuda_sync(device)
    metrics["total_time_sec"] = time.perf_counter()-began
    return results, metrics
