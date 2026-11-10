"""Offline integration: real tiny Qwen3 models, generation, OT, SGD, HF resume.

This is explicitly a synthetic engineering smoke test, never a math result.
"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from b200_experiment.config import load_config, save_config
from b200_experiment.data import epoch_batch_indices, expand_prompt_batch, tokenize_prompts
from b200_experiment.distributed import DistributedContext
from b200_experiment.opd_core import build_student_topk_opd_reference
from b200_experiment.resume import restore_optimizer
from b200_experiment.scoring import generate_on_policy, score_original_rollout
from b200_experiment.trainer import _make_optimizer, _opd_train_step, _save_checkpoint
from .checkpoint import load_pending_rollout, pending_rollout
from .configuration import ROOT
from .engine import TOPDEngine
from .reporting import finalize_training


def tiny_tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
             **{f"token{i}": i for i in range(3, 32)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
        pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %} token3"
    tokenizer.padding_side = "left"
    return tokenizer


def tiny_model(seed=7):
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=512, bos_token_id=1,
        eos_token_id=2, pad_token_id=0, tie_word_embeddings=True,
        attention_dropout=0.0)
    return Qwen3ForCausalLM(config).float()


def smoke_config(output):
    config = load_config(ROOT/"configs/topd_compe_math.yaml")
    config["experiment"].update(output_dir=str(output), require_b200=False)
    config["models"].update(dtype="float32", stop_token_ids=[2])
    config["rollout"].update(batch_size=2, num_responses=2, max_new_tokens=6, backend="hf")
    config["training"].update(micro_batch_size_per_gpu=1, ppo_mini_batch_size=2,
        fused_optimizer=False, gradient_checkpointing=False, learning_rate=1e-3)
    config["topd"].update(candidate_fraction=1., candidate_budget=1,
        future_window=50, future_seed=1000042, generation_batch_size=2,
        loss_micro_batch_size=1, divergence_mode="absolute", ot_threshold=0.)
    config["training_evaluation"]["enabled"] = False
    return config


def run_smoke(output):
    torch.set_num_threads(2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="tiny-qwen3-", dir=output))
    config = smoke_config(run)
    save_config(config, run/"resolved_config.yaml")
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    student, teacher, tokenizer = tiny_model(7), tiny_model(8), tiny_tokenizer()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    initial = {name: p.detach().clone() for name, p in student.named_parameters()}
    initial_teacher = {name: p.detach().clone() for name, p in teacher.named_parameters()}
    optimizer, _ = _make_optimizer(student.parameters(), config["training"])
    engine = TOPDEngine(config, student, teacher, tokenizer, context, run)
    records = [{"problem": f"token{3+i} token7"} for i in range(3)]
    step = 0
    resumed_exactly = False
    for batch_index in range(2):
        indices = epoch_batch_indices(3, 2, batch_index, 42)
        encoded, _ = tokenize_prompts([records[i] for i in indices], tokenizer,
            config["data"], context.device)
        encoded, expanded, response_indices = expand_prompt_batch(encoded, indices, 2)
        rollout = generate_on_policy(student, encoded["input_ids"], encoded["attention_mask"],
            max_new_tokens=6, temperature=1., top_p=1., eos_token_ids=[2],
            pad_token_id=0, seed=42, sample_seed_offset=batch_index*4)
        s = score_original_rollout(student, rollout, False, 8, top_k=4,
            micro_batch_size=1, retain_response_logits=False)
        t = score_original_rollout(teacher, rollout, False, 8, top_k=4,
            candidate_ids=s.top_k_ids, micro_batch_size=1, retain_response_logits=False)
        reference = build_student_topk_opd_reference(s.top_k_ids, s.top_k_log_probs,
            t.candidate_log_probs, rollout.valid_mask, top_k=4)
        student._topd_pending_rollout = pending_rollout(rollout, s, t, batch_index, context)
        engine.dataset_indices, engine.response_indices = expanded, response_indices
        if batch_index == 0:
            # Commit halfway through a main rollout, then resume its remaining
            # PPO group from the saved frozen batch, not a newly sampled suffix.
            def checkpoint_callback(current_step, metrics):
                _save_checkpoint(student, tokenizer, optimizer, run, current_step,
                    False, True, context)
            first = _opd_train_step(student, optimizer, rollout, rollout.valid_mask.float(),
                reference, config, context.device, context,
                trajectory_group_ids=response_indices, max_optimizer_steps=1,
                optimizer_step_start=step, on_optimizer_step=checkpoint_callback,
                trajectory_engine=engine)
            step += first["optimizer_steps"]
            checkpoint = run/f"checkpoint-{step:06d}"
            # Uninterrupted and resumed branches must produce identical final
            # parameters and Adam moments for the next PPO update.
            uninterrupted = copy.deepcopy(student)
            uninterrupted_optimizer, _ = _make_optimizer(uninterrupted.parameters(), config["training"])
            uninterrupted_optimizer.load_state_dict(optimizer.state_dict())
            uninterrupted_engine = TOPDEngine(config, uninterrupted, teacher, tokenizer, context)
            _opd_train_step(uninterrupted, uninterrupted_optimizer, rollout,
                rollout.valid_mask.float(), reference, config, context.device, context,
                trajectory_group_ids=response_indices, ppo_minibatch_offset=1,
                optimizer_step_start=step, trajectory_engine=uninterrupted_engine)
            restored = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
            student.load_state_dict(restored.state_dict())
            restore_optimizer(optimizer, checkpoint, context.device, model=student, distributed=context)
            resumed_rollout, resumed_s, resumed_t = load_pending_rollout(checkpoint, batch_index, context)
            if not torch.equal(resumed_rollout.input_ids, rollout.input_ids):
                raise AssertionError("Original rollout changed on resume")
            if not torch.equal(resumed_s.top_k_log_probs, s.top_k_log_probs):
                raise AssertionError("Frozen student OPD scores changed on resume")
            next_result = _opd_train_step(student, optimizer, resumed_rollout,
                resumed_rollout.valid_mask.float(), reference, config, context.device, context,
                trajectory_group_ids=response_indices, ppo_minibatch_offset=1,
                optimizer_step_start=step, trajectory_engine=engine)
            step += next_result["optimizer_steps"]
            for name, value in student.named_parameters():
                if not torch.allclose(value, dict(uninterrupted.named_parameters())[name], atol=1e-7, rtol=1e-6):
                    raise AssertionError(f"Resumed update differs at {name}")
            resumed_exactly = True
        else:
            result = _opd_train_step(student, optimizer, rollout, rollout.valid_mask.float(),
                reference, config, context.device, context,
                trajectory_group_ids=response_indices, optimizer_step_start=step,
                trajectory_engine=engine)
            step += result["optimizer_steps"]
    for name, value in teacher.named_parameters():
        if not torch.equal(value, initial_teacher[name]) or value.grad is not None:
            raise AssertionError("Teacher was updated or received gradients")
    if not any(not torch.equal(p, initial[name]) for name, p in student.named_parameters()):
        raise AssertionError("Student did not update")
    final = _save_checkpoint(student, tokenizer, optimizer, run, step, True, True, context)
    reloaded = AutoModelForCausalLM.from_pretrained(final, local_files_only=True)
    if reloaded.config.model_type != "qwen3" or not reloaded.config.use_cache:
        raise AssertionError("HF checkpoint config/cache was not saved for inference")
    summary = finalize_training(run, 3, 2, 3, context)
    if not summary["counts"]["accepted_candidates"] or not summary["counts"]["trajectory_loss_tokens"]:
        raise AssertionError("Smoke did not exercise OT-aligned trajectory supervision")
    summary.update(status="passed", synthetic_only=True, run_directory=str(run),
        mid_rollout_resume_matches_uninterrupted=resumed_exactly,
        teacher_frozen=True, student_updated=True, future_window=50,
        checkpoint_reloaded=True)
    (output/"smoke_result.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT/"outputs/smoke"))
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output), indent=2))


if __name__ == "__main__":
    main()
