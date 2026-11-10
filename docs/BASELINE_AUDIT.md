# Baseline audit (2026-11-10)

Project root: `BellmanOPD_analysis/`, sibling of `TOPD/`. Actual source bytes,
including uncommitted source, are pinned in `source_manifest.json`.

## Sources inspected before implementation

- `README.md`, `RUN_B200.md`, `HUONG_DAN_CHAY.md`, `requirements.txt`.
- `configs/qwen3_b200_{base,cmt,opd}.yaml`, `scripts/train_cmt_b200.sh`,
  `scripts/common_b200.sh`, `scripts/eval_multiseed_b200.sh`.
- `b200_experiment/trainer.py`: `run_training`, `_opd_train_step`,
  `_make_optimizer`, `_save_checkpoint`, training evaluation callbacks.
- `models.py`: `load_models`, `inspect_model_assets`,
  `assert_tokenizer_compatibility`, `validate_shared_tokenizer_protocol`.
- `data.py`: `read_records`, `render_record_prompt`,
  `filter_overlong_prompt_records`, `expand_prompt_batch`, `epoch_batch_indices`.
- `math_prompts.py`: single canonical math instruction and no-think chat template.
- `scoring.py`: `generate_on_policy`, `rollout_forward_inputs`,
  `score_original_rollout`, `score_student_teacher_rollout`.
- `opd_core.py`: `build_student_topk_opd_reference`,
  `gather_candidate_log_probs`, `topk_candidate_ppo_loss`, `weighted_token_sums`.
- `distributed.py`, `fsdp.py`, `vllm_rollout.py`, `resume.py`, `metadata.py`,
  `preflight.py`, `training_cost.py`: FSDP, CUDA IPC, tail masks, checkpoint/RNG.
- `selectors/cmt_selector.py`, `selectors/pgt_selector.py`,
  `selectors/opd_selector.py`: CMT is the current main method; RAC is legacy.
- `evaluation.py`, `vllm_evaluation.py`, `multiseed_evaluation.py`:
  shared math-verify grading, Avg@8 and Pass@8 from identical eight samples.

## End-to-end protocol

Load local HF student/teacher; freeze teacher; verify exact vocabulary mapping.
Student tokenizer alone renders both models' context IDs, with
`enable_thinking=false`. Read only the configured training parquet; preserve
file order during rendered-prompt filtering (>1024 tokens excluded). Shuffle
using Python `Random(seed + epoch)`. Global prompt batches contain 64 examples,
each expanded prompt-major into four separately seeded vLLM requests. Per-rank
TP=1 engines synchronize current HF weights via CUDA IPC, sleep during scoring
and backward. Score frozen student Top-16 support and teacher logits on the
same response predictors, `prompt_width - 1 + response_index`. Include EOS,
exclude prompts/padding and inactive rank fillers.

CMT builds local projected-gradient opportunity and a raw truncated common-mass
successor correction on the student/teacher support union. Bounded Gibbs weights
(mean 1, bounds 0.5/2, final KL budget .02, gamma/lambda 1) allocate learning
within response-index-major global PPO groups. The policy loss stays on student
Top-16. Each 256-trajectory rollout produces four optimizer updates of 64
trajectories, microbatch 16/GPU, automatic accumulation. AdamW lr 5e-6,
betas (.9,.95), weight decay 0, BF16, clip norm 1, gradient checkpointing,
PPO clips .2/.28, dual clip 3; **no LR scheduler**. Save model, tokenizer,
optimizer and RNG every 150 updates and at end. FSDP FULL_SHARD student/teacher.

## Discrepancies and limits

1. Checked-in math launch defaults use **Qwen3-8B -> Qwen3-1.7B-Base, 3 epochs**,
   not the requested **Qwen3-4B -> Qwen3-1.7B, 1 epoch**. There is no completed
   math run's resolved config/checkpoint in this checkout proving the latter.
   TOPD obeys the requested identities/duration; the comparison CMT must be
   rerun with those same overrides. Never claim historical runs are matched.
2. Base YAML says lr 1e-6, n=1, response 7168, PPO=16, one epoch. CMT YAML and
   real launcher override these to lr 5e-6, n=4, response 4096, PPO=64, 3 epochs.
3. README describes a reverse-KL recipe, but single-turn code freezes **full
   vocabulary** log probabilities and then conditionalizes **current** policy
   probabilities on Top-16. The unchanged-policy PPO ratio is 1 / p(Top-16),
   not 1. Clipping may suppress gradients. ALFWorld v2 repairs this only for
   that setting. TOPD preserves the math code exactly for fair comparison;
   this is a baseline objective limitation, not a mathematically exact direct
   reverse-KL derivative. A repaired objective would require both baselines
   to change and must be reported as a separate experiment.
4. CMT PPO groups are response-index-major; plain OPD uses rank-interleaved
   order. TOPD deliberately adopts CMT group membership/update cadence.
5. Original resume can regenerate a partially consumed rollout using updated
   student weights. TOPD must persist that rollout and frozen scores when
   checkpointing within a rollout, rather than alter its remaining examples.
6. Runtime assets at `/workspace/storage-shared` are absent here. Machine has
   one RTX 3090; two-GPU/B200 production training cannot be validated locally.
   Actual filtered example count and training costs must be measured at launch.
7. The evaluation parser falls back to basic boxed-text equality on exceptions.
   TOPD requires `math-verify` and verifies representative numeric, fraction,
   nested-box and incorrect answers before evaluation; it copies the parser
   unchanged. This does not prove correctness on every symbolic expression.
8. `RUN_B200.md` still shows an older n=1/PPO=16/micro=8/one-epoch command,
   whereas the newer CMT launcher and `HUONG_DAN_CHAY.md` use n=4/PPO=64/micro=16.
   The latter runbook explicitly exports training-evaluation seed 42, while
   the launcher/base YAML default to 1234. No actual shell environment/run
   config is available to decide which was used in a historical experiment.
   TOPD defaults to the current launcher; `BASELINE_CONFIG` inherits actual
   shared settings, including evaluation seeds, when supplied.
9. `common_b200.sh::build_training_args` derives vLLM max sequences and request
   concurrency from `ceil(global_prompts * responses / world_size)`. With two
   GPUs this is 128, overriding CMT YAML's static 64. TOPD reproduces this
   launch-time derivation; generation sampling/request seeds stay unchanged.

## Intentional copied-source adaptations

The user subsequently confirmed the B200 model paths `models/Qwen3-4B` and
`models/Qwen3-1.7B`, both relative to `/workspace/storage-shared`. These are the
active TOPD defaults and the required CMT comparison overrides. Shared dataset
paths and resolver semantics remain those of the original B200 project.

Only `trainer.py`, `resume.py`, and `evaluation.py` differ from their audited
copies. The trainer adds TOPD hooks, CMT group membership for TOPD, exact
mid-rollout persistence, explicit coverage/cost metadata, and masked
collective-only execution when a tail rank has no real row. Empty CUDA RNG
state supports CPU checkpoint smoke only; CUDA behavior is unchanged.
Resume additionally checks TOPD/epoch/filter/clipping settings and rewinds
TOPD metrics. Evaluation only appends the `TOPD` model display name to its
whitelist; decoding, math parser and metric functions are identical.

Original source files remain unchanged; `source_manifest.json` records the
inspected bytes, and `FILES_CREATED.md` lists the standalone implementation.
