# Implementation and reduction contract

Paper source: [Jiang & Ferraro, v2, Section 3](https://arxiv.org/pdf/2606.00305v2).
The mathematical components implemented here are same-prefix teacher/student
future sampling, OT divergence detection, transported teacher token guidance,
and forward KL added to the common OPD loss. The future window is 50 tokens.
Thresholds, representations and reduction details are reproduction decisions
listed separately in `PAPER_AMBIGUITIES.md`.

## Common OPD

`b200_experiment/opd_core.py` is byte-identical to the audited comparison source.
Student Top-16 candidate IDs, detached rewards, clipping, masks and global
token normalization are preserved. In particular, the existing single-turn
full-vocabulary frozen / conditional-current normalization discrepancy is
preserved and must accompany any scientific report (see `BASELINE_AUDIT.md`).
This code is not presented as a repaired direct reverse-KL optimizer.

TOPD uses uniform valid-response weights. Main prompts, rendered no-think
templates, vLLM generation, teacher scoring, original rollout seeds and
precision remain shared. PPO membership is CMT's response-index-major order,
with one update for every global group of at most 64 trajectories.

## Candidate states and fresh futures

`topd/algorithm.py::select_candidates` ranks the existing frozen local OPD
statistic `-sum_k advantage[t,k]`, also used as the source pipeline's divergence
proxy. This is the Student-Top-K weighted full-logprob gap before PPO clipping,
not a direct full-vocabulary KL. Select the highest `ceil(.02 * valid_length)`
per response. Stable ties prefer earlier response positions. Optional absolute
budget and minimum loss are configurable. Ranks/thresholds exclude inactive
rank fillers and response padding. The score is frozen with the original
rollout, exactly like its OPD reference; it is not refreshed after each of the
four PPO updates within a full rollout.

For response position `t`, `extract_prefix` selects physical IDs before
`prompt_width + t`, removes left padding, and excludes token `t` itself.
`TOPDEngine.prepare` runs immediately before each optimizer update: frozen
teacher and current pre-update student independently generate from these exact
IDs. No reference/gold response enters generation. Per-request RNG streams are
disjoint between teacher and student and do not consume the main rollout RNG.

HF causal KV caches are reused within each future; generation batches prefixes.
It forwards through the FSDP wrapper rather than materializing both whole models
or unwrapping decoder forward calls. All ranks execute identical numbers of
generation rounds and synchronized EOS exits; ranks without candidates use
collective-only fillers. Include the first EOS, stop there, exclude later pads.
Suppress a distinct padding ID during future sampling; an EOS sharing that ID
remains sampleable. Main vLLM rollout behavior is unchanged.

## OT convention and targets

The ground cost is `C[i,j] = 1 - cosine(E_teacher[s_i], E_teacher[t_j])`:
**student rows, teacher columns**, one frozen common input embedding table.
Zero/nonfinite embeddings raise an error. Tokenizer vocabularies and IDs must
match exactly. Only required vectors are gathered from the teacher table; FSDP
root-only parameter materialization avoids gathering every decoder block.

Uniform marginals use actual valid future lengths `n` and `m`:
`P.sum(1)=1/n`, `P.sum(0)=1/m`. Equal lengths use exact Hungarian assignment
scaled by `1/n`, an exact vertex solution of balanced uniform OT. Unequal
lengths use SciPy HiGHS sparse linear programming. Optional log-domain Sinkhorn
is explicitly approximate and must satisfy tolerance; nonconvergence raises.
Plans are detached, nonnegative, finite and validated against both marginals.

Across candidate windows in the current global PPO group, accept OT distance
at least its median and strictly above `1e-8`. Positive percentile ties are
accepted. Identical trajectories with zero OT receive no correction. Absolute
threshold mode is available. This calibration consumes training candidates
only, with no persistent statistics or benchmark-based tuning.

Conditional probabilities are `Q[i,j] = P[i,j] / sum_j P[i,j]`.
For every distinct teacher vocabulary ID `v`, use
`target[i,v] = sum_{j:teacher_id[j]=v} Q[i,j]`. Repeated teacher IDs are combined
**before** computing target entropy. This yields at most 50 unique target IDs
and a `[student_length, unique_teacher_ids]` sparse representation; no dense
vocabulary-sized target is created. Every target row sums to one.

## Student update

For each accepted window use its independently generated student tokens.
`rollout_forward_inputs` predicts future token `i` at physical logit position
`padded_prefix_width - 1 + i`. Inputs to that predictor contain the original
prefix and generated student tokens strictly before `i`; no teacher suffix is
teacher-forced into the student context. Student forwards and backward happen
with autograd enabled, with stable FP32 logsumexp over the full vocabulary:

```text
KL_i = sum_v target_i(v) log target_i(v)
       - sum_v target_i(v) log p_student(v | original_prefix, student_future_<i)

OPD = common global token-mean clipped surrogate
trajectory = sum_{accepted windows} sum_{valid future positions} KL_i
             / global_accepted_window_count
total = OPD + 0.01 * trajectory
```

The primary `window_mean_sum` keeps the sum over future positions. Optional
`token_mean` divides by global future-token count; `sum` divides by one;
`original_token_mean` divides by the common original valid-response token count.
Each choice changes lambda's scale and is logged. Empty acceptance sets produce
zero trajectory loss without changing OPD. Solver failures never silently do so.

OPD gradients and trajectory gradients accumulate before the **single shared**
gradient clipping/AdamW update. FSDP/DDP average gradients across ranks, so local
trajectory sums are multiplied by `world_size / global_denominator`. No extra
division by microbatch or accumulation count occurs. Per-token effective
coefficient `lambda/global_denominator` is recorded.

Overlapping windows are separate sampled contexts: each accepted window enters
once; repeated contextual supervision is intentional. Their losses add, then
use the selected global denominator. No cached future or OT plan survives an
optimizer update. Rank fillers have exactly zero loss, count and coverage.

## Checkpoints, coverage and diagnostics

Checkpoint staging saves HF model/tokenizer/config, optimizer/RNG, and each
rank's frozen main rollout/scoring tensors atomically before directory commit.
Mid-rollout resume restores these tensors and only processes remaining PPO
groups. This bounded original rollout cache is not a stale future cache.
Configuration changes to TOPD, epochs, models, data or objective are rejected.

`topd_metrics.jsonl` stores consumed `(filtered_dataset_index,response_index)`
pairs for each committed update. Finalization requires exactly the complete
Cartesian product of examples and response indices, each with multiplicity one,
and every expected optimizer step. It raises on missing/repeated coverage.

Candidate thresholds/loss quantiles, OT quantiles/cutoff, accept/reject counts,
window lengths, original/additional rollouts/tokens, physical filler costs,
losses, gradient norm, peak allocated GPU bytes and phase times are recorded.
Optional `log_candidate_details` writes window IDs and prefix hashes, not large
logit/embedding matrices. Original training-cost logs also measure co-located
device usage, main-rollout throughput and wall time; completed process-session
wall time includes setup/checkpoint/evaluation and is labeled separately.
