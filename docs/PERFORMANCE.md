# TOPD throughput optimization and measurement

The previous approximately 900 seconds/optimizer step is user-observed. The
local machine exposes one RTX 3090, not two B200s; verification uses CPU PyTorch.
No B200 before/after seconds, tokens/second or memory measurements are available.
The B200 speedup is **unverified**. No full training was launched.

## What changed

| Phase | Previous implementation | Optimized implementation |
|---|---|---|
| Future model weights | FSDP forwards for every prefill/decode token of each four-request chunk | One recursive full-parameter gather per role/update; bypass all nested FSDP forwards within the inference-only context, then restore wrappers and reshard |
| Prefix compute | Recompute a long causal prefix for every candidate | Prefill once per selected original response; fork independent causal KV slices at each candidate position |
| Decode batching | Four requests at a time; filler generation on inactive ranks | Length-bucketed batches spanning original responses, live-VRAM/token bounds and a bounded source-KV pool; no decode collectives or filler samples |
| Sampling host overhead | GPU scalar read for every row/token | Accumulate token/mask tensors and transfer once per decode batch; retain per-request torch.multinomial generators |
| Trajectory backward | Fixed one-window microbatches | VRAM/token-bounded, length-bucketed batches; sparse batched KL with chunked FP32 vocabulary reductions and analytic backward |
| Gradient communication | Repeated gradient synchronization | Memory-guarded no_sync accumulation across original OPD and trajectory micros, followed by one synchronized backward per update |
| OT embeddings | Repeated root gathers and duplicate vector transfers in each tiny generation chunk | One root gather/update, transfer only new unique teacher IDs, cache immutable FP64-normalized CPU vectors |
| OT solver | Rebuild rectangular sparse constraints per window | Shape-cached constraints; parallel square assignment/cost work, serialized independent HiGHS calls; exact Hungarian/HiGHS objectives unchanged |
| Checkpoints | Every 150 steps | Complete immutable saves at 20, 40, 60, 80, … plus final; retain both names if final is an interval boundary |

The old decoder nested FSDP synchronization within many candidate chunks and
up to 50 decode iterations. This is a structural bottleneck visible in source,
not a measured allocation of the user's 900 seconds. Actual phase dominance
must be read from a B200 trace.

The main vLLM rollout engine and CUDA-IPC synchronization are unchanged. vLLM
was considered for futures, but replacing the torch sampler would change the
seed-to-sample mapping. Instead, future inference uses the current HF parameters
after every optimizer update, with no stale student replica or KV retained
across updates. Only normalized frozen teacher token embeddings persist.

## Mathematical and distributed contract

Candidate ranking/budget/ties, K=50, EOS and distinct-pad masking, seed formula,
independent teacher/student streams, uniform OT marginals, median acceptance,
duplicate-token target normalization, lambda=.01, mean-of-window-sums and the
original Top-16 OPD/PPO loss are unchanged. Main model initialization, data,
global prompt/trajectory/PPO sizes, AdamW settings and evaluation are unchanged.
Effective gradient normalization still uses global counts and the world-size
factor for FSDP/DDP rank-averaged reductions.

Gathering just the FSDP root and then calling its decoder would still execute
child FSDP forwards. The optimized context explicitly bypasses every wrapper's
forward while all parameters are materialized. It restores each method before
leaving the context; inference graphs never reach backward. All ranks enter
the same two role contexts, even with zero local candidates. Decode and safe
OOM retries are local. Backward rounds and the memory decision for no_sync
are agreed across ranks; empty ranks execute differentiable zero-loss fillers.

Scoring uses no_grad rather than inference_mode so reusable FSDP collective
buffers remain normal tensors when shared inference and training resume. The
score arithmetic remains byte-identical to the audited reference after this
decorator substitution.

Tests compare seeded future token sequences for greedy and stochastic top-p,
different adaptive batches and independent source prefixes; exact costs/plans;
FP32/BF16 sparse KL values/gradients; a full AdamW update; and two-process
FULL_SHARD forward/backward with a rank holding zero candidates. FP32 full-update
parameter tolerance is atol 3e-7 / rtol 3e-6. In the two-rank CPU test using the
production AdamW hyperparameters, observed FP32 relative gradient L2 error is
1.05e-7 and maximum updated-parameter error is 1.86e-9. BF16 relative gradient
error is .00367 (rounding from batch/accumulation layout), loss error 4.58e-7,
and maximum updated-parameter error 4.77e-7 at lr=5e-6. Tests explicitly bound
BF16 gradient error rather than claim bitwise equivalence. These measurements
do not establish B200 CUDA kernel equivalence. Floating-point kernels may
change rounding with batch shape; bitwise CUDA identity across different batch
layouts/hardware is not promised. Requests keep the same independent RNG seeds,
sampling distributions and candidate identity regardless of scheduling.

## Memory controls

Defaults are generation batch `auto` (cap 64, 131072 padded tokens), backward
batch `auto` (cap 32, 65536 padded tokens), 8 GiB headroom and workspace fraction
.65. The generation source-KV pool is bounded to .25 of estimated workspace.
The planner accounts for full current parameters, live allocator usage and
cached blocks. Automatic decisions only change execution microbatches.

Decode OOM retries halve the batch and reinitialize each request's original
generator; candidates are never dropped. A one-request failure raises. The
backward planner is conservative; arbitrary CUDA/workspace fragmentation or
other processes cannot be universally predicted. It does not attempt an unsafe
one-rank FSDP-backward retry. Lower TOPD_LOSS_MAX_BATCH_SIZE or token budget if
the hardware requires it. Both ranks use the same conservative backward cap.
No precision or context length is changed automatically.

## Profiling

`topd_metrics.jsonl` includes generation total, role-specific weight-gather time,
prefill tokens/calls, decode batches, chosen batch sizes and OOM retries;
embedding extraction, OT total, trajectory forward/backward total, original OPD
forward/backward, gradient clipping, optimizer, checkpoint time and peak allocated
and reserved GPU memory. Original main-rollout and scoring costs are amortized
over their PPO groups in `amortized_step_time_sec`. Device allocator peaks are
reset per update. Other-process/device totals remain in the common cost logs.

CUDA events time OPD micros without host synchronization after every backward.
For actual NCCL kernel timings and CPU launch/idle gaps, explicitly set
`TOPD_PROFILE_STEPS=2,3`. Per-rank Chrome traces go under `profiles/`.
`communication_gpu_time_sec` is a sum of NCCL kernel durations (may overlap other
work); unprofiled updates report null, never an invented zero. Profiler runs
have instrumentation overhead. Phase totals are inclusive and should not all
be added (embedding extraction is part of OT; gathers are part of generation).

## Bounded B200 benchmark (run only when requested)

From `TOPD/`, after installing dependencies, use the same real model and MATH
paths as training. One main rollout supplies the default one warmup plus three
measured updates, with exactly the production global batch and 50-token futures.
No scientific completed-epoch claim or production checkpoint is made.

```bash
export CUDA_VISIBLE_DEVICES=0,1 TRAIN_SEED=42
export PYTHON_BIN="$(command -v python3)"
bash scripts/benchmark_b200.sh --output results/throughput_probe_01 \
  --mode reference --warmup 1 --steps 3
bash scripts/benchmark_b200.sh --output results/throughput_probe_01 \
  --mode optimized --warmup 1 --steps 3
```

Both modes restart from the original model weights and deterministic data order.
`reference` retains the previous small-batch HF/FSDP generation and scalar KL
backward, under the corrected current infrastructure; it is not a byte-for-byte
old-run timing. Compare `reference/benchmark.json` with `optimized/benchmark.json`.
They contain update time, main-rollout/scoring amortization, phase times, token
and trajectory throughput, and memory. Use a separate output with `--profile`
for an operator trace of the first measured update. GPU count/type, output
immutability and a hard bound of at most 12 total updates are checked. This
script is never called automatically by training.

Remaining possible bottlenecks: exact long-prefix gradients still require
student computation, full-vocabulary LM heads, Python HF decode/sampling kernel
launches, rectangular CPU LPs, rank imbalance and shared-storage checkpoint I/O.
Measure these on B200 before attributing a remaining slowdown or changing kernels.

## Start a new complete training run

```bash
unset RESUME_FROM_CHECKPOINT RESUME BASELINE_CONFIG
export CUDA_VISIBLE_DEVICES=0,1 TRAIN_SEED=42 SAVE_INTERVAL=20
export RUN_NAME=topd_b200_fast_seed42
export PYTHON_BIN="$(command -v python3)"
bash scripts/train_compe_math_4b_17b_1epoch.sh
```

The resolved YAML contains `training.save_interval: 20`; new checkpoints contain
student weights/tokenizer, AdamW state, per-rank torch/CUDA/Python/NumPy RNG,
resolved config, training-state metadata, and any pending frozen main rollout.
Do not reuse the failed/slow old run directory. Evaluation stays separate:
`bash scripts/eval_multiseed_b200.sh` evaluates final with seeds 42/43/44.
