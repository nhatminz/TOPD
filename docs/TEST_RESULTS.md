# Validation results — 2026-11-10

**69 tests passed, 0 failed (7.77 seconds)**. Machine-readable evidence:
[`test_results.xml`](test_results.xml), [`test_results.json`](test_results.json).
No full training or benchmark accuracy evaluation was started.

## Executed commands

In an isolated verification environment with the package versions recorded in
`test_results.json`:

```bash
PYTHON_BIN=/path/to/verification/python \
  bash TOPD/scripts/smoke_test.sh --junitxml=docs/test_results.xml

CUDA_VISIBLE_DEVICES=0,1 PYTHON_BIN=/path/to/verification/python \
  bash TOPD/scripts/train_compe_math_4b_17b_1epoch.sh --dry-run

PYTHON_BIN=/path/to/verification/python \
  bash TOPD/scripts/eval_compe_math_4b_17b_1epoch.sh --help

for script in TOPD/scripts/*.sh; do bash -n "$script"; done
```

The train preview correctly resolved teacher/student identities, one epoch,
seed 42, n=4, response 4096, PPO=64, lr=5e-6 and two-GPU vLLM concurrency 128.
The configuration-only preview is not a successful production preflight.

## What was actually exercised

- Valid-token candidate ranking, budgets, thresholds, deterministic ties and
  nonfinite rejection; prompt/pad exclusion and candidate-token prefix indexing.
- A manually computed 2×2 OT permutation with zero optimal cost and known
  targets; exact/sinkhorn rectangular marginals, nonnegative finite plans,
  orientation, row normalization and duplicate teacher IDs.
- Exact forward-KL value and `p - target` logit gradient; detached targets,
  masked gradients, own-student-prefix contexts and additive overlapping windows.
- Empty candidates preserve OPD; forced OT failures raise; EOS truncation and
  variable-length continuations; no silent ordinary-OPD fallback.
- Primary non-Base identity, official weight-shard hash verification using
  controlled fixtures, one-epoch guard, dataset/question leakage detection,
  frozen source semantics, math parser sanity, Avg@8 and Pass@8.
- Two real CPU processes using Gloo/DDP, with accepted window counts [2,0]:
  gradients equal a single-process reference within atol 2e-6/rtol 2e-5 and
  are identical on both ranks. Also tested a dataset tail with real data on
  only one rank through common OPD, fresh generation, OT and trajectory update;
  filler rows contributed no scientific samples or loss.
- The complete copied `run_training` orchestration using tiny Qwen3 models;
  only model/asset loading and CUDA hardware calls were substituted for CPU.
- Reference OPD/scoring regression tests copied from the source project.
- Comparison JSON/Markdown output and sample-standard-deviation aggregation
  operate on synthetic fixtures and reject mismatched/repeated runs. Those
  fixture scores are not experimental benchmark results.

## Offline integration smoke result

Three synthetic examples, two original rollouts/example, PPO group size two,
50-token maximum future, FP32 tiny random Qwen3 teacher/student:

| Measurement | Observed |
|---|---:|
| Original student rollouts | 6 |
| Actual optimizer updates | 3 |
| Extra teacher/student continuations | 6 / 6 |
| Teacher/student generated future tokens | 96 / 225 |
| Trajectory-loss tokens | 225 |
| Accepted OT candidates | 6 |
| Coverage / repeated / missing pairs | 1.0 / 0 / 0 |

The smoke intentionally uses an absolute positive-distance acceptance threshold
and a one-candidate budget to exercise trajectory loss on tiny data. It is
separate from the primary 2%-candidate/median-OT production configuration.
Student parameters changed; teacher parameters remained unchanged with no
teacher gradients. HF config/tokenizer/weights and optimizer/RNG were saved and
reloaded. Resume from the first update inside an unconsumed rollout matched
uninterrupted parameters after the next update within atol 1e-7/rtol 1e-6.
Adam state is restored by the same checkpoint loader. Completed checkpoints at
rollout boundaries do not store unnecessary frozen-rollout sidecars.

The final smoke JSON is `outputs/smoke/smoke_result.json`. Its phase times and
zero GPU-memory values are CPU engineering measurements, not training cost for
4B/1.7B or a claimed overhead ratio.

## What remains unverified

There is only one physical RTX 3090 here, with no `/workspace/storage-shared`
Competition-MATH/model assets and no compatible local vLLM production runtime.
The verification interpreter uses CPU PyTorch. Full-size BF16/FSDP/NCCL, CUDA IPC,
teacher root embedding gathers under FSDP, B200 memory/throughput, real data
counts and benchmark accuracy were not tested. These require the supplied
launch command on the actual training server. The implementation is not
represented as an exact author-code reproduction; paper ambiguities and the
shared OPD normalization limitation remain documented.

All **67** audited source/config/document files in `BellmanOPD_analysis` retain
their recorded SHA256. Its Git working tree remains unchanged.
