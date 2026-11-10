# Validation results — 2026-11-10

**98 tests passed, 0 failed (15.26 seconds)**. Machine-readable evidence:
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

The B200 train preview correctly resolved the shared storage asset paths, one epoch,
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

The verification session uses CPU PyTorch and synthetic fixtures. Production
configuration now requires B200 and existing local model/data assets from the
same storage paths as BellmanOPD_analysis. Full-size BF16/FSDP/NCCL, CUDA IPC,
teacher root embedding gathers under FSDP, B200 memory/throughput, real data
counts and benchmark accuracy were not tested. These require the supplied
launch command on the actual training server. The implementation is not
represented as an exact author-code reproduction; paper ambiguities and the
shared OPD normalization limitation remain documented.

All **67** audited source/config/document files in `BellmanOPD_analysis` retain
their recorded SHA256. Its Git working tree remains unchanged.

## B200 path follow-up validation

Nine added tests cover identical shared data/eval paths, original B200 environment
alias precedence, absolute paths, relocated checkouts, method-specific run/result
layout and offline local checkpoint handling. An additional test verifies that
comparison discovers standalone evaluation under `results/<run>/eval/topd`.
Default preparation never queries HF or downloads models; optional official
metadata checks preserve local paths. No production training was launched.

The user confirmed `models/Qwen3-4B` and `models/Qwen3-1.7B`. Default-path tests
check both absolute paths under `/workspace/storage-shared`; model identities
remain 4B/non-Base. The historical CMT 8B/Base defaults are not selected.

## Startup environment and evaluation-seed follow-up

Nine additional tests exercise the reported mixed-venv startup path and the
new standalone multiseed wrapper. The complete suite now has 88 passing tests.

- Two real torchrun workers launched with an invalid inherited PYTHON_EXEC both
  used the explicitly selected verification interpreter after sourcing the helper.
- The rollout command executed an installed console-entry-point fixture using
  the trainer's Python despite a foreign PATH, forwarding its serving arguments.
- Independent vLLM children keep the existing torchrun rendezvous isolation and
  matching GPU assignment; the child PYTHON_EXEC is pinned to the trainer.
- A synthetic engine error more than 250 lines before the wrapper traceback
  remains visible in the raised exception; stale causes from older attempts do not.
- The new script plans final/ with evaluation seeds 42/43/44 on all six datasets;
  original CLI run/seed overrides work. A synthetic three-seed evaluation verifies
  both metrics, sample std, preservation of the resolved config and result reuse.

The first production trace lacked the engine-worker root cause. These checks
verify the environment correction and diagnostics, not successful vLLM startup
on B200. No production rollout or real benchmark evaluation was run locally.

## Confirmed FlashInfer version mismatch

The follow-up B200 log establishes that flashinfer-cubin 0.6.6 and
flashinfer-python 0.6.4 fail the upstream import version check. Ten more tests
verify early rejection even with the bypass environment variable set, matching
versions and optional CUDA-suffixed JIT-cache metadata, targeted repair command
selection, no subprocess execution during repair dry-run, an unchanged correct
installation, and mandatory import verification after repair. The preparation
test fails before model access; a real shell dry-run reads controlled package
metadata with the selected interpreter. The suite now has 98 passing tests.

No FlashInfer wheel was installed or replaced in the CPU verification environment.
The B200 repair command is supplied in `B200_STARTUP.md`; successful production
startup remains to be verified on that server after replacing the mismatched wheel.
