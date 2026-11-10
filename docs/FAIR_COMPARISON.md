# Comparison contract

The current math CMT launcher is the default reference, with the user-required
model identities and duration applied to **both** methods. There is no matching
completed CMT run in this checkout. Historical 8B/Base/3-epoch results cannot be
claimed as the matched reference for the implementation's requested experiment.

| Setting | Current CMT launcher defaults | Primary TOPD | Required CMT comparison |
|---|---|---|---|
| Teacher | local Qwen3-8B | official Qwen/Qwen3-4B | same 4B bytes |
| Initial student | Qwen3-1.7B-Base | official Qwen/Qwen3-1.7B | same non-Base bytes |
| Competition-MATH epochs | 3 | exactly 1 | exactly 1 |
| Training data/order | local train parquet, seed 42, rendered-prompt filter | same | same |
| Template | student tokenizer, `enable_thinking=false` | same | same |
| Main rollout | 64 prompts × 4 responses; T=1, p=1 | same | same |
| Max prompt/response | 1024 / 4096 | same | same |
| PPO group | 64, response-index-major | same | same |
| Training micro / scoring micro | 16 / 8 per GPU | same | same |
| Accumulation | automatic over local PPO rows | same | same |
| Optimizer | AdamW 5e-6, (.9,.95), decay 0 | same | same |
| LR schedule | none; constant LR | same | same |
| BF16 / gradient checkpointing | enabled | same | same |
| Gradient clipping | global norm 1 after accumulation | same | same |
| OPD vocabulary support | student Top-16, pinned code | same | same |
| PPO clips | .2 / .28, dual clip 3 | same | same |
| Distributed | FSDP FULL_SHARD, one rank/GPU | same | same |
| Two-GPU vLLM request capacity | 128/rank, reservation .6, context 5200 | same derivation | same |
| Save/eval cadence | step 0 eval, every 150 updates, final | same | same |
| Eval | eight samples, T=.7, p=.95, length 7168 | same | same |
| Eval seed | 1234 default; runbook also exports 42 | 1234 unless actual config supplied | identical actual value |
| Token weighting | CMT bounded Gibbs | uniform original OPD + trajectory KL | method difference |
| Future guidance | CMT common-mass successor statistic | fresh teacher/student K=50, OT, transported targets | method difference |

## Use an actual comparison run

`BASELINE_CONFIG=/absolute/path/cmt_run/resolved_config.yaml` loads that run's
shared settings before adding TOPD. Mandatory teacher/student identities,
one-epoch duration and no-test-training constraints still apply. A mismatch
fails before training. Subsequent `comparison_manifest.json` records each shared
field and intentional differences. It distinguishes an actual historical config
from a generated config for a future matched CMT run.

The comparison checks configuration equality. An old run lacking weight/data
attestations cannot prove the bytes used historically just from its path.
TOPD records SHA256 of the existing local model/config/tokenizer and training
files. Optional VERIFY_OFFICIAL_WEIGHTS=true compares official metadata without
changing model paths. Keep initial checkpoint/data hashes for the CMT rerun as well.
`prior_initialization_attested` in the manifest makes missing prior evidence
visible; it is not silently inferred from a matching directory name.

Without an existing matched run, prepare writes `cmt_matched_config.yaml` in
the TOPD output directory. It has CMT's selector, the same inputs/settings and
an isolated CMT output. It can be run using the original project:

```bash
PYTHONPATH="$PWD/BellmanOPD_analysis" CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m b200_experiment.cli train \
  --config "$PWD/TOPD/outputs/topd_compe_math_4b_17b_seed42/topd/cmt_matched_config.yaml"
```

This config becomes available after TOPD asset preparation; the command is not
run automatically. Alternatively use the original launcher with explicit values:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_EPOCHS=1 SEED=42 ROLLOUT_SEED=42 \
TEACHER_MODEL=models/Qwen3-4B \
STUDENT_MODEL=models/Qwen3-1.7B \
STORAGE_ROOT=/workspace/storage-shared \
OUTPUT_DIR="$PWD/TOPD/outputs/cmt_matched_seed42" \
bash BellmanOPD_analysis/scripts/train_cmt_b200.sh
```

Keep a copy of the TOPD `model_provenance` and confirm those source files are the
same files used to initialize CMT. Do not compare trained CMT checkpoint weights
to fresh TOPD initialization.

## Dataset duration and compute

One epoch is a complete traversal of filtered train records according to
`epoch_batch_indices`. No `max_steps` shortcut is accepted by production TOPD.
If `N` rows remain and batch size is 64, each full batch gives four updates;
the final partial batch gives `ceil(4 * tail / 64)` updates. Runtime preflight
prints exact N, 4N original trajectories, expected updates, file hashes and
sampler-order hash. For illustration only, N=7500 would mean 30000 original
rollouts and 469 updates; actual filtering may change N. Final coverage is
verified from committed example/response IDs, including resumed runs.

TOPD's additional continuations are part of its method. Neither method receives
extra original rollouts to equalize compute. Logs distinguish valid scientific
requests from collective filler requests and include physical generated tokens,
phase times and memory. Do not transfer a published runtime ratio to these
models; no full training cost has been measured here. CPU/Gloo tests validate
reduction arithmetic, not B200 FSDP/vLLM runtime or memory capacity.

The shared math OPD implementation has a frozen/full versus current/conditional
logprob discrepancy. This baseline preserves it, as described in the audit.
A repaired direct reverse-KL experiment must repair both methods and carry a
separate name/config; no accuracy claim from that variant is made here.

`compare_with_baseline.sh` returns a machine-readable comparison and exits 2 on
configuration mismatch. It does not fabricate absent evaluations or multi-seed
statistics. Training seeds 42/43/44 are supported through `TRAIN_SEED`; report
mean and **sample** standard deviation only after all declared runs complete.
