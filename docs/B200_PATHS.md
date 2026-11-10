# B200 path contract

Production targets the same B200 storage/mount and environment as
`BellmanOPD_analysis/scripts/common_b200.sh`. The copied
`b200_experiment/config.py::resolve_runtime_paths` is unchanged. There are no
workstation paths in runtime source/config defaults and no automatic downloads.

## Shared assets

`STORAGE_ROOT` defaults to `/workspace/storage-shared` in both projects. Relative
model/train/evaluation paths resolve under that mount; absolute paths retain
the original resolver behavior. Override precedence matches the original:

| Asset | Environment overrides, in precedence order |
|---|---|
| Teacher | `TEACHER_MODEL`, `TEACHER_MODEL_PATH`, `TEACHER_PATH` |
| Student/tokenizer | `STUDENT_MODEL`, `STUDENT_MODEL_PATH`, `STUDENT_PATH` |
| Training parquet | `TRAIN_DATA`, `TRAIN_DATA_PATH` |
| Prompt key | `PROMPT_KEY`, `TRAIN_PROMPT_KEY` |
| Split | `TRAIN_DATA_SPLIT` (`null` for the single train parquet) |

The Competition-MATH train/test, MATH-500, AIME24, AIME25, GPQA-Diamond and AMC23
paths are identical to `configs/qwen3_b200_base.yaml`. No dataset is copied,
downloaded, relocated, truncated or switched to a local development fixture.

The user confirmed these exact B200 model paths:

| Role | Relative path | Default absolute path |
|---|---|---|
| Teacher | `models/Qwen3-4B` | `/workspace/storage-shared/models/Qwen3-4B` |
| Student | `models/Qwen3-1.7B` | `/workspace/storage-shared/models/Qwen3-1.7B` |

Both TOPD and the intended CMT comparison use these checkpoints. The original
CMT launcher's historical 8B/Base defaults need explicit overrides for this
experiment. Configuration and local checkpoint bytes are attested.

Missing local checkpoints raise before any network request. The default
verification hashes local weights/config/tokenizer and checks architecture and
tokenizer compatibility. `VERIFY_OFFICIAL_WEIGHTS=true` additionally queries
official LFS metadata; it never downloads or redirects a model to HF cache.

## Method outputs

Same structure, with isolated method-specific destinations:

| Kind | Original CMT | TOPD |
|---|---|---|
| Run output | `$OUTPUT_ROOT/$RUN_NAME/cmt_opd` | `$OUTPUT_ROOT/$RUN_NAME/topd` |
| Final model | run output + `/final` | same |
| Checkpoints | run output + `/checkpoint-XXXXXX` | same |
| Periodic eval | run output + `/training_eval` | same |
| TensorBoard | run output + `/tensorboard` | same |
| Standalone eval | `$RESULTS_DIR/eval/cmt_opd` | `$RESULTS_DIR/eval/topd` |
| Default output root | project checkout + `/outputs` | same |
| Default results root | project checkout + `/results/$COMPARISON_NAME` | same |
| Launch YAML | B200 script-resolved config | `results/launch_configs/$RUN_NAME.yaml` |
| Base eval cache | `outputs/.base_eval_cache` | same relative setting |

`OUTPUT_DIR`/`TOPD_OUTPUT_DIR`/`TOPD_RUN_DIR`, `TOPD_CHECKPOINT` and
`TOPD_EVAL_OUTPUT` support explicit overrides. Outputs never default into the
sibling CMT checkout. `RESUME_FROM_CHECKPOINT` and `RESUME` are both accepted.

`python -m topd.cli paths` displays output/result locations. Train `--dry-run`
displays the resolved teacher, student and training parquet and writes the
reviewable B200 launch YAML. Moving the checkout changes project-relative
output paths; storage asset paths stay on the same shared mount.

Production requires B200, BF16 and the existing FSDP/vLLM infrastructure.
CPU synthetic smoke remains an isolated test mode, not a production preset.
Runtime `outputs/` and `results/` are ignored when pushing source to B200.
