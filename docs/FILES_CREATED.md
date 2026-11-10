# Exact implementation inventory

Paths are relative to `TOPD/`. Original BellmanOPD_analysis files were not
modified. Git internals, runtime outputs/results and verification environments
are excluded.

## Adapted copied modules

- `b200_experiment/evaluation.py`
- `b200_experiment/resume.py`
- `b200_experiment/trainer.py`
- `b200_experiment/vllm_rollout.py`

## All copied modules

- `b200_experiment/__init__.py`
- `b200_experiment/alfworld_cli.py`
- `b200_experiment/alfworld_diagnostics.py`
- `b200_experiment/alfworld_env.py`
- `b200_experiment/alfworld_evaluation.py`
- `b200_experiment/alfworld_logprob_debug.py`
- `b200_experiment/alfworld_protocol.py`
- `b200_experiment/alfworld_rollout.py`
- `b200_experiment/autotune.py`
- `b200_experiment/checkpoint_evaluation.py`
- `b200_experiment/cli.py`
- `b200_experiment/config.py`
- `b200_experiment/data.py`
- `b200_experiment/diagnostics.py`
- `b200_experiment/distributed.py`
- `b200_experiment/eval_schedule.py`
- `b200_experiment/evaluation.py`
- `b200_experiment/evaluation_cache.py`
- `b200_experiment/evaluation_context.py`
- `b200_experiment/evaluation_history.py`
- `b200_experiment/fsdp.py`
- `b200_experiment/math_prompts.py`
- `b200_experiment/metadata.py`
- `b200_experiment/mmlu_pro.py`
- `b200_experiment/mmlu_pro_evaluation.py`
- `b200_experiment/models.py`
- `b200_experiment/multiseed_evaluation.py`
- `b200_experiment/opd_core.py`
- `b200_experiment/passk_experiment.py`
- `b200_experiment/passk_plotting.py`
- `b200_experiment/plotting.py`
- `b200_experiment/preflight.py`
- `b200_experiment/rebuild_evaluation_history.py`
- `b200_experiment/resume.py`
- `b200_experiment/scoring.py`
- `b200_experiment/selector_logging.py`
- `b200_experiment/selectors/__init__.py`
- `b200_experiment/selectors/base.py`
- `b200_experiment/selectors/budget.py`
- `b200_experiment/selectors/cmt_selector.py`
- `b200_experiment/selectors/opd_selector.py`
- `b200_experiment/selectors/pgt_selector.py`
- `b200_experiment/selectors/rac_selector.py`
- `b200_experiment/selectors/ta_selector.py`
- `b200_experiment/tensorboard_logging.py`
- `b200_experiment/trainer.py`
- `b200_experiment/training_cost.py`
- `b200_experiment/vllm_evaluation.py`
- `b200_experiment/vllm_rollout.py`

## Other files created

- `.gitignore`
- `README.md`
- `configs/qwen3_b200_base.yaml`
- `configs/qwen3_b200_cmt.yaml`
- `configs/topd_compe_math.yaml`
- `docs/B200_PATHS.md`
- `docs/B200_STARTUP.md`
- `docs/BASELINE_AUDIT.md`
- `docs/FAIR_COMPARISON.md`
- `docs/FILES_CREATED.md`
- `docs/PAPER_AMBIGUITIES.md`
- `docs/TEST_RESULTS.md`
- `docs/TOPD_METHOD.md`
- `docs/entrypoint_checks.json`
- `docs/implementation_manifest.json`
- `docs/source_manifest.json`
- `docs/test_results.json`
- `docs/test_results.xml`
- `requirements.txt`
- `scripts/common_b200.sh`
- `scripts/compare_with_baseline.sh`
- `scripts/eval_compe_math_4b_17b_1epoch.sh`
- `scripts/eval_multiseed_b200.sh`
- `scripts/fix_flashinfer_b200.sh`
- `scripts/smoke_test.sh`
- `scripts/train_compe_math_4b_17b_1epoch.sh`
- `tests/conftest.py`
- `tests/test_algorithm.py`
- `tests/test_b200_paths.py`
- `tests/test_b200_runtime.py`
- `tests/test_dependencies.py`
- `tests/test_distributed_topd.py`
- `tests/test_multiseed_topd.py`
- `tests/test_pipeline.py`
- `tests/test_preparation.py`
- `tests/test_reference_opd_core.py`
- `tests/test_reference_scoring.py`
- `tests/test_reporting.py`
- `topd/__init__.py`
- `topd/algorithm.py`
- `topd/checkpoint.py`
- `topd/cli.py`
- `topd/configuration.py`
- `topd/dependencies.py`
- `topd/engine.py`
- `topd/reporting.py`
- `topd/smoke.py`

The two qwen3_b200 config copies and two reference test copies remain unchanged.
The standalone B200 helper accepts the original model/data/path aliases and pins
torchrun to the selected Python. The rollout adaptation changes interpreter
selection and failure diagnostics; sampling and IPC flags are preserved.
The explicit FlashInfer repair script aligns the cubin wheel with the installed
flashinfer-python; train/eval check metadata without installing packages.
