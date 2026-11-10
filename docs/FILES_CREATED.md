# Exact implementation inventory

All paths below are relative to `TOPD/`. The original project was not modified.
Temporary dependency environments are outside this folder. Runtime tiny smoke
artifacts are listed by `outputs/smoke/smoke_result.json`; full model/data caches
were not copied.

## Copied source modules

49 Python modules preserve the original self-contained package and its lazy
infrastructure imports. Exactly three copied modules are adapted:

- `b200_experiment/evaluation.py`
- `b200_experiment/resume.py`
- `b200_experiment/trainer.py`

All copied module paths:

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
- `scripts/compare_with_baseline.sh`
- `scripts/eval_compe_math_4b_17b_1epoch.sh`
- `scripts/smoke_test.sh`
- `scripts/train_compe_math_4b_17b_1epoch.sh`
- `tests/conftest.py`
- `tests/test_algorithm.py`
- `tests/test_distributed_topd.py`
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
- `topd/engine.py`
- `topd/reporting.py`
- `topd/smoke.py`

`configs/qwen3_b200_base.yaml` and `configs/qwen3_b200_cmt.yaml` are unchanged
configuration copies. The active overlay is `configs/topd_compe_math.yaml`.
Reference tests `test_reference_opd_core.py` and `test_reference_scoring.py` are
unchanged copies; all other TOPD tests and modules are new.
