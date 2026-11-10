# vLLM startup failure and environment consistency

The first reported log ended with `Engine core initialization failed` on both ranks.
This is vLLM's API-server wrapper error. The follow-up engine log identifies
the actual import failure: `flashinfer-cubin==0.6.6` versus
`flashinfer-python==0.6.4`. FlashInfer rejects that mismatch while vLLM is
importing its attention backends, before any rollout can run.
The original terminal exception printed only the last 40 server-log lines,
which discarded the earlier engine traceback. The OMP_NUM_THREADS notice is
torchrun's normal default and does not explain this failure.

The matching-version requirement is explicit in
[FlashInfer v0.6.4](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.4/flashinfer/jit/env.py).
[vLLM v0.17.1 CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.17.1/requirements/cuda.txt)
pin `flashinfer-python==0.6.4`; `flashinfer-cubin==0.6.4` is
[published on PyPI](https://pypi.org/project/flashinfer-cubin/0.6.4/).

From the TOPD directory on B200, with the intended training venv active:

```bash
PYTHON_BIN="$(command -v python3)" bash scripts/fix_flashinfer_b200.sh
```

The helper prints the selected Python/package versions, replaces only the
cubin wheel with the installed flashinfer-python version using `--no-deps`,
and verifies an actual FlashInfer import with its version guard enabled.
For this log, the repair command is equivalent to:

```bash
python3 -m pip install --no-deps --force-reinstall flashinfer-cubin==0.6.4
```

`--dry-run` prints the repair plan without installing or importing FlashInfer.
No package is installed automatically by train/eval. Their metadata checks
now reject this mismatch before starting the GPU engine. Missing optional
cubin/JIT-cache wheels are permitted by the check; an installed JIT cache may
have its upstream-supported CUDA version suffix.

The trace also shows torchrun imported from `TA-OPD-B200/.venv`, while workers
and the API server used `BellmanOPD/.venv`. Previously, inherited PYTHON_EXEC
could override the launcher Python and a `vllm` executable on PATH could select
another venv through its shebang. These are now controlled:

- `common_b200.sh` resolves PYTHON_BIN to its sys.executable and sets PYTHON_EXEC
  to that value before starting torchrun. It does not resolve venv symlinks.
- The rollout server invokes the installed `vllm` console entry point through
  the trainer's sys.executable. It does not search PATH for a vllm executable.
- The independent rollout child keeps the existing removal of torchrun
  rendezvous variables and gets the same PYTHON_EXEC.
- Startup exceptions include earlier engine errors plus the last 100 log lines,
  restricted to the current attempt in append-only logs. Full logs and startup
  JSON remain under the run directory.

After repairing dependencies, run from the workspace parent and retry in a new run directory:

```bash
CUDA_VISIBLE_DEVICES=0,1 TRAIN_SEED=42 \
PYTHON_BIN="$(command -v python3)" \
RUN_NAME=topd_compe_math_4b_17b_seed42_retry1 \
ROLLOUT_VLLM_LOGGING_LEVEL=INFO \
bash TOPD/scripts/train_compe_math_4b_17b_1epoch.sh
```

Use a new RUN_NAME because the failed setup may already have written a resolved
config. Do not delete the failed run's logs. There is no resumable training
checkpoint in the supplied setup-stage trace. If a later run has a checkpoint,
use the documented resume flow instead.

If startup still fails, inspect the engine error in:

```bash
tail -n 250 TOPD/outputs/topd_compe_math_4b_17b_seed42_retry1/topd/vllm_rollout_server.rank-00000.log
```

Changing startup_timeout does not repair this version mismatch. The segfault
appears during cleanup after the import failure; fix the explicit package
mismatch first, then investigate further only if a fresh run still fails.
The repair/check orchestration is CPU-tested with controlled package metadata;
actual wheel replacement and B200 engine startup must run on the B200 server.

Use the same RUN_NAME to evaluate that retry:

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_NAME=topd_compe_math_4b_17b_seed42_retry1 \
bash TOPD/scripts/eval_multiseed_b200.sh
```

This evaluates final with sampling seeds 42, 43, 44 on all six datasets. It
does not retrain the model.
