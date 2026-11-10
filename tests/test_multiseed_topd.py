import argparse
import copy
import json
import os
import statistics
import subprocess
import sys

import pytest

from b200_experiment.config import load_config, save_config
from b200_experiment.multiseed_evaluation import BENCHMARKS, execute
from topd.configuration import ROOT


@pytest.fixture
def run(tmp_path):
    run = tmp_path/"topd"
    (run/"final").mkdir(parents=True)
    (run/"final/config.json").write_text('{"max_position_embeddings":40960}')
    (run/"latest.json").write_text(json.dumps({"checkpoint": "final", "step": 750}))
    config = load_config(ROOT/"configs/topd_compe_math.yaml")
    config["experiment"]["output_dir"] = str(run)
    save_config(config, run/"resolved_config.yaml")
    return run


def test_multiseed_wrapper_plans_final_six_datasets_and_three_fresh_seeds(run):
    environment = dict(os.environ, PYTHON_BIN=sys.executable, PYTHON_EXEC="/foreign/python",
                       TOPD_RUN_DIR=str(run))
    result = subprocess.run(["bash", str(ROOT/"scripts/eval_multiseed_b200.sh"), "--dry-run"],
                            env=environment, text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["new_seeds"] == [42, 43, 44]
    assert len(plan["plan"]) == 1
    item = plan["plan"][0]
    assert item["checkpoint"] == str(run/"final")
    assert item["baseline"] == {}
    assert item["settings"]["benchmark_names"] == list(BENCHMARKS)
    assert item["settings"]["num_responses"] == 8


def test_multiseed_wrapper_accepts_original_cli_overrides(run):
    environment = dict(os.environ, PYTHON_BIN=sys.executable, TOPD_RUN_DIR="/unused/default")
    result = subprocess.run(["bash", str(ROOT/"scripts/eval_multiseed_b200.sh"),
                             "--run-output", str(run), "--seeds", "1234", "43", "44", "--dry-run"],
                            env=environment, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["new_seeds"] == [1234, 43, 44]


def test_topd_fresh_seeds_compute_both_metrics_and_resume_committed_results(run):
    original = (run/"resolved_config.yaml").read_bytes()
    args = argparse.Namespace(run_output=run, checkpoints=["final"], seeds=[42, 43, 44],
        fresh_seeds=True, baseline_seed=None, tag="seeds_42_43_44", dry_run=False,
        world_size=2, max_num_seqs=None, max_model_len=None, gpu_memory_utilization=None)
    calls = []

    def evaluate(name, checkpoint, config, destination, settings, resolved, world_size):
        calls.append(copy.deepcopy(settings))
        seed = settings["vllm"]["seed"]
        assert world_size == 2 and resolved.is_file()
        assert config["experiment"]["method"] == "topd"
        return {"benchmarks": {benchmark: {
            "avg_at_8": {42: .25, 43: .5, 44: .75}[seed], "pass_at_8": 1.,
            "problems": 2, "samples_per_problem": 8} for benchmark in BENCHMARKS},
            "parameters": {key: settings[key] for key in
                ("temperature", "top_p", "num_responses", "max_new_tokens")} | {"seed": seed}}

    summary = execute(args, evaluator=evaluate)
    assert len(calls) == 3
    assert len(summary["results"]) == 12
    for row in summary["results"]:
        assert row["n"] == 3 and row["seeds"] == [42, 43, 44]
        if row["metric"] == "avg@8":
            assert row["mean"] == .5
            assert row["std"] == statistics.stdev([.25, .5, .75])
    assert (run/"eval_summary_seeds_42_43_44.md").is_file()
    assert (run/"eval_summary_seeds_42_43_44.csv").is_file()
    assert (run/"resolved_config.yaml").read_bytes() == original
    assert execute(args, evaluator=lambda *a, **kw: pytest.fail("committed results must be reused")) == summary
