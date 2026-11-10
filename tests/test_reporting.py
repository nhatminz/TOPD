import copy
import json
import math

import pytest

from b200_experiment.config import load_config, save_config
from topd.configuration import ROOT
from topd.reporting import aggregate_training_seeds, compare_runs


def fixture_run(root, seed, value, method="topd"):
    root.mkdir()
    c = load_config(ROOT/"configs/topd_compe_math.yaml")
    c["experiment"].update(seed=seed, method=method)
    c["rollout"]["seed"] = seed
    c["topd"]["future_seed"] = 1000000+seed
    save_config(c, root/"resolved_config.yaml")
    (root/"topd_summary.json").write_text(json.dumps({"completed_epochs": 1, "epoch_coverage": 1.}))
    (root/"final").mkdir()
    (root/"final/config.json").write_text("{}")
    (root/"evaluation").mkdir()
    (root/"evaluation/summary.json").write_text(json.dumps({
        "parameters": {"seed": 1234, "num_responses": 8},
        "benchmarks": {"Synthetic-fixture": {"avg_at_8": value, "pass_at_8": value}}}))


def test_aggregate_only_distinct_completed_training_seeds(tmp_path):
    a, b = tmp_path/"a", tmp_path/"b"
    fixture_run(a, 42, .1)
    fixture_run(b, 43, .3)
    result = aggregate_training_seeds([a, b], tmp_path/"aggregate.json")
    score = result["benchmarks"]["Synthetic-fixture"]["avg_at_8"]
    assert score["mean"] == pytest.approx(.2)
    assert score["sample_std"] == pytest.approx(math.sqrt(.02))
    with pytest.raises(ValueError, match="Repeated training seed"):
        aggregate_training_seeds([a, a], tmp_path/"bad.json")
    with pytest.raises(ValueError, match="at least two"):
        aggregate_training_seeds([a], tmp_path/"bad.json")


def test_compare_summary_and_mismatch_are_explicit(tmp_path):
    a, b = tmp_path/"topd", tmp_path/"cmt"
    fixture_run(a, 42, .3)
    fixture_run(b, 42, .1, "cmt")
    result = compare_runs(a, b, tmp_path/"comparison.json")
    assert result["matched"]
    assert result["benchmark_comparison"][0]["delta_TOPD_minus_CMT"] == pytest.approx(.2)
    assert (tmp_path/"comparison.md").is_file()
    c = load_config(b/"resolved_config.yaml")
    c["training"]["epochs"] = 3
    save_config(c, b/"resolved_config.yaml")
    assert not compare_runs(a, b, tmp_path/"mismatched.json")["matched"]
