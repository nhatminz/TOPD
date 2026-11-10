from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path

from b200_experiment.config import load_config, resolve_runtime_paths


SHARED_KEYS = (
    "models.student_path", "models.teacher_path", "models.dtype",
    "models.teacher_no_think", "data.path", "data.split", "data.prompt_key",
    "data.prefer_source_prompt", "data.chat_template_kwargs", "data.max_prompt_tokens",
    "data.overlong_prompt_policy", "experiment.seed", "rollout.seed", "rollout.backend",
    "rollout.batch_size", "rollout.num_responses", "rollout.max_new_tokens",
    "rollout.temperature", "rollout.top_p", "opd", "selector.top_k",
    "rollout.vllm",
    "training.epochs", "training.learning_rate", "training.adam_betas",
    "training.weight_decay", "training.ppo_mini_batch_size", "training.grad_accum_steps",
    "training.ppo_clip_low", "training.ppo_clip_high", "training.ppo_dual_clip",
    "training.max_grad_norm", "training.use_lora", "training.gradient_checkpointing",
    "distributed.strategy", "distributed.fsdp", "training_evaluation", "evaluation",
)


def get(config, dotted):
    value = config
    for key in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def comparison_manifest(current, reference, *, reference_path, historical=False):
    # Standalone final evaluation replaces training-time callbacks by request.
    # Record that schedule difference separately while comparing all decoding,
    # data and optimization settings as before.
    schedule_keys = ("enabled", "eval_at_start", "eval_at_end")
    def shared_value(config, key):
        value = get(config, key)
        if key == "training_evaluation" and isinstance(value, dict):
            return {name: item for name, item in value.items() if name not in schedule_keys}
        return value
    rows = [{"setting": key, "baseline": shared_value(reference, key), "topd": shared_value(current, key),
             "identical": shared_value(current, key) == shared_value(reference, key)} for key in SHARED_KEYS]
    return dict(reference_path=str(reference_path), historical_run_supplied=historical,
        matched=all(row["identical"] for row in rows), settings=rows,
        training_evaluation_schedule={name: {key: get(config, f"training_evaluation.{key}")
            for key in schedule_keys} for name, config in (("baseline", reference), ("topd", current))},
        match_scope="configuration equality excluding recorded training-evaluation enable/start/end flags; historical weight/data bytes need prior attestations",
        prior_initialization_attested=bool(historical and reference.get("model_provenance")),
        intentional_differences=["CMT Gibbs token weighting versus uniform OPD + TOPD trajectory guidance",
            "TOPD independently generated teacher/student futures and OT",
            "TOPD disables training-evaluation callbacks and evaluates final separately",
            "Additional compute is measured, not equalized"],
        update_partition="CMT response-index-major in both methods",
        schedule="constant learning rate; no scheduler in source",
        checkpoint_bytes_verified_against_official_models=all(
            current.get("model_provenance", {}).get(role, {}).get("verification") == "official_sha256"
            for role in ("student", "teacher")))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finalize_training(output, dataset_rows, num_responses, expected_steps, distributed):
    result = None
    if distributed.is_main:
        events = rows(Path(output)/"topd_metrics.jsonl")
        steps = [event["optimizer_step"] for event in events]
        if steps != list(range(1, expected_steps+1)):
            raise AssertionError("TOPD metrics do not cover every optimizer step exactly once")
        consumed = Counter(tuple(pair) for event in events for pair in event["consumed_example_responses"])
        expected = {(row, response) for row in range(dataset_rows) for response in range(num_responses)}
        if set(consumed) != expected or any(count != 1 for count in consumed.values()):
            raise AssertionError("One-epoch coverage failed: repeated/skipped example-response pairs")
        counts = {key: sum(event[key] for event in events) for key in (
            "original_student_rollouts", "extra_teacher_continuations", "extra_student_continuations",
            "extra_teacher_tokens", "extra_student_tokens", "trajectory_loss_tokens",
            "extra_generated_tokens_physical", "extra_teacher_continuations_physical",
            "extra_student_continuations_physical",
            "high_loss_candidates", "accepted_candidates", "rejected_candidates")}
        times = {key: sum(event[key] for event in events) for key in (
            "prepare_time_sec", "future_generation_time_sec", "ot_time_sec", "trajectory_update_time_sec")}
        result = dict(dataset_examples=dataset_rows, original_rollouts=len(consumed),
            optimizer_steps=expected_steps, epoch_coverage=1.0, completed_epochs=1,
            unique_example_response_pairs=len(consumed), repeated_pairs=0, missing_pairs=0,
            peak_gpu_memory_bytes=max(event["peak_gpu_memory_bytes"] for event in events),
            counts=counts, times=times)
        if (Path(output)/"training_sessions.jsonl").is_file():
            result["completed_process_sessions_wall_clock_seconds"] = sum(
                event["wall_clock_seconds"] for event in rows(Path(output)/"training_sessions.jsonl"))
        if (Path(output)/"metrics.jsonl").is_file():
            original = rows(Path(output)/"metrics.jsonl")
            # Each original event amortizes one rollout across its PPO updates.
            result["original_rollout_time_amortized_seconds"] = sum(
                event.get("rollout_time", 0)/max(event.get("optimizer_steps_in_rollout", 1), 1)
                for event in original)
        (Path(output)/"topd_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return distributed.broadcast_object(result)


def compare_runs(topd_output, baseline_output, destination):
    topd_output, baseline_output = Path(topd_output), Path(baseline_output)
    current = resolve_runtime_paths(load_config(topd_output/"resolved_config.yaml"))
    reference = resolve_runtime_paths(load_config(baseline_output/"resolved_config.yaml"))
    result = comparison_manifest(current, reference,
        reference_path=baseline_output/"resolved_config.yaml", historical=True)
    if reference["experiment"]["method"] != "cmt":
        raise ValueError("Primary comparison requires a CMT reference run")
    # Results are linked to training seeds, not treated as new training runs.
    result["training_seeds"] = {"TOPD": current["experiment"]["seed"], "CMT": reference["experiment"]["seed"]}
    for name, root in (("TOPD", topd_output), ("CMT", baseline_output)):
        result[name] = {}
        for file in ("summary.json", "topd_summary.json", "evaluation/summary.json", "training_cost_summary.json"):
            if (root/file).is_file():
                result[name][file] = json.loads((root/file).read_text())
    suites = {name: completed_evaluation(root) for name, root in
              (("TOPD", topd_output), ("CMT", baseline_output))}
    result["benchmark_comparison"] = []
    if all(suites.values()):
        for benchmark in suites["TOPD"]["benchmarks"]:
            if benchmark not in suites["CMT"]["benchmarks"]:
                continue
            for metric in ("avg_at_8", "pass_at_8"):
                values = {name: suite["benchmarks"][benchmark].get(metric)
                          for name, suite in suites.items()}
                if all(isinstance(value, (int, float)) for value in values.values()):
                    result["benchmark_comparison"].append(dict(benchmark=benchmark,
                        metric=metric, **values, delta_TOPD_minus_CMT=values["TOPD"]-values["CMT"]))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["# TOPD / CMT comparison", "",
        f"Configuration matched: {result['matched']}. Historical initialization attested: {result['prior_initialization_attested']}.",
        "", "| Benchmark | Metric | CMT | TOPD | Delta (percentage points) |",
        "|---|---|---:|---:|---:|"]
    for row in result["benchmark_comparison"]:
        lines.append(f"| {row['benchmark']} | {row['metric']} | {row['CMT']*100:.2f} | {row['TOPD']*100:.2f} | {row['delta_TOPD_minus_CMT']*100:+.2f} |")
    if not result["benchmark_comparison"]:
        lines.extend(["", "No paired completed evaluation was found; no accuracy comparison is reported."])
    destination.with_suffix(".md").write_text("\n".join(lines)+"\n")
    return result


def completed_evaluation(root):
    root = Path(root)
    location = root/"evaluation_location.json"
    if location.is_file():
        marker = json.loads(location.read_text())
        candidate = Path(marker["summary"])
        if Path(marker["checkpoint"]).resolve() == (root/"final").resolve() and candidate.is_file():
            return json.loads(candidate.read_text())
    candidate = root/"evaluation/summary.json"
    if candidate.is_file():
        return json.loads(candidate.read_text())
    if (root/"summary.json").is_file():
        step = json.loads((root/"summary.json").read_text())["steps"]
        candidate = root/"training_eval"/f"step-{step:06d}"/"summary.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())
    return None


def aggregate_training_seeds(run_dirs, output):
    if len(run_dirs) < 2:
        raise ValueError("Sample standard deviation requires at least two completed training seeds")
    seeds, suites, protocols, training_protocols = [], [], [], []
    for run in map(Path, run_dirs):
        config = load_config(run/"resolved_config.yaml")
        if config["experiment"]["method"] != "topd":
            raise ValueError("Only TOPD training seeds can be aggregated by this entrypoint")
        summary = json.loads((run/"topd_summary.json").read_text())
        if summary["completed_epochs"] != 1 or summary["epoch_coverage"] != 1:
            raise ValueError(f"Run is not a completed one-epoch TOPD run: {run}")
        if not (run/"final/config.json").is_file():
            raise FileNotFoundError(f"Run has no final checkpoint: {run}")
        suite = completed_evaluation(run)
        if suite is None:
            raise FileNotFoundError(f"Run has no completed evaluation: {run}")
        seeds.append(config["experiment"]["seed"])
        suites.append(suite)
        protocols.append({key: suite.get("parameters", {}).get(key) for key in
            ("backend", "seed", "temperature", "top_p", "num_responses", "metric", "max_new_tokens", "limit")})
        training_protocols.append({key: get(config, key) for key in SHARED_KEYS
            if key not in {"experiment.seed", "rollout.seed"}} | {
                "topd": {key: value for key, value in config["topd"].items() if key != "future_seed"}})
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated training seed is not an independent training run")
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("Evaluation protocols differ between training seeds")
    if any(protocol != training_protocols[0] for protocol in training_protocols[1:]):
        raise ValueError("Training protocols differ beyond random seeds")
    results = {}
    for benchmark in suites[0]["benchmarks"]:
        results[benchmark] = {}
        for metric in ("avg_at_8", "pass_at_8"):
            values = [suite["benchmarks"][benchmark][metric] for suite in suites]
            results[benchmark][metric] = dict(mean=statistics.mean(values),
                sample_std=statistics.stdev(values), values=values)
    result = dict(training_seeds=seeds, completed_runs=list(map(str, run_dirs)),
        standard_deviation="sample (ddof=1)", benchmarks=results)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result
