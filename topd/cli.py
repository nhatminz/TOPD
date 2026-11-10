from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

from b200_experiment.config import load_config, resolve_runtime_paths, save_config
from .configuration import ROOT, b200_run_paths, launch_config, resolve_official_models, sha256_file, validate_config
from .reporting import aggregate_training_seeds, compare_runs, comparison_manifest


def parser_selfcheck():
    from b200_experiment.evaluation import _grade, evaluation_problem_scores
    import math_verify  # Fail here instead of silently selecting basic equality.
    cases = [(r"The answer is \boxed{2}.", "2", True),
             (r"\boxed{\frac{1}{2}}", "0.5", True),
             (r"\boxed{\frac{2}{4}}", r"\frac{1}{2}", True),
             (r"\boxed{3}", "2", False)]
    failures = [(prediction, gold) for prediction, gold, expected in cases
                if _grade(prediction, gold) != expected]
    if failures:
        raise RuntimeError(f"Shared mathematical answer parser failed sanity checks: {failures}")
    scores = evaluation_problem_scores([True]+[False]*7, "avg@8")
    if scores["avg@8"] != .125 or scores["pass@8"] != 1.:
        raise AssertionError("Shared Avg@8/Pass@8 semantics changed")
    return dict(status="passed", parser="unchanged math-verify", examples=len(cases))


def data_audit(config):
    from b200_experiment.data import (
        read_records, filter_overlong_prompt_records, epoch_batch_indices,
        record_messages, render_record_prompt, validate_prompt_records,
    )
    from b200_experiment.models import load_student_tokenizer
    tokenizer = load_student_tokenizer(config)
    records, files = read_records(config["data"]["path"], config["data"].get("split"))
    validate_prompt_records(records, config["data"])
    original_count = len(records)
    records, filtering = filter_overlong_prompt_records(records, tokenizer, config["data"])
    if not records:
        raise ValueError("No training examples remain after prompt filtering")
    n, batch, responses = len(records), config["rollout"]["batch_size"], config["rollout"]["num_responses"]
    order = [index for step in range(math.ceil(n/batch))
             for index in epoch_batch_indices(n, batch, step, config["experiment"]["seed"])]
    if sorted(order) != list(range(n)):
        raise AssertionError("Sampler failed exact one-epoch coverage")
    prompt_digest = hashlib.sha256()
    for begin in range(0, n, 256):
        rendered = [render_record_prompt(row, tokenizer, config["data"]) for row in records[begin:begin+256]]
        encoded = tokenizer(rendered, add_special_tokens=False, truncation=False, padding=False)
        for token_ids in encoded["input_ids"]:
            prompt_digest.update((json.dumps(token_ids, separators=(",", ":"))+"\n").encode())
    # Compare rendered question content, never gold/reference solutions.
    from b200_experiment.evaluation import load_benchmark
    def question_text(row, key):
        return " ".join(record_messages(row, prompt_key=key)[0]["content"].split())
    training_questions = {question_text(row, config["data"]["prompt_key"]) for row in records}
    overlaps = {}
    for name, spec in config["evaluation"]["benchmarks"].items():
        if name == "GPQA-Diamond":
            continue
        eval_rows, _schema = load_benchmark(name, spec)
        # load_benchmark canonicalizes all datasets into problem/answer keys.
        questions = {question_text(row, "problem") for row in eval_rows}
        overlaps[name] = len(training_questions & questions)
    if any(overlaps.values()):
        raise ValueError(f"Train/evaluation problem overlap detected: {overlaps}")
    return dict(original_examples=original_count, examples=n, prompt_filter=filtering,
        original_rollouts=n*responses,
        optimizer_updates=sum(math.ceil(len(epoch_batch_indices(n,batch,i,config["experiment"]["seed"]))*responses/
            config["training"]["ppo_mini_batch_size"]) for i in range(math.ceil(n/batch))),
        order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        rendered_prompt_ids_sha256=prompt_digest.hexdigest(),
        data_files={str(path): sha256_file(path) for path in files},
        train_eval_question_overlap=overlaps, intended_epochs=1)


def prepare(args):
    config = launch_config()
    if args.dry_run:
        save_config(config, args.output)
        print(json.dumps({"status": "configuration_only", "config": str(args.output),
            "output": config["experiment"]["output_dir"], "assets_verified": False,
            "teacher": config["models"]["teacher_path"],
            "student": config["models"]["student_path"],
            "training_data": config["data"]["path"]}, indent=2))
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA devices visible. Use --dry-run for configuration checks or scripts/smoke_test.sh for CPU tests.")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if mask and len(mask.split(",")) != torch.cuda.device_count():
        raise RuntimeError(f"Requested CUDA_VISIBLE_DEVICES={mask}, but PyTorch sees {torch.cuda.device_count()} GPUs")
    prior_provenance = config.get("model_provenance")
    provenance = resolve_official_models(config,
        verify_official=bool(config["topd"].get("verify_official_weights", False)))
    if config["training"].get("resume_from_checkpoint") and prior_provenance != provenance:
        raise ValueError("Resume would change verified model/tokenizer/config bytes")
    config["model_provenance"] = provenance
    # Local identity checks, tokenizer mapping and dependencies before any
    # vLLM/HF model loading, plus complete dataset coverage and leak checks.
    from b200_experiment.preflight import run_preflight
    output = Path(config["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = config["training"].get("resume_from_checkpoint")
    if not checkpoint and ((output/"resolved_config.yaml").exists() or (output/"metrics.jsonl").exists()):
        raise FileExistsError(f"Existing training run {output}; use RESUME_FROM_CHECKPOINT=auto or another OUTPUT_DIR")
    previous_report = None
    if checkpoint:
        if not (output/"preflight.json").is_file():
            raise FileNotFoundError("Resume requires original preflight.json to verify unchanged dataset bytes")
        previous_report = json.loads((output/"preflight.json").read_text())
    report_path = output/("preflight.resume.json" if checkpoint else "preflight.json")
    report = run_preflight(config, report_path)
    report["math_parser"] = parser_selfcheck()
    report["dataset_coverage"] = data_audit(config)
    if previous_report and previous_report["dataset_coverage"] != report["dataset_coverage"]:
        raise ValueError("Resume would change data bytes, rendered prompt IDs, filtering or sample order")
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    baseline_config = os.environ.get("BASELINE_CONFIG")
    if baseline_config:
        reference = resolve_runtime_paths(load_config(baseline_config))
        if reference["experiment"]["method"] != "cmt":
            raise ValueError("BASELINE_CONFIG must identify a CMT comparison run")
        manifest = comparison_manifest(config, reference, reference_path=baseline_config, historical=True)
        if not manifest["matched"]:
            different = [row["setting"] for row in manifest["settings"] if not row["identical"]]
            raise ValueError(f"Unfair comparison with BASELINE_CONFIG; differences: {different}")
    else:
        # Concrete configuration to use when rerunning CMT. Do not claim an
        # unspecified historical baseline matches the requested model pair.
        reference = copy.deepcopy(config)
        reference["experiment"].update(method="cmt", output_dir=str(output.parent/f"cmt_matched_seed{config['experiment']['seed']}"))
        reference["topd"]["enabled"] = False
        path = output/"cmt_matched_config.yaml"
        save_config(reference, path)
        manifest = comparison_manifest(config, reference, reference_path=path, historical=False)
    (output/"comparison_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    save_config(config, args.output)
    print(json.dumps({"status": "passed", "gpus": torch.cuda.device_count(),
        "config": str(args.output), "output": str(output),
        "coverage": report["dataset_coverage"]}, indent=2))


def evaluate(args):
    parser_selfcheck()
    paths = b200_run_paths()
    run = Path(args.run or paths["run_output"]).resolve()
    config = resolve_runtime_paths(load_config(run/"resolved_config.yaml"))
    if config["experiment"]["method"] != "topd":
        raise ValueError("Evaluation expects a TOPD run's resolved_config.yaml")
    validate_config(config)
    model = Path(args.model).resolve() if args.model else (
        Path(paths["checkpoint"]) if os.environ.get("TOPD_CHECKPOINT") else run/"final")
    if not (model/"config.json").is_file():
        raise FileNotFoundError(f"No completed HF checkpoint at {model}")
    if args.seed is not None:
        config["evaluation"]["vllm"]["seed"] = args.seed
    # Call the original evaluator orchestration, with its actual argparse
    # namespace and original decoding/parser/metric implementation.
    from b200_experiment.cli import _evaluate_checkpoint
    resolved = run/f".topd_eval_seed{config['evaluation']['vllm']['seed']}.yaml"
    save_config(config, resolved)
    namespace = argparse.Namespace(config=str(resolved), overlay=[], overrides=[],
        name="TOPD", model=str(model), output=args.output or paths["eval_output"],
        benchmarks=None, history_run_output=None)
    try:
        result = _evaluate_checkpoint(namespace)
        (run/"evaluation_location.json").write_text(json.dumps({
            "summary": str(Path(namespace.output).resolve()/"summary.json"),
            "checkpoint": str(model),
        }, indent=2)+"\n")
        print(json.dumps(result, indent=2))
    finally:
        resolved.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="Independent TOPD reproduction")
    commands = p.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument("--dry-run", action="store_true")
    paths = commands.add_parser("paths", help="Resolve B200 run/result paths without accessing assets")
    paths.add_argument("--key", choices=list(b200_run_paths()))
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--run")
    evaluation.add_argument("--model")
    evaluation.add_argument("--output")
    evaluation.add_argument("--seed", type=int)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--topd", required=True)
    comparison.add_argument("--baseline", required=True)
    comparison.add_argument("--output", required=True)
    aggregate = commands.add_parser("aggregate-seeds")
    aggregate.add_argument("--runs", nargs="+", required=True)
    aggregate.add_argument("--output", required=True)
    commands.add_parser("parser-selfcheck")
    args = p.parse_args(argv)
    if args.command == "paths":
        resolved = b200_run_paths()
        print(resolved[args.key] if args.key else json.dumps(resolved, indent=2))
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        from b200_experiment.trainer import run_training
        config = resolve_runtime_paths(load_config(args.config))
        validate_config(config)
        # All normal launches go through prepare, which verifies weights,
        # dependencies, tokenization, leakage and source configuration.
        if not config.get("model_provenance"):
            raise ValueError("Run scripts/train_compe_math_4b_17b_1epoch.sh to prepare verified assets first")
        run_training(config)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "compare":
        result = compare_runs(args.topd, args.baseline, args.output)
        print(json.dumps({"output": args.output, "matched": result["matched"]}))
        return 0 if result["matched"] else 2
    elif args.command == "aggregate-seeds":
        print(json.dumps(aggregate_training_seeds(args.runs, args.output), indent=2))
    else:
        print(json.dumps(parser_selfcheck()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
