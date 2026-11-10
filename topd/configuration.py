from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

from b200_experiment.config import load_config, resolve_runtime_paths

ROOT = Path(__file__).resolve().parents[1]


def validate_config(config):
    c, t = config, config["topd"]
    expected = {"teacher_source": "Qwen/Qwen3-4B", "student_source": "Qwen/Qwen3-1.7B"}
    for key, value in expected.items():
        if t.get(key) != value:
            raise ValueError(f"TOPD requires {key}={value}")
    for key in ("student_path", "teacher_path"):
        if "base" in Path(str(c["models"][key])).name.casefold():
            raise ValueError("Qwen3-1.7B-Base is forbidden; use Qwen/Qwen3-1.7B")
    if c["experiment"]["method"] != "topd" or t.get("enabled") is not True:
        raise ValueError("TOPD entrypoint requires method=topd, topd.enabled=true")
    if c["training"]["epochs"] != 1 or c["training"].get("max_steps") is not None:
        raise ValueError("Primary TOPD requires exactly one complete epoch and max_steps=null")
    if c["training"].get("resume_allow_config_mismatch", False):
        raise ValueError("Controlled TOPD resume cannot allow configuration mismatches")
    if c["training"].get("use_lora", False):
        raise ValueError("Primary comparison requires full-parameter training")
    if c["models"]["dtype"] != "bfloat16":
        raise ValueError("Primary comparison requires BF16")
    if c["models"].get("teacher_no_think") is not True or c["data"]["chat_template_kwargs"].get("enable_thinking") is not False:
        raise ValueError("Both models must share the no-think student template")
    if t["future_window"] != 50:
        raise ValueError("TOPD future_window must be 50 (separate from vocabulary Top-K)")
    if t["ground_cost"] != "teacher_embedding_cosine":
        raise ValueError("Only common teacher embedding cosine cost is implemented")
    if t["ot_solver"] not in {"exact", "sinkhorn"}:
        raise ValueError("ot_solver must be exact or sinkhorn")
    if t["divergence_mode"] not in {"percentile", "absolute"}:
        raise ValueError("divergence_mode must be percentile or absolute")
    for key in ("candidate_fraction", "ot_quantile", "future_top_p"):
        if not math.isfinite(float(t[key])) or not 0 <= t[key] <= 1:
            raise ValueError(f"Invalid {key}")
    for key in ("lambda_topd", "future_temperature", "ot_threshold", "min_ot_distance"):
        if not math.isfinite(float(t[key])) or t[key] < 0:
            raise ValueError(f"Invalid {key}")
    for key in ("ot_tolerance", "sinkhorn_epsilon", "trajectory_temperature"):
        if not math.isfinite(float(t[key])) or t[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if t["future_top_p"] == 0:
        raise ValueError("future_top_p must be positive")
    for key in ("generation_batch_size", "loss_micro_batch_size", "ot_max_iterations"):
        if int(t[key]) != t[key] or t[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if t["candidate_budget"] is not None and (int(t["candidate_budget"]) != t["candidate_budget"] or t["candidate_budget"] < 0):
        raise ValueError("candidate_budget must be nonnegative integer or null")
    from .algorithm import trajectory_denominator
    trajectory_denominator(t["trajectory_normalization"], 1, 1, 1)
    data_path = Path(c["data"]["path"])
    if "test" in data_path.name.casefold() or c["data"].get("split") == "test":
        raise ValueError("Training cannot read a test split")
    if c["data"].get("split") not in (None, "train"):
        raise ValueError("Competition-MATH optimization requires train split only")
    if data_path.is_dir() and c["data"].get("split") is None:
        raise ValueError("Directory training data needs split=train to exclude test files")
    output = Path(c["experiment"]["output_dir"]).resolve()
    source = ROOT.parent/"BellmanOPD_analysis"
    if output == source or source in output.parents:
        raise ValueError("TOPD cannot write into BellmanOPD_analysis")
    return c


def launch_config():
    c = load_config(ROOT/"configs/topd_compe_math.yaml")
    if os.environ.get("RESUME_FROM_CHECKPOINT"):
        resume = os.environ["RESUME_FROM_CHECKPOINT"]
        root = (Path(os.environ.get("OUTPUT_DIR", ROOT/"outputs"/
            f"topd_compe_math_4b_17b_seed{os.environ.get('TRAIN_SEED', '42')}"))
            if resume == "auto" else Path(resume).resolve().parent)
        saved = root/"resolved_config.yaml"
        if not saved.is_file():
            raise FileNotFoundError(f"Resume requires original resolved configuration: {saved}")
        c = load_config(saved)
        if "OUTPUT_DIR" in os.environ and Path(os.environ["OUTPUT_DIR"]).resolve() != root.resolve():
            raise ValueError("Resume must write to the original TOPD run directory")
        c["experiment"]["output_dir"] = str(root.resolve())
    elif os.environ.get("BASELINE_CONFIG"):
        reference = resolve_runtime_paths(load_config(os.environ["BASELINE_CONFIG"]))
        if reference["experiment"]["method"] != "cmt":
            raise ValueError("BASELINE_CONFIG must be a resolved CMT configuration")
        topd_settings = copy.deepcopy(c["topd"])
        c = copy.deepcopy(reference)
        c["topd"] = topd_settings
        c["experiment"].update(method="topd", require_b200=False, allow_existing_output=False)
        c["training"].update(resume_from_checkpoint=None, resume_allow_config_mismatch=False)
        # Keep the historical duration visible: validate_config rejects a
        # 3-epoch baseline instead of claiming it matches this 1-epoch task.
    # All entrypoint environment variables are read here, with typed parsing;
    # there is no shell interpolation into Python/YAML snippets.
    mapping = {
        "STORAGE_ROOT": ("paths.storage_root", str),
        "STUDENT_MODEL_PATH": ("models.student_path", str),
        "TEACHER_MODEL_PATH": ("models.teacher_path", str),
        "TRAIN_DATA": ("data.path", str),
        "TRAIN_SEED": ("experiment.seed", int),
        "ROLLOUT_SEED": ("rollout.seed", int),
        "OUTPUT_DIR": ("experiment.output_dir", str),
        "MICRO_BATCH_SIZE_PER_GPU": ("training.micro_batch_size_per_gpu", int),
        "SCORE_MICRO_BATCH_SIZE": ("selector.score_micro_batch_size", int),
        "RESUME_FROM_CHECKPOINT": ("training.resume_from_checkpoint", str),
        "TOPD_LAMBDA": ("topd.lambda_topd", float),
        "TOPD_CANDIDATE_FRACTION": ("topd.candidate_fraction", float),
        "TOPD_OT_QUANTILE": ("topd.ot_quantile", float),
        "TOPD_OT_THRESHOLD": ("topd.ot_threshold", float),
        "TOPD_DIVERGENCE_MODE": ("topd.divergence_mode", str),
        "TOPD_NORMALIZATION": ("topd.trajectory_normalization", str),
        "TOPD_OT_SOLVER": ("topd.ot_solver", str),
        "TOPD_GENERATION_BATCH_SIZE": ("topd.generation_batch_size", int),
        "TOPD_LOSS_MICRO_BATCH_SIZE": ("topd.loss_micro_batch_size", int),
        "FSDP_TEACHER_CPU_OFFLOAD": ("distributed.fsdp.teacher_cpu_offload", lambda x: x.lower() == "true"),
    }
    for env, (key, cast) in mapping.items():
        if env in os.environ and os.environ[env] != "":
            target = c
            parts = key.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = cast(os.environ[env])
    # Independent deterministic run per training seed; evaluation sampling
    # seed remains 1234, exactly as in the original single-run protocol.
    seed = c["experiment"]["seed"]
    if ("ROLLOUT_SEED" not in os.environ and not os.environ.get("RESUME_FROM_CHECKPOINT")
            and not os.environ.get("BASELINE_CONFIG")):
        c["rollout"]["seed"] = seed
    c["topd"]["future_seed"] = 1000000 + seed
    if "OUTPUT_DIR" not in os.environ and not os.environ.get("RESUME_FROM_CHECKPOINT"):
        c["experiment"]["output_dir"] = str(ROOT/"outputs"/f"topd_compe_math_4b_17b_seed{seed}")
    if not os.environ.get("BASELINE_CONFIG") and not os.environ.get("RESUME_FROM_CHECKPOINT"):
        # common_b200.sh derives these from the actual launch world size,
        # overriding the static CMT YAML's four-GPU defaults.
        import torch
        visible = torch.cuda.device_count()
        if not visible:
            visible = max(1, len([v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v]))
        local_trajectories = math.ceil(c["rollout"]["batch_size"]*c["rollout"]["num_responses"]/visible)
        c["rollout"]["vllm"].update(max_num_seqs=local_trajectories,
            max_concurrent_requests=local_trajectories, return_log_probs=False)
    c = resolve_runtime_paths(c)
    validate_config(c)
    return c


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_official_models(config, *, download=False):
    """Pin actual HF commits and verify all official safetensors, including Base.

    A local directory name/config alone cannot distinguish renamed Base weights.
    Compare bytes to the official repository's LFS sha256 for every shard.
    No model download is performed by tests or a configuration-only dry run.
    """
    from huggingface_hub import model_info, snapshot_download
    provenance = {}
    for role in ("student", "teacher"):
        repo = config["topd"][f"{role}_source"]
        revision = config["topd"][f"{role}_revision"]
        info = model_info(repo, revision=revision, files_metadata=True)
        config["topd"][f"{role}_revision"] = info.sha
        path = Path(config["models"][f"{role}_path"])
        if not (path/"config.json").is_file():
            if not download:
                raise FileNotFoundError(f"Missing {role} {path}; set DOWNLOAD_MODELS=1 or {role.upper()}_MODEL_PATH")
            path = Path(snapshot_download(repo, revision=info.sha,
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"]))
            config["models"][f"{role}_path"] = str(path)
        config_json = json.loads((path/"config.json").read_text())
        declared = str(config_json.get("_name_or_path", ""))
        if "base" in declared.casefold() or "base" in path.name.casefold():
            raise ValueError(f"Forbidden Base model: {path}")
        expected_shape = (2048, 28) if role == "student" else (2560, 36)
        if (config_json.get("hidden_size"), config_json.get("num_hidden_layers")) != expected_shape:
            raise ValueError(f"{role} architecture does not match {repo}")
        weights = {}
        for sibling in info.siblings:
            if not sibling.rfilename.endswith(".safetensors"):
                continue
            if sibling.lfs is None:
                raise RuntimeError("Official model weight has no verifiable LFS SHA256")
            local = path/sibling.rfilename
            if not local.is_file():
                raise FileNotFoundError(f"Missing official {role} shard: {local}")
            actual = sha256_file(local)
            if actual != sibling.lfs.sha256:
                raise ValueError(f"{role} weights differ from {repo}@{info.sha}: {local}. Base or another initialization is not allowed.")
            weights[sibling.rfilename] = actual
        if not weights:
            raise RuntimeError(f"No official safetensors found for {repo}")
        assets = {name: sha256_file(path/name) for name in (
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "chat_template.jinja") if (path/name).is_file()}
        provenance[role] = dict(source=repo, revision=info.sha, path=str(path.resolve()),
            weights_sha256=weights, tokenizer_and_config_sha256=assets)
    return provenance
