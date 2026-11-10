from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path

from b200_experiment.config import load_config, resolve_runtime_paths

ROOT = Path(__file__).resolve().parents[1]


def first_environment(*names):
    """Match common_b200.sh's nonempty environment override precedence."""
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def boolean(value):
    if value.lower() not in {"true", "false", "1", "0"}:
        raise ValueError(f"Expected true/false, got {value!r}")
    return value.lower() in {"true", "1"}


def b200_run_paths(seed=None):
    """Same OUTPUT_ROOT/RUN_NAME/method and RESULTS_DIR layout as CMT.

    All project-relative outputs are anchored to this standalone checkout,
    while model/data assets use STORAGE_ROOT through the unchanged resolver.
    """
    seed = int(seed if seed is not None else first_environment("TRAIN_SEED", "SEED", "EXPERIMENT_SEED") or 42)
    name = first_environment("TOPD_RUN_NAME", "RUN_NAME") or f"topd_compe_math_4b_17b_seed{seed}"
    comparison = first_environment("COMPARISON_NAME") or name
    for value in (name, comparison):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            raise ValueError("Run names may contain only letters, numbers, dot, underscore, and dash")

    def project_path(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else ROOT/path).resolve()

    output_root = project_path(first_environment("OUTPUT_ROOT") or ROOT/"outputs")
    run = project_path(first_environment("TOPD_RUN_DIR", "OUTPUT_DIR", "TOPD_OUTPUT_DIR") or output_root/name/"topd")
    results = project_path(first_environment("RESULTS_DIR") or ROOT/"results"/comparison)
    return dict(run_name=name, run_output=str(run),
        checkpoint=str(project_path(first_environment("TOPD_CHECKPOINT") or run/"final")),
        eval_output=str(project_path(first_environment("TOPD_EVAL_OUTPUT") or results/"eval/topd")),
        launch_config=str(ROOT/"results/launch_configs"/f"{name}.yaml"),
        base_eval_cache=str(project_path(first_environment("TRAIN_EVAL_BASE_CACHE_DIR") or "outputs/.base_eval_cache")))


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
        if key != "ot_max_iterations" and t[key] == "auto":
            continue
        if int(t[key]) != t[key] or t[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if t.get("future_backend", "shared_prefix") not in {"shared_prefix", "reference"}:
        raise ValueError("Invalid future_backend")
    for key in ("generation_max_batch_size", "generation_token_budget", "loss_max_batch_size", "loss_token_budget", "ot_workers", "kl_chunk_tokens"):
        if key in t and (int(t[key]) != t[key] or t[key] < 1):
            raise ValueError(f"Invalid {key}")
    for key in ("workspace_fraction", "prefix_pool_fraction"):
        if key in t and not 0 < float(t[key]) <= 1:
            raise ValueError(f"Invalid {key}")
    if float(t.get("memory_headroom_gib", 8)) < 0:
        raise ValueError("Invalid memory_headroom_gib")
    if int(c["training"].get("save_interval", 0)) <= 0:
        raise ValueError("training.save_interval must be positive")
    if not c["training"].get("save_optimizer", True) or not c["training"].get("save_checkpoints", True):
        raise ValueError("TOPD requires complete model/optimizer/RNG checkpoints")
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
    resume = first_environment("RESUME_FROM_CHECKPOINT", "RESUME")
    if resume:
        root = (Path(b200_run_paths()["run_output"])
            if resume == "auto" else Path(resume).expanduser().resolve().parent)
        saved = root/"resolved_config.yaml"
        if not saved.is_file():
            raise FileNotFoundError(f"Resume requires original resolved configuration: {saved}")
        c = load_config(saved)
        if first_environment("TOPD_RUN_DIR", "OUTPUT_DIR", "TOPD_OUTPUT_DIR") and Path(b200_run_paths()["run_output"]) != root.resolve():
            raise ValueError("Resume must write to the original TOPD run directory")
        c["experiment"]["output_dir"] = str(root.resolve())
    elif os.environ.get("BASELINE_CONFIG"):
        reference = resolve_runtime_paths(load_config(os.environ["BASELINE_CONFIG"]))
        if reference["experiment"]["method"] != "cmt":
            raise ValueError("BASELINE_CONFIG must be a resolved CMT configuration")
        topd_settings = copy.deepcopy(c["topd"])
        c = copy.deepcopy(reference)
        c["topd"] = topd_settings
        c["experiment"].update(method="topd", require_b200=True, allow_existing_output=False)
        c["training"].update(resume_from_checkpoint=None, resume_allow_config_mismatch=False)
        c["training"].update(save_interval=20, save_checkpoints=True, save_optimizer=True)
        # Keep the historical duration visible: validate_config rejects a
        # 3-epoch baseline instead of claiming it matches this 1-epoch task.
    # All entrypoint environment variables are read here, with typed parsing;
    # there is no shell interpolation into Python/YAML snippets.
    mapping = {
        "STORAGE_ROOT": ("paths.storage_root", str),
        "ROLLOUT_SEED": ("rollout.seed", int),
        "ROLLOUT_VLLM_LOGGING_LEVEL": ("rollout.vllm.logging_level", str),
        "ROLLOUT_VLLM_STARTUP_TIMEOUT": ("rollout.vllm.startup_timeout", float),
        "TRAIN_DATA_SPLIT": ("data.split", lambda value: None if value.lower() == "null" else value),
        "TRAIN_PREFER_SOURCE_PROMPT": ("data.prefer_source_prompt", boolean),
        "MICRO_BATCH_SIZE_PER_GPU": ("training.micro_batch_size_per_gpu", int),
        "SCORE_MICRO_BATCH_SIZE": ("selector.score_micro_batch_size", int),
        "SAVE_INTERVAL": ("training.save_interval", int),
        "TOPD_LAMBDA": ("topd.lambda_topd", float),
        "TOPD_CANDIDATE_FRACTION": ("topd.candidate_fraction", float),
        "TOPD_OT_QUANTILE": ("topd.ot_quantile", float),
        "TOPD_OT_THRESHOLD": ("topd.ot_threshold", float),
        "TOPD_DIVERGENCE_MODE": ("topd.divergence_mode", str),
        "TOPD_NORMALIZATION": ("topd.trajectory_normalization", str),
        "TOPD_OT_SOLVER": ("topd.ot_solver", str),
        "TOPD_GENERATION_BATCH_SIZE": ("topd.generation_batch_size", lambda value: value if value == "auto" else int(value)),
        "TOPD_LOSS_MICRO_BATCH_SIZE": ("topd.loss_micro_batch_size", lambda value: value if value == "auto" else int(value)),
        "TOPD_FUTURE_BACKEND": ("topd.future_backend", str),
        "TOPD_GENERATION_MAX_BATCH_SIZE": ("topd.generation_max_batch_size", int),
        "TOPD_LOSS_MAX_BATCH_SIZE": ("topd.loss_max_batch_size", int),
        "TOPD_GENERATION_TOKEN_BUDGET": ("topd.generation_token_budget", int),
        "TOPD_LOSS_TOKEN_BUDGET": ("topd.loss_token_budget", int),
        "TOPD_MEMORY_HEADROOM_GIB": ("topd.memory_headroom_gib", float),
        "TOPD_OT_WORKERS": ("topd.ot_workers", int),
        "TOPD_PROFILE_STEPS": ("topd.profile_steps", lambda value: [int(item) for item in value.split(",") if item.strip()]),
        "FSDP_TEACHER_CPU_OFFLOAD": ("distributed.fsdp.teacher_cpu_offload", boolean),
        "VERIFY_OFFICIAL_WEIGHTS": ("topd.verify_official_weights", boolean),
    }
    aliases = {
        ("STUDENT_MODEL", "STUDENT_MODEL_PATH", "STUDENT_PATH"): ("models.student_path", str),
        ("TEACHER_MODEL", "TEACHER_MODEL_PATH", "TEACHER_PATH"): ("models.teacher_path", str),
        ("TRAIN_DATA", "TRAIN_DATA_PATH"): ("data.path", str),
        ("PROMPT_KEY", "TRAIN_PROMPT_KEY"): ("data.prompt_key", str),
        ("TRAIN_SEED", "SEED", "EXPERIMENT_SEED"): ("experiment.seed", int),
    }
    for names, (key, cast) in aliases.items():
        value = first_environment(*names)
        if value is not None:
            section, field = key.split(".")
            c[section][field] = cast(value)
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
    if resume:
        c["training"]["resume_from_checkpoint"] = resume
    if ("ROLLOUT_SEED" not in os.environ and not resume
            and not os.environ.get("BASELINE_CONFIG")):
        c["rollout"]["seed"] = seed
    c["topd"]["future_seed"] = 1000000 + seed
    if not resume:
        c["experiment"]["output_dir"] = b200_run_paths(seed)["run_output"]
    if first_environment("TRAIN_EVAL_BASE_CACHE_DIR"):
        c["training_evaluation"]["base_cache_dir"] = b200_run_paths(seed)["base_eval_cache"]
    if not os.environ.get("BASELINE_CONFIG") and not resume:
        # common_b200.sh derives these from the actual launch world size,
        # overriding the static CMT YAML's four-GPU defaults.
        import torch
        visible = torch.cuda.device_count()
        if not visible:
            visible = max(1, len([v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v]))
        local_trajectories = math.ceil(c["rollout"]["batch_size"]*c["rollout"]["num_responses"]/visible)
        c["rollout"]["vllm"].update(max_num_seqs=local_trajectories,
            max_concurrent_requests=local_trajectories, return_log_probs=False)
    # User-requested execution schedule also overrides inherited baseline and
    # older resume configs. Keep the decoding protocol for standalone eval.
    c["training_evaluation"].update(enabled=False, eval_at_start=False, eval_at_end=False)
    c = resolve_runtime_paths(c)
    validate_config(c)
    return c


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_official_models(config, *, verify_official=False):
    """Inspect local B200 checkpoints without downloading or rewriting paths.

    Local SHA256/config/tokenizer attestations work offline. An explicit
    verify_official=True additionally compares local safetensors to HF LFS
    metadata; even that mode never downloads weights or selects a cache path.
    """
    local_paths = {role: Path(config["models"][f"{role}_path"]).expanduser().resolve()
                   for role in ("student", "teacher")}
    # Check both paths first. Missing B200 assets must not start a network call.
    for role, path in local_paths.items():
        if not (path/"config.json").is_file():
            raise FileNotFoundError(
                f"Missing local {role} checkpoint: {path}. Set {role.upper()}_MODEL "
                f"or {role.upper()}_MODEL_PATH to the existing B200 checkpoint. "
                "TOPD never downloads or redirects model paths.")
    provenance = {}
    for role, path in local_paths.items():
        repo = config["topd"][f"{role}_source"]
        requested_revision = config["topd"][f"{role}_revision"]
        config_json = json.loads((path/"config.json").read_text())
        declared = str(config_json.get("_name_or_path", ""))
        if "base" in declared.casefold() or "base" in path.name.casefold():
            raise ValueError(f"Forbidden Base model: {path}")
        expected_shape = (2048, 28) if role == "student" else (2560, 36)
        if (config_json.get("hidden_size"), config_json.get("num_hidden_layers")) != expected_shape:
            raise ValueError(f"{role} architecture does not match {repo}: {path}")
        names = None
        for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if (path/index).is_file():
                names = sorted(set(json.loads((path/index).read_text())["weight_map"].values()))
                break
        if names is None:
            names = [name for name in ("model.safetensors", "pytorch_model.bin") if (path/name).is_file()]
        if not names:
            raise FileNotFoundError(f"No Hugging Face model weights/index found at {path}")
        weights = {}
        for name in names:
            shard = (path/name).resolve()
            # HF cache symlinks to blobs are valid local assets; the configured
            # snapshot directory itself is never replaced with a cache path.
            if not shard.is_file():
                raise FileNotFoundError(f"Missing local {role} weight shard: {path/name}")
            weights[name] = sha256_file(shard)
        revision = None
        if verify_official:
            from huggingface_hub import model_info
            info = model_info(repo, revision=requested_revision, files_metadata=True)
            expected = {item.rfilename: item.lfs.sha256 for item in info.siblings
                        if item.rfilename.endswith(".safetensors") and item.lfs is not None}
            if not expected or expected != weights:
                raise ValueError(f"{role} weights differ from {repo}@{info.sha}: {path}. Base or another initialization is not allowed.")
            revision = info.sha
            config["topd"][f"{role}_revision"] = revision
        assets = {name: sha256_file(path/name) for name in (
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "chat_template.jinja") if (path/name).is_file()}
        provenance[role] = dict(source=repo, revision=revision, path=str(path),
            verification="official_sha256" if verify_official else "local_checkpoint_sha256",
            weights_sha256=weights, tokenizer_and_config_sha256=assets)
    return provenance
