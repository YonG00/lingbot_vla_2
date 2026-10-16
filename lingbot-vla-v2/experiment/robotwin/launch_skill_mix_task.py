#!/usr/bin/env python3
"""单卡 AL 技能探针：默认只打印计划；--prepare-only 落配置；--execute 训练。

预检不导入 torch（来源训练配置为 YAML 时需 PyYAML）；复用现有 trainer / AL hook。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lingbotvla.auto_learning.config import AutoLearningConfig

REPO = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_training_config(path):
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except ValueError:
        import yaml
        return yaml.safe_load(text)


def validate_tokenizer_compatibility(source, selected):
    from tokenizers import Tokenizer

    normalized = []
    runtime = []
    for path in (source, selected):
        try:
            tokenizer = Tokenizer.from_file(str(path))
            # HF 保存时会固化训练的 padding/truncation，并可能将 BPE merges
            # 从字符串改存二元数组。让同一运行库规范化，保留全部词表/规则语义；
            # 长度设置由训练 loader 负责，不以序列化格式判断模型是否兼容。
            raw = read_json(path)
            runtime.append({key: raw.get(key) for key in ("padding", "truncation")})
            tokenizer.no_padding()
            tokenizer.no_truncation()
            normalized.append(json.loads(tokenizer.to_str()))
        except Exception as exc:
            raise ValueError(f"cannot validate tokenizer: {path}: {exc}") from exc
    if normalized[0] != normalized[1]:
        keys = sorted(key for key in set(normalized[0]) | set(normalized[1])
                      if normalized[0].get(key) != normalized[1].get(key))
        raise ValueError(f"source HF tokenizer disagrees with selected Qwen3VL tokenizer: {keys}")
    return dict(normalized_sha256=hashlib.sha256(
        json.dumps(normalized[0], sort_keys=True).encode()).hexdigest(),
        ignored_runtime_settings=runtime)


def checkpoint_inventory(model):
    index_path = model / "model.safetensors.index.json"
    index = read_json(index_path)
    expected = index["weight_map"]
    if not expected:
        raise ValueError("empty checkpoint index")
    inventory = []
    seen = {}
    for shard in sorted(set(expected.values())):
        path = model / shard
        if path.resolve().parent != model.resolve():
            raise ValueError("checkpoint shard must be inside model directory")
        with path.open("rb") as f:
            size_bytes = f.read(8)
            if len(size_bytes) != 8:
                raise ValueError(f"truncated safetensors: {path}")
            header_size = struct.unpack("<Q", size_bytes)[0]
            if not 0 < header_size < min(path.stat().st_size, 32 * 1024 * 1024):
                raise ValueError(f"invalid safetensors header: {path}")
            header = json.loads(f.read(header_size))
        payload = path.stat().st_size - 8 - header_size
        dtypes = set()
        for key, tensor in header.items():
            if key == "__metadata__":
                continue
            lo, hi = tensor["data_offsets"]
            if key in seen or not 0 <= lo <= hi <= payload:
                raise ValueError(f"invalid/duplicate tensor {key} in {shard}")
            seen[key] = shard
            dtypes.add(tensor["dtype"])
        inventory.append(dict(file=shard, size=path.stat().st_size,
                              header_sha256=hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest(),
                              dtypes=sorted(dtypes)))
    if seen != expected:
        raise ValueError("checkpoint tensor headers disagree with index weight_map")
    if any(t in {"F32", "F16"} for row in inventory for t in row["dtypes"]):
        raise ValueError("this experiment requires a BF16 source checkpoint")
    return inventory


def build_plan(args):
    cfg = getattr(args, "al_config_data", None) or read_json(args.al_config)
    AutoLearningConfig(**cfg)
    if cfg.get("sample_scan_method", "hardness") == "target_change_window" and cfg["hardness_scan_enabled"]:
        if not cfg.get("target_change_cache_dir") or not cfg.get("target_change_embodiment_config"):
            raise ValueError("target_change_window requires cache directory and actual embodiment config")
        cfg["target_change_cache_dir"] = str(Path(cfg["target_change_cache_dir"]).expanduser().resolve())
        cfg["target_change_embodiment_config"] = str(Path(cfg["target_change_embodiment_config"]).expanduser().resolve())
        cfg["target_change_dataset_root"] = str(args.dataset_root.resolve())
    learning = args.learning_tasks if args.learning_tasks is not None else cfg["learning_task_names"]
    replay = args.replay_tasks if args.replay_tasks is not None else cfg["replay_task_names"]
    tasks = list(learning) + list(replay)
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("learning / replay tasks must be nonempty, unique and disjoint")
    if args.steps <= 0 or args.steps % cfg["eval_interval_steps"]:
        raise ValueError("steps must be a positive multiple of eval_interval_steps")
    if cfg["min_steps_before_defer"] <= args.steps or cfg["early_defer_on_overfit"]:
        raise ValueError("experiment requires defer budget > total steps and early defer disabled")
    # 2026-10-16：**允许** `hardness_scan_enabled=true` —— 新增"周期存档后重扫"触发机制后，
    # 单任务实验也能按当前模型难度重新加权采样。其余扫描/PASS/池导入/复查仍须关闭（隔离前提）。
    if any(cfg[k] for k in ("task_scan_enabled", "pass_transition_enabled",
                             "import_pool_state", "review_after_task_transitions",
                             "rescan_candidates_after_transition")):
        raise ValueError("skill probe requires PASS/task-scan/import/review disabled")
    if args.micro < 1 or args.gas < 1:
        raise ValueError("micro/gas must be positive")
    if (not replay and args.new_ratio != 1.0) or not 0 < args.new_ratio <= 1:
        raise ValueError("new-ratio must be in (0,1]; without replay it must be 1")
    if args.out.exists():
        raise ValueError(f"use a fresh output directory: {args.out}")
    model = args.model.resolve()
    if args.out.resolve() == model or args.out.resolve() in model.parents or model in args.out.resolve().parents:
        raise ValueError("source checkpoint and experiment output must be independent")
    for path in (args.train_config, args.dataset_root / "meta", args.qwen3vl):
        if not path.exists():
            raise ValueError(f"missing input: {path}")
    inventory = checkpoint_inventory(model)
    model_config = read_json(model / "config.json")
    source_args = read_training_config(args.train_config)
    norm = source_args.get("data", {}).get("norm_stats_file")
    if not norm:
        raise ValueError("source training config must explicitly record data.norm_stats_file")
    norm = Path(norm).expanduser()
    if not norm.is_absolute():
        norm = REPO / norm
    if not norm.is_file():
        raise ValueError(f"source normalization file missing: {norm}")
    if not (model / "tokenizer_config.json").is_file():
        raise ValueError("source HF must include its tokenizer/processor assets")
    tokenizer_validation = validate_tokenizer_compatibility(
        model / "tokenizer.json", args.qwen3vl / "tokenizer.json")
    # build_foundation_model 当前按路径识别 Qwen3 patch；HF 的 step 目录名没有型号。
    qwen_name = str(args.qwen3vl).lower()
    if "qwen3" not in qwen_name or "vl" not in qwen_name:
        raise ValueError("Qwen3-VL path must identify Qwen3VL for existing model patch selection")
    # 复用来源架构/动作配置，只明确清掉权重部分加载及 DCP 恢复入口。
    run_config = json.loads(json.dumps(source_args))
    run_config.setdefault("model", {}).update(vlm_repo_id=None, post_training=True, incremental_training=False)
    run_config.setdefault("train", {}).update(load_checkpoint_path=None, enable_resume=False)
    run_config["train"]["low_noise_fm_enabled"] = cfg.get("low_noise_fm_enabled", False)
    manifest = read_json(args.manifest)
    missing = set(tasks) - set(manifest["tasks"])
    if missing:
        raise ValueError(f"unknown tasks: {sorted(missing)}")
    train_ids, val_ids = [], []
    counts = {}
    for task in tasks:
        row = manifest["tasks"][task]
        train, val = row["train_ids"], row["val_ids"]
        if (not train or not val or set(train) & set(val)
                or len(set(train + val)) != row["n_total"]):
            raise ValueError(f"invalid train/val split: {task}")
        train_ids.extend(train)
        val_ids.extend(val)
        counts[task] = dict(train=len(train), val=len(val))
    if len(set(train_ids + val_ids)) != len(train_ids + val_ids):
        raise ValueError("cross-task episode overlap")
    thresholds = Path(cfg["pass_thresholds_file"])
    if not thresholds.is_absolute():
        thresholds = REPO / thresholds
    reference = read_json(thresholds)
    judge_n = max(cfg.get("global_scout_val_trajs", 2), cfg["active_val_probe_trajs"])
    for task in tasks:
        if reference["trajectory_ids"][task] != manifest["tasks"][task]["val_ids"][:judge_n]:
            raise ValueError(f"reference/eval trajectories disagree: {task}")
        if not math.isfinite(reference["tasks"][task]) or reference["tasks"][task] <= 0:
            raise ValueError(f"invalid reference: {task}")
    cfg.update(task_names=tasks, learning_task_names=list(learning), replay_task_names=list(replay),
               max_global_steps=args.steps, batch_size=args.micro * args.gas, new_ratio=args.new_ratio,
               pass_thresholds_file=str(thresholds.resolve()), data_audit_file=str(args.audit_report.resolve()))
    cfg["data_audit_instruction_patterns"] = {
        k: v for k, v in cfg.get("data_audit_instruction_patterns", {}).items() if k in tasks}
    out = args.out.resolve()
    command = [args.python, "-m", "torch.distributed.run", "--nproc_per_node=1",
               f"--master_port={args.master_port}", "tasks/vla/train_lingbotvla.py",
               str(out / "trainer_config.json")]
    options = {
        "model.model_path": str(model), "data.train_path": str(out / "datasets.txt"),
        "model.config_path": str(model / "config.json"),
        "model.tokenizer_path": str(args.qwen3vl.resolve()), "data.norm_stats_file": str(norm.resolve()),
        "data.episode_ids_file": str(out / "train_ids.json"), "data.image_augment": False,
        "train.output_dir": str(out), "train.micro_batch_size": args.micro,
        "train.gradient_accumulation_steps": args.gas, "train.global_batch_size": args.micro * args.gas,
        "train.num_train_epochs": args.steps + 1, "train.max_steps": args.steps,
        "train.save_steps": 250, "train.save_epochs": 0, "train.save_hf_weights": True,
        "train.async_save_hf_weights": False, "train.hf_pass_interval": 0,
        "train.hf_export_dtype": "bf16", "train.dcp_save_mode": "hf_only",
        "train.dcp_keep_last": 0, "train.enable_resume": False, "train.reset_step_counter": True,
        "train.step_offset": 0, "train.train_expert_only": False,
        "train.seed": cfg.get("seed", 3), "train.open_loop_eval_steps": 0,
        "train.smoke_no_checkpoint": False, "train.auto_learning_baseline": "",
        "train.freeze_vision_encoder": True, "train.enable_mixed_precision": False,
        "train.enable_fp32": False, "train.use_compile": False,
        "train.rmpad": False, "train.rmpad_with_pos_ids": False,
        "train.disk_guard": True, "train.disk_guard_margin": 1.1, "train.disk_check_interval": 50,
        "train.data_parallel_mode": "fsdp2", "train.data_parallel_replicate_size": 1,
        "train.data_parallel_shard_size": 1, "train.auto_learning": str(out / "al_config.json"),
        "train.auto_learning_manifest": str(args.manifest.resolve()),
    }
    for key, value in options.items():
        command += ["--" + key, str(value).lower() if isinstance(value, bool) else str(value)]
    source_files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        model / "config.json", model / "model.safetensors.index.json", args.train_config,
        args.manifest, thresholds, norm, model / "tokenizer.json", args.qwen3vl / "tokenizer.json")}
    return dict(config=cfg, training_config=run_config, command=command, train_ids=sorted(train_ids), counts=counts,
                source_model=str(model), checkpoint_inventory=inventory, source_files=source_files,
                tokenizer_validation=tokenizer_validation,
                source_model_config=model_config,
                note="fresh optimizer/LR/step; HF-only; historical finished-model reference reports only")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--train-config", type=Path, required=True,
                   help="来源训练 run 的 lingbotvla_cli.yaml（含原归一化路径/动作配置）")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--qwen3vl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--audit-report", type=Path, required=True)
    p.add_argument("--al-config", type=Path, default=REPO / "configs/auto_learning/learn_skill_mix_task.json")
    p.add_argument("--manifest", type=Path, default=REPO / "refs/al_artifacts/al/task_splits_50/manifest.json")
    p.add_argument("--learning-tasks", nargs="+")
    p.add_argument("--replay-tasks", nargs="*")
    p.add_argument("--new-ratio", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--micro", type=int, default=1)
    p.add_argument("--gas", type=int, default=10)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--master-port", type=int, default=62510)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    plan = build_plan(args)
    print(json.dumps({k: v for k, v in plan.items() if k not in ("train_ids", "source_model_config", "training_config")},
                     ensure_ascii=False, indent=2))
    print("COMMAND: " + shlex.join(plan["command"]))
    if not (args.prepare_only or args.execute):
        return
    if args.execute and plan["config"]["hardness_scan_enabled"] and plan["config"].get("sample_scan_method") == "target_change_window":
        # Finish CPU preparation before creating the run or loading the model.
        # A bad cache must not leave an unusable, nonempty output directory.
        from experiment.robotwin.prepare_target_change_windows import prepare
        prepare(AutoLearningConfig(**plan["config"]), args.dataset_root,
                read_json(args.manifest), plan["config"]["task_names"])
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in (("al_config.json", plan["config"]), ("trainer_config.json", plan["training_config"]),
                        ("train_ids.json", plan["train_ids"]),
                        ("provenance.json", plan)):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (args.out / "datasets.txt").write_text(f"robotwin {args.dataset_root.resolve()}\n")
    if args.execute:
        # 入口在 tasks/vla，子进程不会自动把包根加入 sys.path；安装的 editable
        # 包可能仍指向生产目录。必须让 torchrun 及其 worker 优先导入实验源码。
        env = dict(os.environ, QWEN3VL_PATH=str(args.qwen3vl.resolve()),
                   PYTHONPATH=os.pathsep.join(filter(None, (str(REPO), os.environ.get("PYTHONPATH", "")))),
                   AL_POOL_STATE_NO_IMPORT="1", AL_POOL_STATE_MIRROR="",
                   AL_SCAN_SKIP_HARDNESS="0",
                   AL_HARDNESS_CACHE_FILE="", AL_HARDNESS_CACHE="",
                   AL_SCOUT_CACHE_MODE="off", AL_SCOUT_CACHE_FILE="", AL_HARDNESS_SELFTEST_IDS="0",
                   HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        if len(env["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
            raise ValueError("select exactly one CUDA device for this launcher")
        os.chdir(REPO)
        os.execvpe(plan["command"][0], plan["command"], env)


if __name__ == "__main__":
    main()
