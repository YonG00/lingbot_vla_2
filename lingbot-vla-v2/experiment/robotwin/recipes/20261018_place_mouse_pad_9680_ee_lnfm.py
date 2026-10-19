#!/usr/bin/env python3
"""鼠标单任务：原始9680 + 500步；默认CPU预览，--execute才后台训练。"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "mouse_pad_continue", ROOT / "experiment/robotwin/continue_skill_training.py"
)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)
launcher = continuation.launcher
TASK = "place_mouse_pad"
SOURCE = Path("/data/outputs/al_v79_poolfix_7300")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=SOURCE / "checkpoints/global_step_9680/hf_ckpt")
    p.add_argument("--train-config", type=Path, default=SOURCE / "lingbotvla_cli.yaml")
    p.add_argument("--dataset-root", type=Path, default=Path(
        "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"))
    p.add_argument("--qwen3vl", type=Path, default=Path(
        "/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct"))
    p.add_argument("--al-config", type=Path, default=ROOT / "configs/auto_learning/place_mouse_pad_ee_lnfm.json")
    p.add_argument("--manifest", type=Path, default=ROOT / "refs/al_artifacts/al/task_splits_50/manifest.json")
    p.add_argument("--ee-fk-spec", type=Path, default=ROOT / "configs/kinematics/robotwin_aloha_agilex_ee.json")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--hf-save-steps", type=int, default=250, help="HF周期步数；设为总步数时只在结束保存")
    p.add_argument("--micro", type=int, default=20)
    p.add_argument("--gas", type=int, default=1)
    p.add_argument("--ee-position-weight", type=float, default=3.0)
    p.add_argument("--ee-orientation-weight", type=float, default=0.8)
    p.add_argument("--ee-gripper-weight", type=float, default=1.5)
    p.add_argument("--train-all-episodes", action=argparse.BooleanOptionalAction, default=False,
                   help="把 manifest 的 train+val 共50条全部纳入训练；关闭独立 val/PASS/gap")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--alignment-loss-scale", type=float, default=1.0,
                   help="对来源中当前深度/未来深度/视频loss权重乘同一倍率，不改变内部视频损失配比")
    p.add_argument("--shutdown", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--output-root", type=Path, default=Path("/data/outputs"))
    p.add_argument("--runtime-dir", type=Path, default=Path("/data/outputs/skill_launches"))
    p.add_argument("--cache-root", type=Path, default=Path("/data/al_cache"))
    p.add_argument("--run-name")
    p.add_argument("--auxiliary-manifest", type=Path)
    p.add_argument("--auxiliary-primary-ratio", type=float, default=0.7)
    p.add_argument("--auxiliary-grasp-weight", type=float, default=3.0)
    p.add_argument("--auxiliary-approach-seconds", type=float, default=1.0)
    p.add_argument("--master-port", type=int, default=62510)
    p.add_argument("--execute", action="store_true")
    return p


def build_job(args):
    if not math.isfinite(args.alignment_loss_scale) or args.alignment_loss_scale <= 0:
        raise ValueError("alignment-loss-scale must be finite and positive")
    label = f"all50_p{args.ee_position_weight:g}_" if args.train_all_episodes else ""
    name = args.run_name or f"al_mouse_pad_{label}9680_ee_lnfm_add{args.steps}_{datetime.now():%Y%m%d_%H%M%S}"
    # Same filename contract as the existing continuation entry.
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name):
        raise ValueError("run-name must be 1-100 safe filename characters")
    cfg = launcher.read_json(args.al_config)
    cfg.update(task_names=[TASK], learning_task_names=[TASK], replay_task_names=[],
               train_all_episodes=args.train_all_episodes,
               low_noise_fm_enabled=True, hardness_scan_enabled=True,
               sample_scan_method="target_change_window",
               target_change_cache_dir=str((args.cache_root / name).resolve()),
               data_audit_instruction_patterns={TASK: "mouse"})
    if args.auxiliary_manifest:
        cfg.update(auxiliary_manifest=str(args.auxiliary_manifest.resolve()),
                   auxiliary_primary_ratio=args.auxiliary_primary_ratio,
                   auxiliary_grasp_weight=args.auxiliary_grasp_weight,
                   auxiliary_approach_seconds=args.auxiliary_approach_seconds)
    out = (args.output_root / name).resolve()
    runtime = args.runtime_dir.resolve()
    training_snapshot = runtime / f"{name}.training.json"
    plan_args = SimpleNamespace(
        model=args.model.resolve(), train_config=args.train_config.resolve(),
        dataset_root=args.dataset_root.resolve(), qwen3vl=args.qwen3vl.resolve(),
        al_config=args.al_config.resolve(), al_config_data=cfg,
        manifest=args.manifest.resolve(), out=out,
        audit_report=(args.cache_root / name / "data_audit.json").resolve(),
        learning_tasks=[TASK], replay_tasks=[], steps=args.steps, micro=args.micro,
        gas=args.gas, hf_save_steps=args.hf_save_steps, new_ratio=1.0, master_port=args.master_port, python=sys.executable,
        ee_fk_loss=True, ee_fk_eval=True, ee_fk_spec=args.ee_fk_spec,
        ee_position_weight=args.ee_position_weight,
        ee_orientation_weight=args.ee_orientation_weight,
        ee_gripper_weight=args.ee_gripper_weight,
        freeze_vision_encoder=False, gradient_checkpointing=args.gradient_checkpointing,
    )
    plan = launcher.build_plan(plan_args)
    if args.alignment_loss_scale != 1.0:
        align = plan['training_config']['train'].get('align_params', {})
        video = align.get('video', {})
        if not align.get('use_future_video') or not align.get('depth', {}).get('use_future_depth'):
            raise ValueError('alignment scaling requires enabled depth/future-depth/video supervision')
        for container, key in ((align, 'depth_loss_weight'), (align, 'future_depth_loss_weight'),
                               (video, 'future_video_loss_weight')):
            value = container.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'explicit positive source alignment weight required: {key}')
            container[key] = value * args.alignment_loss_scale
    # Two independent visual freeze switches exist. The dedicated source
    # snapshot disables both, and LoRA, before the ordinary launcher rebuilds
    # its plan in the worker. Model weights still come exclusively from --model.
    plan["training_config"]["train"].update(
        freeze_vit=False, freeze_vision_encoder=False, use_lora=False,
        train_expert_only=False,
    )
    plan["command"] += ["--train.freeze_vit", "false", "--train.use_lora", "false"]
    command = [sys.executable, str(Path(launcher.__file__).resolve())]
    for flag in ("model", "dataset_root", "qwen3vl", "al_config", "manifest", "out",
                 "audit_report", "steps", "micro", "gas", "hf_save_steps", "master_port", "ee_fk_spec",
                 "ee_position_weight", "ee_orientation_weight", "ee_gripper_weight"):
        command += ["--" + flag.replace("_", "-"), str(getattr(plan_args, flag))]
    command += ["--train-config", str(training_snapshot), "--learning-tasks", TASK,
                "--replay-tasks", "--ee-fk-loss", "--ee-fk-eval",
                "--no-freeze-vision-encoder",
                "--gradient-checkpointing" if args.gradient_checkpointing else "--no-gradient-checkpointing",
                "--python", sys.executable, "--execute"]
    job = dict(command=command, source_model=str(args.model.resolve()), output_dir=str(out),
               steps=args.steps, shutdown=args.shutdown, log=str(runtime / f"{name}.log"),
               job_file=str(runtime / f"{name}.json"), status_file=str(runtime / f"{name}.status.json"))
    return job, plan, training_snapshot


def main(argv=None):
    args = parser().parse_args(argv)
    job, plan, snapshot = build_job(args)
    print(json.dumps(dict(
        source_model=job["source_model"], tasks=plan["config"]["task_names"],
        split=plan["counts"], added_steps=args.steps, micro=args.micro, gas=args.gas,
        freeze_vit=False, freeze_vision_encoder=False, train_expert_only=False, use_lora=False,
        low_noise_fm=True, ee_loss=True,
        ee_weights=[args.ee_position_weight,args.ee_orientation_weight,args.ee_gripper_weight],
        gradient_checkpointing=args.gradient_checkpointing,
        sampling=plan["config"]["sample_scan_method"],
        train_all_episodes=args.train_all_episodes,
        auxiliary=plan.get('auxiliary'),
        cache=plan["config"]["target_change_cache_dir"],
        output=job["output_dir"], log=job["log"], shutdown=args.shutdown,
    ), ensure_ascii=False, indent=2))
    print("COMMAND: " + shlex.join(job["command"]))
    if not args.execute:
        return 0
    continuation.check_machine(job, plan)
    artifacts = [snapshot, Path(job["job_file"]), Path(job["job_file"]).with_suffix(".al.json"),
                 Path(job["log"]), Path(job["status_file"])]
    for artifact in artifacts:
        if artifact.exists():
            raise FileExistsError(f"launch artifact already exists: {artifact}")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    continuation.write_json(snapshot, plan["training_config"])
    pid = continuation.start_background(job, plan)
    print(f"后台启动 PID={pid}\n日志：{job['log']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
