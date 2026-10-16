#!/usr/bin/env python3
"""从 HF 权重便捷续练：默认预览，--execute 后台启动，--shutdown 成功后关机。"""
from __future__ import annotations

import argparse
from datetime import datetime
import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "skill_training_launcher", Path(__file__).with_name("launch_skill_mix_task.py"))
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)

AL_CLI_FIELDS = {
    "hardness_scan_enabled": "sample_scan",
    "eval_interval_steps": "eval_interval_steps",
    "min_steps_before_defer": "defer_steps",
    "defer_retry_steps": "defer_retry_steps",
    "refresh_hardness_on_continue": "refresh_hardness_on_continue",
}


def make_job(args):
    model = args.from_hf.expanduser().resolve()
    if model.name != "hf_ckpt" or model.parent.parent.name != "checkpoints":
        raise ValueError("--from-hf must be <run>/checkpoints/global_step_N/hf_ckpt")
    source = model.parents[2]
    training = source / "lingbotvla_cli.yaml"
    if not training.is_file():
        training = source / "trainer_config.json"
    source_cfg = launcher.read_training_config(training)
    rows = [line.split(maxsplit=1) for line in (source / "datasets.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][0] != "robotwin":
        raise ValueError("source datasets.txt must contain exactly one robotwin dataset")
    dataset = Path(rows[0][1]).expanduser()
    if not dataset.is_absolute():
        raise ValueError("source dataset path must be absolute; use the full launcher otherwise")
    old_al = launcher.read_json(source / "al_config.json")
    qwen = args.qwen3vl or source_cfg.get("model", {}).get("tokenizer_path")
    if not qwen:
        raise ValueError("source config has no model.tokenizer_path; specify --qwen3vl")
    audit = old_al.get("data_audit_file")
    if not audit or not Path(audit).is_absolute():
        raise ValueError("source al_config.json must record an absolute data_audit_file")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.run_name or f"al_skill_from_{model.parent.name}_add{args.steps}_{stamp}_{os.getpid()}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name):
        raise ValueError("--run-name must be 1-100 safe filename characters")
    out = (args.out or args.output_root / name).expanduser().resolve()
    al_config = launcher.read_json(args.al_config)
    for field, attr in AL_CLI_FIELDS.items():
        value = getattr(args, attr, None)
        if value is not None:
            al_config[field] = value
    protected = {"enabled", "task_names", "learning_task_names", "replay_task_names",
                 "data_audit_file", "max_global_steps"}
    fields = {field.name for field in dataclasses.fields(launcher.AutoLearningConfig)}
    for item in getattr(args, "al_set", []):
        if "=" not in item:
            raise ValueError(f"--al-set expects key=JSON_VALUE: {item!r}")
        key, raw = item.split("=", 1)
        if key in protected or key not in fields:
            raise ValueError(f"--al-set cannot override unknown or launcher-owned AL field: {key}")
        try:
            al_config[key] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--al-set value must be valid JSON for {key}: {raw!r}") from exc
    launcher.AutoLearningConfig(**al_config)
    config = SimpleNamespace(
        model=model, train_config=training, dataset_root=dataset, qwen3vl=Path(qwen),
        out=out, audit_report=Path(audit), al_config=args.al_config.expanduser().resolve(),
        al_config_data=al_config,
        manifest=args.manifest.expanduser().resolve(), learning_tasks=args.learning_tasks, replay_tasks=None,
        steps=args.steps, micro=args.micro, gas=args.gas, new_ratio=1.0,
        python=sys.executable, master_port=args.master_port)
    plan = launcher.build_plan(config)  # 同一 CPU 预检、来源配置与数据划分契约。
    command = [sys.executable, str(Path(launcher.__file__).resolve())]
    for flag in ("model", "train_config", "dataset_root", "qwen3vl", "out", "audit_report",
                 "al_config", "manifest", "steps", "micro", "gas", "master_port"):
        command += ["--" + flag.replace("_", "-"), str(getattr(config, flag))]
    if args.learning_tasks is not None:
        command += ["--learning-tasks", *args.learning_tasks]
    command += ["--python", sys.executable, "--execute"]
    runtime = (args.runtime_dir or args.output_root / "skill_launches").expanduser().resolve()
    job = dict(command=command, output_dir=str(out), source_model=str(model), steps=args.steps,
               shutdown=args.shutdown, log=str(runtime / f"{name}.log"),
               job_file=str(runtime / f"{name}.json"),
               status_file=str(runtime / f"{name}.status.json"))
    return job, plan


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def check_machine(job, plan):
    # GPU 有任务时不启动第二份训练；不停止或清理任何现有任务。
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise RuntimeError(f"GPU compute processes are still active: {result.stdout.strip()}")
    parent = Path(job["output_dir"]).parent
    while not parent.exists():
        parent = parent.parent
    count = math.ceil(job["steps"] / 250)
    weight_bytes = sum(row["size"] for row in plan["checkpoint_inventory"])
    needed = math.ceil(count * weight_bytes * 1.1) + 1024**3
    free = shutil.disk_usage(parent).free
    if free < needed:
        raise RuntimeError(f"not enough disk for {count} HF checkpoints: "
                           f"need {needed / 1024**3:.1f} GiB, free {free / 1024**3:.1f} GiB")


def start_background(job, plan):
    path = Path(job["job_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # 冻结本次 AL 配置；以后改模板只影响后续启动。
    snapshot = path.with_suffix(".al.json")
    for artifact in (path, snapshot, Path(job["log"]), Path(job["status_file"])):
        if artifact.exists():
            raise FileExistsError(f"launch artifact already exists: {artifact}")
    write_json(snapshot, plan["config"])
    job["command"][job["command"].index("--al-config") + 1] = str(snapshot)
    write_json(path, job)
    write_json(job["status_file"], dict(state="queued", training_complete=False,
                                       shutdown=job["shutdown"]))
    with open(job["log"], "xb") as log:
        child = subprocess.Popen(
            ["setsid", "nohup", sys.executable, str(Path(__file__).resolve()), "--job", str(path)],
            cwd=launcher.REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    return child.pid


def verify_final(job):
    model = Path(job["output_dir"]) / "checkpoints" / f"global_step_{job['steps']}" / "hf_ckpt"
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (model / name).is_file():
            raise RuntimeError(f"final checkpoint asset missing: {model / name}")
    launcher.checkpoint_inventory(model)
    source = launcher.read_json(Path(job["source_model"]) / "model.safetensors.index.json")
    final = launcher.read_json(model / "model.safetensors.index.json")
    if set(source["weight_map"]) != set(final["weight_map"]):
        raise RuntimeError("final checkpoint tensor names disagree with source model")
    return model


def run_job(job):
    status = dict(state="running", training_complete=False, shutdown=job["shutdown"],
                  output_dir=job["output_dir"])
    write_json(job["status_file"], status)
    try:
        result = subprocess.run(job["command"], cwd=launcher.REPO, check=False)
        status["returncode"] = result.returncode
        if result.returncode != 0:
            status["state"] = "training_failed"
            write_json(job["status_file"], status)
            print("训练失败，保留服务器，不自动关机。", flush=True)
            return 1
        final = verify_final(job)
        status.update(state="complete", training_complete=True, final_hf=str(final))
        write_json(job["status_file"], status)
        print(f"训练正常结束，最终 HF 已校验：{final}", flush=True)
        if job["shutdown"]:
            status["state"] = "shutdown_requested"
            write_json(job["status_file"], status)
            # AutoDL 的 shutdown 文件可能没有 shebang，必须显式交给 bash。
            print("执行自动关机：bash /usr/bin/shutdown", flush=True)
            subprocess.run(["bash", "/usr/bin/shutdown"], check=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        status.update(state="shutdown_failed" if status["training_complete"] else "verification_or_run_failed",
                      error=repr(exc))
        write_json(job["status_file"], status)
        print(f"未能完成流程：{exc!r}；不重试关机，请检查日志。", flush=True)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-hf", type=Path, help="来源 run/checkpoints/global_step_N/hf_ckpt")
    parser.add_argument("--steps", type=int, default=500, help="新增训练步数，默认500")
    parser.add_argument("--micro", type=int, default=24)
    parser.add_argument("--gas", type=int, default=1)
    parser.add_argument("--learning-tasks", nargs="+")
    for field, name in AL_CLI_FIELDS.items():
        if field in {"eval_interval_steps", "min_steps_before_defer", "defer_retry_steps"}:
            parser.add_argument("--" + name.replace("_", "-"), dest=name, type=int)
        else:
            parser.add_argument("--" + name.replace("_", "-"), dest=name,
                                action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--al-set", action="append", default=[], metavar="KEY=JSON_VALUE",
                        help="任意AutoLearningConfig字段覆盖，可重复；不可覆盖任务/预算上限等启动器字段")
    parser.add_argument("--al-config", type=Path, default=launcher.REPO / "configs/auto_learning/learn_skill_mix_task.json")
    parser.add_argument("--manifest", type=Path, default=launcher.REPO / "refs/al_artifacts/al/task_splits_50/manifest.json")
    parser.add_argument("--qwen3vl", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("/data/outputs"))
    parser.add_argument("--out", type=Path, help="可选独立输出目录；必须不存在")
    parser.add_argument("--run-name", help="固定本次运行名，便于预览和执行使用同一路径")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--master-port", type=int, default=62510)
    parser.add_argument("--shutdown", action=argparse.BooleanOptionalAction, default=False,
                        help="正常结束且最终HF完整后自动关机；默认关闭")
    parser.add_argument("--execute", action="store_true", help="预检后后台启动；不加则仅预览")
    parser.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.job is not None:
        return run_job(launcher.read_json(args.job))
    if args.from_hf is None:
        parser.error("--from-hf is required")
    job, plan = make_job(args)
    print(json.dumps(dict(source_model=job["source_model"], added_steps=args.steps,
        micro=args.micro, gas=args.gas, tasks=plan["config"]["learning_task_names"],
        hardness_scan_enabled=plan["config"]["hardness_scan_enabled"], shutdown=args.shutdown,
        al={key: plan["config"].get(key) for key in (
            "hardness_scan_enabled", "task_scan_enabled", "pass_transition_enabled",
            "eval_interval_steps", "min_steps_before_defer", "defer_retry_steps",
            "early_defer_on_overfit", "review_after_task_transitions",
            "refresh_hardness_on_continue")},
        output_dir=job["output_dir"], log=job["log"], status_file=job["status_file"]),
        ensure_ascii=False, indent=2))
    if args.execute:
        check_machine(job, plan)
        print(f"后台启动 PID={start_background(job, plan)}\n日志：{job['log']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
