#!/usr/bin/env python3
"""Run the RoboTwin benchmark through this repository's NVIDIA launcher."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_CONFIGS = ("demo_clean", "demo_randomized")
SUCCESS_RE = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([\d.]+)%")
TASK_LIST_RE = re.compile(r"^task_list_all=\((.*)\)$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RoboTwin benchmark using one NVIDIA inference GPU."
    )
    parser.add_argument("--gpu-count", type=int, default=1, choices=(1,),
                        help="This adapter uses one NVIDIA GPU (default: 1).")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-tasks", type=int, default=50,
                        help="Run the first N tasks from RoboTwin's official 50-task list.")
    parser.add_argument("--task-config", default="both",
                        choices=("both", *TASK_CONFIGS))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runtime-dir", type=Path,
                        default=Path("/data/eval_results/clean_benchmark"))
    parser.add_argument("--robotwin-root", type=Path,
                        default=Path("/data/code/RoboTwin-lingbot"))
    parser.add_argument("--conda-sh", type=Path,
                        default=Path("/data/miniconda3/etc/profile.d/conda.sh"))
    parser.add_argument("--inference-env", default="lingbotvla")
    parser.add_argument("--sim-env", default="RoboTwin")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="HF checkpoint directory, e.g. global_step_N/hf_ckpt.")
    parser.add_argument("--qwen3vl-path", type=Path,
                        default=Path(os.environ.get("QWEN3VL_PATH", "/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct")))
    parser.add_argument("--base-port", type=int, default=9330)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def benchmark_configs(task_config: str) -> tuple[str, ...]:
    return TASK_CONFIGS if task_config == "both" else (task_config,)


def load_tasks() -> list[str]:
    launcher = REPO_ROOT / "experiment/robotwin/start_robotwin_infer_and_eval.sh"
    text = launcher.read_text(encoding="utf-8")
    match = TASK_LIST_RE.search(text)
    if not match:
        raise RuntimeError(f"Could not find the official task list in {launcher}")
    tasks = shlex.split(match.group(1))
    if len(tasks) != 50 or len(set(tasks)) != 50 or not all(isinstance(t, str) for t in tasks):
        raise ValueError(f"Expected 50 unique task names in {launcher}")
    return tasks


def read_result(log_path: Path) -> tuple[int, int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text).replace("\r", "\n")
    matches = SUCCESS_RE.findall(text)
    if not matches:
        raise RuntimeError(f"Missing success-rate line in {log_path}")
    successes, episodes, _ = matches[-1]
    return int(successes), int(episodes)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _manifest(args: argparse.Namespace, tasks: list[str]) -> dict[str, Any]:
    model_path = args.model_path.expanduser().resolve()
    index = model_path / "model.safetensors.index.json"
    return {
        "model_path": str(model_path),
        "model_index_size": index.stat().st_size,
        "model_index_mtime_ns": index.stat().st_mtime_ns,
        "qwen3vl_path": str(args.qwen3vl_path.expanduser().resolve()),
        "robotwin_root": str(args.robotwin_root.expanduser().resolve()),
        "episodes": args.episodes,
        "num_tasks": len(tasks),
        "task_configs": list(benchmark_configs(args.task_config)),
        "tasks": tasks,
        "precision": args.precision,
        "use_compile": args.use_compile,
        "video": args.video,
        "inference_env": args.inference_env,
        "sim_env": args.sim_env,
        "conda_sh": str(args.conda_sh.expanduser().resolve()),
        "base_port": args.base_port,
    }


def _launcher_command(args: argparse.Namespace, run_dir: Path, config: str,
                      tasks_file: Path) -> list[str]:
    launcher = REPO_ROOT / "experiment/robotwin/start_robotwin_infer_and_eval.sh"
    use_bf16 = args.precision == "bf16"
    return [
        "bash", str(launcher),
        "--inference_workdir", str(REPO_ROOT),
        "--model_path", str(args.model_path.expanduser().resolve()),
        "--eval_workdir", str(args.robotwin_root.expanduser().resolve()),
        "--output_base", str(run_dir / "launcher_outputs" / config),
        "--start_port", str(args.base_port),
        "--task_list_file", str(tasks_file),
        "--test_num", str(args.episodes),
        "--num_gpus", "1", "--num_per_gpu", "1",
        "--use_bf16", str(use_bf16), "--use_fp32", str(not use_bf16),
        "--use_compile", str(args.use_compile),
        "--task_config", config,
        "--inference_env", args.inference_env,
        "--sim_env", args.sim_env,
        "--conda_sh", str(args.conda_sh.expanduser().resolve()),
        "--no_video" if not args.video else "--enable_video",
    ]


def _done_path(run_dir: Path, config: str, task: str) -> Path:
    return run_dir / "done" / f"{config}__{task}.json"


def _completed_tasks(run_dir: Path, config: str, tasks: list[str], episodes: int) -> set[str]:
    completed: set[str] = set()
    for path in (run_dir / "done").glob(f"{config}__*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        task = path.stem.removeprefix(f"{config}__")
        if (task not in tasks or row.get("task_config") != config or row.get("task") != task
                or row.get("episodes") != episodes or not 0 <= row.get("successes", -1) <= episodes):
            raise RuntimeError(f"Invalid or incompatible completion marker: {path}")
        completed.add(task)
    return completed


def _record_task(run_dir: Path, config: str, task: str, episodes: int,
                 launcher_run: Path) -> tuple[int, int] | None:
    log_path = launcher_run / "eval_logs" / f"{task}.log"
    try:
        successes, completed = read_result(log_path)
    except (OSError, RuntimeError):
        return None
    if completed != episodes:
        return None
    _atomic_json(_done_path(run_dir, config, task), {
        "task_config": config,
        "task": task,
        "successes": successes,
        "episodes": completed,
        "log_path": str(log_path),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    })
    return successes, completed


def _find_new_launcher_run(output_base: Path, before: set[Path]) -> Path | None:
    created = set(output_base.iterdir()) - before if output_base.is_dir() else set()
    directories = [path for path in created if path.is_dir()]
    if len(directories) > 1:
        raise RuntimeError(f"Expected at most one new launcher output, found {directories}")
    return directories[0] if directories else None


def _run_config(args: argparse.Namespace, run_dir: Path, config: str,
                pending: list[str]) -> None:
    if not pending:
        return
    tasks_file = run_dir / f"{config}_pending_tasks.txt"
    tasks_file.write_text("\n".join(pending) + "\n", encoding="utf-8")
    output_base = run_dir / "launcher_outputs" / config
    output_base.mkdir(parents=True, exist_ok=True)
    before = set(output_base.iterdir())
    launcher_log = run_dir / f"launcher_{config}.log"
    command = _launcher_command(args, run_dir, config, tasks_file)
    env = os.environ.copy()
    env["QWEN3VL_PATH"] = str(args.qwen3vl_path.expanduser().resolve())
    print(f"Starting {config}: {len(pending)} tasks, one NVIDIA GPU; log={launcher_log}", flush=True)
    with launcher_log.open("ab") as log_stream:
        result = subprocess.run(command, cwd=REPO_ROOT, env=env,
                                stdout=log_stream, stderr=subprocess.STDOUT, check=False)
    launcher_run = _find_new_launcher_run(output_base, before)
    recorded: list[str] = []
    if launcher_run is not None:
        for task in pending:
            if _record_task(run_dir, config, task, args.episodes, launcher_run) is not None:
                recorded.append(task)
    if result.returncode != 0:
        raise RuntimeError(
            f"NVIDIA launcher exited {result.returncode}; completed tasks saved={recorded}; "
            f"inspect {launcher_log}"
        )
    missing = sorted(set(pending) - set(recorded))
    if missing:
        raise RuntimeError(
            f"Launcher returned success but these tasks lack {args.episodes} completed episodes: "
            f"{missing}; inspect {launcher_log}"
        )


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not 1 <= args.num_tasks <= 50:
        raise ValueError("--num-tasks must be in [1, 50]")
    if not 1 <= args.base_port <= 65535:
        raise ValueError("--base-port must be in [1, 65535]")

    root = args.robotwin_root.expanduser().resolve()
    model = args.model_path.expanduser().resolve()
    if not (model / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Incomplete HF checkpoint: {model}")
    if not args.qwen3vl_path.expanduser().is_dir():
        raise FileNotFoundError(f"Qwen3-VL checkpoint not found: {args.qwen3vl_path}")
    tasks = load_tasks()[:args.num_tasks]
    run_name = args.run_name or (
        f"{args.task_config}_{len(benchmark_configs(args.task_config)) * len(tasks)}x"
        f"{args.episodes}_{model.parent.name}"
    )
    run_dir = args.runtime_dir.expanduser().resolve() / "outputs" / run_name
    manifest = _manifest(args, tasks)
    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.resume:
            raise RuntimeError(f"{run_dir} is not empty; choose a new --run-name or pass --resume")
        if not manifest_path.is_file():
            raise RuntimeError(f"Cannot resume without benchmark manifest: {manifest_path}")
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_manifest != manifest:
            raise RuntimeError("Resume parameters differ from run_manifest.json; use a new --run-name")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "done").mkdir()
        _atomic_json(manifest_path, manifest)

    summaries: dict[str, dict[str, int]] = {}
    for config in benchmark_configs(args.task_config):
        completed = _completed_tasks(run_dir, config, tasks, args.episodes)
        pending = [task for task in tasks if task not in completed]
        print(f"{config}: completed={len(completed)} pending={len(pending)}")
        _run_config(args, run_dir, config, pending)
        rows = [json.loads(_done_path(run_dir, config, task).read_text(encoding="utf-8"))
                for task in tasks]
        summaries[config] = {
            "successes": sum(row["successes"] for row in rows),
            "episodes": sum(row["episodes"] for row in rows),
        }

    _atomic_json(run_dir / "summary.json", summaries)
    for config, result in summaries.items():
        rate = 100.0 * result["successes"] / result["episodes"]
        print(f"{config}: {result['successes']}/{result['episodes']} = {rate:.2f}%")
    print(f"results: {run_dir}")


if __name__ == "__main__":
    main()
