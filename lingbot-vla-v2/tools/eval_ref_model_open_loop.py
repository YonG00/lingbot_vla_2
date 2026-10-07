#!/usr/bin/env python3
"""参考模型（成品）→ 各任务 open-loop MSE → eval.jsonl（compute_pass_thresholds.py 的输入）。

关键：**一次加载模型、批量测全部任务**，避免「每个任务单独跑 open_loop_eval 反复加载
6B 模型（每次 2–3 分钟）」。做法 = 把 50 个任务的 val 轨迹 id 拼成一份，单次调用
``scripts/open_loop_eval.py``，再按轨迹 id 映射回任务汇总。

用法（远端，/data/code/lingbot-vla-v2 下，lingbotvla env）：
    /data/miniconda3/envs/lingbotvla/bin/python tools/eval_ref_model_open_loop.py \
        --model_path /data/models/lingbot-vla-v2-6b-robotwin/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt \
        --out /data/eval_results/open_loop/ref50k/eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET = "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
DEFAULT_SPLIT = "/data/train/task_splits_50"
DEFAULT_QWEN3VL = "/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct"


def load_task_val_ids(split_dir: Path, val_trajs: int) -> dict:
    """读 per-task *.val_ids.json → {task: [traj_id...]}（每个任务取前 val_trajs 条）。

    ⚠️ 排除 `combined.val_ids.json`（全任务合并文件，不是单个任务）。
    """
    out: dict = {}
    for p in sorted(split_dir.glob("*.val_ids.json")):
        if p.name.startswith("combined"):
            continue
        task = p.name[: -len(".val_ids.json")]
        ids = json.loads(p.read_text())
        out[task] = ids[:val_trajs]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="参考模型 → 各任务 open-loop MSE → eval.jsonl")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", default=DEFAULT_DATASET)
    ap.add_argument("--robo_name", default="robotwin")
    ap.add_argument("--split_dir", default=DEFAULT_SPLIT)
    ap.add_argument("--val_trajs", type=int, default=10, help="每任务取 val 前几条（默认 10 = 全部）")
    ap.add_argument("--use_length", type=int, default=50)
    ap.add_argument("--qwen3vl", default=DEFAULT_QWEN3VL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot_dir", default="/tmp/ref_open_loop_plots")
    a = ap.parse_args()

    split_dir = Path(a.split_dir)
    task_ids = load_task_val_ids(split_dir, a.val_trajs)
    if not task_ids:
        sys.exit(f"❌ {split_dir} 下没有 *.val_ids.json")

    flat: list[int] = []
    id2task: dict[int, str] = {}
    for task, ids in task_ids.items():
        for t in ids:
            flat.append(t)
            id2task[t] = task

    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "open_loop_eval.py"
    if not script.is_file():
        sys.exit(f"❌ 找不到 {script}")

    env = dict(os.environ)
    # 成品 50k 的 yaml 里 tokenizer_path 是占位符 ⇒ 必须真的 export（见 infra.md）
    env["QWEN3VL_PATH"] = a.qwen3vl

    cmd = [
        sys.executable, str(script),
        "--model_path", a.model_path,
        "--robo_name", a.robo_name,
        "--data_path", a.data_path,
        "--use_length", str(a.use_length),
        "--traj_ids", *[str(t) for t in flat],
        "--save_plot_path", a.plot_dir,
        "--no_plot",  # 批量标定：跳过逐条绘图，省 ~数秒/条
    ]
    print(f"▶ 一次评测 {len(task_ids)} 个任务 × {a.val_trajs} 条 = {len(flat)} 条轨迹（模型只加载一次）")

    # 子进程 stdout 实时落盘到日志（可 tail 看进度），跑完再从日志解析逐条 MSE
    mse_log = Path(a.out).parent / "open_loop_eval.log"
    mse_log.parent.mkdir(parents=True, exist_ok=True)
    with open(mse_log, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = mse_log.read_text(errors="ignore")[-2000:] if mse_log.is_file() else "(无日志)"
        sys.exit(f"❌ open_loop_eval 失败 rc={proc.returncode}\n--- 日志尾部 ---\n{tail}")

    mse_by_id: dict[int, float] = {}
    for line in mse_log.read_text(errors="ignore").splitlines():
        m = re.search(r"MSE for trajectory (\d+):\s*([-+\d.eE]+)", line)
        if m:
            mse_by_id[int(m.group(1))] = float(m.group(2))

    rows = []
    missing = []
    for task in sorted(task_ids):
        ids = task_ids[task]
        ms = [mse_by_id[t] for t in ids if t in mse_by_id]
        if len(ms) < len(ids):
            missing.append(task)
        if ms:
            rows.append({
                "task": task,
                "split": "active_val",
                "mse": round(sum(ms) / len(ms), 6),
                "nmse": None,  # 由 compute_pass_thresholds 用 baseline_mse 自洽校验
                "n_traj": len(ms),
            })

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 写出 {len(rows)} 个任务 → {out}")
    if missing:
        print(f"⚠️ 轨迹不全的任务（部分 id 无结果，已按有的算）: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
