#!/usr/bin/env python3
"""汇总开环评测结果 -> registry.json（基线登记表）。

扫描 `<root>/<tag>/<task>/eval.log`：
  * 解析逐条 `MSE for trajectory <id>: <mse>, MAE: <mae>`
  * 用 `/data/train/task_splits/manifest.json` 的 `train_ids` / `val_ids` 分组
  * 计算该任务「只输出常数均值」的 MSE 下界（该任务全部回合、各维 `std²` 的均值）

写入 `<root>/registry.json`。**以后每跑完一个 ckpt 重跑本脚本即可自动并入。**

用法:
    python tools/collect_open_loop.py                 # 扫描默认 root 并写 registry.json
    python tools/collect_open_loop.py --print         # 只打印表格，不写文件
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robotwin_curriculum as rc  # noqa: E402

DEFAULT_ROOT = "/data/eval_results/open_loop"
SPLIT_DIR = "/data/train/task_splits"
DATASET = "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
EPT = int(getattr(rc, "EPISODES_PER_TASK", 50))
TASK_ORDER = list(rc.TASK_ORDER)

_floor_cache: dict[str, float | None] = {}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def parse_log(path: Path) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for ln in path.read_text(errors="ignore").splitlines():
        m = re.search(r"trajectory (\d+): ([-\d.eE+]+), MAE: ([-\d.eE+]+)", ln)
        if m:
            out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def task_floor(task: str) -> float | None:
    """该任务「只输出常数均值」的 MSE 下界 = 各维 std² 的均值（物理单位）。"""
    if task in _floor_cache:
        return _floor_cache[task]
    try:
        b = TASK_ORDER.index(task)
        files = sorted(glob.glob(f"{DATASET}/data/**/*.parquet", recursive=True))
        if not files:
            raise RuntimeError("找不到 data 分片")
        df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action"]) for f in files], ignore_index=True)
        sub = df[df["episode_index"].between(b * EPT, b * EPT + EPT - 1)]
        if len(sub) == 0:
            raise RuntimeError("该任务无数据")
        A = np.stack(sub["action"].to_numpy())
        val = float(A.var(axis=0).mean())
        _floor_cache[task] = val
        log(f"下界 {task}: {val:.4f}（{len(sub)} 帧）")
        return val
    except Exception as exc:  # noqa: BLE001
        log(f"⚠️ 下界计算失败（{task}）: {type(exc).__name__}: {exc}")
        _floor_cache[task] = None
        return None


def split_of(task: str) -> dict:
    p = Path(SPLIT_DIR) / "manifest.json"
    if not p.is_file():
        return {}
    try:
        m = json.loads(p.read_text())
        return m.get("tasks", {}).get(task, {})
    except Exception:  # noqa: BLE001
        return {}


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def sem(xs: list[float]) -> float | None:
    return st.stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--print", dest="do_print", action="store_true", help="只打印表格，不写文件")
    a = ap.parse_args()

    logs = sorted(Path(a.root).glob("*/*/eval.log"))
    if not logs:
        log(f"❌ {a.root} 下没有 */<task>/eval.log")
        return 1

    runs = []
    for lg in logs:
        tag, task = lg.parent.parent.name, lg.parent.name
        per = parse_log(lg)
        if not per:
            log(f"⚠️ 跳过（无结果行）: {lg}")
            continue
        sp = split_of(task)
        tr_ids = sp.get("train_ids") or []
        va_ids = sp.get("val_ids") or []
        other = [i for i in per if i not in tr_ids and i not in va_ids]
        fl = task_floor(task)

        def agg(ids):
            xs = [per[i] for i in ids if i in per]
            if not xs:
                return None
            return {"n": len(xs), "mse": mean([x[0] for x in xs]), "mse_sem": sem([x[0] for x in xs]),
                    "mae": mean([x[1] for x in xs])}

        runs.append({
            "tag": tag, "task": task,
            "log": str(lg),
            "n_traj": len(per),
            "groups": {"train": agg(tr_ids), "val": agg(va_ids),
                       **({"other": agg(other)} if other else {})},
            "per_traj": {str(k): {"mse": v[0], "mae": v[1]} for k, v in sorted(per.items())},
            "floor_mse": fl,
            "r2_val": (None if not fl or not agg(va_ids) else 1 - agg(va_ids)["mse"] / fl),
        })

    reg = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(a.root),
        "note": "开环诊断用；spec 明确 open-loop 不作为模型选择依据，选型看闭环成功率。"
                "floor_mse = 该任务「只输出常数均值」的 MSE（各维 std² 均值）⇒ MSE < floor 才算学到东西。",
        "runs": runs,
    }

    # 打印表
    print("=" * 96)
    print("开环基线登记表（按 val MSE 排序；R²>0 才算比「输出常数」好）")
    print("%-24s %-12s %10s %10s %10s %10s %10s" % ("tag", "task", "floor", "train MSE", "val MSE", "val SEM", "val R²"))
    print("-" * 96)
    for r in sorted(runs, key=lambda x: (x["groups"].get("val") or {}).get("mse", 9e9)):
        g = r["groups"]
        f = lambda k, fld: ("%.4f" % g[k][fld]) if g.get(k) else "-"
        print("%-24s %-12s %10s %10s %10s %10s %10s" % (
            r["tag"], r["task"],
            ("%.4f" % r["floor_mse"]) if r["floor_mse"] else "-",
            f("train", "mse"), f("val", "mse"),
            ("%.4f" % g["val"]["mse_sem"]) if g.get("val") and g["val"]["mse_sem"] else "-",
            ("%+.3f" % r["r2_val"]) if r["r2_val"] is not None else "-"))
    print("=" * 96)

    if not a.do_print:
        out = Path(a.root) / "registry.json"
        out.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log(f"写出 {out}（{len(runs)} 条 run）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
