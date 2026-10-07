#!/usr/bin/env python3
"""按 task 划分「训练 / 验证」回合，并给出开环评测用的轨迹索引。

设计要点
--------
* **独立于课程**：只复用 `tools/robotwin_curriculum.py` 里的两个常量
  （`TASK_ORDER` = 50 个任务名、`EPISODES_PER_TASK` = 每任务回合数），
  **不读课程 yaml、不碰 skill_levels / phases**。
* 数据集块结构：`block_id = episode_index // EPISODES_PER_TASK`，
  `task = TASK_ORDER[block_id]` ⇒ 每个任务恰好 50 个回合，整块属于该任务。
* 划分是**任务内**的：在每个任务自己的 50 个回合里按 `--val-ratio` 切出验证集。
* **确定性**：同样输入 + 同样参数 ⇒ 同样输出（`--strategy seed` 时 seed 固定并写入 manifest）。

产出
----
    <out>/<task>.train_ids.json   裸列表，直接喂 `--data.episode_ids_file`
    <out>/<task>.val_ids.json     裸列表，开环评测用
    <out>/manifest.json           各任务明细 + 参数 + sha256（审计 / 复现）

用法
----
    # 1) 生成划分（默认 --task all 之外必须显式给 --task）
    python tools/task_split.py --task click_bell
    python tools/task_split.py --task all

    # 2) 取开环轨迹索引（stdout 只有数字，日志走 stderr ⇒ 可直接内联）
    python tools/task_split.py --task click_bell --pick-train 5
    python tools/task_split.py --task click_bell --pick-val 10

    # 3) 串起来
    python scripts/open_loop_eval.py --model_path ... --robo_name robotwin \
        --data_path /data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30 \
        --traj_ids $(python tools/task_split.py --task click_bell --pick-val 10)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import robotwin_curriculum as rc  # noqa: E402

TASK_ORDER = list(rc.TASK_ORDER)
EPISODES_PER_TASK = int(getattr(rc, "EPISODES_PER_TASK", 50))
EXPECTED_TOTAL_EPISODES = int(getattr(rc, "EXPECTED_TOTAL_EPISODES", len(TASK_ORDER) * EPISODES_PER_TASK))
EXPECTED_TOTAL_FRAMES = int(getattr(rc, "EXPECTED_TOTAL_FRAMES", 548893))

DEFAULT_DATASET = "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
DEFAULT_OUT = "/data/train/task_splits"
DEFAULT_SEED = 20261004
STRATEGIES = ("quantile", "seed", "stride")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> "None":
    log(f"❌ {msg}")
    sys.exit(1)


# --------------------------------------------------------------------------
# 数据集：回合 -> (任务, 帧数)
# --------------------------------------------------------------------------
def load_episodes(dataset_root: str) -> dict[int, int]:
    """返回 {episode_index: length}，并对数据集结构做硬校验。"""
    files = sorted(Path(dataset_root).glob("meta/episodes/**/*.parquet"))
    if not files:
        die(f"找不到回合元数据: {dataset_root}/meta/episodes/**/*.parquet")

    meta = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "length"]) for f in files],
        ignore_index=True,
    )
    meta = meta.sort_values("episode_index").drop_duplicates("episode_index").reset_index(drop=True)

    ids = meta["episode_index"].to_numpy(dtype=int)
    if len(meta) != EXPECTED_TOTAL_EPISODES or not np.array_equal(ids, np.arange(EXPECTED_TOTAL_EPISODES)):
        die(f"回合数/编号不符: 期望 0..{EXPECTED_TOTAL_EPISODES - 1}，实际 {len(meta)}")
    total_frames = int(meta["length"].sum())
    if total_frames != EXPECTED_TOTAL_FRAMES:
        die(f"总帧数不符: 期望 {EXPECTED_TOTAL_FRAMES}，实际 {total_frames}")

    log(f"数据集校验通过：{len(meta)} 回合 / {total_frames} 帧")
    return dict(zip(ids.tolist(), meta["length"].astype(int).tolist()))


def task_ids(task: str, lengths: dict[int, int]) -> list[int]:
    """某任务的全部回合号（升序）。"""
    if task not in TASK_ORDER:
        die(f"未知任务 '{task}'；用 --list 看全部 {len(TASK_ORDER)} 个任务名")
    b = TASK_ORDER.index(task)
    lo, hi = b * EPISODES_PER_TASK, (b + 1) * EPISODES_PER_TASK
    ids = [i for i in range(lo, hi) if i in lengths]
    if len(ids) != EPISODES_PER_TASK:
        die(f"任务 {task} 应有 {EPISODES_PER_TASK} 个回合，实际 {len(ids)}")
    return ids


# --------------------------------------------------------------------------
# 划分
# --------------------------------------------------------------------------
def split(ids: list[int], lengths: dict[int, int], val_ratio: float,
          strategy: str, seed: int) -> tuple[list[int], list[int]]:
    """返回 (train_ids, val_ids)，均升序。

    * ``quantile``（默认）：**按长度分层** —— 把该任务的回合按长度升序排好、等分成
      ``n_val`` 层，每层取层内中位那一条。⇒ val 的长度分布与整体一致（10 个样本时
      这比「随机不随机」重要得多），且完全确定性。
    * ``seed``：固定种子的确定性抽样。无周期偏置，但小样本下分布可能聚簇。
    * ``stride``：每 ``1/val_ratio`` 条取 1 条。最均匀，但若数据生成有周期会同相偏置。
    """
    n_ids = len(ids)
    if val_ratio <= 0:
        n_val = 0
    elif val_ratio >= 1:
        n_val = n_ids - 1
    else:
        n_val = max(1, min(int(round(n_ids * val_ratio)), n_ids - 1))

    if strategy == "stride":
        step = max(2, int(round(1.0 / val_ratio)))
        val = [x for i, x in enumerate(ids) if i % step == 0][:n_val]
    elif strategy == "quantile":
        order = sorted(ids, key=lambda i: (lengths[i], i))
        val = []
        for j in range(n_val):
            seg = order[j * n_ids // n_val:(j + 1) * n_ids // n_val]
            if seg:
                val.append(seg[len(seg) // 2])
        val = sorted(set(val))
    else:  # seed
        val = sorted(random.Random(seed + ids[0]).sample(ids, n_val))

    val_set = set(val)
    train = [x for x in ids if x not in val_set]
    return sorted(train), sorted(val)


def pick_even(ids: list[int], n: int) -> list[int]:
    """从升序 ids 里等间隔取 n 条（含首尾），确定性。"""
    if n <= 0:
        return []
    if n >= len(ids):
        return list(ids)
    if n == 1:
        return [ids[len(ids) // 2]]
    pos = [int(round(i * (len(ids) - 1) / (n - 1))) for i in range(n)]
    out, seen = [], set()
    for p in pos:
        if p not in seen:
            seen.add(p)
            out.append(ids[p])
    return out


def sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="按 task 划分训练/验证回合，并提供开环评测轨迹索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--task", default=None,
                    help="任务名、逗号分隔的多个任务名，或 'all'（全部 %d 个）" % len(TASK_ORDER))
    ap.add_argument("--list", action="store_true", help="列出全部任务名后退出")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2 (=1/5)")
    ap.add_argument("--strategy", choices=STRATEGIES, default="quantile",
                    help="划分策略：quantile=按长度分层(默认) / seed=固定种子随机 / stride=等间隔")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="--strategy seed 的随机种子")
    ap.add_argument("--out", default=DEFAULT_OUT, help="划分产物目录")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="LeRobot 数据集根")
    ap.add_argument("--pick-train", type=int, default=0, metavar="N", help="从 train 里等间隔抽 N 条打印到 stdout")
    ap.add_argument("--pick-val", type=int, default=0, metavar="N", help="从 val 里等间隔抽 N 条打印到 stdout")
    args = ap.parse_args()

    if args.list:
        for i, t in enumerate(TASK_ORDER):
            print(f"{i:2d}  block {i:2d}  ep {i * EPISODES_PER_TASK:4d}-{i * EPISODES_PER_TASK + EPISODES_PER_TASK - 1:4d}  {t}")
        return 0

    if not args.task:
        die("必须给 --task <名字|all>（或用 --list 查看全部任务）")

    lengths = load_episodes(args.dataset)
    if args.task == "all":
        tasks = list(TASK_ORDER)
    else:
        # 逗号分隔的多个任务（如 `--task click_bell,click_alarmclock`）——
        # 多任务 Auto Learning / 课程子集都需要「一份只含这几个任务的 manifest」，
        # 因为 `TaskCatalog.attach_samples()` 要求 manifest 里**每个**任务在
        # 当前数据集白名单里都有样本（多列任务会直接报错）。
        tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASK_ORDER]
    if unknown:
        die(f"未知任务 {unknown}（用 --list 查看全部 {len(TASK_ORDER)} 个任务名）")
    if not tasks:
        die("--task 解析后是空的")

    # --pick-* 需要划分结果：优先读 manifest，读不到就现算
    manifest_path = Path(args.out) / "manifest.json"
    cached = {}
    if manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8")).get("tasks", {})
        except Exception as exc:  # noqa: BLE001
            log(f"⚠️  manifest 解析失败（{exc}），改为现算")

    records = {}
    for task in tasks:
        ids = task_ids(task, lengths)
        if task in cached and cached[task].get("n_total") == len(ids) and \
                abs(cached[task].get("val_ratio", -1) - args.val_ratio) < 1e-9 and \
                cached[task].get("strategy") == args.strategy:
            train, val = cached[task]["train_ids"], cached[task]["val_ids"]
        else:
            train, val = split(ids, lengths, args.val_ratio, args.strategy, args.seed)

        records[task] = {
            "n_total": len(ids),
            "n_train": len(train),
            "n_val": len(val),
            "train_frames": sum(lengths[i] for i in train),
            "val_frames": sum(lengths[i] for i in val),
            "train_ids": train,
            "val_ids": val,
            "sha256_train": sha256_ids(train),
            "sha256_val": sha256_ids(val),
            "val_ratio": args.val_ratio,
            "strategy": args.strategy,
        }

    # --pick-* 模式：只往 stdout 吐数字，日志都在 stderr
    if args.pick_train or args.pick_val:
        if len(tasks) != 1:
            die("--pick-* 一次只能对一个 task 用")
        rec = records[tasks[0]]
        ids = pick_even(rec["train_ids"], args.pick_train) if args.pick_train else pick_even(rec["val_ids"], args.pick_val)
        which = "train" if args.pick_train else "val"
        log(f"{tasks[0]} / {which}：从 {rec['n_train'] if which == 'train' else rec['n_val']} 条里取 {len(ids)} 条")
        print(" ".join(str(i) for i in ids))
        return 0

    # 写盘
    out = Path(args.out)
    for task, rec in records.items():
        write_json(out / f"{task}.train_ids.json", rec["train_ids"])
        write_json(out / f"{task}.val_ids.json", rec["val_ids"])
    if len(records) > 1:
        # 多任务时额外给一份**合并**白名单，直接喂 `--data.episode_ids_file`
        # （单任务场景不写，避免和历史行为产生多余文件）
        write_json(out / "combined.train_ids.json",
                   sorted(i for r in records.values() for i in r["train_ids"]))
        write_json(out / "combined.val_ids.json",
                   sorted(i for r in records.values() for i in r["val_ids"]))

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": args.dataset,
        "episodes_per_task": EPISODES_PER_TASK,
        "val_ratio": args.val_ratio,
        "strategy": args.strategy,
        "seed": args.seed,
        "note": "开环诊断用；spec 明确 open-loop 不作为模型选择依据，选型看闭环成功率",
        "tasks": records,
    }
    write_json(out / "manifest.json", manifest)

    log(f"写出 {len(records)} 个任务 -> {out}")
    log(f"{'task':<26}{'train':>6}{'val':>6}{'train帧':>10}{'val帧':>9}   val_ids")
    for task, r in records.items():
        head = " ".join(str(i) for i in r["val_ids"][:5])
        more = " ..." if r["n_val"] > 5 else ""
        log(f"{task:<26}{r['n_train']:>6}{r['n_val']:>6}{r['train_frames']:>10}{r['val_frames']:>9}   {head}{more}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
