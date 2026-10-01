"""RoboTwin 课程数据映射 —— 共享模块。

本模块集中维护「回合号 -> 任务 -> 技能等级」的唯一映射逻辑, 供
`build_robotwin_curriculum_manifest.py` 与 `prepare_phase.py` 共用,
避免 TASK_ORDER 出现两个真源。

关键不变量
----------
RoboTwin_lerobot_v30 是把 50 个任务各 50 个回合合并而成的单份数据集,
且**按任务连续分块**:

    block_id = episode_index // 50
    task     = TASK_ORDER[block_id]

`derive_episode_table()` 会断言这一结构, 一旦数据集换了布局就会立即失败,
而不是静默地把等级算错。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# 50 个 RoboTwin 任务的固定顺序, 与数据集的分块顺序一一对应。
# 与 build_robotwin_curriculum_manifest.py 中的 TASK_ORDER 保持一致。
TASK_ORDER = [
    "adjust_bottle",
    "click_bell",
    "hanging_mug",
    "move_stapler_pad",
    "place_a2b_left",
    "place_can_basket",
    "place_fan",
    "place_phone_stand",
    "rotate_qrcode",
    "stack_blocks_two",
    "beat_block_hammer",
    "dump_bin_bigbin",
    "lift_pot",
    "open_laptop",
    "place_a2b_right",
    "place_cans_plasticbox",
    "place_mouse_pad",
    "place_shoe",
    "scan_object",
    "stack_bowls_three",
    "blocks_ranking_rgb",
    "grab_roller",
    "move_can_pot",
    "open_microwave",
    "place_bread_basket",
    "place_container_plate",
    "place_object_basket",
    "press_stapler",
    "shake_bottle",
    "stack_bowls_two",
    "blocks_ranking_size",
    "handover_block",
    "move_pillbottle_pad",
    "pick_diverse_bottles",
    "place_bread_skillet",
    "place_dual_shoes",
    "place_object_scale",
    "put_bottles_dustbin",
    "shake_bottle_horizontally",
    "stamp_seal",
    "click_alarmclock",
    "handover_mic",
    "move_playingcard_away",
    "pick_dual_bottles",
    "place_burger_fries",
    "place_empty_cup",
    "place_object_stand",
    "put_object_cabinet",
    "stack_blocks_three",
    "turn_switch",
]

EPISODES_PER_TASK = 50
LEVELS = ["L1", "L2", "L3", "L4"]

EXPECTED_TOTAL_EPISODES = 2500
EXPECTED_TOTAL_FRAMES = 548893


# ---------------------------------------------------------------- 配置读取


def load_curriculum(yaml_path) -> dict:
    """读取课程配置 yaml。"""
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_skill_levels(cfg: dict) -> dict[str, str]:
    """从课程配置得到 task -> 等级 的映射。

    若有任务重复出现在多个等级, 或任务集合与 TASK_ORDER 不一致, 直接报错。
    """
    task_to_level: dict[str, str] = {}
    for level, info in cfg["skill_levels"].items():
        for task in info["tasks"]:
            if task in task_to_level:
                raise RuntimeError(
                    f"任务 {task} 同时出现在 {task_to_level[task]} 和 {level} 中"
                )
            task_to_level[task] = level

    missing = set(TASK_ORDER) - set(task_to_level)
    extra = set(task_to_level) - set(TASK_ORDER)
    if missing or extra:
        raise RuntimeError(
            f"课程配置与 TASK_ORDER 不一致:\n"
            f"  配置中缺少: {sorted(missing)}\n"
            f"  TASK_ORDER 中缺少: {sorted(extra)}"
        )
    return task_to_level


def load_phase_defs(cfg: dict) -> dict:
    """从课程配置得到阶段定义, 形如 {"P1": {"levels": [...], "dir_name": ...}}。"""
    if "phases" not in cfg:
        raise RuntimeError(
            "课程配置缺少 `phases:` 段。请把旧的 C1..C8 8阶段设计替换为 4 个 phase。"
        )
    return cfg["phases"]


def derive_episode_table(dataset_root, task_to_level: dict[str, str]) -> pd.DataFrame:
    """从 meta/episodes 构建「回合 -> 任务 -> 等级」表。

    返回列: episode_index, block_id, task, skill_level, length, instruction
    """
    root = Path(dataset_root)
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"未找到回合元数据: {root}/meta/episodes/**/*.parquet")

    meta = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    meta = (
        meta.sort_values("episode_index")
        .drop_duplicates("episode_index")
        .reset_index(drop=True)
    )

    # ---- 结构断言: 这些是等级映射的地基, 不成立就不能继续 ----
    if len(meta) != EXPECTED_TOTAL_EPISODES:
        raise RuntimeError(
            f"回合数不符: 期望 {EXPECTED_TOTAL_EPISODES}, 实际 {len(meta)}"
        )

    actual_ids = meta["episode_index"].to_numpy(dtype=int)
    if not np.array_equal(np.arange(EXPECTED_TOTAL_EPISODES), actual_ids):
        raise RuntimeError("episode_index 不是恰好 0..2499")

    total_frames = int(meta["length"].sum())
    if total_frames != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError(
            f"总帧数不符: 期望 {EXPECTED_TOTAL_FRAMES}, 实际 {total_frames}"
        )

    # ---- 块映射 ----
    meta["block_id"] = (meta["episode_index"] // EPISODES_PER_TASK).astype(int)
    if meta["block_id"].max() != len(TASK_ORDER) - 1:
        raise RuntimeError(
            f"块数不符: 最大 block_id={meta['block_id'].max()}, "
            f"期望 {len(TASK_ORDER) - 1}"
        )

    meta["task"] = meta["block_id"].map(lambda x: TASK_ORDER[x])

    counts = meta.groupby("task").size()
    if meta["task"].nunique() != len(TASK_ORDER) or not (counts == EPISODES_PER_TASK).all():
        raise RuntimeError(
            f"每个任务应恰好 {EPISODES_PER_TASK} 个回合, 实际分布: "
            f"{sorted(set(counts.values))}"
        )

    meta["skill_level"] = meta["task"].map(task_to_level)
    if meta["skill_level"].isna().any():
        bad = sorted(meta.loc[meta["skill_level"].isna(), "task"].unique())
        raise RuntimeError(f"以下任务没有等级: {bad}")

    # instruction: meta.episodes 的 tasks 字段是长度 1 的列表
    if "tasks" in meta.columns:
        meta["instruction"] = meta["tasks"].map(
            lambda x: (x[0] if isinstance(x, (list, tuple, np.ndarray)) and len(x) else str(x))
        )
    else:
        meta["instruction"] = ""

    return meta[
        ["episode_index", "block_id", "task", "skill_level", "length", "instruction"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------- 阶段解析


def resolve_phase_episodes(table: pd.DataFrame, phase_levels) -> list[int]:
    """给定等级列表, 返回该阶段的回合号 (升序)。"""
    sub = table[table["skill_level"].isin(list(phase_levels))]
    if sub.empty:
        raise RuntimeError(f"等级 {phase_levels} 没有匹配到任何回合")
    return sorted(int(x) for x in sub["episode_index"])


def summarize(table: pd.DataFrame, phase_levels) -> dict:
    """阶段概要: 任务数 / 回合数 / 帧数。"""
    sub = table[table["skill_level"].isin(list(phase_levels))]
    return {
        "levels": list(phase_levels),
        "tasks": int(sub["task"].nunique()),
        "episodes": int(len(sub)),
        "frames": int(sub["length"].sum()),
    }
