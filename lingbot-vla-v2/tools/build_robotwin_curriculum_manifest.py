from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DATASET_ROOT = Path(
    "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
)

CONFIG = Path(
    "/data/code/lingbot-vla-v2/configs/curriculum/robotwin_curriculum_v1.yaml"
)

OUT = Path(
    "/data/results/robotwin_dataset_analysis/curriculum_v1"
)

OUT.mkdir(parents=True, exist_ok=True)


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


FIRST_STAGE = {
    ("L1", "short"): ("C1", 1),
    ("L1", "long"):  ("C2", 2),
    ("L2", "short"): ("C3", 3),
    ("L2", "long"):  ("C4", 4),
    ("L3", "short"): ("C5", 5),
    ("L3", "long"):  ("C6", 6),
    ("L4", "short"): ("C7", 7),
    ("L4", "long"):  ("C8", 8),
}


def first_item(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()

    if isinstance(x, (list, tuple)):
        return x[0] if len(x) else ""

    if x is None:
        return ""

    return str(x)


# ============================================================
# 1. Load curriculum config
# ============================================================

cfg = yaml.safe_load(CONFIG.read_text())

task_to_level = {}

for level, info in cfg["skill_levels"].items():
    for task in info["tasks"]:
        if task in task_to_level:
            raise RuntimeError(
                f"Task appears in multiple levels: {task}"
            )
        task_to_level[task] = level


if set(task_to_level) != set(TASK_ORDER):
    missing = set(TASK_ORDER) - set(task_to_level)
    extra = set(task_to_level) - set(TASK_ORDER)

    raise RuntimeError(
        f"Task mismatch.\n"
        f"Missing in YAML: {sorted(missing)}\n"
        f"Extra in YAML: {sorted(extra)}"
    )


# ============================================================
# 2. Load episode metadata
# ============================================================

files = sorted(
    (DATASET_ROOT / "meta" / "episodes").rglob("*.parquet")
)

if not files:
    raise RuntimeError("No episode metadata found.")

meta = pd.concat(
    [pd.read_parquet(p) for p in files],
    ignore_index=True,
)

meta = (
    meta.sort_values("episode_index")
        .drop_duplicates("episode_index")
        .reset_index(drop=True)
)


if len(meta) != 2500:
    raise RuntimeError(
        f"Expected 2500 episodes, got {len(meta)}"
    )


expected_ids = np.arange(2500)

actual_ids = meta["episode_index"].to_numpy(dtype=int)

if not np.array_equal(expected_ids, actual_ids):
    raise RuntimeError(
        "episode_index is not exactly 0..2499"
    )


# ============================================================
# 3. Assign canonical task
# ============================================================

meta["block_id"] = (
    meta["episode_index"] // 50
).astype(int)

meta["task_round"] = (
    meta["episode_index"] % 50 + 1
).astype(int)

meta["task"] = meta["block_id"].map(
    lambda x: TASK_ORDER[x]
)

meta["skill_level"] = meta["task"].map(
    task_to_level
)


if "tasks" in meta.columns:
    meta["instruction"] = meta["tasks"].apply(first_item)
else:
    meta["instruction"] = ""


# ============================================================
# 4. Rank length inside each task
# ============================================================

parts = []

for block_id in range(50):

    g = meta[
        meta["block_id"] == block_id
    ].copy()

    if len(g) != 50:
        raise RuntimeError(
            f"Block {block_id} has {len(g)} episodes"
        )

    g = g.sort_values(
        ["length", "episode_index"],
        ascending=[True, True],
    ).copy()

    g["length_rank"] = np.arange(1, 51)

    g["length_group"] = np.where(
        g["length_rank"] <= 25,
        "short",
        "long",
    )

    parts.append(g)


manifest = pd.concat(
    parts,
    ignore_index=True,
)


# ============================================================
# 5. Assign first curriculum stage
# ============================================================

stage_names = []
stage_nums = []

for level, group in zip(
    manifest["skill_level"],
    manifest["length_group"],
):

    stage_name, stage_num = FIRST_STAGE[
        (level, group)
    ]

    stage_names.append(stage_name)
    stage_nums.append(stage_num)


manifest["first_stage"] = stage_names
manifest["first_stage_num"] = stage_nums


# ============================================================
# 6. Keep only useful columns
# ============================================================

manifest = manifest[
    [
        "episode_index",
        "block_id",
        "task_round",
        "task",
        "skill_level",
        "length",
        "length_rank",
        "length_group",
        "first_stage",
        "first_stage_num",
        "instruction",
    ]
]

manifest = (
    manifest.sort_values("episode_index")
            .reset_index(drop=True)
)


# ============================================================
# 7. Validation
# ============================================================

if manifest["task"].nunique() != 50:
    raise RuntimeError("Expected 50 tasks")


task_counts = (
    manifest.groupby("task")
            .size()
)

if not (task_counts == 50).all():
    raise RuntimeError(
        "Every task must contain exactly 50 episodes"
    )


split_counts = (
    manifest
    .groupby(["task", "length_group"])
    .size()
    .unstack(fill_value=0)
)

if not (
    (split_counts["short"] == 25).all()
    and
    (split_counts["long"] == 25).all()
):
    raise RuntimeError(
        "Every task must contain 25 short + 25 long"
    )


expected_stage_counts = {
    1: 350,
    2: 700,
    3: 1075,
    4: 1450,
    5: 1725,
    6: 2000,
    7: 2250,
    8: 2500,
}


for stage_num, expected_count in expected_stage_counts.items():

    actual_count = int(
        (manifest["first_stage_num"] <= stage_num).sum()
    )

    if actual_count != expected_count:
        raise RuntimeError(
            f"C{stage_num}: "
            f"expected {expected_count}, "
            f"got {actual_count}"
        )


# ============================================================
# 8. Save ONE manifest
# ============================================================

output_path = (
    OUT / "curriculum_manifest_v1.csv"
)

manifest.to_csv(
    output_path,
    index=False,
)


# ============================================================
# 9. Print summary
# ============================================================

print()
print("=" * 80)
print("RoboTwin Curriculum Manifest V1")
print("=" * 80)

print(f"Output: {output_path}")
print()
print(f"Total episodes: {len(manifest)}")
print(f"Total tasks: {manifest['task'].nunique()}")

print("\nSkill levels:")

for level in ["L1", "L2", "L3", "L4"]:

    n_tasks = (
        manifest.loc[
            manifest["skill_level"] == level,
            "task",
        ]
        .nunique()
    )

    n_eps = int(
        (manifest["skill_level"] == level).sum()
    )

    print(
        f"  {level}: "
        f"{n_tasks} tasks, "
        f"{n_eps} episodes"
    )


print("\nShort / Long:")
print("  Every task = 25 short + 25 long")


print("\nCumulative stages:")

for i in range(1, 9):

    n = int(
        (manifest["first_stage_num"] <= i).sum()
    )

    print(f"  C{i}: {n} episodes")


print()
print("ALL VALIDATION CHECKS: PASS")
print("=" * 80)
