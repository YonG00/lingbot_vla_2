from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(
    "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
)

OUT_DIR = Path("/data/results/robotwin_dataset_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUT_DIR / "robotwin_50_blocks.csv"
TXT_OUT = OUT_DIR / "robotwin_50_blocks_examples.txt"

FPS = 15
EPISODES_PER_TASK = 50


# --------------------------------------------------
# Load episode metadata
# --------------------------------------------------

files = sorted(
    (ROOT / "meta" / "episodes").rglob("*.parquet")
)

dfs = [pd.read_parquet(p) for p in files]
ep = pd.concat(dfs, ignore_index=True)

if "episode_index" not in ep.columns:
    raise RuntimeError(
        f"episode_index missing. columns={ep.columns.tolist()}"
    )

if "length" not in ep.columns:
    raise RuntimeError(
        f"length missing. columns={ep.columns.tolist()}"
    )

ep = (
    ep.sort_values("episode_index")
      .drop_duplicates("episode_index")
      .reset_index(drop=True)
)


# --------------------------------------------------
# Extract instruction
# --------------------------------------------------

def first_item(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()

    if isinstance(x, (list, tuple)):
        return x[0] if len(x) else ""

    return "" if x is None else str(x)


if "tasks" in ep.columns:
    ep["instruction"] = ep["tasks"].apply(first_item)

elif "task" in ep.columns:
    ep["instruction"] = ep["task"].astype(str)

else:
    ep["instruction"] = ""


# --------------------------------------------------
# Sanity check
# --------------------------------------------------

print("Total episodes:", len(ep))
print("Min episode:", ep["episode_index"].min())
print("Max episode:", ep["episode_index"].max())

if len(ep) != 2500:
    print(
        "WARNING: expected 2500 episodes, got",
        len(ep)
    )


# --------------------------------------------------
# Build sequential blocks of 50 episodes
# --------------------------------------------------

ep["block_id"] = ep["episode_index"] // EPISODES_PER_TASK


def q25(x):
    return np.percentile(x, 25)


def q75(x):
    return np.percentile(x, 75)


rows = []
example_sections = []

for block_id, g in ep.groupby("block_id"):

    g = g.sort_values("episode_index")

    lengths = g["length"]

    unique_inst = (
        g["instruction"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    # 取前3和后2条不同指令帮助判断真实task
    examples = unique_inst[:3]

    if len(unique_inst) > 3:
        examples += unique_inst[-2:]

    mean = lengths.mean()
    median = lengths.median()
    std = lengths.std()

    row = {
        "block_id": int(block_id),
        "episode_start": int(g["episode_index"].min()),
        "episode_end": int(g["episode_index"].max()),
        "episodes": len(g),

        "mean_frames": mean,
        "median_frames": median,
        "std_frames": std,

        "min_frames": lengths.min(),
        "p25_frames": q25(lengths),
        "p75_frames": q75(lengths),
        "max_frames": lengths.max(),

        "iqr_frames": q75(lengths) - q25(lengths),

        "cv": std / mean if mean else np.nan,

        "median_sec": median / FPS,

        "unique_instructions": len(unique_inst),

        "example_1": examples[0] if len(examples) > 0 else "",
        "example_2": examples[1] if len(examples) > 1 else "",
        "example_3": examples[2] if len(examples) > 2 else "",
        "example_4": examples[3] if len(examples) > 3 else "",
        "example_5": examples[4] if len(examples) > 4 else "",
    }

    rows.append(row)

    example_sections.append(
        "\n".join(
            [
                f"BLOCK {block_id}",
                f"Episodes: {g['episode_index'].min()} - {g['episode_index'].max()}",
                f"Count: {len(g)}",
                f"Median: {median:.1f} frames",
                f"Mean: {mean:.1f} frames",
                f"Range: {lengths.min()} - {lengths.max()}",
                "",
            ]
            +
            [
                f"  - {x}"
                for x in examples
            ]
            +
            [
                "",
                "-" * 80,
                "",
            ]
        )
    )


stats = pd.DataFrame(rows)

# 同时给一个按照 median 排序后的 rank
rank_map = (
    stats
    .sort_values(["median_frames", "mean_frames"])
    .reset_index(drop=True)
)

rank_map["length_rank"] = (
    np.arange(len(rank_map)) + 1
)

stats = stats.merge(
    rank_map[["block_id", "length_rank"]],
    on="block_id",
)

stats = stats[
    [
        "block_id",
        "length_rank",
        "episode_start",
        "episode_end",
        "episodes",
        "mean_frames",
        "median_frames",
        "std_frames",
        "min_frames",
        "p25_frames",
        "p75_frames",
        "max_frames",
        "iqr_frames",
        "cv",
        "median_sec",
        "unique_instructions",
        "example_1",
        "example_2",
        "example_3",
        "example_4",
        "example_5",
    ]
]

stats.to_csv(CSV_OUT, index=False)

with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(example_sections))


print()
print("Blocks:", len(stats))
print("Episodes:", stats["episodes"].sum())

print()
print("Saved:")
print(CSV_OUT)
print(TXT_OUT)
