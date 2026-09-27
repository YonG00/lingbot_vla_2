from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(
    "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
)
FPS = 15

OUT_DIR = Path("/data/results/robotwin_dataset_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUT_DIR / "robotwin_clean_length_stats.csv"
SUMMARY_OUT = OUT_DIR / "summary.txt"

# --------------------------------------------------
# Load episode metadata
# --------------------------------------------------
episode_files = sorted((ROOT / "meta" / "episodes").rglob("*.parquet"))

if not episode_files:
    raise RuntimeError("No episode parquet files found.")

dfs = [pd.read_parquet(p) for p in episode_files]
ep = pd.concat(dfs, ignore_index=True)

# --------------------------------------------------
# Load task metadata if available
# --------------------------------------------------
tasks_path = ROOT / "meta" / "tasks.parquet"

if tasks_path.exists():
    tasks_df = pd.read_parquet(tasks_path)
else:
    tasks_df = None


def first_task(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()

    if isinstance(x, (list, tuple)):
        return x[0] if len(x) else None

    return x


# --------------------------------------------------
# Resolve task name / key
# --------------------------------------------------
if "tasks" in ep.columns:
    ep["task_key"] = ep["tasks"].apply(first_task)

elif "task" in ep.columns:
    ep["task_key"] = ep["task"]

elif "task_index" in ep.columns and tasks_df is not None:
    task_text_col = None

    for candidate in ["task", "task_name", "name"]:
        if candidate in tasks_df.columns:
            task_text_col = candidate
            break

    if task_text_col is None:
        raise RuntimeError(
            "Found task_index but could not identify task name column "
            f"in tasks.parquet. Columns: {tasks_df.columns.tolist()}"
        )

    mapping = dict(
        zip(tasks_df["task_index"], tasks_df[task_text_col])
    )

    ep["task_key"] = ep["task_index"].map(mapping)

else:
    raise RuntimeError(
        "Could not determine task names from episode metadata. "
        f"Episode columns: {ep.columns.tolist()}"
    )


# --------------------------------------------------
# Stats helpers
# --------------------------------------------------
def percentile(q):
    def func(x):
        return np.percentile(x, q)
    return func


# --------------------------------------------------
# Per-task statistics
# --------------------------------------------------
stats = (
    ep.groupby("task_key")["length"]
    .agg(
        episodes="count",
        total_frames="sum",
        mean_frames="mean",
        median_frames="median",
        std_frames="std",
        min_frames="min",
        p10_frames=percentile(10),
        p25_frames=percentile(25),
        p75_frames=percentile(75),
        p90_frames=percentile(90),
        max_frames="max",
    )
    .reset_index()
)

stats["iqr_frames"] = stats["p75_frames"] - stats["p25_frames"]
stats["range_frames"] = stats["max_frames"] - stats["min_frames"]
stats["cv"] = stats["std_frames"] / stats["mean_frames"]

stats["mean_sec"] = stats["mean_frames"] / FPS
stats["median_sec"] = stats["median_frames"] / FPS
stats["p90_sec"] = stats["p90_frames"] / FPS

stats = stats.sort_values(
    ["median_frames", "mean_frames"]
).reset_index(drop=True)

stats.insert(
    0,
    "length_rank",
    np.arange(1, len(stats) + 1)
)

stats.to_csv(CSV_OUT, index=False)

# --------------------------------------------------
# Summary
# --------------------------------------------------
counts = ep.groupby("task_key").size()

with open(SUMMARY_OUT, "w") as f:
    f.write("RoboTwin clean dataset trajectory analysis\n")
    f.write("========================================\n\n")

    f.write(f"Dataset root: {ROOT}\n")
    f.write(f"FPS: {FPS}\n")
    f.write(f"Episode parquet files: {len(episode_files)}\n")
    f.write(f"Total episodes: {len(ep)}\n")
    f.write(f"Unique tasks: {ep['task_key'].nunique()}\n")
    f.write(f"Total frames: {int(ep['length'].sum())}\n\n")

    f.write("Episodes per task:\n")
    f.write(counts.value_counts().sort_index().to_string())
    f.write("\n\n")

    f.write("Overall episode length distribution:\n")

    for q in [0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]:
        frames = ep["length"].quantile(q)
        f.write(
            f"P{int(q * 100):03d}: "
            f"{frames:.1f} frames "
            f"({frames / FPS:.1f} sec)\n"
        )

    f.write("\n")

    f.write(
        f"Overall mean: {ep['length'].mean():.1f} frames\n"
    )
    f.write(
        f"Overall std:  {ep['length'].std():.1f} frames\n"
    )

    f.write("\nShortest 10 tasks by median length:\n")
    f.write(
        stats[
            [
                "length_rank",
                "task_key",
                "median_frames",
                "mean_frames",
                "cv",
            ]
        ]
        .head(10)
        .round(2)
        .to_string(index=False)
    )

    f.write("\n\nLongest 10 tasks by median length:\n")
    f.write(
        stats[
            [
                "length_rank",
                "task_key",
                "median_frames",
                "mean_frames",
                "cv",
            ]
        ]
        .tail(10)
        .round(2)
        .to_string(index=False)
    )

print("Done.")
print("CSV:", CSV_OUT)
print("Summary:", SUMMARY_OUT)
