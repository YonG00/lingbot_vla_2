from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(
    "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
)

OUT = Path("/data/results/robotwin_dataset_analysis/qa_v1")
OUT.mkdir(parents=True, exist_ok=True)

# 这里只用于检测“几乎完全不动”
# 不用于判定数据好坏，所以设置得比较保守
MOTION_EPS = 1e-4

# robust z-score 超过该值才标记为统计异常
ROBUST_Z_THRESHOLD = 3.5


def first_item(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) else ""
    if x is None:
        return ""
    return str(x)


def max_true_run(mask):
    best = 0
    cur = 0
    for x in mask:
        if x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def reversal_ratio(x, eps=MOTION_EPS):
    """
    统计各维度运动方向反转比例。
    仅作为异常特征，不直接判定 retry。
    """
    if len(x) < 3:
        return 0.0

    d = np.diff(x, axis=0)

    a = d[:-1]
    b = d[1:]

    valid = (np.abs(a) > eps) & (np.abs(b) > eps)

    denom = valid.sum()
    if denom == 0:
        return 0.0

    reverse = ((a * b) < 0) & valid

    return float(reverse.sum() / denom)


def robust_z_transform(series):
    """
    每个 task 内部计算 robust z-score。
    优先使用 MAD，MAD=0 时依次退化到 IQR/std。
    """
    x = series.to_numpy(dtype=float)

    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < 1e-12:
        q25, q75 = np.nanpercentile(x, [25, 75])
        iqr = q75 - q25
        scale = iqr / 1.349 if iqr > 0 else 0.0

    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.nanstd(x)

    if not np.isfinite(scale) or scale < 1e-12:
        z = np.zeros_like(x)
    else:
        z = (x - med) / scale

    return pd.Series(z, index=series.index)


print("=" * 80)
print("RoboTwin Dataset QA v1")
print("Dataset:", ROOT)
print("=" * 80)


# ============================================================
# 1. Load episode metadata
# ============================================================

meta_files = sorted((ROOT / "meta" / "episodes").rglob("*.parquet"))

if not meta_files:
    raise RuntimeError("No meta/episodes parquet files found.")

meta = pd.concat(
    [pd.read_parquet(p) for p in meta_files],
    ignore_index=True,
)

if "episode_index" not in meta.columns:
    raise RuntimeError(
        f"episode_index missing from metadata: {meta.columns.tolist()}"
    )

if "length" not in meta.columns:
    raise RuntimeError(
        f"length missing from metadata: {meta.columns.tolist()}"
    )

meta = (
    meta.sort_values("episode_index")
        .drop_duplicates("episode_index")
        .reset_index(drop=True)
)

if "tasks" in meta.columns:
    meta["instruction"] = meta["tasks"].apply(first_item)
elif "task" in meta.columns:
    meta["instruction"] = meta["task"].apply(first_item)
else:
    meta["instruction"] = ""

# 数据集当前排列为每个 canonical task 连续 50 episodes
meta["block_id"] = meta["episode_index"] // 50

expected_length = dict(
    zip(meta["episode_index"], meta["length"])
)

instruction_map = dict(
    zip(meta["episode_index"], meta["instruction"])
)

block_map = dict(
    zip(meta["episode_index"], meta["block_id"])
)

print("Metadata episodes:", len(meta))
print("Blocks:", meta["block_id"].nunique())


# ============================================================
# 2. Load frame-level parquet
# ============================================================

data_files = sorted((ROOT / "data").rglob("*.parquet"))

if not data_files:
    raise RuntimeError("No data parquet files found.")

print("Data parquet files:", len(data_files))
print("Loading frame data...")

wanted = [
    "episode_index",
    "frame_index",
    "timestamp",
    "observation.state",
    "action",
]

pieces = []

for i, p in enumerate(data_files, 1):
    # 先尝试只读需要的列
    try:
        df = pd.read_parquet(p, columns=wanted)
    except Exception:
        df = pd.read_parquet(p)

        missing = [c for c in wanted if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"Missing columns in {p}: {missing}\n"
                f"Available: {df.columns.tolist()}"
            )

        df = df[wanted]

    pieces.append(df)

    if i % 10 == 0 or i == len(data_files):
        print(f"  loaded {i}/{len(data_files)}")

frames = pd.concat(pieces, ignore_index=True)

print("Total frame rows:", len(frames))


# ============================================================
# 3. Episode motion QA
# ============================================================

rows = []

episode_groups = frames.groupby("episode_index", sort=True)

print("Computing episode metrics...")

for count, (episode_index, g) in enumerate(episode_groups, 1):

    episode_index = int(episode_index)

    if "frame_index" in g.columns:
        g = g.sort_values("frame_index")

    actual_frames = len(g)
    expected_frames = int(
        expected_length.get(episode_index, actual_frames)
    )

    # --------------------------------------------------------
    # Frame indices
    # --------------------------------------------------------

    frame_idx = g["frame_index"].to_numpy()

    frame_index_unique = (
        len(np.unique(frame_idx)) == actual_frames
    )

    expected_frame_idx = np.arange(actual_frames)

    frame_index_contiguous = (
        frame_index_unique
        and np.array_equal(frame_idx, expected_frame_idx)
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    ts = g["timestamp"].to_numpy(dtype=float)

    timestamp_finite = bool(np.isfinite(ts).all())

    if len(ts) > 1 and timestamp_finite:
        dt = np.diff(ts)
        timestamp_monotonic = bool((dt > 0).all())
        timestamp_dt_median = float(np.median(dt))
        timestamp_dt_max = float(np.max(dt))
    else:
        timestamp_monotonic = True
        timestamp_dt_median = np.nan
        timestamp_dt_max = np.nan

    # --------------------------------------------------------
    # Stack state/action
    # --------------------------------------------------------

    state_arrays = [
        np.asarray(x, dtype=np.float64).reshape(-1)
        for x in g["observation.state"]
    ]

    action_arrays = [
        np.asarray(x, dtype=np.float64).reshape(-1)
        for x in g["action"]
    ]

    state_dims = {len(x) for x in state_arrays}
    action_dims = {len(x) for x in action_arrays}

    state_dim_consistent = len(state_dims) == 1
    action_dim_consistent = len(action_dims) == 1

    state_dim = next(iter(state_dims)) if state_dim_consistent else -1
    action_dim = next(iter(action_dims)) if action_dim_consistent else -1

    if state_dim_consistent:
        state = np.stack(state_arrays)
    else:
        state = None

    if action_dim_consistent:
        action = np.stack(action_arrays)
    else:
        action = None

    # defaults
    state_nonfinite = -1
    action_nonfinite = -1

    state_path_length = np.nan
    action_path_length = np.nan

    state_jump_mean = np.nan
    state_jump_p95 = np.nan
    state_jump_max = np.nan

    action_jump_mean = np.nan
    action_jump_p95 = np.nan
    action_jump_max = np.nan

    state_idle_ratio = np.nan
    action_idle_ratio = np.nan
    both_idle_ratio = np.nan
    max_both_idle_run = np.nan

    state_reversal_ratio = np.nan
    action_reversal_ratio = np.nan

    # --------------------------------------------------------
    # State metrics
    # --------------------------------------------------------

    if state is not None:

        state_nonfinite = int(
            (~np.isfinite(state)).sum()
        )

        if state_nonfinite == 0 and len(state) > 1:

            ds = np.diff(state, axis=0)
            state_step = np.linalg.norm(ds, axis=1)

            state_path_length = float(state_step.sum())

            state_jump_mean = float(np.mean(state_step))
            state_jump_p95 = float(np.percentile(state_step, 95))
            state_jump_max = float(np.max(state_step))

            state_idle = state_step < MOTION_EPS
            state_idle_ratio = float(np.mean(state_idle))

            state_reversal_ratio = reversal_ratio(state)

    # --------------------------------------------------------
    # Action metrics
    # --------------------------------------------------------

    if action is not None:

        action_nonfinite = int(
            (~np.isfinite(action)).sum()
        )

        if action_nonfinite == 0 and len(action) > 1:

            da = np.diff(action, axis=0)
            action_step = np.linalg.norm(da, axis=1)

            action_path_length = float(action_step.sum())

            action_jump_mean = float(np.mean(action_step))
            action_jump_p95 = float(np.percentile(action_step, 95))
            action_jump_max = float(np.max(action_step))

            action_idle = action_step < MOTION_EPS
            action_idle_ratio = float(np.mean(action_idle))

            action_reversal_ratio = reversal_ratio(action)

    # --------------------------------------------------------
    # Both state + action idle
    # --------------------------------------------------------

    if (
        state is not None
        and action is not None
        and state_nonfinite == 0
        and action_nonfinite == 0
        and len(state) > 1
    ):
        ds = np.linalg.norm(np.diff(state, axis=0), axis=1)
        da = np.linalg.norm(np.diff(action, axis=0), axis=1)

        both_idle = (
            (ds < MOTION_EPS)
            & (da < MOTION_EPS)
        )

        both_idle_ratio = float(np.mean(both_idle))
        max_both_idle_run = int(max_true_run(both_idle))

    rows.append(
        {
            "episode_index": episode_index,
            "block_id": int(
                block_map.get(
                    episode_index,
                    episode_index // 50
                )
            ),
            "instruction": instruction_map.get(
                episode_index, ""
            ),

            "expected_frames": expected_frames,
            "actual_frames": actual_frames,
            "length_match": actual_frames == expected_frames,

            "frame_index_unique": frame_index_unique,
            "frame_index_contiguous": frame_index_contiguous,

            "timestamp_finite": timestamp_finite,
            "timestamp_monotonic": timestamp_monotonic,
            "timestamp_dt_median": timestamp_dt_median,
            "timestamp_dt_max": timestamp_dt_max,

            "state_dim": state_dim,
            "action_dim": action_dim,

            "state_dim_consistent": state_dim_consistent,
            "action_dim_consistent": action_dim_consistent,

            "state_nonfinite": state_nonfinite,
            "action_nonfinite": action_nonfinite,

            "state_path_length": state_path_length,
            "action_path_length": action_path_length,

            "state_jump_mean": state_jump_mean,
            "state_jump_p95": state_jump_p95,
            "state_jump_max": state_jump_max,

            "action_jump_mean": action_jump_mean,
            "action_jump_p95": action_jump_p95,
            "action_jump_max": action_jump_max,

            "state_idle_ratio": state_idle_ratio,
            "action_idle_ratio": action_idle_ratio,
            "both_idle_ratio": both_idle_ratio,
            "max_both_idle_run": max_both_idle_run,

            "state_reversal_ratio": state_reversal_ratio,
            "action_reversal_ratio": action_reversal_ratio,
        }
    )

    if count % 100 == 0:
        print(
            f"  processed {count}/{len(episode_groups)} episodes"
        )


qa = pd.DataFrame(rows)


# ============================================================
# 4. Task-relative statistics
# ============================================================

metrics_for_outlier = [
    "actual_frames",
    "state_path_length",
    "action_path_length",
    "both_idle_ratio",
    "max_both_idle_run",
    "state_jump_max",
    "action_jump_max",
    "state_reversal_ratio",
    "action_reversal_ratio",
]

for metric in metrics_for_outlier:

    qa[f"pct_{metric}"] = (
        qa.groupby("block_id")[metric]
          .rank(method="average", pct=True)
    )

    qa[f"rz_{metric}"] = (
        qa.groupby("block_id")[metric]
          .transform(robust_z_transform)
    )


# ============================================================
# 5. Suspicion flags
# ============================================================

reason_list = []
scores = []

for _, r in qa.iterrows():

    reasons = []
    score = 0

    # -------- hard integrity --------

    if not r["length_match"]:
        reasons.append("frame_count_mismatch")
        score += 3

    if not r["frame_index_unique"]:
        reasons.append("duplicate_frame_index")
        score += 3

    if not r["frame_index_contiguous"]:
        reasons.append("noncontiguous_frame_index")
        score += 2

    if not r["timestamp_finite"]:
        reasons.append("timestamp_nonfinite")
        score += 3

    if not r["timestamp_monotonic"]:
        reasons.append("timestamp_nonmonotonic")
        score += 3

    if not r["state_dim_consistent"]:
        reasons.append("state_dim_inconsistent")
        score += 3

    if not r["action_dim_consistent"]:
        reasons.append("action_dim_inconsistent")
        score += 3

    if r["state_nonfinite"] > 0:
        reasons.append("state_nan_inf")
        score += 4

    if r["action_nonfinite"] > 0:
        reasons.append("action_nan_inf")
        score += 4

    # -------- statistical anomalies --------

    rz_length = r["rz_actual_frames"]

    if np.isfinite(rz_length):
        if rz_length > ROBUST_Z_THRESHOLD:
            reasons.append("unusually_long")
            score += 1
        elif rz_length < -ROBUST_Z_THRESHOLD:
            reasons.append("unusually_short")
            score += 1

    positive_metrics = {
        "state_path_length": "large_state_path",
        "action_path_length": "large_action_path",
        "both_idle_ratio": "high_idle_ratio",
        "max_both_idle_run": "long_idle_run",
        "state_jump_max": "large_state_jump",
        "action_jump_max": "large_action_jump",
        "state_reversal_ratio": "high_state_reversal",
        "action_reversal_ratio": "high_action_reversal",
    }

    for metric, name in positive_metrics.items():

        z = r[f"rz_{metric}"]

        if np.isfinite(z) and z > ROBUST_Z_THRESHOLD:
            reasons.append(name)
            score += 1

    reason_list.append(";".join(reasons))
    scores.append(score)


qa["suspicion_score"] = scores
qa["suspicion_reasons"] = reason_list
qa["is_suspicious"] = qa["suspicion_score"] > 0


# ============================================================
# 6. Task summary
# ============================================================

task_summary = (
    qa.groupby("block_id")
      .agg(
          episodes=("episode_index", "count"),
          median_frames=("actual_frames", "median"),
          min_frames=("actual_frames", "min"),
          max_frames=("actual_frames", "max"),

          median_state_path=("state_path_length", "median"),
          median_action_path=("action_path_length", "median"),

          median_idle_ratio=("both_idle_ratio", "median"),
          max_idle_ratio=("both_idle_ratio", "max"),

          median_reversal=("state_reversal_ratio", "median"),

          suspicious=("is_suspicious", "sum"),
      )
      .reset_index()
)


# ============================================================
# 7. Video file-level integrity
# ============================================================

video_rows = []

ffprobe = shutil.which("ffprobe")

video_files = sorted((ROOT / "videos").rglob("*.mp4"))

print("Checking video containers:", len(video_files))

for p in video_files:

    readable = p.is_file() and p.stat().st_size > 0
    size_mb = p.stat().st_size / (1024 ** 2) if p.exists() else 0

    ffprobe_ok = None
    duration = np.nan
    error = ""

    if readable and ffprobe:

        cmd = [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(p),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        ffprobe_ok = result.returncode == 0

        if ffprobe_ok:
            try:
                duration = float(result.stdout.strip())
            except Exception:
                duration = np.nan
        else:
            error = result.stderr.strip()

    video_rows.append(
        {
            "path": str(p.relative_to(ROOT)),
            "size_mb": size_mb,
            "readable": readable,
            "ffprobe_available": ffprobe is not None,
            "ffprobe_ok": ffprobe_ok,
            "duration_sec": duration,
            "error": error,
        }
    )


video_qa = pd.DataFrame(video_rows)


# ============================================================
# 8. Save
# ============================================================

qa = qa.sort_values(
    ["suspicion_score", "block_id", "episode_index"],
    ascending=[False, True, True],
)

suspicious = qa[qa["is_suspicious"]].copy()

qa.to_csv(
    OUT / "episode_qa.csv",
    index=False,
)

suspicious.to_csv(
    OUT / "suspicious_episodes.csv",
    index=False,
)

task_summary.to_csv(
    OUT / "task_qa_summary.csv",
    index=False,
)

video_qa.to_csv(
    OUT / "video_file_qa.csv",
    index=False,
)


# ============================================================
# 9. Human-readable summary
# ============================================================

hard_problem_mask = (
    (~qa["length_match"])
    | (~qa["frame_index_unique"])
    | (~qa["timestamp_finite"])
    | (~qa["timestamp_monotonic"])
    | (~qa["state_dim_consistent"])
    | (~qa["action_dim_consistent"])
    | (qa["state_nonfinite"] > 0)
    | (qa["action_nonfinite"] > 0)
)

reason_counts = (
    suspicious["suspicion_reasons"]
    .str.split(";")
    .explode()
)

reason_counts = (
    reason_counts[reason_counts != ""]
    .value_counts()
)

summary_path = OUT / "dataset_qa_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:

    f.write("RoboTwin Dataset QA v1\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Dataset: {ROOT}\n")
    f.write(f"Metadata episodes: {len(meta)}\n")
    f.write(f"Frame rows: {len(frames)}\n")
    f.write(f"QA episodes: {len(qa)}\n")
    f.write(f"Task blocks: {qa['block_id'].nunique()}\n\n")

    f.write("Structural integrity\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"Hard integrity problems: "
        f"{int(hard_problem_mask.sum())}\n"
    )
    f.write(
        f"Length mismatch: "
        f"{int((~qa['length_match']).sum())}\n"
    )
    f.write(
        f"Non-contiguous frame index: "
        f"{int((~qa['frame_index_contiguous']).sum())}\n"
    )
    f.write(
        f"Timestamp non-monotonic: "
        f"{int((~qa['timestamp_monotonic']).sum())}\n"
    )
    f.write(
        f"State NaN/Inf episodes: "
        f"{int((qa['state_nonfinite'] > 0).sum())}\n"
    )
    f.write(
        f"Action NaN/Inf episodes: "
        f"{int((qa['action_nonfinite'] > 0).sum())}\n"
    )

    f.write("\nMotion anomaly screening\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"Suspicious episodes: {len(suspicious)} / {len(qa)}\n"
    )

    f.write("\nReason counts:\n")

    if len(reason_counts):
        f.write(reason_counts.to_string())
    else:
        f.write("None")

    f.write("\n\nSuspicion score distribution:\n")
    f.write(
        qa["suspicion_score"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    f.write("\n\nVideo files\n")
    f.write("-" * 40 + "\n")
    f.write(f"MP4 files: {len(video_qa)}\n")

    if len(video_qa):
        f.write(
            f"Unreadable files: "
            f"{int((~video_qa['readable']).sum())}\n"
        )

        if ffprobe:
            f.write(
                f"ffprobe failures: "
                f"{int((video_qa['ffprobe_ok'] == False).sum())}\n"
            )
        else:
            f.write("ffprobe: not installed\n")


print()
print("=" * 80)
print("QA COMPLETE")
print("=" * 80)
print("All episodes:")
print(OUT / "episode_qa.csv")
print()
print("Suspicious episodes:")
print(OUT / "suspicious_episodes.csv")
print()
print("Task summary:")
print(OUT / "task_qa_summary.csv")
print()
print("Video QA:")
print(OUT / "video_file_qa.csv")
print()
print("Summary:")
print(OUT / "dataset_qa_summary.txt")
print()
print("IMPORTANT: suspicious != bad data.")
print("No episode has been deleted or modified.")
