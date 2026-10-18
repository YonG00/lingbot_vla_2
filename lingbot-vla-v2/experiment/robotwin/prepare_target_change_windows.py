#!/usr/bin/env python3
"""Prepare readable task caches and aligned HTML reports, without importing torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.sampling.target_change_window import (
    cache_path,
    joint_mapping,
    prepare_task,
    write_report,
)


def prepare(cfg, dataset_root, manifest, tasks, *, refresh=False):
    import yaml
    import pyarrow.dataset as ds

    if not cfg.target_change_cache_dir or not cfg.target_change_embodiment_config:
        raise ValueError(
            "set target_change_cache_dir and actual target_change_embodiment_config first"
        )
    cfg.validate()
    root = Path(dataset_root).resolve()
    if (
        cfg.target_change_dataset_root
        and Path(cfg.target_change_dataset_root).resolve() != root
    ):
        raise ValueError(
            "configured target-change dataset root differs from preparation source"
        )
    info = json.loads((root / "meta/info.json").read_text())
    fps = float(info["fps"])
    embodiment = yaml.safe_load(Path(cfg.target_change_embodiment_config).read_text())
    mapping = joint_mapping(embodiment, cfg.target_change_wrist_joint_names)
    files = sorted(str(p) for p in (root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError("dataset has no data parquet files")
    table = ds.dataset(files, format="parquet")
    columns = [
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "observation.state",
    ]

    def read_episode(ep):
        rows = table.to_table(
            columns=columns, filter=ds.field("episode_index") == ep
        ).to_pylist()
        return sorted(rows, key=lambda r: r["frame_index"])

    seen = set()
    for task in tasks:
        entry = manifest["tasks"][task]
        train, val = entry["train_ids"], entry["val_ids"]
        if (
            not train
            or len(set(train)) != len(train)
            or set(train) & set(val)
            or seen & set(train + val)
        ):
            raise ValueError(f"invalid/cross-task train/val episodes: {task}")
        seen.update(train + val)
        if cfg.train_all_episodes:
            # 与实际训练白名单一致；扩大后旧40条缓存必须失效重建。
            train = sorted(train + val)
        raw = prepare_task(
            cfg, root, task, train, fps, mapping, read_episode, refresh=refresh
        )
        videos = {}
        template = info.get("video_path", "")
        # LeRobot v2 episode videos. v3 chunked video files need their metadata
        # offsets; never guess those offsets or show a misleading video.
        if "{episode_index" in template:
            for ep in train:
                for key, feature in info.get("features", {}).items():
                    if feature.get("dtype") == "video":
                        path = root / template.format(
                            episode_chunk=ep // int(info.get("chunks_size", 1000)),
                            episode_index=ep,
                            video_key=key,
                        )
                        if path.is_file():
                            videos.setdefault(ep, []).append(path.as_uri())
        report = cache_path(cfg.target_change_cache_dir, task).with_suffix(".html")
        write_report(raw, report, videos)
        print(
            json.dumps(
                dict(
                    task=task,
                    cache=str(report.with_suffix(".json")),
                    report=str(report),
                    **raw["summary"],
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--al-config", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument(
        "--refresh",
        action="store_true",
        help="explicitly rebuild existing caches after inspection",
    )
    args = p.parse_args()
    cfg = AutoLearningConfig(**json.loads(args.al_config.read_text()))
    prepare(
        cfg,
        args.dataset_root,
        json.loads(args.manifest.read_text()),
        args.tasks,
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
