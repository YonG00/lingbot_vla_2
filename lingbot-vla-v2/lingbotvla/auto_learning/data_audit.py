"""一次完整映射检查，后续按内容/文件身份复用；不读图像、不 import torch。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from collections import Counter


def audit_mapping(catalog, resolver, *, dataset_root, episode_getter,
                  report_file, instruction_patterns=None, task_indices=None, task_label=None):
    root = Path(dataset_root).resolve()
    patterns = dict(instruction_patterns or {})
    # 轻量身份核对覆盖 metadata 和帧表；视频内容不解码、不声称语义逐帧验收。
    files = sorted(set(root.glob("meta/**/*")) | set(root.glob("data/**/*.parquet")))
    stamps = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
              for p in files if p.is_file()]
    if not stamps:
        raise ValueError(f"data audit: no source metadata: {root}")
    splits = {n: {"train": catalog.entry(n).train_traj_ids,
                  "val": catalog.entry(n).val_traj_ids}
              for n in catalog.task_names()}
    digest = hashlib.sha256()
    digest.update(resolver.episode_map().tobytes())
    digest.update(resolver.frame_map().tobytes())
    if task_indices is not None:
        digest.update(json.dumps([int(i) for i in task_indices]).encode())
    identity = dict(version=1, dataset_root=str(root), files=stamps, splits=splits,
                    task_order=catalog.task_order, patterns=patterns,
                    mapping_sha256=digest.hexdigest())
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(report_file).expanduser()
    try:
        old = json.loads(path.read_text())
        if old.get("signature") == signature and old.get("ok") is True:
            return {**old, "cached": True}
    except (OSError, ValueError):
        pass
    problems = list(catalog.verify()) + list(resolver.verify())
    if problems:
        raise ValueError("data audit: " + "; ".join(map(str, problems)))
    expected = {ep for split in splits.values() for ep in split["train"]}
    actual = Counter(int(ep) for ep in resolver.episode_map())
    if set(actual) != expected:
        raise ValueError(f"data audit: train episodes mismatch: missing="
                         f"{sorted(expected-set(actual))[:8]}, extra={sorted(set(actual)-expected)[:8]}")
    rows = []
    for task, split in splits.items():
        train_samples = 0
        examples = []
        regex = re.compile(patterns[task], re.I) if task in patterns else None
        for ep in split["train"] + split["val"]:
            meta = episode_getter(ep)
            if int(meta["episode_index"]) != ep:
                raise ValueError(f"data audit: metadata episode mismatch for {ep}")
            instructions = meta.get("tasks", [])
            if isinstance(instructions, str):
                instructions = [instructions]
            instructions = [str(x) for x in instructions]
            if not instructions or (regex and not all(regex.search(s) for s in instructions)):
                raise ValueError(f"data audit: task/instruction mismatch: {task}, episode={ep}, {instructions}")
            if catalog.task_of_episode(ep) != task:
                raise ValueError(f"data audit: wrong task for episode={ep}")
            if ep in split["train"]:
                ids = resolver.local_indices_of_episode(ep)
                if len(ids) != int(meta["length"]):
                    raise ValueError(f"data audit: incomplete episode={ep}: "
                                     f"{len(ids)} samples vs metadata length={meta['length']}")
                for sid in ids:
                    ref = resolver.resolve(task, sid)
                    if ref.episode_id != ep:
                        raise ValueError(f"data audit: sample {sid} maps to wrong episode")
                    if task_indices is not None and str(task_label(int(task_indices[sid]))) not in instructions:
                        raise ValueError(f"data audit: frame task_index disagrees with episode={ep}, sample={sid}")
                    if "dataset_from_index" in meta and "dataset_to_index" in meta:
                        if not int(meta["dataset_from_index"]) <= ref.frame_id < int(meta["dataset_to_index"]):
                            raise ValueError(f"data audit: wrong absolute frame {ref.frame_id} for episode={ep}")
                train_samples += len(ids)
                if len(examples) < 3:
                    examples.append(dict(episode=ep, first_local_id=ids[0],
                                         last_local_id=ids[-1], instructions=instructions))
        rows.append(dict(task=task, train_episodes=len(split["train"]),
                         val_episodes=len(split["val"]), train_samples=train_samples,
                         instruction_pattern=patterns.get(task), examples=examples))
    report = dict(ok=True, signature=signature, identity=identity, tasks=rows,
                  note="block order + metadata instructions + complete train mapping; no video semantic inspection")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)
    return {**report, "cached": False}


def audit_dataset(catalog, resolver, dataset, cfg):
    node = dataset
    children = getattr(node, "_datasets", None)
    if children is not None:
        if len(children) != 1:
            raise ValueError("data audit requires one dataset")
        node = children[0]
    node = getattr(node, "dataset", node)
    tasks = node.meta.tasks
    def label(index):
        return tasks.iloc[index].name if hasattr(tasks, "iloc") else tasks[index]
    return audit_mapping(catalog, resolver, dataset_root=node.root,
        episode_getter=lambda ep: node.meta.episodes[ep],
        report_file=cfg.data_audit_file,
        instruction_patterns=cfg.data_audit_instruction_patterns,
        task_indices=node.hf_dataset["task_index"], task_label=label)
