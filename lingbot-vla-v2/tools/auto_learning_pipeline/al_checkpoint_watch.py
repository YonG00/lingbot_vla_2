#!/usr/bin/env python3
"""纯 CPU 监控单个运行的周期 HF；满四份时删除最早一份（保留三份）。

默认只预览；--apply 才清理。仅删除完整 HF + 同步号池文件的目录，
跳过临时目录、完整/正在写的 DCP、受保护路径及符号链接。
"""

import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import struct
import time
import uuid


def complete_periodic_hf(path: Path) -> bool:
    try:
        return _complete_periodic_hf(path)
    except OSError:
        # 训练器可能同时发布/移除目录；本轮不碰，下轮重新检查。
        return False


def _complete_periodic_hf(path: Path) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if not match:
        return False
    # DCP（包括尚未写完的 DCP）由训练器自己的保留策略负责。
    if any((path / name).exists() for name in ("model", "optimizer", "extra_state")):
        return False
    if any(p.name.startswith((".hf_ckpt", "hf_ckpt.stale")) for p in path.iterdir()):
        return False
    hf = path / "hf_ckpt"
    if hf.is_symlink() or not hf.is_dir():
        return False
    index_path = hf / "model.safetensors.index.json"
    config_path = hf / "config.json"
    pool_path = path / "pool_state.json"
    if not pool_path.exists():
        pool_path = hf / "pool_state.json"
    if index_path.is_symlink() or pool_path.is_symlink() or config_path.is_symlink():
        return False
    try:
        if not isinstance(json.loads(config_path.read_text()), dict):
            return False
        pool = json.loads(pool_path.read_text())
        if pool.get("global_step") != int(match[1]):
            return False
        records = (pool.get("auto_learning") or {}).get("registry", {}).get("records")
        if not isinstance(records, dict) or not records:
            return False
        index = json.loads(index_path.read_text())
        weights = index.get("weight_map")
        if not isinstance(weights, dict) or not weights:
            return False
        for name in set(weights.values()):
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
                return False
            shard = hf / name
            if shard.is_symlink():
                return False
            with shard.open("rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                if n > 64 * 1024 * 1024:
                    return False
                header = json.loads(f.read(n))
            end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
            if shard.stat().st_size != 8 + n + end:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError, struct.error):
        # 正在原子发布/尚未写完 ⇒ 留到下一次检查，绝不清理。
        return False


def prune_once(run_dir: Path, *, apply: bool = False, protected=()) -> dict:
    root = run_dir.resolve() / "checkpoints"
    if root.is_symlink():
        raise ValueError("checkpoints 目录不能是符号链接")
    protected = [Path(p).resolve() for p in protected]
    candidates = []
    if root.is_dir():
        for path in root.iterdir():
            # 对受保护 HF 或其所在 checkpoint 整体留存。
            if any(p == path.resolve() or path.resolve() in p.parents for p in protected):
                continue
            if complete_periodic_hf(path):
                candidates.append(path)
    candidates.sort(key=lambda p: int(p.name.removeprefix("global_step_")))
    victims = candidates[:-3]  # 4 份删 1 份；积累更多时也追到只剩最近 3 份。
    removed = []
    for path in victims:
        if not apply:
            continue
        # 删除前再校验一次；只操作本 run/checkpoints 的直接子目录。
        if path.parent != root or not complete_periodic_hf(path):
            continue
        tombstone = root / f".prune_{path.name}_{uuid.uuid4().hex}"
        try:
            path.rename(tombstone)
        except FileNotFoundError:
            continue
        print(json.dumps({"action": "prune_begin", "checkpoint": str(path)}, ensure_ascii=False), flush=True)
        shutil.rmtree(tombstone)
        removed.append(path.name)
    return {
        "action": "checkpoint_watch", "apply": apply, "keep": 3,
        "complete_periodic": [p.name for p in candidates],
        "would_remove": [p.name for p in victims], "removed": removed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--apply", action="store_true", help="执行删除；默认只预览")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--protect", type=Path, action="append", default=[])
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not run.is_dir():
        parser.error("run_dir 必须是已存在的运行目录")
    # 同一个运行最多一个清理进程；不依赖 torch，不访问 GPU。
    with (run / ".checkpoint_watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            result = prune_once(run, apply=args.apply, protected=args.protect)
            signature = json.dumps(result, sort_keys=True)
            if signature != previous:
                print(json.dumps(result, ensure_ascii=False), flush=True)
                previous = signature
            if args.once:
                return
            time.sleep(15)


if __name__ == "__main__":
    main()
