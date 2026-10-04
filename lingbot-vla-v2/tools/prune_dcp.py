#!/usr/bin/env python3
"""剪掉训练存档里的 DCP 部分，只保留 hf_ckpt —— 训练期常驻的「磁盘看门狗」。

## 为什么能剪

训练每存一份 checkpoint 会产生两份内容（源码 `tasks/vla/train_lingbotvla.py`
L1173 + L1177，两者**绑死**，没有跳过 DCP 的开关）：

    <output_dir>/checkpoints/global_step_<N>/
        model/         fp32 权重            ≈ 23.8G   ← DCP，只服务续训
        optimizer/     优化器状态            ≈ 23.7G   ← DCP，只服务续训
        extra_state/   调度器/RNG/dataloader 很小      ← DCP，只服务续训
        hf_ckpt/       HF 格式权重（fp32）   ≈ 23.8G   ← **评测唯一需要的东西**

评测侧（`experiment/robotwin/robotwin_multi_ckpt_eval.py`）只 glob
`checkpoints/global_step_*/hf_ckpt`，校验 index.json + 分片字节数 + config.json，
**从不读 model/ 与 optimizer/**。

所以对「不续训、只要评测」的运行，`global_step_N/` 下除 `hf_ckpt` 之外的一切都是
纯冗余。剪掉后单份存档从 ≈71G 降到 ≈24G，同样 210G 空间能留 3 份而不是 2 份。

## 安全性

* **只有 `hf_ckpt` 通过完整性校验的存档才会被剪** —— 复用调度器里同一份
  `check_hf_ckpt`（importlib 直接加载，判据不会漂移）。
* 存档写入顺序是「先 DCP 后 HF」⇒ 只要 `hf_ckpt` 完整，同目录的 DCP 必然早已写完，
  不存在「剪到正在写的 DCP」的竞态。
* 再加 `--min-age-seconds` 静默期兜底。
* `--keep-last N` 保留最新 N 份的完整 DCP，保住「从最近存档续训」的能力。
* `--dry-run` 只打印计划，不删任何东西。
* 只动 `<...>/checkpoints/global_step_*/` 的**直系子条目**，不递归到别处。

## 用法

    # 常驻（训练脚本会自动拉起；也可手动）
    python tools/prune_dcp.py --ckpt-root /data/outputs/phase1_L1_vit_frozen \
        --interval 120 --keep-last 1

    # 一次性扫一遍
    python tools/prune_dcp.py --ckpt-root /data/outputs/xxx --once

    # 只看会删什么
    python tools/prune_dcp.py --ckpt-root /data/outputs/xxx --once --dry-run

    # 收工后彻底清空所有 DCP（连最新一份也不留）
    python tools/prune_dcp.py --ckpt-root /data/outputs/xxx --once --keep-last 0
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# 与调度器保持一致：<output_dir>/checkpoints/global_step_<N>/hf_ckpt
# --ckpt-root 允许指到 output_dir 本身，或它的上若干层。
CKPT_PATTERNS = (
    "checkpoints/global_step_*",
    "*/checkpoints/global_step_*",
    "*/*/checkpoints/global_step_*",
)

KEEP_NAME = "hf_ckpt"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def _load_check_hf_ckpt():
    """加载调度器里的 check_hf_ckpt —— 完整性判据只维护一份。"""
    here = Path(__file__).resolve()
    cand = here.parents[1] / "experiment" / "robotwin" / "robotwin_multi_ckpt_eval.py"
    if not cand.is_file():
        log(f"⚠️  找不到 {cand}；退化为内置简化校验")
        return None
    spec = importlib.util.spec_from_file_location("_robotwin_multi_ckpt_eval", cand)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # @dataclass 会查 sys.modules[cls.__module__]，必须先注册
    spec.loader.exec_module(mod)          # 该模块 import 期无副作用（只定义常量/函数）
    return mod.check_hf_ckpt


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def human(n: int) -> str:
    return f"{n / 1024 ** 3:.1f}G"


def step_of(p: Path) -> int:
    try:
        return int(p.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def find_step_dirs(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for pat in CKPT_PATTERNS:
            for p in sorted(root.glob(pat)):
                rp = p.resolve()
                if rp in seen or not rp.is_dir():
                    continue
                seen.add(rp)
                found.append(rp)
    return found


def sweep(roots: list[Path], keep_last: int, min_age_seconds: float,
          dry_run: bool, check_hf_ckpt) -> tuple[int, int]:
    """扫一遍。返回 (剪掉的份数, 释放字节)。"""
    now = time.time()
    step_dirs = find_step_dirs(roots)
    if not step_dirs:
        return 0, 0

    # keep_last 按「同一个 checkpoints/ 目录」分组，各自保留最新 N 份
    by_run: dict[Path, list[Path]] = {}
    for d in step_dirs:
        by_run.setdefault(d.parent, []).append(d)
    keep: set[Path] = set()
    for _run, dirs in by_run.items():
        for d in sorted(dirs, key=step_of, reverse=True)[:max(keep_last, 0)]:
            keep.add(d)

    pruned = freed = 0
    for d in sorted(step_dirs, key=lambda p: (str(p.parent), step_of(p))):
        hf = d / KEEP_NAME
        if not hf.is_dir():
            continue
        complete, reasons, warnings, total, newest = check_hf_ckpt(
            hf, min_age_seconds=min_age_seconds, now=now)
        if not complete:
            log(f"跳过 {d.name}：hf_ckpt 未通过完整性校验 -> {'; '.join(reasons)}")
            continue

        victims = sorted(p for p in d.iterdir() if p.name != KEEP_NAME)
        if not victims:
            continue
        if d in keep:
            log(f"保留 {d.name} 的 DCP（在 --keep-last {keep_last} 之内，可续训）")
            continue

        vbytes = sum(dir_size(v) if v.is_dir() else v.stat().st_size for v in victims)
        names = ", ".join(v.name for v in victims)
        if dry_run:
            log(f"[dry-run] 将剪 {d} -> 删除 [{names}]，预计释放 {human(vbytes)}")
            pruned += 1
            freed += vbytes
            continue

        log(f"剪 {d.name} -> 删除 [{names}]（{human(vbytes)}）")
        ok = True
        for v in victims:
            try:
                if v.is_dir():
                    shutil.rmtree(v)
                else:
                    v.unlink()
            except OSError as exc:
                log(f"  ⚠️  删除失败 {v}: {exc!r}")
                ok = False
        if ok:
            pruned += 1
            freed += vbytes
            log(f"  ✅ {d.name} 现在只剩 hf_ckpt（{human(dir_size(d))}）")

    return pruned, freed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="剪掉训练存档的 DCP 部分，只留 hf_ckpt（评测不需要 DCP）")
    ap.add_argument("--ckpt-root", action="append", required=True,
                    help="output_dir 或它的上层目录；可重复")
    ap.add_argument("--keep-last", type=int, default=1,
                    help="每个 output_dir 保留最新 N 份的完整 DCP（默认 1，保住续训能力）")
    ap.add_argument("--min-age-seconds", type=float, default=120.0,
                    help="hf_ckpt 最新 mtime 距今至少这么久才动手（默认 120）")
    ap.add_argument("--interval", type=float, default=120.0,
                    help="常驻时的扫描间隔秒数（默认 120）")
    ap.add_argument("--max-runtime", type=float, default=0.0,
                    help="常驻时最长运行秒数，0 = 不限（默认 0）")
    ap.add_argument("--once", action="store_true", help="只扫一遍就退出")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不删")
    args = ap.parse_args()

    roots = [Path(r).expanduser().resolve() for r in args.ckpt_root]
    for r in roots:
        if not r.is_dir():
            log(f"⚠️  --ckpt-root 不是目录: {r}")

    check_hf_ckpt = _load_check_hf_ckpt()
    if check_hf_ckpt is None:
        log("没有可用的完整性校验，退出（宁可不剪也不误删）")
        return 2

    log(f"看门狗启动 | roots={[str(r) for r in roots]} keep_last={args.keep_last} "
        f"min_age={args.min_age_seconds:g}s interval={args.interval:g}s "
        f"once={args.once} dry_run={args.dry_run}")

    t0 = time.time()
    total_pruned = total_freed = 0
    while True:
        try:
            p, f = sweep(roots, args.keep_last, args.min_age_seconds,
                         args.dry_run, check_hf_ckpt)
        except Exception as exc:                       # noqa: BLE001
            log(f"⚠️  本轮扫描异常（继续下一轮）: {type(exc).__name__}: {exc}")
            p = f = 0
        total_pruned += p
        total_freed += f
        if p:
            log(f"本轮剪掉 {p} 份，释放 {human(f)}；累计 {total_pruned} 份 / {human(total_freed)}")

        if args.once:
            break
        if args.max_runtime > 0 and (time.time() - t0) >= args.max_runtime:
            log(f"达到 --max-runtime {args.max_runtime:g}s，退出")
            break
        time.sleep(max(args.interval, 1.0))

    log(f"看门狗退出 | 共剪 {total_pruned} 份，释放 {human(total_freed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
