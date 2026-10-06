"""把已有 run 的 `tb_scalars.jsonl` 转成**真正的 TensorBoard event 文件**。

    # 转换一个 run（写到 <run>/tb/）
    python -m auto_learning.tools.tb_from_jsonl runs/walkthrough

    # 一次转多个 run，然后用一个 TB 看全部（不同 run 会并列显示）
    python -m auto_learning.tools.tb_from_jsonl runs/demo_4task runs/demo_12task runs/demo_50task

    # 指定输出目录
    python -m auto_learning.tools.tb_from_jsonl runs/walkthrough --logdir /tmp/tb

为什么需要它：`EventLogger` 现在会**直接**写 event 文件，但早先跑出来的 run
（以及没有装 tensorboard 的环境）只有 `tb_scalars.jsonl`，这个脚本负责补上。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

from ..obs.logger import TensorboardSink


def convert(run_dir: str, logdir: Optional[str] = None) -> Tuple[str, int, Dict[str, int]]:
    """返回 `(logdir, 写入的标量数, {tag: 点数})`。"""
    src = os.path.join(run_dir, "tb_scalars.jsonl")
    if not os.path.exists(src):
        raise FileNotFoundError(f"找不到 {src}（这个 run 可能没开 logger）")
    out = logdir or os.path.join(run_dir, "tb")
    os.makedirs(out, exist_ok=True)

    sink = TensorboardSink(out)
    if not sink.ok:
        raise RuntimeError(
            "环境里既没有 tensorboard 也没有 torch，无法写 event 文件；"
            "请先 pip install tensorboard"
        )

    n = 0
    per_tag: Dict[str, int] = {}
    with open(src, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sink.add_scalar(row["tag"], float(row["value"]), int(row["step"]))
            per_tag[row["tag"]] = per_tag.get(row["tag"], 0) + 1
            n += 1
    sink.close()
    return out, n, per_tag


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tb_from_jsonl")
    p.add_argument("runs", nargs="+", help="一个或多个 run 目录（含 tb_scalars.jsonl）")
    p.add_argument("--logdir", default=None, help="输出目录；给多个 run 时必填同一个才能并列比较")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.logdir and len(args.runs) > 1:
        pass  # 全部写进同一个 logdir ⇒ TB 里会并列显示多个 run
    total = 0
    for run in args.runs:
        logdir = args.logdir
        if logdir and len(args.runs) > 1:
            logdir = os.path.join(logdir, os.path.basename(os.path.normpath(run)))
        out, n, per_tag = convert(run, logdir)
        total += n
        print(f"[{os.path.basename(os.path.normpath(run))}] {n} 个标量 / {len(per_tag)} 个 tag → {out}")
        if not args.quiet:
            for tag in sorted(per_tag):
                print(f"      {tag}")

    base = args.logdir or os.path.dirname(os.path.normpath(args.runs[0])) or "."
    print(f"\n共写入 {total} 个标量。")
    print("启动 TensorBoard：")
    print(f"    python -m tensorboard.main --logdir {base} --port 6006")
    print("然后浏览器打开 http://localhost:6006")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
