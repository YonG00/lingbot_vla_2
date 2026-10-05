#!/usr/bin/env python3
"""逐值对拍：in-process dump vs 官方 dump（GT / pred / 模型输入）。

判读顺序（**不要跳步**）：
  1. `gt`         —— 数据管线 + 反归一化口径
  2. `in.*`       —— 归一化 / 预处理（**这一层才抓得到 norm_stats 那类 bug，GT 抓不到**）
  3. `pred`       —— 前向链路（**前提：噪声已逐位对齐**，见 tools/open_loop_parity_dump_official.py 文件头）

用法：
    python tools/open_loop_parity_compare.py \
        --ours /data/tmp/dump_inprocess --official /data/tmp/dump_official

对齐关系：我方 `train_monitor_ep0` ↔ 官方 `traj50_chunk0`；`val_ep0` ↔ `traj51_chunk0`
（50 / 51 分别是两个集合的第 1 条）。命名里 `ep{local_idx}` 的 local_idx 是 **chunk 起点**，不是回合号。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

KEYS = ("images", "img_masks", "lang_tokens", "lang_masks", "state", "grid")
# 键名映射：我方 dump 的键 → 官方 dump 的键（两边命名不同，别当成"一方缺失"）
KEY_MAP = {"grid": "image_grid_thw"}
# 必须**逐值**一致的项（噪声无关）
MUST_MATCH = ("gt", "in.img_masks", "in.lang_tokens", "in.lang_masks", "in.state", "in.grid")
# 已知容许差异：图像走两条不同 resize 路径（我方 LeRobot image_transforms=Resize(256)，
# 官方手动 resize_image）⇒ 像素级 ~0.4%，实测 max|Δ| ≈ 3.9e-03
KNOWN_TOL = {"in.images": 1e-2}


def _stat(a, b, name: str, rows: list) -> None:
    if a is None or b is None:
        rows.append((name, "一方缺失", float("nan")))
        return
    note = ""
    if a.shape != b.shape:
        if a.size == b.size:      # 例：grid 我方 (3,3)、官方 (1,3,3) —— 只差 batch 维
            a, b = a.reshape(-1), b.reshape(-1)
            note = f"（形状不同但元素数相同，按展平比；原 {a.shape}）"
        else:
            rows.append((name, f"形状不同 {a.shape} vs {b.shape}", float("nan")))
            return
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    mx = float(d.max())
    tag = "✅ 一致" if mx < 1e-5 else ("≈ 接近" if mx < 1e-2 else "❌ 不同")
    rows.append((name, f"{tag}  max|Δ|={mx:.3e}  mean|Δ|={d.mean():.3e}{note}", mx))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="/data/tmp/dump_inprocess")
    ap.add_argument("--official", default="/data/tmp/dump_official")
    ap.add_argument("--pairs", default="train_monitor:50,val:51",
                    help="我方 tag : 官方 traj_id，逗号分隔")
    ap.add_argument("--chunks", default="0,50", help="chunk 起点，逗号分隔")
    a = ap.parse_args()

    ours, off = Path(a.ours), Path(a.official)
    pairs = [p.split(":") for p in a.pairs.split(",") if p]
    chunks = [int(x) for x in a.chunks.split(",") if x]

    bad: list[str] = []
    for tag, traj in pairs:
        print("=" * 78)
        print(f"traj {traj}（我方 tag={tag}）")
        for idx in chunks:
            p, o = f"{tag}_ep{idx}", f"traj{traj}_chunk{idx}"
            print(f"  --- chunk 起点 {idx} ---")
            rows: list = []
            for kind in ("gt", "pred"):
                # ⚠️ 官方 gt/pred 是**整条轨迹**一个文件（`traj{id}_gt.npy`），**没有** `_chunk{idx}` 后缀
                fa, fb = ours / f"{p}_{kind}.npy", off / f"traj{traj}_{kind}.npy"
                A = np.load(fa) if fa.exists() else None
                B = np.load(fb) if fb.exists() else None
                # ⚠️ 官方存的是**整条轨迹**（T, D），我方存的是**单个 chunk**（horizon, D）
                #    ⇒ 从官方那条里按 chunk 起点切出同样长度再比
                if A is not None and B is not None and B.shape[0] > A.shape[0]:
                    B = B[idx: idx + A.shape[0]]
                _stat(A, B, kind, rows)
            for key in KEYS:
                fa = ours / f"{p}_in_{key}.npy"
                fb = off / f"{o}_in_{KEY_MAP.get(key, key)}.npy"
                _stat(np.load(fa) if fa.exists() else None,
                      np.load(fb) if fb.exists() else None, f"in.{key}", rows)
            for name, txt, mx in rows:
                print(f"    {name:<14} {txt}")
                if name in MUST_MATCH and not (mx < 1e-5):
                    bad.append(f"traj{traj}/chunk{idx}/{name}")
                elif name in KNOWN_TOL and not (mx < KNOWN_TOL[name]):
                    bad.append(f"traj{traj}/chunk{idx}/{name} 超出已知容差 {KNOWN_TOL[name]}")

    print("=" * 78)
    if bad:
        print("❌ 不一致项：")
        for x in bad:
            print(f"    {x}")
        print("判读：gt 不同 ⇒ 查数据管线/反归一化；in.* 不同 ⇒ 查归一化/预处理；"
              "pred 不同但 in.* 全一致 ⇒ 查前向链路与噪声对齐")
        return 1
    print("✅ gt / lang_tokens / lang_masks / img_masks / state / grid 全部逐值一致")
    print(f"   in.images 在已知容差内（两条 resize 路径不同，像素 ~0.4%）")
    print("   pred 见上表：噪声对齐后应到 1e-3 量级；若显著更大，查前向链路")


if __name__ == "__main__":
    raise SystemExit(main())
