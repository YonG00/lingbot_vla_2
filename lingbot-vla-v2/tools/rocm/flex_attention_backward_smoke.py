#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟：判定本机 Triton 能否编译「flex attention 反向」——不跑训练、几分钟出结论。

用途
----
2026-10-10 在 gfx1100 上实测：flex attention 的反向 kernel
（`triton_tem_fused_slice_backward_transpose_view_zeros_*`）会触发 Triton AMDGPU
`OptimizeDotOperands` pass 崩溃 ⇒ 开 `torch.compile` 的第一步 backward 必挂。
本脚本用一个最小同形用例（f32 q/k/v + 因果 block mask）复现，**在起正式训练之前**就知道
这台机器/这个 Triton 版本能不能用 flex 反向。

用法
----
    /opt/robotwin-env/bin/python tools/rocm/flex_attention_backward_smoke.py
    # 指定形状/精度（默认 f32，与线上失败 kernel 一致）
    ... --dtype fp32 --seq 512 --heads 4 --head-dim 64

退出码
------
0 = 前向+反向都通过（flex 反向可用）
3 = 反向触发 Triton 崩溃（需换注意力实现或改 Triton）
4 = torch/ROCm 不可用或其它运行时错误
"""
from __future__ import annotations

import argparse
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser(description="flex attention 反向冒烟")
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16", "fp16"))
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--eager", action="store_true",
                    help="用 eager 路径（不推荐：那是展开实现，与线上不同，会假绿）")
    ap.add_argument("--fullgraph", action="store_true",
                    help="用 torch.compile(flex_attention, fullgraph=True)")
    args = ap.parse_args()

    try:
        import torch
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] 无法导入 torch / flex_attention: {exc!r}")
        return 4

    if not torch.cuda.is_available():
        print("[skip] 无可用 GPU（本脚本要在训练机上跑）")
        return 4

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    dev = "cuda"
    print(f"device={torch.cuda.get_device_name(0)} dtype={args.dtype} "
          f"seq={args.seq} heads={args.heads} head_dim={args.head_dim}")

    def mask(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx

    try:
        block_mask = create_block_mask(mask, args.batch, args.heads, args.seq, args.seq,
                                      device=dev, BLOCK_SIZE=128)
    except TypeError:                      # 旧版本签名不带 BLOCK_SIZE
        block_mask = create_block_mask(mask, args.batch, args.heads, args.seq, args.seq,
                                      device=dev)

    shape = (args.batch, args.heads, args.seq, args.head_dim)
    q = torch.randn(shape, dtype=dtype, device=dev, requires_grad=True)
    k = torch.randn(shape, dtype=dtype, device=dev, requires_grad=True)
    v = torch.randn(shape, dtype=dtype, device=dev, requires_grad=True)

    # 🔴 关键：线上是 `torch.compile(model)` ⇒ flex_attention 走**融合 kernel**（会崩的那条）。
    #    直接调 `flex_attention(...)` 在 eager 下会退回"展开实现"（materialize 全 scores 矩阵），
    #    那条路**不报错**（2026-10-10 实测：假绿）。所以默认必须编译。
    if args.eager:
        print("[warn] --eager：走的是展开实现，与线上融合路径**不同**，结论仅供参考")
        fn = flex_attention
    else:
        print(f"用 torch.compile(flex_attention, fullgraph={bool(args.fullgraph)}) ⇒ 走融合 kernel（同线上）")
        fn = torch.compile(flex_attention, fullgraph=bool(args.fullgraph))

    try:
        out = fn(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()
        print("[ok] 前向通过")
        out.sum().backward()
        torch.cuda.synchronize()
        print("[ok] 反向通过 ⇒ 本机 flex 反向（融合路径）可用，可开 torch.compile")
        return 0
    except Exception as exc:  # noqa: BLE001
        text = traceback.format_exc()
        crash = "PassManager::run failed" in text or "OptimizeDotOperands" in text
        print("[FAIL] 反向失败" + ("（Triton AMDGPU pass 崩溃）" if crash else ""))
        for line in text.splitlines():
            if any(k in line for k in ("Error", "error:", "note:", "make_ttgir", "PassManager")):
                print("   " + line.strip()[:200])
        print("\n⇒ 处置见 docs/rocm_triton_compile_crash_zh.md §6")
        return 3 if crash else 4


if __name__ == "__main__":
    raise SystemExit(main())
