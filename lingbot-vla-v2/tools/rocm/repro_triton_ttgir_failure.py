#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复现 / 判定 Triton AMDGPU 后端 `OptimizeDotOperands` pass 崩溃（ROCm / gfx1100）。

背景
----
2026-10-10 在 7×W7900D（gfx1100、ROCm 7.2.1）上跑正式训练时，第一步 `loss.backward()`
触发的 Inductor 编译**确定性崩溃**（7/7 rank）：

    RuntimeError: PassManager::run failed
      ← triton/backends/amd/compiler.py:262 make_ttgir → pm.run(mod)

从 Triton 缓存反查出**唯一**编不过的 kernel：
`triton_tem_fused_slice_backward_transpose_view_zeros_2`（flex-attention 反向；全缓存 2652 个
kernel 里 2651 个正常）。本脚本把它单独编一遍，把 Triton 自己的诊断（通常被
`SubprocException` 吞掉的那几行）打出来。

用法
----
    # A) 自动从缓存里找"缺 ttgir"的失败 kernel（推荐）
    python tools/rocm/repro_triton_ttgir_failure.py --scan-cache /models/robotwin-persistent/al_cache/triton

    # B) 直接指定某个 .ttir
    python tools/rocm/repro_triton_ttgir_failure.py <path/to/kernel.ttir>

    # C) 只做参数扫描矩阵（默认 num_warps{1,2,4,8} × num_stages{1,2,3}）
    python tools/rocm/repro_triton_ttgir_failure.py <kernel.ttir> --warps 4,8 --stages 1,2,3

退出码
------
0 = 至少一组配置编译成功（说明可绕）
1 = 全部配置失败（说明该 kernel 在当前 Triton 上无法 lowering，需换实现或改 Triton）
2 = 用法/环境错误
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import traceback
from pathlib import Path


def find_broken_kernels(cache_root: Path, limit: int = 20):
    """找出「有 .ttir 但没有 .ttgir」的目录 —— 即编不过的 kernel。

    ⚠️ Triton 缓存布局是 `<cache>/<POS_HASH>/<name>.{ttir,ttgir,llir,hsaco,…}`（**一层**），
    不是两层；早期版本这里写成 `*/*/*.ttir` 会漏掉全部条目、误报"没有失败项"（2026-10-10 实测踩过）。
    """
    broken = []
    for ttir in sorted(cache_root.glob("*/*.ttir")):
        if not list(ttir.parent.glob("*.ttgir")):
            broken.append(ttir)
            if len(broken) >= limit:
                break
    return broken


def try_compile(ttir: Path, warps, stages, verbose: bool):
    """把 ttir 复制成临时文件后编译（⚠️ triton.compile 第一参数是路径，不是源码）。"""
    import triton

    src = ttir.read_text(encoding="utf-8", errors="replace")
    tmpdir = Path(tempfile.mkdtemp(prefix="ttir_repro_"))
    tmp = tmpdir / "kernel.ttir"
    tmp.write_text(src, encoding="utf-8")

    try:
        target = triton.runtime.driver.active.get_current_target()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 取 target 失败（{exc!r}）⇒ 让 triton 自选")
        target = None

    print(f"kernel      : {ttir}")
    print(f"大小        : {len(src)} 字节")
    print(f"target      : {target}")

    results = []
    for nw in warps:
        for ns in stages:
            buf = io.StringIO()
            ok = False
            try:
                with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
                    kwargs = {"options": {"num_warps": nw, "num_stages": ns}}
                    if target is not None:
                        kwargs["target"] = target
                    triton.compile(str(tmp), **kwargs)
                ok = True
            except Exception:  # noqa: BLE001
                buf.write(traceback.format_exc(limit=3))

            text = buf.getvalue()
            diags = [ln.strip() for ln in text.splitlines()
                     if "error:" in ln or "note:" in ln or "PassManager" in ln]
            results.append((nw, ns, ok, diags))
            print(f"  num_warps={nw} num_stages={ns} ⇒ {'成功' if ok else '失败'}")
            for d in diags[:4]:
                print(f"      {d[:220]}")
            if verbose and not ok and not diags:
                for ln in text.splitlines()[-6:]:
                    print(f"      | {ln[:200]}")

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="复现 Triton AMDGPU pass 崩溃")
    ap.add_argument("ttir", nargs="?", help="要编译的 .ttir 路径")
    ap.add_argument("--scan-cache", default=None,
                    help="triton 缓存根目录；自动找出缺 ttgir 的失败 kernel")
    ap.add_argument("--warps", default="1,2,4,8", help="num_warps 列表，默认 1,2,4,8")
    ap.add_argument("--stages", default="1,2,3", help="num_stages 列表，默认 1,2,3")
    ap.add_argument("-v", "--verbose", action="store_true", help="失败时打印原始尾部")
    args = ap.parse_args()

    targets = []
    if args.scan_cache:
        root = Path(args.scan_cache)
        if not root.is_dir():
            print(f"[error] 缓存目录不存在: {root}", file=sys.stderr)
            return 2
        targets = find_broken_kernels(root)
        if not targets:
            print(f"[ok] {root} 下未发现「缺 ttgir」的 kernel ⇒ 当前缓存里没有编译失败项")
            return 0
        print(f"发现 {len(targets)} 个编译失败的 kernel：")
        for t in targets:
            print(f"  - {t}")
        print()
    elif args.ttir:
        targets = [Path(args.ttir)]
        if not targets[0].is_file():
            print(f"[error] 文件不存在: {targets[0]}", file=sys.stderr)
            return 2
    else:
        ap.print_help()
        print("\n[error] 需要 <ttir> 或 --scan-cache", file=sys.stderr)
        return 2

    warps = [int(x) for x in args.warps.split(",") if x.strip()]
    stages = [int(x) for x in args.stages.split(",") if x.strip()]

    all_results = []
    for t in targets[:3]:            # 最多试 3 个，避免刷屏
        res = try_compile(t, warps, stages, args.verbose)
        all_results.extend(res)
        print()

    ok_cfgs = [f"nw={nw},ns={ns}" for nw, ns, ok, _ in all_results if ok]
    bad_cfgs = [f"nw={nw},ns={ns}" for nw, ns, ok, _ in all_results if not ok]
    print("=== 汇总结论 ===")
    print("  能编过的配置:", ok_cfgs or "无")
    print("  编不过的配置:", bad_cfgs or "无")
    if ok_cfgs:
        print("  ⇒ 存在可用配置；可尝试用 inductor 侧参数固定它（见 docs/rocm_triton_compile_crash_zh.md §6-C）")
        return 0
    print("  ⇒ 全部失败：该 kernel 在当前 Triton 上无法 lowering ⇒ 需换注意力实现或改 Triton（§6-A/C）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
