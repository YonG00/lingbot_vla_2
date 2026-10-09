#!/usr/bin/env python3
"""加载加速的**端到端体检**（一键跑，默认只用合成小 ckpt，安全无副作用）。

为什么需要：`lingbotvla/models/module_utils.py` 依赖特定版本的 transformers，
**在开发机（Mac）上无法导入** ⇒ 单元测试只能用 AST 替身；真正的"用仓库自己的加载器
把 BF16 副本读回来"必须在训练机上验证。本脚本就是那一步。

## 两种模式

    # A) 合成端到端（默认；秒级、不碰任何真实产物）
    python tools/load_accel_smoke.py

    # B) 真机实测（只读原目录；--write-copy 才写盘）
    python tools/load_accel_smoke.py --ckpt <hf_ckpt 目录> [--write-copy <新目录>]

A 模式做四件事（全部用**仓库真实实现**，不是替身）：
  1. 造一个 2 分片 F32 小 ckpt（含 int64 / bool / 0 维 / 空张量）；
  2. 用 `tools/make_bf16_ckpt.py` 转成 BF16 副本并**逐张量校验**；
  3. 用仓库自己的 `_resolve_weight_files()` + `StateDictIterator` **把副本读回来**，
     核对 key 集合 / 形状 / dtype（浮点应全为 bf16）；
  4. 调 `_parallel_prewarm_shards()`（并行预读）并报告字节数/线程/耗时。

B 模式额外测：冷读 / 热读 / 并行冷读三条计时 + （可选）真实 BF16 副本转换与校验。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def _import_loader():
    """按**文件路径**直接导入 `module_utils`（绕过 `lingbotvla.models.__init__` 的重依赖）。"""
    import importlib.util

    path = REPO / "lingbotvla/models/module_utils.py"
    spec = importlib.util.spec_from_file_location("loader_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # type: ignore[union-attr]
    return module


def _make_synthetic(root: Path, *, shards: int = 2) -> Dict[str, torch.Tensor]:
    root.mkdir(parents=True, exist_ok=True)
    all_tensors: Dict[str, torch.Tensor] = {}
    weight_map: Dict[str, str] = {}
    for i in range(shards):
        name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        tensors = {
            f"layer{i}.weight": torch.randn(8, 4, dtype=torch.float32),
            f"layer{i}.bias": torch.randn(8, dtype=torch.float32),
            f"layer{i}.scalar": torch.tensor(float(i), dtype=torch.float32),
            f"layer{i}.counter": torch.arange(8, dtype=torch.int64),
            f"layer{i}.mask": torch.tensor([True, False] * 4),
        }
        save_file(tensors, str(root / name), metadata={"format": "pt"})
        for k in tensors:
            weight_map[k] = name
            all_tensors[k] = tensors[k]
    (root / "model.safetensors.index.json").write_text(
        __import__("json").dumps({"metadata": {"total_size":
                                               sum(f.stat().st_size for f in root.glob("*.safetensors"))},
                                  "weight_map": weight_map}), encoding="utf-8")
    (root / "config.json").write_text('{"model_type": "smoke"}', encoding="utf-8")
    return all_tensors


def mode_a(verbose: bool = True) -> int:
    from tools import make_bf16_ckpt as mk

    tmp = Path(tempfile.mkdtemp(prefix="load_accel_smoke_"))
    try:
        src, dst = tmp / "src", tmp / "dst"
        ref = _make_synthetic(src)
        if verbose:
            print(f"[A1] 合成 ckpt：{len(ref)} 张量 / {sum(f.stat().st_size for f in src.glob('*.safetensors'))} 字节")
        rc = mk.main(["--src", str(src), "--dst", str(dst), "--verify", "full"])
        if rc != 0:
            print(f"[A2] ❌ BF16 转换/校验失败 rc={rc}")
            return 2
        if verbose:
            print(f"[A2] ✅ BF16 转换 + 逐张量校验通过；"
                  f"{sum(f.stat().st_size for f in src.glob('*.safetensors'))} → "
                  f"{sum(f.stat().st_size for f in dst.glob('*.safetensors'))} 字节")

        try:
            loader = _import_loader()
        except Exception as exc:  # noqa: BLE001
            print(f"[A3/A4] ⚠️ 本机无法导入仓库加载器（{type(exc).__name__}: {str(exc)[:90]}）")
            print("[A3/A4]    ⇒ 本机只能验证「转换 + 逐张量校验」；端到端读回请在训练机上跑同一脚本")
            print("[A] ⚠️ 部分完成（合成转换 OK，真实加载器未验证）")
            return 3

        files = loader._resolve_weight_files(str(dst))
        got: Dict[str, Any] = {}
        for iterator in loader._load_state_dict(str(dst)):
            for key, tensor in iterator:
                got[key] = tensor
        if set(got) != set(ref):
            print(f"[A3] ❌ key 集合不一致：missing={sorted(set(ref) - set(got))[:3]} "
                  f"extra={sorted(set(got) - set(ref))[:3]}")
            return 2
        bad = []
        for key, want in ref.items():
            have = got[key]
            if want.is_floating_point():
                if have.dtype != torch.bfloat16 or not torch.equal(have, want.to(torch.bfloat16)):
                    bad.append(key)
            elif have.dtype != want.dtype or not torch.equal(have, want):
                bad.append(key)
        if bad:
            print(f"[A3] ❌ 读回后数值/类型不符：{bad[:5]}")
            return 2
        if verbose:
            print(f"[A3] ✅ 用仓库真实加载器读回：{len(got)} 张量 / {len(files)} 文件，"
                  f"逐张量一致（浮点全 bf16）")

        pre = loader._parallel_prewarm_shards(str(dst))
        if not pre["enabled"]:
            print(f"[A4] ⚠️ 并行预读未启用：{pre.get('error')}")
        elif verbose:
            print(f"[A4] ✅ 并行预读：{pre['files']} 文件 / {pre['bytes']} 字节 / "
                  f"{pre['seconds']:.3f}s / {pre['threads']} 线程")
        print("[A] ✅ 合成端到端体检通过")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _timed_read(paths: Sequence[Path], *, parallel: bool, threads: int = 6) -> float:
    from concurrent.futures import ThreadPoolExecutor

    def _one(p: Path) -> None:
        with open(p, "rb", buffering=0) as fh:
            while fh.read(8 * 1024 * 1024):
                pass

    t0 = time.perf_counter()
    if parallel:
        with ThreadPoolExecutor(max_workers=max(1, min(threads, len(paths)))) as pool:
            list(pool.map(_one, paths))
    else:
        for p in paths:
            _one(p)
    return time.perf_counter() - t0


def mode_b(ckpt: Path, *, write_copy: Optional[Path], verbose: bool = True) -> int:
    from tools import make_bf16_ckpt as mk

    if not ckpt.is_dir():
        print(f"[B] ❌ 找不到 ckpt 目录：{ckpt}")
        return 2
    files = sorted(ckpt.glob("*.safetensors"))
    if not files:
        print(f"[B] ❌ 目录里没有 .safetensors：{ckpt}")
        return 2
    total = sum(f.stat().st_size for f in files)
    print(f"[B] ckpt={ckpt} 分片={len(files)} 体积={total / 1024**3:.2f} GiB")
    cold = _timed_read(files, parallel=False)
    warm = _timed_read(files, parallel=False)
    par = _timed_read(files, parallel=True)
    print(f"[B1] 冷读(顺序) {cold:.1f}s ({total / 1024**3 / max(cold, 1e-9):.2f} GiB/s) | "
          f"热读(顺序) {warm:.1f}s | 并行冷读 {par:.1f}s")
    verdict = "并行更快 ✅" if par < cold * 0.8 else "并行无收益 ⚠️（顺序读已够快或是同一份页缓存）"
    print(f"[B1] 判定：{verdict}")
    if write_copy is not None:
        rc = mk.main(["--src", str(ckpt), "--dst", str(write_copy), "--verify", "full"])
        if rc != 0:
            print(f"[B2] ❌ BF16 副本失败 rc={rc}")
            return rc
        print(f"[B2] ✅ BF16 副本 + 逐张量校验通过：{write_copy}")
    else:
        print("[B2] （未加 --write-copy，跳过真实副本转换）")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="加载加速端到端体检（A 合成 / B 真机实测）")
    ap.add_argument("--ckpt", help="真机模式：真实 hf_ckpt 目录（只读）")
    ap.add_argument("--write-copy", help="真机模式：把 BF16 副本写到该目录（必须不存在或为空）")
    a = ap.parse_args(argv)
    rc = mode_a()
    if rc != 0:
        return rc
    if a.ckpt:
        return mode_b(Path(a.ckpt), write_copy=Path(a.write_copy) if a.write_copy else None)
    print("[B] （未提供 --ckpt，跳过真实 ckpt 实测）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
