#!/usr/bin/env python3
"""把 F32 的 HF ckpt 复制成 **BF16 副本**，用于加速训练/评测启动时的权重加载。

## 为什么
实测（2026-10-09，50-task GMean200 正式跑）：`Prepare model → Start training` 共 **129 s**，
其中 `Loading checkpoint shards 0/6→6/6` 占 **84 s**（≈290 MB/s）——
因为这个 ckpt 是 **F32**（1708 个张量全 F32，磁盘上 24 GB），加载后要转成 BF16（12 GB）用。
**只读它一半的体积**，就能把这段读盘时间近似减半，并省掉 F32→BF16 的 CPU 转换。

## 做什么
逐分片流式处理（一次只驻留一个分片的张量）：
* 浮点张量 → `to(bfloat16)`；非浮点（int64/bool 等）**原样保留**；
* 分片文件名与 `model.safetensors.index.json` 的 `weight_map` **保持不变**（只有 `metadata.total_size` 更新）；
* 其余文件（config / processor / tokenizer / *.py 等）按字节复制。

## 安全
* 默认**逐张量全量校验**：重开两份文件，断言
  `dst == src.to(bf16)`（浮点，**逐位相等**）/ `dst == src`（非浮点）；不一致即非零退出；
* 拒绝：目标目录已存在且非空、目标是源目录（或在其内部）、源是 `.bin`（只支持 safetensors）；
* `--dry-run` 只报计划不写盘。

用法：
    python tools/make_bf16_ckpt.py --src <hf_ckpt> --dst <new_dir> [--verify full|sample:N|none]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

#: 与权重同名的索引文件（需要在副本里重写 total_size）。
INDEX_NAME = "model.safetensors.index.json"
#: 逐分片处理时的张量上限（超过就报错，避免误吃内存）。
MAX_SHARD_BYTES = 16 * 1024**3


def list_shards(src: Path) -> List[Path]:
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"[bf16] ❌ 源目录没有 .safetensors：{src}")
    return shards


def plan(src: Path, dst: Path) -> Dict[str, Any]:
    shards = list_shards(src)
    total = sum(p.stat().st_size for p in shards)
    others = sorted(p.name for p in src.iterdir()
                    if p.is_file() and p.suffix != ".safetensors")
    return {"shards": [p.name for p in shards], "shard_count": len(shards),
            "src_bytes": total, "other_files": others}


def _check_paths(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise SystemExit(f"[bf16] ❌ 源目录不存在：{src}")
    if src.resolve() == dst.resolve():
        raise SystemExit("[bf16] ❌ 目标目录不能等于源目录")
    if src.resolve() in dst.resolve().parents:
        raise SystemExit("[bf16] ❌ 目标目录不能在源目录内部")
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"[bf16] ❌ 目标目录已存在且非空，拒绝覆盖：{dst}")
    if any(src.glob("*.bin")):
        raise SystemExit("[bf16] ❌ 检测到 .bin 权重：本工具只支持 safetensors")


def _sanitize_tensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """修掉 safetensors 的两条硬限制（真实 ckpt 一般不触发，但一旦触发就是硬失败）：

    * **非连续张量** ⇒ `save_file` 抛 `ValueError: non contiguous tensor`（`.contiguous()` 解决）；
    * **两个 key 共享存储**（tied weights）⇒ `save_file` 抛 `RuntimeError: Some tensors share memory`
      （把后出现的那个 `clone()` 掉）。

    返回计数 ``{"contiguous_fixed": n, "shared_cloned": m}``。
    """
    stats = {"contiguous_fixed": 0, "shared_cloned": 0}
    seen: Dict[int, str] = {}
    for key in list(tensors):
        t = tensors[key]
        if not t.is_contiguous():
            t = t.contiguous()
            stats["contiguous_fixed"] += 1
        ptr = t.untyped_storage().data_ptr()
        if ptr in seen:
            t = t.clone()
            stats["shared_cloned"] += 1
        else:
            seen[ptr] = key
        tensors[key] = t
    return stats

def convert_one_shard(src_file: Path, dst_file: Path, dtype: torch.dtype) -> Tuple[int, int, int]:
    """转换单个分片。返回 (张量数, 源字节, 目标字节)。"""
    src_bytes = src_file.stat().st_size
    out: Dict[str, torch.Tensor] = {}
    n_float = n_other = 0
    with safe_open(str(src_file), framework="pt", device="cpu") as fh:
        for key in fh.keys():
            tensor = fh.get_tensor(key)
            if tensor.is_floating_point():
                out[key] = tensor.to(dtype)
                n_float += 1
            else:
                out[key] = tensor                       # int64/bool 等原样
                n_other += 1
    stats = _sanitize_tensors(out)
    if stats["contiguous_fixed"] or stats["shared_cloned"]:
        print(f"[bf16]   {src_file.name}: 防御修复 contiguous={stats['contiguous_fixed']} "
              f"shared_cloned={stats['shared_cloned']}")
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst_file), metadata={"format": "pt"})
    del out
    return n_float + n_other, src_bytes, dst_file.stat().st_size


def rewrite_index(src: Path, dst: Path, dst_shards: Dict[str, int]) -> Optional[int]:
    """重写 index.json（weight_map 不变，只更新 total_size）。返回新的 total_size。"""
    index = src / INDEX_NAME
    if not index.is_file():
        return None
    doc = json.loads(index.read_text(encoding="utf-8"))
    total = int(sum(dst_shards.values()))
    doc.setdefault("metadata", {})["total_size"] = total
    (dst / INDEX_NAME).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    return total


def copy_other_files(src: Path, dst: Path, names: Sequence[str]) -> List[str]:
    done = []
    for name in names:
        if name == INDEX_NAME:
            continue
        shutil.copy2(src / name, dst / name)
        done.append(name)
    return done


def verify(src: Path, dst: Path, shards: Sequence[str], dtype: torch.dtype,
           sample: Optional[int] = None) -> Dict[str, Any]:
    """逐张量校验：浮点 `dst == src.to(dtype)`（逐位），非浮点 `dst == src`。

    * 分片缺失/打不开 ⇒ 记为 problem（**返回值 ok=False，不抛异常**）；
    * `sample=N` ⇒ 每片只逐位比对**前 N 个 key**（key 集合仍全量比对）。
    """
    bad: List[Dict[str, str]] = []
    checked = 0
    keys_total = 0
    src_keys: set = set()
    dst_keys: set = set()
    for name in shards:
        try:
            fs = safe_open(str(src / name), framework="pt", device="cpu")
            fd = safe_open(str(dst / name), framework="pt", device="cpu")
        except Exception as exc:  # noqa: BLE001
            bad.append({"shard": name, "key": "<open>", "detail": f"{type(exc).__name__}: {exc}"})
            continue
        with fs, fd:
            ks, kd = set(fs.keys()), set(fd.keys())
            src_keys |= ks
            dst_keys |= kd
            if ks != kd:
                bad.append({"shard": name, "key": "<key-set>",
                            "detail": f"missing={sorted(ks - kd)[:3]} extra={sorted(kd - ks)[:3]}"})
            keys_total += len(ks)
            for i, key in enumerate(sorted(ks & kd)):
                if sample is not None and i >= sample:
                    break
                a = fs.get_tensor(key)
                b = fd.get_tensor(key)
                checked += 1
                if a.is_floating_point():
                    want = a.to(dtype)
                    if b.dtype != dtype or not torch.equal(b, want):
                        bad.append({"shard": name, "key": key,
                                    "detail": f"float mismatch dtype={b.dtype} want={dtype}"})
                else:
                    if b.dtype != a.dtype or not torch.equal(b, a):
                        bad.append({"shard": name, "key": key,
                                    "detail": f"non-float mismatch {b.dtype} vs {a.dtype}"})
    return {"ok": not bad and src_keys == dst_keys, "checked": checked,
            "n_tensors": keys_total, "key_sets_equal": src_keys == dst_keys,
            "problems": bad[:10]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="F32 HF ckpt → BF16 副本（含逐张量校验）")
    ap.add_argument("--src", required=True, help="源 hf_ckpt 目录（F32）")
    ap.add_argument("--dst", required=True, help="新目录（必须不存在或为空）")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--verify", default="full", help="full（默认）| sample:N | none")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    src, dst = Path(a.src), Path(a.dst)
    _check_paths(src, dst)
    p = plan(src, dst)
    dtype = getattr(torch, a.dtype)
    print(f"[bf16] 源 {src}：{p['shard_count']} 片 / {p['src_bytes'] / 1024**3:.2f} GiB；"
          f"其它文件 {len(p['other_files'])} 个；目标 {dst}；dtype={a.dtype}")
    if a.dry_run:
        print("[bf16] DRY-RUN：不写盘。分片 =", ", ".join(p["shards"]))
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    src_sizes: Dict[str, int] = {}
    t0 = time.perf_counter()
    n_tensors = 0
    seen_keys: Dict[str, str] = {}
    for name in p["shards"]:
        with safe_open(str(src / name), framework="pt", device="cpu") as _fh:
            dupes = [k for k in _fh.keys() if k in seen_keys]
        if dupes:
            raise SystemExit(f"[bf16] ❌ 跨分片重复 key（索引会歧义）：{dupes[:5]} @ {name}")
        with safe_open(str(src / name), framework="pt", device="cpu") as _fh:
            for k in _fh.keys():
                seen_keys[k] = name
        n, sb, db = convert_one_shard(src / name, dst / name, dtype)
        n_tensors += n
        src_sizes[name] = db
        print(f"[bf16]   {name}: {n} 张量 {sb / 1024**3:.2f} → {db / 1024**3:.2f} GiB")
    copy_other_files(src, dst, p["other_files"])
    total = rewrite_index(src, dst, src_sizes)
    convert_s = time.perf_counter() - t0
    out_bytes = sum(f.stat().st_size for f in dst.glob("*.safetensors"))
    print(f"[bf16] ✅ 转换完成：{n_tensors} 张量 / {convert_s:.1f} s / "
          f"{p['src_bytes'] / 1024**3:.2f} → {out_bytes / 1024**3:.2f} GiB"
          + (f"（index.total_size={total}）" if total is not None else ""))

    verdict: Dict[str, Any] = {"ok": True, "checked": 0}
    if a.verify != "none":
        sample = None
        if a.verify.startswith("sample:"):
            sample = int(a.verify.split(":", 1)[1])
        t1 = time.perf_counter()
        verdict = verify(src, dst, p["shards"], dtype, sample=sample)
        print(f"[bf16] 🔍 校验（{a.verify}）：checked={verdict['checked']}/"
              f"{verdict['n_tensors']} key_sets_equal={verdict['key_sets_equal']} "
              f"ok={verdict['ok']} / {time.perf_counter() - t1:.1f} s")
        if not verdict["ok"]:
            print("[bf16] ❌ 校验失败（副本与源不一致）⇒ 请勿使用：", verdict["problems"])
            return 2
    print(f"[bf16] 完成。用法：--model.model_path {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
