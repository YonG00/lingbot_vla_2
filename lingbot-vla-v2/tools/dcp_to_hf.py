#!/usr/bin/env python3
"""DCP → HF safetensors 转换（**单进程、不需要 GPU、不需要 torch.distributed**）。

为什么需要它（2026-10-10）
--------------------------
50 任务那轮自动学习训练**只落了 DCP**（启动器里 `SAVE_HF_BOOL=false` 写死），
而**闭环评测**要求 `<CKPT_ROOT>/checkpoints/global_step_N/hf_ckpt`（HuggingFace 目录）
⇒ 没有 HF 就评不了。仓库里原本**没有** DCP→HF 的转换脚本（`tools/` 只有 `prune_dcp.py`）。

本脚本把两个现成件拼起来，**不复制任何写出逻辑**：
* `lingbotvla.checkpoint.format_utils.dcp_to_torch_state_dict()` —— 单进程 `no_dist=True`
  把 DCP 读成普通 state_dict；
* `lingbotvla.utils.direct_hf_checkpoint.export_model_hf_direct(..., snapshot=…)` —— 复用训练器
  自己的 HF 写出路径（分片 + `model.safetensors.index.json` + `model_assets` 资源 + 磁盘余量检查 +
  `verify_hf_weight_files` 头校验 + 临时目录原子发布）。

用法
----
    PY=/data/miniconda3/envs/lingbotvla/bin/python
    # 默认：读 <run>/checkpoints/global_step_750，产出 <run>/checkpoints/global_step_750/hf_ckpt（bf16）
    $PY tools/dcp_to_hf.py --dcp /data/outputs/al_50task_gmean200_gbs24_r2/checkpoints/global_step_750

    # 指定精度 / 资源目录 / 干跑
    $PY tools/dcp_to_hf.py --dcp <dir> --dtype fp32 --model-assets <run>/model_assets --dry-run

退出码：`0` 成功 / `2` 失败（读取、写出或校验失败）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_STEP_RE = re.compile(r"global_step_(\d+)")


def _parse_step(dcp: Path, explicit: Optional[int]) -> int:
    if explicit is not None:
        return int(explicit)
    m = _STEP_RE.search(str(dcp))
    if m:
        return int(m.group(1))
    raise SystemExit(f"[dcp2hf] ❌ 无法从路径推断 step：{dcp}（请用 --step 指定）")


def _default_assets(dcp: Path, explicit: Optional[str]) -> Optional[str]:
    """默认资源目录：先看 `<dcp 同级的 run 根>/model_assets`，再看 `<dcp>/assets`。"""
    if explicit:
        return explicit
    for cand in (dcp.parent.parent / "model_assets", dcp / "model_assets"):
        if cand.is_dir():
            return str(cand)
    return None


def convert(*, dcp: Path, step: int, checkpoint_root: Path, dtype: str,
            model_assets: Optional[str], verify_sample: int = 5,
            dry_run: bool = False, log=print) -> Optional[Path]:
    """执行转换；返回产出的 `hf_ckpt` 目录（dry_run 返回 None）。"""
    from lingbotvla.checkpoint.format_utils import dcp_to_torch_state_dict
    from lingbotvla.utils.direct_hf_checkpoint import export_model_hf_direct

    if not dcp.exists():
        raise SystemExit(f"[dcp2hf] ❌ DCP 目录不存在：{dcp}")
    log(f"[dcp2hf] DCP        = {dcp}")
    log(f"[dcp2hf] step       = {step}")
    log(f"[dcp2hf] 输出根     = {checkpoint_root}（产出 <root>/global_step_{step}/hf_ckpt）")
    log(f"[dcp2hf] 精度       = {dtype}")
    log(f"[dcp2hf] 资源目录   = {model_assets or '(无：只写权重)'}")
    if dry_run:
        log("[dcp2hf] --dry-run ⇒ 只打印计划，不读不写")
        return None

    log("[dcp2hf] ① 读取 DCP → state_dict（单进程 no_dist）…")
    sd = dcp_to_torch_state_dict(str(dcp))
    n_tensors = len(sd)
    n_bytes = sum(int(t.numel()) * int(t.element_size()) for t in sd.values()
                  if hasattr(t, "numel"))
    log(f"[dcp2hf]    读到 {n_tensors} 个张量 / {n_bytes / 1024**3:.2f} GiB（CPU）")

    log("[dcp2hf] ② 复用训练器写出路径导出 HF（原子发布 + 头校验）…")
    # ⚠️ 不传 `model_assets=`：训练器那边的类型是 **ModelAssets 对象列表**（`[model_config, processor]`），
    #    不是路径。离线转换没有这两个对象 ⇒ 先只写权重，随后**按目录拷贝**训练产物里的 `model_assets/`。
    dest = export_model_hf_direct(
        None, snapshot=sd, global_step=step, checkpoint_root=str(checkpoint_root),
        export_dtype=dtype, logger=None,
    )
    if dest is None:
        raise RuntimeError("export_model_hf_direct 返回 None（写出失败）")
    log(f"[dcp2hf]    ✅ 已发布：{dest}")

    copied = _copy_assets(Path(dest), model_assets, log=log)
    if copied:
        log(f"[dcp2hf]    已补入 {copied} 个资源文件（config/tokenizer 等）")

    log("[dcp2hf] ③ 校验：key 集合 + 抽样逐张量比对")
    _verify(Path(dest), sd, dtype, sample=verify_sample, log=log)
    shards = sorted(p.name for p in Path(dest).glob("*.safetensors"))
    size = sum(p.stat().st_size for p in Path(dest).glob("*") if p.is_file())
    log(f"[dcp2hf] ✅ 完成：{len(shards)} 个分片 / {size / 1024**3:.2f} GiB ⇒ {dest}")
    return Path(dest)


def _copy_assets(dest: Path, assets: Optional[str], *, log=print) -> int:
    """把训练产物里的 `model_assets/`（config.json / tokenizer 等）拷进 HF 目录。

    只补**缺失**的文件，不覆盖已写好的权重与索引。
    """
    import shutil

    if not assets:
        return 0
    src = Path(assets)
    if not src.is_dir():
        log(f"[dcp2hf] ⚠️ 资源目录不存在，跳过：{src}")
        return 0
    n = 0
    for item in sorted(src.iterdir()):
        if item.is_dir():
            for sub in sorted(item.rglob("*")):
                if sub.is_file():
                    rel = sub.relative_to(src)
                    target = dest / rel
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(sub, target)
                        n += 1
        elif item.is_file():
            target = dest / item.name
            if not target.exists():
                shutil.copy2(item, target)
                n += 1
    return n


def _verify(dest: Path, sd: Dict[str, Any], dtype: str, *, sample: int, log=print) -> None:
    """key 集合全量比对 + 抽样逐位比对（浮点按目标 dtype 转换后必须逐位相等）。"""
    import torch
    from safetensors import safe_open

    # 🔴 必须查**分片实体**，不能只信 index.json —— 索引与分片不一致（少写/漏写张量）时，
    #    只查索引会"看起来通过"（2026-10-10 测试里就靠这条抓出来的）。
    weight_map = {}
    for shard in sorted(dest.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as fh:
            for k in fh.keys():
                weight_map[k] = shard.name
    index_path = dest / "model.safetensors.index.json"
    if index_path.is_file():
        idx_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        only_index = sorted(set(idx_map) - set(weight_map))
        if only_index:
            raise RuntimeError(f"index.json 里有、分片里没有的 key：{only_index[:3]}")
    missing = sorted(set(sd) - set(weight_map))
    extra = sorted(set(weight_map) - set(sd))
    if missing or extra:
        raise RuntimeError(f"HF key 集合不一致：missing={missing[:3]} extra={extra[:3]}")

    want_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32,
                  "fp16": torch.float16}.get(str(dtype).lower())
    checked = 0
    for key in sorted(sd)[:max(0, int(sample))]:
        with safe_open(str(dest / weight_map[key]), framework="pt", device="cpu") as fh:
            got = fh.get_tensor(key)
        src = sd[key]
        if src.is_floating_point() and want_dtype is not None:
            expect = src.to(want_dtype)
            if got.dtype != want_dtype or not torch.equal(got, expect):
                raise RuntimeError(f"抽样校验失败：{key} dtype={got.dtype} 期望={want_dtype}")
        else:
            if got.dtype != src.dtype or not torch.equal(got, src):
                raise RuntimeError(f"抽样校验失败（非浮点）：{key} {got.dtype} vs {src.dtype}")
        checked += 1
    log(f"[dcp2hf]    校验通过：{len(weight_map)} 个 key 全在；抽样 {checked} 个逐位一致")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DCP → HF safetensors（单进程，无需 GPU）")
    ap.add_argument("--dcp", required=True, help="DCP 目录，如 <run>/checkpoints/global_step_750")
    ap.add_argument("--step", type=int, default=None, help="默认从路径 global_step_N 推断")
    ap.add_argument("--checkpoint-root", default=None,
                    help="输出根目录，默认 = DCP 的父目录（即 <run>/checkpoints）")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16", "native"],
                    help="浮点权重导出精度（默认 bf16；非浮点原样保留）")
    ap.add_argument("--model-assets", default=None, help="默认自动找 <run>/model_assets")
    ap.add_argument("--verify-sample", type=int, default=5, help="抽样逐位比对的张量数（0=只查 key）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    dcp = Path(a.dcp).resolve()
    step = _parse_step(dcp, a.step)
    root = Path(a.checkpoint_root).resolve() if a.checkpoint_root else dcp.parent
    assets = _default_assets(dcp, a.model_assets)
    try:
        convert(dcp=dcp, step=step, checkpoint_root=root, dtype=a.dtype,
                model_assets=assets, verify_sample=a.verify_sample, dry_run=a.dry_run)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[dcp2hf] ❌ 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
