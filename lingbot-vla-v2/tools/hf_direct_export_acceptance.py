#!/usr/bin/env python
"""HF 直出**独立验收入口**（不伪造 Scheduler PASS 事件）。

用法：与训练器完全相同的参数，用 torchrun 起（1 进程即可）：

    torchrun --nproc-per-node 1 tools/hf_direct_export_acceptance.py \
        /data/train/configs/robotwin_official_paths.yaml \
        --model.model_path … --train.output_dir … --train.hf_pass_interval 1 \
        --train.save_hf_weights false --train.async_save_hf_weights false \
        --train.dcp_save_mode always --train.save_steps 1000 --train.save_epochs 0 …

它做三件事：

1. **只在决策层打一个一次性补丁**：第一次进入 PASS 里程碑判定时强制返回 `"hf"`，
   从而**按需触发真实导出**。Scheduler / Registry / PASS 事件**完全不动**（不伪造任何 PASS）。
2. 调用训练器的 `main()`（真实 FSDP2/MoE 模型、真实导出路径、真实原子发布）。
3. 导出后**立刻**再取一次全量 CPU state，与**已经写到磁盘上的 safetensors** 逐张量比对，
   并打印：分片数、dtype、最大绝对差、导出耗时、进程 RSS 峰值（VmHWM）。

判据：`max_abs_diff == 0`（或 F32↔bf16 转换后的容差内）⇒ 导出的 HF 与活模型一致。
"""

from __future__ import annotations

import glob
import importlib.util
import os
import pathlib
import sys
import time

FIRED = {"done": False}
CAP: dict = {}
TM: dict = {}


def _install_decision_patch() -> None:
    """只改"要不要导"这一个决策；导出实现本身用原版（外面再包一层做计时/取证）。"""
    import lingbotvla.utils.al_checkpoint_policy as policy
    import lingbotvla.utils.direct_hf_checkpoint as dhc

    def one_shot_action(**kwargs):  # noqa: ANN003, ARG001
        if not FIRED["done"]:
            FIRED["done"] = True
            return "hf"
        return "none"

    policy.milestone_action = one_shot_action  # type: ignore[assignment]

    original_export = dhc.export_model_hf_direct

    def timed_export(model, **kwargs):  # noqa: ANN001, ANN003
        started = time.time()
        try:
            return original_export(model, **kwargs)
        finally:
            TM["export_seconds"] = time.time() - started
            try:
                import torch

                with torch.no_grad():
                    CAP["snapshot"] = dhc.collect_full_model_on_cpu(model)
            except Exception as exc:  # noqa: BLE001
                CAP["snapshot_error"] = repr(exc)

    dhc.export_model_hf_direct = timed_export  # type: ignore[assignment]
    print("[acceptance] 已安装一次性里程碑决策补丁（仅触发一次真实 HF 直出）", flush=True)


def _arg_value(flag: str) -> str | None:
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None


def _rss_peak_gb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024 / 1024  # kB → GiB
    except OSError:
        return None
    return None


def _verify_export(output_dir: str) -> int:
    print("\n" + "=" * 78)
    print("[acceptance] HF 直出验收结果")
    print("=" * 78)
    found = sorted(glob.glob(os.path.join(output_dir, "hf_milestones", "global_step_*", "hf_ckpt")))
    if not found:
        print("  ❌ 没有找到任何 hf_milestones/global_step_*/hf_ckpt")
        return 1
    target = found[-1]
    print(f"  导出目录      : {target}")
    shards = sorted(glob.glob(os.path.join(target, "*.safetensors")))
    print(f"  分片数        : {len(shards)}")
    print(f"  导出耗时      : {TM.get('export_seconds', float('nan')):.1f} s（同步暂停训练的时长）")
    peak = _rss_peak_gb()
    print(f"  进程 RSS 峰值 : {peak:.1f} GiB" if peak else "  进程 RSS 峰值 : (不可读)")
    print(f"  临时目录残留  : {glob.glob(os.path.join(os.path.dirname(target), '.hf_ckpt.tmp.*')) or '无 ✅'}")

    snapshot = CAP.get("snapshot")
    if snapshot is None:
        print(f"  ⚠️ 未能取得活模型快照（{CAP.get('snapshot_error')}）⇒ 跳过数值比对")
        return 1
    try:
        from safetensors.torch import load_file
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ safetensors 不可用（{exc}）⇒ 跳过数值比对")
        return 1

    import collections

    import torch

    # ---- 分块比对：逐个分片读、逐块算，避免"整模型 upcast 到 fp32"的集中大分配 ----
    # 每块元素数：4M ⇒ fp32 下约 16 MB/块，内存与张量总量无关。
    CHUNK = 1 << 22
    live_dtypes = collections.Counter()
    disk_dtypes = collections.Counter()
    seen: set = set()
    missing: list = []
    shape_bad: list = []
    worst_key, worst = None, -1.0
    compared = 0
    upcast_only = True

    def _max_diff_chunked(live_t, disk_t) -> float:
        """按块比较 (disk - live_as_disk_dtype) 的最大绝对值；不整体 upcast。"""
        a = live_t.detach().reshape(-1)
        b = disk_t.detach().reshape(-1)
        if a.numel() != b.numel():
            return float("inf")
        best = 0.0
        for i in range(0, a.numel(), CHUNK):
            ca = a[i:i + CHUNK].to(dtype=disk_t.dtype, device="cpu")
            cb = b[i:i + CHUNK].to(device="cpu")
            if cb.dtype != ca.dtype:
                cb = cb.to(dtype=ca.dtype)
            d = (cb.to(torch.float32) - ca.to(torch.float32)).abs().max().item()
            if d > best:
                best = d
            del ca, cb
        return best

    for shard in shards:
        part = load_file(shard)
        for key, disk_t in part.items():
            seen.add(key)
            live_t = snapshot.get(key)
            if live_t is None:
                continue
            live_dtypes[str(live_t.dtype)] += 1
            disk_dtypes[str(disk_t.dtype)] += 1
            if tuple(disk_t.shape) != tuple(live_t.shape):
                shape_bad.append((key, tuple(live_t.shape), tuple(disk_t.shape)))
                continue
            if disk_t.is_floating_point():
                if disk_t.dtype != live_t.dtype:
                    upcast_only = upcast_only and (disk_t.element_size() >= live_t.element_size())
                d = _max_diff_chunked(live_t, disk_t)
            else:
                d = 0.0 if torch.equal(disk_t, live_t) else float("inf")
            compared += 1
            if d > worst:
                worst_key, worst = key, d
        del part  # 只保留当前分片，别把 24 GB 全摊在内存里

    missing = [k for k in snapshot if k not in seen]
    extra = [k for k in seen if k not in snapshot]
    print(f"  张量数        : 活模型 {len(snapshot)} / 磁盘 {len(seen)}｜缺 {len(missing)}｜多 {len(extra)}"
          f"｜已比对 {compared}")
    if missing[:3]:
        print(f"    缺失示例: {missing[:3]}")
    print(f"  dtype（直方） : 活模型 {dict(live_dtypes.most_common(3))} / 磁盘 {dict(disk_dtypes.most_common(3))}")
    print(f"  精度方向      : {'升/同精度（无损）' if upcast_only else '⚠️ 存在降精度（有意为之，并非无损）'}")
    print(f"  形状不符      : {shape_bad[:2] if shape_bad else '无 ✅'}")
    if worst < 0:
        print("  ❌ 没有任何可比对张量")
        return 1
    print(f"  数值最大绝对差: {worst:.3e}（最差张量 {worst_key}；分块比较，块大小 {CHUNK}）")
    ok = (not missing) and (not extra) and (not shape_bad) and worst == 0.0
    print(f"  判定          : {'✅ 与活模型逐位一致（按磁盘 dtype 语义）' if ok else '⚠️ 需人工确认容差'}")
    return 0 if ok else 1


def main() -> int:
    _install_decision_patch()
    output_dir = _arg_value("--train.output_dir") or ""
    trainer = pathlib.Path(__file__).resolve().parents[1] / "tasks" / "vla" / "train_lingbotvla.py"
    spec = importlib.util.spec_from_file_location("train_lingbotvla", trainer)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # 必须先注册进 sys.modules：否则 `@dataclass` 在 exec 期间解析
    # `sys.modules[cls.__module__]` 会拿到 None（报 NoneType has no __dict__）。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    print(f"[acceptance] 调用真实训练器 main()（{trainer}）", flush=True)
    rc = 0
    try:
        module.main()
    except SystemExit as exc:  # 训练器正常结束可能走 SystemExit
        rc = int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001
        print(f"[acceptance] 训练器抛异常（导出验收仍继续）: {exc!r}")
        rc = 1
    return _verify_export(output_dir) or rc


if __name__ == "__main__":
    sys.exit(main())
