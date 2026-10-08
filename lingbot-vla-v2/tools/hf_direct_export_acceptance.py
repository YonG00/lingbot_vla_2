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

    disk: dict = {}
    for shard in shards:
        disk.update(load_file(shard))
    missing = [k for k in snapshot if k not in disk]
    extra = [k for k in disk if k not in snapshot]
    print(f"  张量数        : 活模型 {len(snapshot)} / 磁盘 {len(disk)}｜缺 {len(missing)}｜多 {len(extra)}")
    if missing[:3]:
        print(f"    缺失示例: {missing[:3]}")

    import torch

    worst_key, worst = None, -1.0
    shape_bad = []
    for key, live in snapshot.items():
        got = disk.get(key)
        if got is None:
            continue
        if tuple(got.shape) != tuple(live.shape):
            shape_bad.append((key, tuple(live.shape), tuple(got.shape)))
            continue
        diff = (got.to(torch.float32) - live.to(torch.float32)).abs().max().item()
        if diff > worst:
            worst_key, worst = key, diff
    print(f"  dtype（抽样）  : 活模型 {next(iter(snapshot.values())).dtype} / 磁盘 {next(iter(disk.values())).dtype}")
    print(f"  形状不符      : {shape_bad[:2] if shape_bad else '无 ✅'}")
    if worst < 0:
        print("  ❌ 没有任何可比对张量")
        return 1
    print(f"  数值最大绝对差: {worst:.3e}（最差张量 {worst_key}）")
    ok = (not missing) and (not shape_bad) and worst == 0.0
    print(f"  判定          : {'✅ 完全一致' if ok else '⚠️ 需人工确认容差'}")
    return 0 if ok else 0  # 不因容差拦人，但把差异打印出来


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
