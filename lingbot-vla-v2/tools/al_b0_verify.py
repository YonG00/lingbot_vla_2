#!/usr/bin/env python3
"""Stage B0 新代码路径的**真实数据**验证（无卡可跑，不需要权重）。

验证三件事：
  V1  TaskCatalog 读**真实** manifest（含 TASK_ORDER 分块结构自检）
  V2  `collect_gt_chunks` 在真实数据集上能收 GT（与 evaluator 同一条口径路径）
  V3  Fixed Baseline 的**口径**在真实数据上与手算参照逐值一致

用法::

    python -u tools/al_b0_verify.py [--n-train 40] [--task click_bell]
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

MANIFEST = "/data/train/task_splits/manifest.json"
CLI_YAML = "/data/outputs/single/click_bell/lingbotvla_cli.yaml"
HF_CKPT = "/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt"
QWEN = "/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct"


class _Logger:
    def info_rank0(self, msg, *a, **k):
        print(f"[verify] {msg}", flush=True)

    def info(self, msg, *a, **k):
        print(f"[verify] {msg}", flush=True)

    def warning(self, msg, *a, **k):
        print(f"[verify][WARN] {msg}", flush=True)


def _rss_mb() -> int:
    """当前进程 RSS（MB）。cgroup 上限只有 2 GiB，必须盯着。"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) // 1024
    except Exception:  # noqa: BLE001
        pass
    return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="click_bell")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--out-dir", default="/data/tmp/al_verify_ids")
    ap.add_argument("--no-processor", action="store_true",
                    help="不加载 AutoProcessor（省 ~150MB；GT 反归一化不需要它）")
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[rss] start = {_rss_mb()} MB", flush=True)

    print("=" * 78)
    print("V1  TaskCatalog 读真实 manifest")
    print("=" * 78)
    from lingbotvla.auto_learning.catalog import catalog_from_task_split

    cat = catalog_from_task_split(MANIFEST, strict=True)
    problems = cat.verify()
    print(f"  任务数 = {len(cat)}；names[:4] = {cat.names()[:4]}")
    print(f"  自检问题 = {problems if problems else '无 ✓'}")
    e = cat.entry(a.task)
    print(f"  {a.task}: train={e.n_train} val={e.n_val} n_total={e.n_total} "
          f"sha256_train={e.sha256_train}")
    print(f"  task_of_episode(0)={cat.task_of_episode(0)}  (50)={cat.task_of_episode(50)}")
    assert not problems, "真实 manifest 自检未通过"
    assert e.n_train + e.n_val == 50
    print("  V1 通过 ✓")

    print()
    print("=" * 78)
    print("V2  collect_gt_chunks（真实数据集，model=None 不加载权重）")
    print("=" * 78)
    from transformers import AutoProcessor

    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )
    from lingbotvla.utils.open_loop_validation import OpenLoopValidator

    cfg = LingbotVLAV2Config.from_pretrained(HF_CKPT)
    print(f"  HF config: chunk_size={cfg.chunk_size} n_action_steps={cfg.n_action_steps} "
          f"max_action_dim={cfg.max_action_dim} use_cache={cfg.use_cache} "
          f"loss_type={getattr(cfg, 'loss_type', None)}", flush=True)
    print(f"[rss] after cfg = {_rss_mb()} MB", flush=True)
    if a.no_processor:
        processor = None
        print("  processor=None（--no-processor）")
    else:
        try:
            processor = AutoProcessor.from_pretrained(QWEN, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ AutoProcessor 失败({type(exc).__name__}) ⇒ processor=None 继续")
            processor = None
        print(f"[rss] after processor = {_rss_mb()} MB", flush=True)

    raw = yaml.safe_load(open(CLI_YAML, encoding="utf-8"))
    ns = types.SimpleNamespace(**dict(raw.get("data", {})))
    if not hasattr(ns, "chunk_size"):
        ns.chunk_size = int(cfg.chunk_size)
    if not hasattr(ns, "num_episode"):
        ns.num_episode = None
    align = (raw.get("train", {}) or {}).get("align_params") or {}

    class _NS:
        pass

    args = _NS()
    args.data = ns
    args.model = cfg
    args.train = _NS()
    args.train.output_dir = a.out_dir
    args.train.global_rank = 0

    v = OpenLoopValidator(
        model=None, model_config=cfg, args=args, processor=processor,
        use_depth_align=bool(align), writer=None, logger=_Logger(),
    )
    train_ids = list(e.train_traj_ids)[: a.n_train]
    chunks, keys = v.collect_gt_chunks(train_ids, f"al_verify_{a.task}")
    print(f"[rss] after collect_gt_chunks = {_rss_mb()} MB", flush=True)
    print(f"  train_ids={len(train_ids)} ⇒ chunks={len(chunks)}  action_keys={keys}")
    print(f"  每 chunk 形状: {[tuple(g.shape) for _, g in chunks[:4]]} ...")
    eps = sorted({k for k, _ in chunks})
    print(f"  覆盖回合数 = {len(eps)}（应 == {len(train_ids)}）")
    assert chunks, "collect_gt_chunks 返回空"
    assert len(eps) == len(train_ids), f"回合覆盖不全: {len(eps)} vs {len(train_ids)}"
    # 🔴 立刻释放数据集缓存（`_ds_cache` 只增不减）：chunks 已经在内存里，
    #    后面的口径计算不需要数据集 ⇒ 峰值内存只有「一份数据集」。
    freed = v.clear_dataset_cache()
    print(f"[rss] 释放数据集缓存 {freed} 份 ⇒ {_rss_mb()} MB")
    print("  V2 通过 ✓")

    print()
    print("=" * 78)
    print("V3  Fixed Baseline 口径（与手算参照逐值比对）")
    print("=" * 78)
    from lingbotvla.auto_learning.baseline import (
        MU_GLOBAL, build_baseline, compute_mu,
    )

    # ⚠️ **复用 V2 已经收好的 chunks**，不再建第二个数据集 ——
    #    无卡模式容器只有 2 GiB，同时持两个数据集会被杀。
    #    集合路径本身已由 V2 证明；`compute_fixed_baseline` 就是
    #    `collect_gt_chunks` + `build_baseline` 两行的组合。
    b = build_baseline(a.task, chunks, fingerprint="verify", action_keys=keys,
                       n_train_episodes=len(train_ids), mu_weighting=MU_GLOBAL)
    mu = compute_mu(chunks, weighting=MU_GLOBAL)

    # 手算参照：按回合分组 → 每条轨迹 mean((gt-μ)²) → 轨迹等权
    per: dict = {}
    for k, g in chunks:
        per.setdefault(k, []).append(g)
    per_traj = []
    traj_frames = []
    for k in sorted(per):
        gts = np.concatenate(per[k], axis=0)
        per_traj.append(float(np.mean((gts - mu) ** 2)))
        traj_frames.append(int(gts.shape[0]))
    manual = float(np.mean(per_traj))
    frame_weighted = float(np.mean((np.concatenate([g for _, g in chunks], axis=0) - mu) ** 2))

    print(f"  μ_task[:4]        = {[round(float(x), 6) for x in mu[:4]]}")
    print(f"  baseline(工具)     = {b.mse:.8f}")
    print(f"  baseline(手算参照) = {manual:.8f}   Δ={abs(b.mse - manual):.3e}")
    print(f"  [对照] 按帧加权    = {frame_weighted:.8f}")
    print(f"  各轨迹帧数        = {traj_frames}")
    print(f"  轨迹数={len(per_traj)} chunk数={b.n_chunks} 帧数={b.n_frames} 维数={b.dims}")
    assert abs(b.mse - manual) < 1e-9, "baseline 与手算参照不一致"

    # 轨迹等权 vs 按帧加权：
    #   * 各轨迹帧数**相同** ⇒ 两者数学上必然相等（不是 bug）
    #   * 帧数**不同** ⇒ 必须不同，否则说明口径没生效
    if len(set(traj_frames)) == 1:
        print(f"  [说明] 各轨迹帧数相同({traj_frames[0]}) ⇒ 两种口径必然相等")
        assert abs(b.mse - frame_weighted) < 1e-12
    else:
        assert abs(b.mse - frame_weighted) > 1e-12, (
            f"轨迹帧数不同 {traj_frames} 却与按帧加权相等 ⇒ 口径没生效？")
        print(f"  [说明] 帧数不等 ⇒ 两种口径必然不同 Δ={abs(b.mse - frame_weighted):.3e} ✓")

    # 再确认工具内部真的走了 evaluator 的 aggregate_chunks
    from lingbotvla.utils.open_loop_validation import aggregate_chunks
    ref = aggregate_chunks([(k, g, np.broadcast_to(mu, g.shape).copy()) for k, g in chunks])["mse"]
    print(f"  aggregate_chunks  = {ref:.8f}   Δ={abs(b.mse - ref):.3e}")
    assert abs(b.mse - ref) < 1e-12, "没有走 evaluator 的 aggregate_chunks"
    print("  V3 通过 ✓")

    print()
    print("=" * 78)
    print(f"✅ B0 新代码路径全部通过（task={a.task}, n_train={len(train_ids)}）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
