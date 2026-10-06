#!/usr/bin/env python3
"""一次性预计算 **Fixed Task BaselineMSE**（task-global-mean + trajectory-balanced）。

口径
----
    1) 固定 **train split**（manifest 的 train_ids，默认 40 条）收集 GT chunk
    2) μ_task = 所有 train chunk 的所有帧、所有维的均值
    3) 每条 train trajectory：mse_i = mean((gts_i - μ_task)²)
    4) baseline_mse = mean_i(mse_i)          ← 轨迹等权

与 evaluator **共用同一条 GT 收集路径**（`OpenLoopValidator.collect_gt_chunks`）
⇒ action space / normalization / valid region 完全一致。

**不需要模型权重、不需要 GPU** —— 只用到 HF config + processor + 数据集。

用法
----
    cd /data/code/lingbot-vla-v2
    export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct
    python -u -m lingbotvla.auto_learning.tools.compute_task_baseline \
        --manifest   /data/train/task_splits/manifest.json \
        --config     /data/outputs/<run>/lingbotvla_cli.yaml \
        --model-path /data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct \
        --out-dir    /data/train/task_splits/_baseline_ids \
        --out        /data/train/task_splits/task_baseline.json

    # 只算部分任务 / 只看会做什么
    ... --tasks click_bell,turn_switch
    ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]          # lingbotvla/auto_learning/tools -> 仓库根
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class _Logger:
    """最小 logger（validator 只用 info_rank0 / info / warning）。"""

    def info_rank0(self, msg, *a, **k):
        print(f"[baseline] {msg}", flush=True)

    def info(self, msg, *a, **k):
        print(f"[baseline] {msg}", flush=True)

    def warning(self, msg, *a, **k):
        print(f"[baseline][WARN] {msg}", flush=True)


def _build_args(data_cfg: dict, hf_cfg, out_dir: str):
    """按 `tools/open_loop_eval_inprocess.py` 的同一套路构造 validator 需要的 args。"""
    ns = types.SimpleNamespace(**data_cfg)
    if not hasattr(ns, "chunk_size"):
        ns.chunk_size = int(getattr(hf_cfg, "chunk_size", 50))
    if not hasattr(ns, "num_episode"):
        ns.num_episode = None

    class _NS:
        pass

    args = _NS()
    args.data = ns
    # ⚠️ 训练侧 `build_vla_dataset(model_config=args.model, config=model.config)`：
    #    离线路径下两者都用 HF config（与 open_loop_eval_inprocess.py 一致）
    args.model = hf_cfg
    args.train = _NS()
    args.train.output_dir = out_dir
    args.train.global_rank = 0
    return args


def main() -> int:
    ap = argparse.ArgumentParser(
        description="预计算 Fixed Task BaselineMSE（无需权重 / 无需 GPU）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", required=True, help="tools/task_split.py 的 manifest.json")
    ap.add_argument("--config", required=True, help="训练用的 yaml（取它的 data 段）")
    ap.add_argument("--model-path", required=True,
                    help="HF config / processor 的来源（**不加载权重**）")
    ap.add_argument("--out", default=None,
                    help="输出 json（默认 <manifest 目录>/task_baseline.json）")
    ap.add_argument("--out-dir", default=None,
                    help="临时回合白名单的落地目录（默认 <manifest 目录>/_baseline_ids）")
    ap.add_argument("--tasks", default=None, help="逗号分隔；默认全部")
    ap.add_argument("--mu-weighting", choices=["global", "trajectory"], default="global",
                    help="μ_task 的加权方式（默认 global = 所有帧 pooled 平均）")
    ap.add_argument("--recompute", action="store_true",
                    help="忽略已有缓存，全部重算")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做输入校验与自检，不算 baseline")
    a = ap.parse_args()

    import yaml

    from lingbotvla.auto_learning.baseline import (
        MU_GLOBAL, MU_TRAJECTORY, BaselineStore, baseline_fingerprint,
        compute_fixed_baseline,
    )
    from lingbotvla.auto_learning.catalog import catalog_from_task_split

    manifest = Path(a.manifest).resolve()
    if not manifest.is_file():
        print(f"❌ 找不到 manifest: {manifest}", file=sys.stderr)
        return 2
    out = Path(a.out).resolve() if a.out else manifest.parent / "task_baseline.json"
    out_dir = str(Path(a.out_dir).resolve() if a.out_dir else manifest.parent / "_baseline_ids")
    os.makedirs(out_dir, exist_ok=True)

    raw = yaml.safe_load(open(a.config, encoding="utf-8"))
    data_cfg = dict(raw.get("data", {}))
    align_params = (raw.get("train", {}) or {}).get("align_params") or {}
    use_depth_align = bool(align_params)

    # ---- task split：严格自检（划分正确 / 无泄漏 / 分块结构）----
    cat = catalog_from_task_split(str(manifest), strict=True)
    problems = cat.verify()
    if problems:
        print("❌ task split 自检未通过：", file=sys.stderr)
        for p in problems:
            print(f"   - {p}", file=sys.stderr)
        return 3

    tasks = cat.names() if not a.tasks else [t.strip() for t in a.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in cat]
    if unknown:
        print(f"❌ 未知任务: {unknown}", file=sys.stderr)
        return 2

    fp = baseline_fingerprint(
        dataset_root=data_cfg.get("train_path"),
        sha256_train=None,                     # 逐任务不同，见下
        norm_stats_file=data_cfg.get("norm_stats_file"),
        cameras=data_cfg.get("cameras"),
        joints=data_cfg.get("joints"),
        chunk_size=data_cfg.get("chunk_size"),
        img_size=data_cfg.get("img_size"),
        per_episode_stride=True,
        mu_weighting=a.mu_weighting,
    )
    print(f"[baseline] manifest = {manifest}")
    print(f"[baseline] tasks    = {len(tasks)} 个；mu_weighting={a.mu_weighting}")
    print(f"[baseline] 配置指纹 = {fp}（逐任务再叠加 sha256_train）")
    print(f"[baseline] 输出     = {out}")
    if a.dry_run:
        print("[baseline] --dry-run：输入与自检均通过，未计算。")
        return 0

    # ---- 建 validator（**model=None** ⇒ 只收 GT，不推理）----
    from transformers import AutoConfig, AutoProcessor

    from lingbotvla.utils.open_loop_validation import OpenLoopValidator

    hf_cfg = AutoConfig.from_pretrained(a.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(a.model_path, trust_remote_code=True)
    args = _build_args(data_cfg, hf_cfg, out_dir)
    validator = OpenLoopValidator(
        model=None, model_config=hf_cfg, args=args, processor=processor,
        use_depth_align=use_depth_align, writer=None, logger=_Logger(),
    )

    store = BaselineStore.load(str(out), fingerprint=fp)
    n_done = n_skip = 0
    for name in tasks:
        entry = cat.entry(name)
        task_fp = baseline_fingerprint(
            dataset_root=data_cfg.get("train_path"),
            sha256_train=entry.sha256_train,      # ← 逐任务叠加：划分变了分母就不可比
            norm_stats_file=data_cfg.get("norm_stats_file"),
            cameras=data_cfg.get("cameras"), joints=data_cfg.get("joints"),
            chunk_size=data_cfg.get("chunk_size"), img_size=data_cfg.get("img_size"),
            per_episode_stride=True, mu_weighting=a.mu_weighting,
        )
        if not a.recompute:
            cached = store.get(name)
            if cached is not None and cached.fingerprint == task_fp:
                print(f"[baseline] {name}: 命中缓存 mse={cached.mse:.6f}（跳过）")
                n_skip += 1
                continue
        b = compute_fixed_baseline(
            validator, name, entry.train_ids, fingerprint=task_fp,
            tag=f"baseline_{name}", mu_weighting=a.mu_weighting,
        )
        store.put(b)
        store.save()          # 逐任务落盘 ⇒ 中断了也不用从头再来
        n_done += 1
        print(f"[baseline] {name}: mse={b.mse:.6f} "
              f"({b.n_train_episodes} 回合 / {b.n_chunks} chunk / {b.n_frames} 帧 × {b.dims} 维)"
              f"  μ[:3]={[round(float(x), 5) for x in b.mu[:3]]}")

    store.save()
    print(f"\n[baseline] 完成：新算 {n_done} 个 / 跳过 {n_skip} 个 ⇒ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
