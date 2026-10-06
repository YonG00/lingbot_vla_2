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

**不需要模型权重、不需要 GPU** —— 只用到 LingbotVLAV2Config + processor + 数据集。

配置构造（review v0.1 #3）
-------------------------
照抄 deploy `load_vla()` 的路径：`LingbotVLAV2Config(**{**yaml['model'], **yaml['train']})`，
processor 走 `build_processor(config.tokenizer_path)`（`QWEN3VL_PATH` 优先）。
**不是** `AutoConfig.from_pretrained(基座目录)` —— 那给的是 Qwen config，
缺 `max_state_dim` / `max_action_dim` / `chunk_size`，`FeatureTransform` 会直接崩。

用法
----
    cd /data/code/lingbot-vla-v2
    export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct
    python -u -m lingbotvla.auto_learning.tools.compute_task_baseline \
        --manifest /data/train/task_splits/manifest.json \
        --config   /data/outputs/single/click_bell/lingbotvla_cli.yaml \
        --out      /data/train/task_splits/task_baseline.json

    # 再跑一次 ⇒ 应当**全部命中缓存**（秒级返回）
    # 只算部分任务 / 只看会做什么 / 强制重算
    ... --tasks click_bell     ... --dry-run     ... --recompute
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


def _rss_mb() -> int:
    """当前进程 RSS（MB）。容器 cgroup 常只有 2 GiB，必须盯着。"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) // 1024
    except Exception:  # noqa: BLE001
        pass
    return -1


def _build_args(data_cfg: dict, lingbot_cfg, out_dir: str):
    """按 `tools/open_loop_eval_inprocess.py` 的同一套路构造 validator 需要的 args。"""
    ns = types.SimpleNamespace(**data_cfg)
    if not hasattr(ns, "chunk_size"):
        ns.chunk_size = int(getattr(lingbot_cfg, "chunk_size", 50))
    if not hasattr(ns, "num_episode"):
        ns.num_episode = None

    class _NS:
        pass

    args = _NS()
    args.data = ns
    # 训练侧是 `build_vla_dataset(model_config=args.model, config=model.config)`；
    # 离线 GT-only 路径下两者都用 **LingbotVLAV2Config**（deploy 也是同一个 config 对象）
    args.model = lingbot_cfg
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
    ap.add_argument("--config", required=True,
                    help="训练用的 lingbotvla_cli.yaml（取它的 model + train + data 段）")
    ap.add_argument("--out", default=None,
                    help="输出 json（默认 <manifest 目录>/task_baseline.json）")
    ap.add_argument("--out-dir", default=None,
                    help="临时回合白名单的落地目录（默认 <manifest 目录>/_baseline_ids）")
    ap.add_argument("--tasks", default=None, help="逗号分隔；默认全部")
    ap.add_argument("--mu-weighting", choices=["global", "trajectory"], default="global",
                    help="μ_task 的加权方式（默认 global = 所有帧 pooled 平均）")
    ap.add_argument("--recompute", action="store_true", help="忽略已有缓存，全部重算")
    ap.add_argument("--allow-missing-curriculum", action="store_true",
                    help="加载不到 robotwin_curriculum 时允许跳过 block 自检")
    ap.add_argument("--dry-run", action="store_true", help="只做输入校验与自检，不算 baseline")
    # ---- 省内存 / 可定制（容器 cgroup 只有 2 GiB）----
    ap.add_argument("--max-train-episodes", type=int, default=0, metavar="N",
                    help="每个 task 只用前 N 条 train 回合（0=全部）。**仅供 smoke / 省内存**，"
                         "算出来的分母不是正式的。")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式：等价于 --max-train-episodes 2 + **不落盘**")
    ap.add_argument("--no-clear-cache", action="store_true",
                    help="不在每个 task 后释放数据集缓存（默认会释放，省内存）")
    a = ap.parse_args()
    if a.smoke and a.max_train_episodes == 0:
        a.max_train_episodes = 2

    import yaml

    from lingbotvla.auto_learning.baseline import (
        BaselineStore, baseline_fingerprint, compute_fixed_baseline,
    )
    from lingbotvla.auto_learning.catalog import catalog_from_task_split
    from lingbotvla.auto_learning.model_config import effective_chunk_size

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
    cat = catalog_from_task_split(
        str(manifest), strict=True,
        allow_missing_curriculum=a.allow_missing_curriculum,
    )
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

    # ---- 构造真实 config（照抄 deploy 的路径；**不加载权重**）----
    # 必须先有 config 才能算准 effective chunk_size（`data.chunk_size` 通常不存在）。
    from lingbotvla.auto_learning.model_config import (
        effective_chunk_size, load_config_and_processor,
    )

    lingbot_cfg, processor, _raw = load_config_and_processor(a.config)
    eff_chunk = effective_chunk_size(raw, lingbot_cfg)
    print(f"[baseline] LingbotVLAV2Config: chunk_size={lingbot_cfg.chunk_size} "
          f"max_state_dim={getattr(lingbot_cfg, 'max_state_dim', None)} "
          f"max_action_dim={getattr(lingbot_cfg, 'max_action_dim', None)}")
    print(f"[baseline] tokenizer_path = {lingbot_cfg.tokenizer_path}")
    print(f"[baseline] effective chunk_size = {eff_chunk}")

    # ---- 配置级指纹（**不含** per-task sha256_train）----
    config_fp = baseline_fingerprint(
        dataset_root=data_cfg.get("train_path"),
        sha256_train=None,
        norm_stats_file=data_cfg.get("norm_stats_file"),
        cameras=data_cfg.get("cameras"),
        joints=data_cfg.get("joints"),
        chunk_size=eff_chunk,
        img_size=data_cfg.get("img_size"),
        per_episode_stride=True,
        mu_weighting=a.mu_weighting,
    )
    print(f"[baseline] manifest = {manifest}")
    print(f"[baseline] tasks    = {len(tasks)} 个；mu_weighting={a.mu_weighting}")
    print(f"[baseline] config指纹 = {config_fp}（逐任务再叠加 sha256_train）")
    print(f"[baseline] 输出     = {out}")

    store = BaselineStore.load(str(out), config_fingerprint=config_fp)

    if a.dry_run:
        hit = [t for t in tasks if store.get(t, cat.entry(t).sha256_train) is not None]
        print(f"[baseline] --dry-run：输入与自检通过；已有缓存命中 {len(hit)}/{len(tasks)} 个。")
        return 0

    # ---- 建 validator（**model=None** ⇒ 只收 GT，不推理）----
    from lingbotvla.utils.open_loop_validation import OpenLoopValidator

    args = _build_args(data_cfg, lingbot_cfg, out_dir)
    validator = OpenLoopValidator(
        model=None, model_config=lingbot_cfg, args=args, processor=processor,
        use_depth_align=use_depth_align, writer=None, logger=_Logger(),
    )

    n_done = n_skip = 0
    for name in tasks:
        entry = cat.entry(name)
        if not a.recompute and not a.smoke:
            cached = store.get(name, entry.sha256_train)
            if cached is not None:
                print(f"[baseline] {name}: 命中缓存 mse={cached.mse:.6f}（跳过）")
                n_skip += 1
                continue
        train_ids = list(entry.train_traj_ids)          # ⚠️ B1 起字段名是 *_traj_ids
        if a.max_train_episodes:
            train_ids = train_ids[: a.max_train_episodes]
            print(f"[baseline] ⚠️ smoke 模式：只用前 {len(train_ids)} 条 train 回合"
                  f"（正式值是 {entry.n_train} 条）")
        task_fp = store.expected_fingerprint(entry.sha256_train)
        b = compute_fixed_baseline(
            validator, name, train_ids, fingerprint=task_fp,
            tag=f"baseline_{name}", mu_weighting=a.mu_weighting, strict=True,
        )
        if not a.smoke:
            store.put(b)
            store.save()      # 逐任务落盘 ⇒ 中断了也不用从头再来
        n_done += 1
        print(f"[baseline] {name}: mse={b.mse:.6f} "
              f"({b.n_train_episodes} 回合 / {b.n_chunks} chunk / {b.n_frames} 帧 × {b.dims} 维)"
              f"  μ[:3]={[round(float(x), 5) for x in b.mu[:3]]}"
              f"  [rss={_rss_mb()}MB]")
        if not a.no_clear_cache:
            # 🔴 必须释放：`_ds_cache` 只增不减，多任务连跑会 OOM
            freed = validator.clear_dataset_cache()
            print(f"[baseline]   已释放数据集缓存 {freed} 份 ⇒ rss={_rss_mb()}MB")

    if a.smoke:
        print(f"\n[baseline] **smoke 模式：未落盘**（新算 {n_done} 个）。"
              f"要产出正式分母请去掉 --smoke。")
        return 0
    store.save()
    print(f"\n[baseline] 完成：新算 {n_done} 个 / 跳过 {n_skip} 个 ⇒ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
