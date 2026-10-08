"""Auto Learning 的**总装**（Stage B1）。

把真实组件串成一条链（见 `docs/stage_b1_integration_design.md` §4）：

    TaskCatalog(manifest) + SampleResolver(训练数据集)
        → attach_samples（回合 → dataset local_idx）
        → RealEvaluator / RealHardnessScorer
        → Backend → Scheduler
        → AutoLearnSampler（静态 slots / 动态全局 NEW:Replay 比例）
        → AutoLearnLoopHook（训练循环接线）

**两段式构造**（因为训练脚本里 dataloader 早于模型）::

    parts = build_auto_learning_parts(args, train_dataset)   # 不需要模型
    ... build_dataloader(..., sampler=parts.lazy_sampler) ...
    ... model = build_parallelize_model(...) ...
    finish_auto_learning(parts, model=model, processor=processor, writer=..., logger=...)

`auto_learning.enabled=false` ⇒ `build_auto_learning_parts()` 返回 **None**，
一个对象都不创建。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..catalog import catalog_from_task_split
from ..config import AutoLearningConfig
from ..orchestration.scheduler import Scheduler
from ..resolver import SampleResolver
from ..sampling.sampler import BatchSampler
from .backend import RealEvaluator, build_real_backend, build_real_hardness
from .hook import AutoLearnLoopHook
from .sampler import AutoLearnSampler, LazyAutoLearnSampler


class SchedulerLoggerAdapter:
    """Scheduler 需要 `log_event()` / `log_metrics()`（Stage A 的 `EventLogger` 有），
    而真实仓库的 `Logger` **没有** ⇒ 这层适配器补齐。

    * 消息（`info_rank0` / `info` / `warning`）转发给仓库 logger
    * `log_event` → 追加 JSONL（B1 计划 §11 要求的事件流）
    * `log_metrics` → 写 TB（有 writer 时）并落 JSONL
    """

    def __init__(self, repo_logger: Any, *, writer: Any = None,
                 event_path: Optional[str] = None):
        self._log = repo_logger
        self._writer = writer
        self._event_path = event_path
        # Scheduler.global_step 是本次 AL run 的相对步数；真实训练器（及其
        # training/* TensorBoard 标签）可以从 step_offset=500 等绝对步数开始。
        # 这个偏移只用于日志，不得改写 Scheduler 的预算/持久化计数。
        self._tb_step_offset = 0
        if event_path:
            os.makedirs(os.path.dirname(os.path.abspath(event_path)), exist_ok=True)

    def set_tb_step_offset(self, *, train_global_step: int,
                           al_global_step: int) -> None:
        """同步两套日志横轴；resume 后按实际恢复步数重算，不猜配置。"""
        self._tb_step_offset = int(train_global_step) - int(al_global_step)

    # -- 转发 ---------------------------------------------------------------
    def info_rank0(self, msg, *a, **k):
        return self._log.info_rank0(msg, *a, **k)

    def info(self, msg, *a, **k):
        return self._log.info(msg, *a, **k)

    def warning(self, msg, *a, **k):
        return self._log.warning(msg, *a, **k)

    # -- Auto Learning 专用 --------------------------------------------------
    def _append(self, obj: Dict[str, Any]) -> None:
        if not self._event_path:
            return
        try:
            with open(self._event_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        except Exception:  # noqa: BLE001 —— 写事件失败不该打断训练
            pass

    def log_event(self, event: Dict[str, Any]) -> None:
        obj = {"kind": "event", **(event or {})}
        if obj.get("step") is not None:
            obj["tb_step"] = int(obj["step"]) + self._tb_step_offset
        self._append(obj)

    def log_text(self, step: int, name: str, value: str) -> None:
        """记录人可读的任务名；JSONL 永远可读，TB Writer 支持时也落 Text。"""
        tb_step = int(step) + self._tb_step_offset
        self._append({"kind": "text", "step": int(step), "tb_step": tb_step,
                      "name": name, "value": str(value)})
        if self._writer is not None and hasattr(self._writer, "add_text"):
            try:
                self._writer.add_text(name, str(value), tb_step)
            except Exception:  # noqa: BLE001 -- 诊断日志不能打断训练
                pass

    def log_metrics(self, step: int, name: str, value: Any) -> None:
        # 真实训练器每个 optimizer step 都写 training/loss；Scheduler 产出的是
        # *unit 平均* loss。不能用同名 tag 在同一步写两种不同统计量。
        name = "auto_learning/unit_loss" if name == "training/loss" else name
        tb_step = int(step) + self._tb_step_offset
        self._append({"kind": "metric", "step": int(step), "tb_step": tb_step,
                      "name": name, "value": value})
        if self._writer is not None:
            try:
                if value is not None:
                    self._writer.add_scalar(name, float(value), tb_step)
            except Exception:  # noqa: BLE001
                pass


@dataclass
class AutoLearningParts:
    """第一段（不需要模型）的产物。"""

    cfg: AutoLearningConfig
    catalog: Any                       # 已 attach_samples
    resolver: Any
    lazy_sampler: LazyAutoLearnSampler
    manifest_path: str
    baseline_path: Optional[str] = None
    baseline_store: Any = None
    notes: Dict[str, Any] = field(default_factory=dict)


def load_auto_learning_config(path: str) -> AutoLearningConfig:
    """从 yaml 读 Auto Learning 配置。

    支持两种写法：顶层就是配置，或包一层 `auto_learning:`。
    """
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    body = raw.get("auto_learning", raw)
    body = {k: v for k, v in body.items() if k in AutoLearningConfig.__dataclass_fields__}
    cfg = AutoLearningConfig(**body)
    return cfg


def auto_learning_enabled(config_path: Optional[str]) -> bool:
    """Auto Learning 到底开没开（review v0.2 #1）。

    🔴 `args.train.auto_learning` 是**配置文件路径字符串**，不是 config 对象 ⇒
    `getattr(_al_cfg, "enabled", False)` 对 str 恒为 False，整个 AL 保护会静默失效。
    唯一正确的判法：把配置文件**真解析**一遍。
    """
    if not config_path:
        return False
    return bool(load_auto_learning_config(config_path).enabled)


def check_rmpad_for_auto_learning(enabled: bool, rmpad: bool,
                                  rmpad_with_pos_ids: bool) -> None:
    """v0 语义要求 rmpad=false；不满足 ⇒ **启动前 fail-fast**（review v0.2 #1）。

    旧实现是「静默把 rmpad 强改成 False」——实际运行配置与命令行/日志不一致，
    后面所有排查都对不上号。
    """
    if enabled and (rmpad or rmpad_with_pos_ids):
        raise ValueError(
            "[auto_learning] v0 要求 rmpad=false 且 rmpad_with_pos_ids=false"
            "（保固定 micro batch 与 7 NEW + 3 Replay 语义）；"
            f"当前 rmpad={rmpad}, rmpad_with_pos_ids={rmpad_with_pos_ids}。"
            "请显式关掉它们，不要依赖静默强制。")


def build_auto_learning_parts(
    *,
    config_path: Optional[str],
    manifest_path: Optional[str],
    train_dataset: Any,
    baseline_path: Optional[str] = None,
    logger: Any = None,
) -> Optional[AutoLearningParts]:
    """第一段：解析配置 + 建 catalog/resolver + 造延迟绑定 sampler。

    `config_path` 为 None 或配置里 `enabled=false` ⇒ 返回 **None**（零副作用）。
    """
    if not config_path:
        return None
    cfg = load_auto_learning_config(config_path)
    if not cfg.enabled:
        return None
    if not manifest_path:
        raise ValueError(
            "auto_learning.enabled=true 时必须提供 task split manifest "
            "（tools/task_split.py 的产物）")

    # ---- task split（严格自检：划分正确 / 无泄漏 / 分块结构）----
    cat = catalog_from_task_split(manifest_path, strict=True)
    problems = cat.verify()
    if problems:
        raise RuntimeError(
            "task split 自检未通过：\n  - " + "\n  - ".join(str(p) for p in problems))

    # ---- 可选：只跑 manifest 里的一个子集（smoke 用）----
    if cfg.task_names:
        want = list(cfg.task_names)
        missing = [t for t in want if t not in cat]
        if missing:
            raise ValueError(f"auto_learning.task_names 里有未知任务: {missing}")
        cat = type(cat)({t: cat.entry(t) for t in want}, meta=cat.meta,
                        episodes_per_task=cat.episodes_per_task, task_order=cat.task_order)

    # ---- 回合 → dataset local_idx ----
    resolver = SampleResolver.from_dataset(train_dataset, task_of_episode=cat.task_of_episode)
    cat = cat.attach_samples(resolver)
    if logger is not None:
        e0 = cat.entry(cat.task_names()[0])
        logger.info_rank0(
            f"[auto_learning] catalog: {len(cat)} 个任务；"
            f"{e0.name}: {e0.n_train_trajs} train 回合 → {len(e0.train_sample_ids)} 样本；"
            f"resolver 覆盖 {len(resolver)} 条")

    # ---- Fixed Baseline store ----
    store = None
    if baseline_path and os.path.isfile(baseline_path):
        from ..baseline import BaselineStore

        # config_fingerprint 由文件自己带回来（CLI 写进去的）⇒ 与 CLI 算的 task 指纹一致
        store = BaselineStore.load(baseline_path)
        if logger is not None:
            logger.info_rank0(
                f"[auto_learning] baseline store: {baseline_path}"
                f"（config指纹={store.config_fingerprint or '空'}，{len(store.tasks)} 个任务）")
    else:
        # 🔴 review v0.2 #6：正式 Auto Learning **必须有** baseline。
        #    否则 Scheduler 拿到 NMSE=None ⇒ 任务被静默排除出候选池 ——
        #    表现是「训练照跑、但什么任务都不选」，而不是任何报错。
        if not cfg.allow_missing_baseline:
            raise ValueError(
                "[auto_learning] enabled=true 但没有可用的 baseline store "
                f"（auto_learning_baseline={baseline_path!r}）⇒ 拒绝启动。\n"
                "  缺 baseline 时 NMSE 恒为 None，所有任务会被排除出候选池"
                "（不会报错，只是永远不选任务）。\n"
                "  先跑：python -m lingbotvla.auto_learning.tools.compute_task_baseline "
                "--manifest <manifest.json> --config <lingbotvla_cli.yaml>\n"
                "  确实要无 baseline 跑 smoke：显式设 "
                "auto_learning.allow_missing_baseline=true（仅供 smoke/单测）。")
        if logger is not None:
            logger.warning(
                "[auto_learning] ⚠️ allow_missing_baseline=true ⇒ NMSE=None，"
                "所有任务会被排除出候选池（仅供 smoke / 单测）")

    # ---- 按任务的通过阈值表（pass_metric="mse" 时必须有，fail-fast）----
    _attach_pass_thresholds(cfg, cat, store, logger)

    return AutoLearningParts(
        cfg=cfg, catalog=cat, resolver=resolver, lazy_sampler=LazyAutoLearnSampler(),
        manifest_path=manifest_path, baseline_path=baseline_path, baseline_store=store,
        notes={"n_tasks": len(cat)},
    )


def finish_auto_learning(
    parts: AutoLearningParts,
    *,
    model: Any,
    processor: Any,
    args: Any,
    writer: Any = None,
    logger: Any = None,
    event_logger: Any = None,
    use_depth_align: bool = False,
) -> AutoLearnLoopHook:
    """第二段：有了模型之后，把 Evaluator / Hardness / Scheduler / Sampler / Hook 建好。

    返回值是 hook；同时会把真 sampler `bind()` 进 `parts.lazy_sampler`
    （DataLoader 拿到的就是它）。
    """
    from lingbotvla.utils.open_loop_validation import OpenLoopValidator

    from ..eval_context import resolve_logger
    from ..evaluator import EvaluatorAdapter

    log = resolve_logger(logger)
    cfg = parts.cfg
    cat = parts.catalog

    # ---- 启动前 fail-fast 三项（review v0.2 §14 Gate）----
    validate_batch_alignment(cfg, args, log)
    _verify_baseline_fingerprint(parts, cfg, args, model, log)
    _guard_image_augment(cfg, args, log)

    # ---- Evaluator（复用 OpenLoopValidator + safe_eval_context）----
    first = cat.entry(cat.task_names()[0])
    validator = OpenLoopValidator(
        model=model, args=args, processor=processor, use_depth_align=use_depth_align,
        writer=writer, logger=log,
        train_monitor_ids=first.train_traj_ids, val_ids=first.val_traj_ids,
    )
    adapter = EvaluatorAdapter(validator, cat, parts.baseline_store, logger=log,
                               require_baseline=False)
    # 有 store ⇒ 缺 baseline 直接报错（B1 计划 §12 fail-fast）；没 store 时允许 nmse=None
    adapter.require_baseline = parts.baseline_store is not None
    if parts.baseline_store is not None:
        miss = [t for t in cat.task_names()
                if parts.baseline_store.get(t, cat.entry(t).sha256_train) is None]
        if miss:
            raise RuntimeError(
                f"这些任务缺 Fixed Baseline（Auto Learning 启动即 fail-fast）: {miss}\n"
                f"  先跑：python -m lingbotvla.auto_learning.tools.compute_task_baseline ...")

    # ---- Hardness（用**训练数据集**取样本；index 空间与 sample_id 一致）----
    hardness = build_real_hardness(model, _dataset_for_hardness(model, args, processor),
                                   logger=log)

    # ---- Backend / Scheduler ----
    backend = build_real_backend(
        catalog=cat, resolver=parts.resolver, adapter=adapter, hardness=hardness,
        dataset=None, baseline_store=parts.baseline_store)
    # Scheduler 需要 log_event/log_metrics ⇒ 用适配器（真实 Logger 没有这两个方法）
    al_logger = SchedulerLoggerAdapter(
        log, writer=writer,
        event_path=os.path.join(getattr(args.train, "output_dir", "."),
                                "auto_learning_events.jsonl"))
    scheduler = Scheduler(backend, cfg, seed=cfg.seed, logger=al_logger)

    # ---- Sampler（scheduler.rng 是唯一权威随机源）----
    batch_sampler = BatchSampler(cfg, parts.resolver, cat, scheduler.rng)
    sampler = AutoLearnSampler(batch_sampler, batch_size=cfg.batch_size)
    parts.lazy_sampler.bind(sampler)

    hook = AutoLearnLoopHook(scheduler=scheduler, sampler=sampler, cfg=cfg,
                             logger=log, event_logger=event_logger)
    log.info_rank0(
        f"[auto_learning] 已就绪：{len(cat)} 任务；batch={cfg.batch_size}"
        f"（{cfg.new_slots} NEW + {cfg.replay_slots} Replay）；"
        f"eval_interval={cfg.eval_interval_steps} 步；defer_train={scheduler.defer_train}")
    return hook


def validate_batch_alignment(cfg: AutoLearningConfig, args: Any, log: Any) -> None:
    """AL logical batch == per-rank DataLoader batch, not global GBS.

    错配的后果：sampler 按 `cfg.batch_size` 产出「7 NEW + 3 Replay」共 10 个 index，
    但 DataLoader 按 `dataloader_batch_size` 消费 ⇒ 前 8 个进 optimizer batch #1、
    「剩 2 个 + 下一组前 6 个」进 batch #2 …… 日志里的 7+3 依然「看起来正确」，
    但**真实训练 batch 已经被切碎**。这种错配不会自己报警，必须启动时查。
    """
    tr = getattr(args, "train", None)

    def _g(key):
        return getattr(tr, key, None) if tr is not None else None

    mbs, gbs, dls = _g("micro_batch_size"), _g("global_batch_size"), _g("dataloader_batch_size")
    if cfg.new_ratio is not None:
        from ..batch_ratio import ratio_plan
        from lingbotvla.distributed.parallel_state import get_parallel_state
        ps = get_parallel_state()  # fail-fast if DP topology not available
        dp_size, dp_rank = int(ps.dp_size), int(ps.dp_rank)
        if (gbs is None or dls is None or mbs is None
                or _g("gradient_accumulation_steps") is None):
            raise ValueError("[auto_learning] ratio mode requires resolved GBS/micro/GAS/local batch")
        if (int(gbs) != int(dls) * dp_size
                or int(dls) != int(mbs) * int(_g("gradient_accumulation_steps"))):
            raise ValueError("[auto_learning] ratio mode: GBS != local_batch * DP, or local_batch != micro * GAS")
        if _g("data_parallel_size") is not None and dp_size != int(_g("data_parallel_size")):
            raise ValueError("[auto_learning] ratio mode: DP topology mismatch")
        plan = ratio_plan(global_batch_size=int(gbs), dp_size=dp_size,
                          dp_rank=dp_rank, new_ratio=cfg.new_ratio)
        # cfg is per-process: scheduler issues per-rank TrainRequests.
        cfg.batch_size, cfg.new_slots, cfg.replay_slots = (
            plan.local_batch_size, plan.local_new, plan.local_replay)
        # Runner derives independent sampling per (AL optimizer step, DP rank).
        # All ranks keep scheduler RNG identical; rank seeds avoid identical streams.
        cfg._ratio_dp_rank = dp_rank
        cfg._ratio_dp_size = dp_size
        cfg._ratio_global_batch_size = int(gbs)
        log.info_rank0(
            f"[auto_learning] global ratio plan: GBS={gbs}, DP={dp_size}, "
            f"NEW={plan.global_new} Replay={plan.global_replay}; "
            f"rank0 local={plan.local_new}+{plan.local_replay} "
            "(no PASS replay pool => actual samples all NEW)")
    if dls is not None and int(dls) != int(cfg.batch_size):
        raise ValueError(
            f"[auto_learning] AL logical batch_size({cfg.batch_size}) != "
            f"train.dataloader_batch_size({dls}) ⇒ 拒绝启动。\n"
            f"  真实 DataLoader 按 dataloader_batch_size 消费 sampler 的 index ⇒ 会把"
            f"「{cfg.new_slots} NEW + {cfg.replay_slots} Replay」切碎"
            f"（日志仍显示 7+3，但真实训练 batch 已经错了）。\n"
            f"  请令 auto_learning.batch_size == --train.dataloader_batch_size。")

    dp = None
    try:
        from lingbotvla.distributed.parallel_state import get_parallel_state

        ps = get_parallel_state()
        dp = getattr(ps, "dp_size", None) or getattr(ps, "dp", None)
    except Exception:  # noqa: BLE001 —— 无卡单测环境没有 parallel state
        dp = None
    num_micro = (int(dls) // int(mbs)) if (mbs and dls and int(mbs) > 0) else None
    log.info_rank0(
        "[auto_learning] batch 对齐检查通过："
        f"AL logical batch_size={cfg.batch_size}"
        f"（{cfg.new_slots} NEW + {cfg.replay_slots} Replay）"
        f" | micro_batch_size={mbs} global_batch_size={gbs} "
        f"dataloader_batch_size={dls} dp_size={dp} num_micro_batch={num_micro}；"
        "1 个 DataLoader logical batch = 1 个 optimizer step")


def _verify_baseline_fingerprint(parts: AutoLearningParts, cfg: AutoLearningConfig,
                                 args: Any, model: Any, log: Any) -> None:
    """baseline 的 config 指纹必须与**本次真实运行配置**一致（review v0.2 #6）。

    原实现直接 `BaselineStore.load(path)`，把文件里自带的旧指纹当成「当前指纹」⇒
    「昨天 chunk=50/norm=A 算的 baseline，今天 chunk=25/norm=B」也能命中，
    NMSE 的分母是错的尺子且**无任何报错**。
    """
    store = parts.baseline_store
    if store is None:
        return
    from ..baseline import runtime_config_fingerprint

    runtime_fp = runtime_config_fingerprint(args, getattr(model, "config", None))
    if not store.config_fingerprint:
        raise RuntimeError(
            "[auto_learning] baseline store 里没有 config_fingerprint ⇒ 无法确认它与"
            "本次运行配置一致。请用 tools/compute_task_baseline.py --recompute 重算。")
    if runtime_fp != store.config_fingerprint:
        raise RuntimeError(
            "[auto_learning] baseline 的 config 指纹与**本次运行配置**不一致 ⇒ 拒绝启动。\n"
            f"  baseline 文件里 : {store.config_fingerprint}\n"
            f"  本次运行算出来 : {runtime_fp}\n"
            "  说明 数据路径 / 归一化统计 / 相机 / joints / chunk_size / img_size 变了 ⇒ "
            "NMSE 的分母不再可比。\n"
            "  请重算：python -m lingbotvla.auto_learning.tools.compute_task_baseline "
            "--manifest <manifest.json> --config <lingbotvla_cli.yaml> --recompute")
    log.info_rank0(f"[auto_learning] baseline config 指纹对拍通过: {runtime_fp}")


def _attach_pass_thresholds(
    cfg: AutoLearningConfig,
    cat: Any,
    store: Any,
    logger: Any = None,
) -> None:
    """`pass_metric="mse"` 时加载按任务的通过阈值表，并做三道 fail-fast。

    1. 文件存在 + ``metric`` 口径匹配（防止把 nmse 单位的表当 mse 用）
    2. ``config_fingerprint`` 与 baseline store 对拍（baseline 重算而阈值表没重算
       ⇒ 两个数错位且**不报错**，必须拒绝启动）
    3. 阈值表必须**覆盖 catalog 里全部任务**（缺的任务会被静默判成"永不 PASS"，
       白白烧算力）
    """
    from ..decision.thresholds import (
        PASS_METRIC_MSE, PASS_METRIC_GMEAN, PassThresholds, active_metric,
        verify_threshold_stat_compatible,
    )

    metric = active_metric(cfg)
    if metric not in (PASS_METRIC_MSE, PASS_METRIC_GMEAN):
        return
    if not getattr(cfg, "pass_thresholds_file", None):
        raise ValueError(
            "[auto_learning] pass_metric='mse' 但没有 pass_thresholds_file ⇒ 拒绝启动。\n"
            "  先跑：python -m lingbotvla.auto_learning.tools.compute_pass_thresholds ...")
    expect_fp = getattr(store, "config_fingerprint", None) if store is not None else None
    thresholds = PassThresholds.load(
        cfg.pass_thresholds_file,
        expect_fingerprint=expect_fp,
        require_metric=PASS_METRIC_MSE,
        allow_fingerprint_mismatch=cfg.allow_thresholds_fingerprint_mismatch,
    )
    verify_threshold_stat_compatible(thresholds, pass_metric=metric)
    missing = [t for t in cat.task_names() if t not in thresholds]
    if missing:
        raise ValueError(
            "[auto_learning] 阈值表没有覆盖全部任务 ⇒ 拒绝启动。\n"
            f"  缺阈值的任务: {missing}\n"
            "  （这些任务会被静默判成 needs_calibration、永远不 PASS，白白烧算力）\n"
            "  请用 tools/compute_pass_thresholds.py 对全部任务重算（含闭环失败的写 null）。")
    # 挂在 cfg 上（阈值模块用 getattr 取）；挂之前清掉可能残留的旧表
    cfg.pass_thresholds = thresholds
    if logger is not None:
        logger.info_rank0(
            f"[auto_learning] pass thresholds: {cfg.pass_thresholds_file}"
            f"（metric=mse，{thresholds.n_usable}/{len(thresholds.tasks)} 个任务有可用线，"
            f"config指纹={thresholds.config_fingerprint or '空'}）")


def _guard_image_augment(cfg: AutoLearningConfig, args: Any, log: Any) -> None:
    """v1 要求训练数据集 `image_augment=false`（review v0.2 #8）。"""
    data = getattr(args, "data", None)
    aug = bool(getattr(data, "image_augment", False)) if data is not None else False
    if not aug:
        return
    if not cfg.allow_image_augment:
        raise ValueError(
            "[auto_learning] v1 要求训练数据集 `image_augment=false`。\n"
            "  开了增强时，同一个 sample_id 两次 hardness 扫描可能拿到不同图 ⇒ "
            "难度不可复现、硬度权重抖动。\n"
            "  请把 --data.image_augment 设为 false；确实要开请显式设 "
            "auto_learning.allow_image_augment=true。")
    log.warning(
        "[auto_learning] ⚠️ image_augment=true：hardness 扫描会临时关增强并还原 RNG，"
        "但「同一 sample_id 的难度」仍可能漂移（allow_image_augment=true）")


def _dataset_for_hardness(model: Any, args: Any, processor: Any) -> Any:
    """Hardness 要用的数据集 = **与训练同一份**（同一索引空间）。

    直接复用训练数据集对象即可 —— 由调用方通过 `args` 传入（见 train_lingbotvla 的接线）。
    """
    ds = getattr(args, "_auto_learning_train_dataset", None)
    if ds is None:
        raise RuntimeError(
            "需要把**训练数据集对象**挂到 args._auto_learning_train_dataset 上"
            "（Hardness 的 sample_id 是它的 local_idx）")
    return ds


__all__ = ["AutoLearningParts", "SchedulerLoggerAdapter",
           "build_auto_learning_parts", "finish_auto_learning",
           "load_auto_learning_config", "auto_learning_enabled",
           "check_rmpad_for_auto_learning", "validate_batch_alignment"]
