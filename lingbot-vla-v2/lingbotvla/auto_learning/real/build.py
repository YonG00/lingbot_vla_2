"""Auto Learning 的**总装**（Stage B1）。

把真实组件串成一条链（见 `docs/stage_b1_integration_design.md` §4）：

    TaskCatalog(manifest) + SampleResolver(训练数据集)
        → attach_samples（回合 → dataset local_idx）
        → RealEvaluator / RealHardnessScorer
        → Backend → Scheduler
        → AutoLearnSampler（7 NEW + 3 Replay）
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
        if event_path:
            os.makedirs(os.path.dirname(os.path.abspath(event_path)), exist_ok=True)

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
        self._append({"kind": "event", **(event or {})})

    def log_metrics(self, step: int, name: str, value: Any) -> None:
        self._append({"kind": "metric", "step": int(step), "name": name, "value": value})
        if self._writer is not None:
            try:
                if value is not None:
                    self._writer.add_scalar(name, float(value), int(step))
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
    elif logger is not None:
        logger.warning("[auto_learning] 没有提供 baseline store ⇒ NMSE 会是 None（不推荐）")

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
    hardness = build_real_hardness(model, _dataset_for_hardness(model, args, processor))

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
           "load_auto_learning_config"]
