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
from pathlib import Path
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


def _is_rank0() -> bool:
    """本进程是否是 rank0（无 dist / 未初始化 ⇒ 视为 rank0）。"""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()) == 0
    except Exception:  # noqa: BLE001
        pass
    return True


class SchedulerLoggerAdapter:
    """Scheduler 需要 `log_event()` / `log_metrics()`（Stage A 的 `EventLogger` 有），
    而真实仓库的 `Logger` **没有** ⇒ 这层适配器补齐。

    * 消息（`info_rank0` / `info` / `warning`）转发给仓库 logger
    * `log_event` → 追加 JSONL（B1 计划 §11 要求的事件流）
    * `log_metrics` → 写 TB（有 writer 时）并落 JSONL
    """

    def __init__(self, repo_logger: Any, *, writer: Any = None,
                 event_path: Optional[str] = None,
                 write_events: Optional[bool] = None):
        self._log = repo_logger
        self._writer = writer
        self._event_path = event_path
        # 🔴 多卡：**只有 rank0 写事件 JSONL**。
        #    调度器在所有 rank 上都会构建/运行（决策一致），若每个 rank 都 append，
        #    事件流会重复 N 份并互相交错。`write_events=None` ⇒ 按 dist rank 自动判定。
        if write_events is None:
            write_events = _is_rank0()
        self._write_events = bool(write_events)
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
        if not self._event_path or not self._write_events:
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
    config_path: Optional[str] = None  # `--train.auto_learning` 指向的 yaml（scout 指纹要用）
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
        # ✅ 例外（用户口径「统一到一把尺子」）：`pass_metric="gmean_mse"` **且**打开
        #    候选池 ratio 筛选时，筛选/判定都只看 GMean 与参考线，**不需要** NMSE
        #    ⇒ 允许缺 baseline（降级为 warning，不再拒绝启动）。
        if _gmean_ratio_pool_mode(cfg):
            if logger is not None:
                logger.warning(
                    "[auto_learning] ⚠️ 没有 baseline store，但 "
                    "pass_metric='gmean_mse' + pool_filter_by_gmean_ratio=true ⇒ "
                    "不拒绝启动：候选池筛选与 PASS 判定都走 GMean/参考线，NMSE 只是 "
                    "legacy 显示量（会一直是 n/a）。")
        elif not cfg.allow_missing_baseline:
            raise ValueError(
                "[auto_learning] enabled=true 但没有可用的 baseline store "
                f"（auto_learning_baseline={baseline_path!r}）⇒ 拒绝启动。\n"
                "  缺 baseline 时 NMSE 恒为 None，所有任务会被排除出候选池"
                "（不会报错，只是永远不选任务）。\n"
                "  先跑：python -m lingbotvla.auto_learning.tools.compute_task_baseline "
                "--manifest <manifest.json> --config <lingbotvla_cli.yaml>\n"
                "  确实要无 baseline 跑 smoke：显式设 "
                "auto_learning.allow_missing_baseline=true（仅供 smoke/单测）；\n"
                "  或者改用 GMean 尺子：pass_metric='gmean_mse' + "
                "pool_filter_by_gmean_ratio=true + pass_thresholds_file=...。")
        else:
            if logger is not None:
                logger.warning(
                    "[auto_learning] ⚠️ allow_missing_baseline=true ⇒ NMSE=None，"
                    "所有任务会被排除出候选池（仅供 smoke / 单测）")

    # ---- 按任务的通过阈值表（pass_metric="mse" 时必须有，fail-fast）----
    _attach_pass_thresholds(cfg, cat, store, logger)

    return AutoLearningParts(
        cfg=cfg, catalog=cat, resolver=resolver, lazy_sampler=LazyAutoLearnSampler(),
        manifest_path=manifest_path, baseline_path=baseline_path, baseline_store=store,
        config_path=config_path,
        notes={"n_tasks": len(cat)},
    )


def _looks_like_repo_root(path: Path) -> bool:
    """仓库根的判据：能同时看到 `lingbotvla/` 包与打包标记之一。"""
    return ((path / 'lingbotvla' / '__init__.py').is_file()
            and ((path / 'pyproject.toml').is_file() or (path / 'setup.py').is_file()))


def resolve_repo_root(start: Optional[Path] = None) -> Path:
    """从 `start`（默认本文件）向上找**仓库根**。

    🔴 为什么不能写死层数（2026-10-10 真机实测踩坑，代价 = 缓存指纹静默降级）：
    本文件位于 `<repo>/lingbotvla/auto_learning/real/build.py`，需要 **3** 层才到仓库根，
    而原实现写的是 `Path(__file__).resolve().parents[2]` ⇒ 只到 `<repo>/lingbotvla/`。
    于是 `source_manifest()` 拿这个假根去拼 `EVAL_SOURCES` 的 14 条相对路径，
    **全部被判成"文件不存在"并静默跳过**：指纹从 20 个 source 退化成 6 个
    （真机 worker 日志原话：「指纹纳入 6 个 source（0 个评测链代码文件）」）。
    两个后果都不报错：
      ① 指纹不再覆盖评测链代码 ⇒ 改了评测代码也能命中旧缓存（拿过期扫描判 PASS，危险）；
      ② 与启动器 `al_launch.py`（按仓库根算，20 个 source）**永远算不到同一指纹**
         ⇒ 启动器预扫的 50 条缓存永远不被训练命中，每次启动白扫一遍。

    现在改为**向上找标记**，再退回"能解析出 `EVAL_SOURCES` 第一条"的层，最后才回退原路径。
    """
    here = Path(start or __file__).resolve()
    for candidate in [here, *here.parents]:
        if _looks_like_repo_root(candidate):
            return candidate
    try:                                  # 兜底：不依赖打包标记
        from ..scout_cache import EVAL_SOURCES
        rel = EVAL_SOURCES[0][1]
    except Exception:                     # noqa: BLE001 —— 清单不可用则不做这项兜底
        rel = None
    if rel:
        for candidate in here.parents:
            if (candidate / rel).is_file():
                return candidate
    return here


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
    # ✅ 例外：GMean ratio 候选池模式下 NMSE 不参与判据 ⇒ 不再要求逐任务 baseline。
    _relax_baseline = _gmean_ratio_pool_mode(cfg)
    adapter.require_baseline = parts.baseline_store is not None and not _relax_baseline
    if parts.baseline_store is not None and not _relax_baseline:
        miss = [t for t in cat.task_names()
                if parts.baseline_store.get(t, cat.entry(t).sha256_train) is None]
        if miss:
            raise RuntimeError(
                f"这些任务缺 Fixed Baseline（Auto Learning 启动即 fail-fast）: {miss}\n"
                f"  先跑：python -m lingbotvla.auto_learning.tools.compute_task_baseline ...")
    elif parts.baseline_store is not None and _relax_baseline:
        miss = [t for t in cat.task_names()
                if parts.baseline_store.get(t, cat.entry(t).sha256_train) is None]
        if miss:
            log.warning(
                f"[auto_learning] ⚠️ 这些任务缺 Fixed Baseline: {miss} ⇒ NMSE 为 n/a；"
                "当前是 GMean ratio 候选池模式，筛选/判定都不需要 NMSE，继续启动。")

    # ---- Hardness（用**训练数据集**取样本；index 空间与 sample_id 一致）----
    # 🔴 逐样本 loss 的磁盘缓存（2026-10-10）：跨 run 复用，避免同一任务反复重扫
    #   （实测 al_v15/v19/v21/v22/v23 各扫了一遍 place_dual_shoes，白烧 25–40 分钟 GPU）。
    #   指纹 = 权重内容 + 评测链源码语义 hash + 噪声/损失口径 + 数据集身份；
    #   **刻意不含** probe_fraction/batch —— 不同 probe 扫的是不同子集，按样本做并集互补。
    hardness_cache = None
    _hc_root = os.environ.get('AL_HARDNESS_CACHE')
    if _hc_root:
        try:
            from ..hardness import DEFAULT_FLOW_TIME, DEFAULT_SEED
            from ..hardness_cache import HardnessSampleCache, compute_hardness_fingerprint
            _ckpt = (os.environ.get('AL_SCOUT_CACHE_CHECKPOINT')
                     or getattr(getattr(args, 'model', None), 'model_path', ''))
            _fp, _fp_info = compute_hardness_fingerprint(
                checkpoint_dir=Path(_ckpt),
                repo_root=resolve_repo_root(__file__),
                noise_seed=int(DEFAULT_SEED),
                flow_time=float(DEFAULT_FLOW_TIME),
                loss_type='L1_fm',
                extra={'dataset': str(os.environ.get('AL_SCOUT_CACHE_MANIFEST', 'unknown'))},
            )
            hardness_cache = HardnessSampleCache(
                _hc_root, _fp, enabled=True, write_enabled=_is_rank0(), logger=log)
            log.info_rank0(
                f'[hardness_cache] 启用：root={_hc_root} fp={_fp[:16]}…'
                f'（{_fp_info["sources_included"]} 个源码 + {len(_fp_info["weight_shards"])} 个权重分片；'
                f'noise_seed={_fp_info["options"]["noise_seed"]}）')
        except Exception as _exc:  # noqa: BLE001 —— 缓存不可用就退回"每轮重扫"，绝不拦启动
            log.warning(f'[hardness_cache] 初始化失败（{_exc!r}）⇒ 本轮不启用缓存，按原行为重扫')
            hardness_cache = None
    hardness = build_real_hardness(model, _dataset_for_hardness(model, args, processor),
                                   logger=log, cache=hardness_cache)

    # ---- Backend / Scheduler ----
    # Optional strict-provenance cache: off by default; never Rescan/Review.
    scout_cache = None
    if os.environ.get('AL_SCOUT_CACHE_MODE', 'off') == 'bootstrap':
        if cfg.pass_metric != 'gmean_mse':
            raise RuntimeError('Scout cache only supports GMean (gmean_mse)')
        # [ROCM-PORT 2026-10-10] 原实现把 step_offset==500 当硬门槛（AutoDL step-500 续训场景）。
        # 我们从原始 base（step_offset=0）起跑，同样需要复用扫描缓存 ⇒ 降级为警告；
        # 缓存的 fingerprint 已含权重内容（见 provenance），跨起点不会串味。
        if int(getattr(args.train, 'step_offset', -1)) != 500:
            log.warning('[scout_cache] step_offset=%s != 500：仍启用缓存（fingerprint 含模型身份）',
                        getattr(args.train, 'step_offset', None))
        from ..scout_cache import BootstrapScoutCache, provenance, source_manifest
        ckpt = Path(os.environ['AL_SCOUT_CACHE_CHECKPOINT']).resolve()
        loaded = getattr(args.model, 'model_path', None)
        if loaded is None or Path(loaded).resolve() != ckpt:
            raise RuntimeError('Scout cache checkpoint must exactly equal args.model.model_path')
        p0 = next(model.parameters(), None)
        actual_dtype = str(p0.dtype).removeprefix('torch.') if p0 is not None else 'unknown'
        configured_dtype = os.environ['AL_SCOUT_CACHE_DTYPE'].lower()
        if configured_dtype != actual_dtype:
            raise RuntimeError(f'Scout cache dtype mismatch: {configured_dtype} != {actual_dtype}')
        if bool(getattr(args.train, 'enable_resume', False)):
            raise RuntimeError('Scout cache forbidden when Resume is enabled')
        shards = sorted(ckpt.glob('*.safetensors'))
        # ---- 评测 source 清单（唯一事实来源见 `scout_cache.EVAL_SOURCES`）----
        # 🔴 2026-10-10：原实现只列了 6 个代码文件，漏了**数据集构造链**（multi_vla_dataset /
        #    base_dataset / dataset.py / utils.py …）与 **AL 配置文件本身** ⇒ 改这些文件会让
        #    评测数字变化而指纹不变（拿旧扫描结果判 PASS，危险）；反过来改一行注释又会
        #    让 50 个结果全废（真机实测今天两次，重扫 ~5 分钟）。
        #    ⇒ ① 清单补全（宁可多列）；② `.py` 改用 AST 语义 hash（注释/空行不再失效）。
        # 🔴 用 `resolve_repo_root()` 而不是写死 `parents[N]`：写死层数会静默丢掉 14 条
        #    评测链代码（详因见该函数 docstring，2026-10-10 真机实测）。
        repo_root = resolve_repo_root(__file__)
        if not (repo_root / 'lingbotvla' / 'utils' / 'open_loop_validation.py').is_file():
            log.warning(
                '[scout_cache] ⚠️ 推出来的仓库根看起来不对：%s（找不到评测链文件）'
                '⇒ 指纹会退化成"只有数据/配置文件"，请检查仓库布局', repo_root)
        _al_cfg_path = getattr(parts, 'config_path', None) or os.environ.get('AL_AUTO_LEARNING_CONFIG')
        src_extra = {
            'manifest': Path(os.environ['AL_SCOUT_CACHE_MANIFEST']),
            'norm': Path(os.environ['AL_SCOUT_CACHE_NORM']),
            'thresholds': Path(cfg.pass_thresholds_file),
            'baseline': Path(os.environ['AL_SCOUT_CACHE_BASELINE']),
            # [2026-10-10] **不**把 AL 配置文件整份计入指纹：8 卡并行预扫描用 al_shard0..7.yaml、
            # 正式 run 用主配置 ⇒ 若按文件 hash 会得到 9 个不同指纹 ⇒ 分片结果无法被正式 run 命中。
            # 配置中真正影响评测结果的字段（scout_trajs / noise_seed / stride / image_augment /
            # inference_dtype）已经在 provenance 的 options 里逐项列出，安全性不受影响。
            'checkpoint_config': ckpt / 'config.json',
            'checkpoint_tokenizer': ckpt / 'tokenizer.json',
        }
        src_extra = {k: v for k, v in src_extra.items() if v is not None}
        src, missing = source_manifest(
            repo_root, src_extra, strict=False,
            on_missing=lambda key, reason: log.warning(
                '[scout_cache] ⚠️ 评测 source 缺失，已从指纹中跳过: %s（%s）', key, reason))
        if missing:
            log.warning(
                '[scout_cache] ⚠️ 共 %d 个 source 未纳入指纹: %s ⇒ 若这些文件其实在评测路径上，'
                '请先修好再启用缓存（缺文件的指纹"看起来合法"但语义不完整）',
                len(missing), [k for k, _ in missing])
        log.info_rank0(
            f'[scout_cache] 指纹纳入 {len(src)} 个 source'
            f'（{len([k for k in src if k not in src_extra])} 个评测链代码文件 + '
            f'{len(src_extra)} 个数据/配置文件）+ {len(shards)} 个权重分片；'
            '`.py` 用 AST 语义 hash（注释/空行/docstring 不再失效）')
        fp = provenance(weight_files=shards, sources=src,
                        options={'inference_dtype': os.environ['AL_SCOUT_CACHE_DTYPE'],
                                 'noise_seed': 1234,
                                 'scout_trajs': cfg.global_scout_val_trajs,
                                 'stride': 'per_episode',
                                 'image_augment': bool(getattr(args.data, 'image_augment', False))})
        # [2026-10-10 用户要求] 两个显式开关，操作者说了算（默认仍走自动指纹 ✓）：
        #   AL_SCOUT_CACHE_FINGERPRINT=<fp>  ⇒ 直接用指定指纹目录（跳过自动指纹；= "指定缓存重载"）
        #   AL_SCOUT_CACHE_FORCE_RESCAN=1    ⇒ 忽略已有条目、强制重扫（写到一个带时间戳的新目录）
        _fp_override = os.environ.get('AL_SCOUT_CACHE_FINGERPRINT', '').strip()
        if _fp_override:
            log.warning('[scout_cache] 使用显式指纹 %s（跳过自动指纹计算；有效性由操作者负责）',
                        _fp_override)
            fp = _fp_override
        if os.environ.get('AL_SCOUT_CACHE_FORCE_RESCAN', '') == '1':
            import time as _time
            _forced = f'{fp}-force-{int(_time.time())}'
            log.warning('[scout_cache] FORCE_RESCAN=1 ⇒ 忽略已有缓存，强制重扫（写入 %s）', _forced)
            fp = _forced
        scout_cache = BootstrapScoutCache(os.environ['AL_SCOUT_CACHE_ROOT'], fingerprint=fp)
        log.info_rank0(f'[auto_learning] strict Bootstrap Scout cache enabled; fp={fp[:12]}…; '
                       'never used in Rescan')
    backend = build_real_backend(
        catalog=cat, resolver=parts.resolver, adapter=adapter, hardness=hardness,
        dataset=None, baseline_store=parts.baseline_store)
    if scout_cache is not None:
        backend.evaluator.scout_cache = scout_cache
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
    # 验收钩子（默认关闭）：零 optimizer.step 时 Hardness 扫描不会被执行，
    # 这里在模型加载完成后、训练开始前，用**同一个**真实 scorer 跑一次逐样本对拍。
    # 复用 RealHardnessScorer 的阶段与报告逻辑（B8 → B1 → B8），不复制任何算法。
    # 零 optimizer.step 时 Hardness 扫描不会执行；本钩子**默认关闭**且必须同时设
    # AL_HARDNESS_REPORT_OUT（正式训练不会设）⇒ 只用于独立验收运行。
    # 单卡限制：多卡 FSDP2 明确拒绝，避免 collective 死锁。
    _selftest_n = int(os.environ.get('AL_HARDNESS_SELFTEST_IDS', '0') or 0)
    if _selftest_n > 0 and os.environ.get('AL_HARDNESS_REPORT_OUT'):
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
                raise RuntimeError(
                    'hardness selftest 不支持多卡 FSDP2（避免 collective 死锁）；请用单卡验收')
            import time as _time
            _task = (os.environ.get('AL_HARDNESS_SELFTEST_TASK') or str(next(iter(cat))))
            _ids = list(range(1, _selftest_n + 1))
            _t0 = _time.perf_counter()
            hardness.score(_task, _ids)
            logger.info_rank0(
                f'[auto_learning][acceptance] hardness selftest: task={_task} '
                f'samples={len(_ids)} seconds={_time.perf_counter() - _t0:.2f} '
                f'(zero-step hook, phases from AL_HARDNESS_REPLAY_BATCH/REPEAT)')
        except Exception as _exc:  # noqa: BLE001
            try:
                logger.warning(f'[auto_learning][acceptance] hardness selftest failed: {_exc!r}')
            except Exception:
                pass
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


def _gmean_ratio_pool_mode(cfg: AutoLearningConfig) -> bool:
    """是否「GMean 一把尺子」模式：``pass_metric='gmean_mse'`` + 候选池 ratio 开关。

    只有这个模式下才允许 baseline 缺失 / 指纹不一致降级为 warning ——
    此时候选池筛选与 PASS 判定都只依赖 ``pass_thresholds_file`` 里的每任务参考线，
    ``task_baseline.json`` 的 NMSE 分母不再参与任何判据（仅剩 legacy 显示用途）。

    开关关闭（默认）⇒ 恒为 False ⇒ 所有守卫逐字维持原样（fail-fast）。
    """
    from ..decision.thresholds import pool_filter_enabled

    return bool(pool_filter_enabled(cfg))


def _verify_baseline_fingerprint(parts: AutoLearningParts, cfg: AutoLearningConfig,
                                 args: Any, model: Any, log: Any) -> None:
    """baseline 的 config 指纹必须与**本次真实运行配置**一致（review v0.2 #6）。

    原实现直接 `BaselineStore.load(path)`，把文件里自带的旧指纹当成「当前指纹」⇒
    「昨天 chunk=50/norm=A 算的 baseline，今天 chunk=25/norm=B」也能命中，
    NMSE 的分母是错的尺子且**无任何报错**。

    ✅ 例外：GMean ratio 候选池模式下 NMSE 不参与任何判据 ⇒ 指纹不一致只 warning
    （不 fail-fast），但仍然**逐字记录**两边的指纹，便于事后审计。
    """
    store = parts.baseline_store
    if store is None:
        return
    from ..baseline import runtime_config_fingerprint

    runtime_fp = runtime_config_fingerprint(args, getattr(model, "config", None))
    relaxed = _gmean_ratio_pool_mode(cfg)
    if not store.config_fingerprint:
        if relaxed:
            log.warning(
                "[auto_learning] ⚠️ baseline store 里没有 config_fingerprint；"
                "当前是 GMean ratio 候选池模式（NMSE 不参与判据）⇒ 只 warning 不拒绝启动。"
                f"  本次运行算出来 : {runtime_fp}")
            return
        raise RuntimeError(
            "[auto_learning] baseline store 里没有 config_fingerprint ⇒ 无法确认它与"
            "本次运行配置一致。请用 tools/compute_task_baseline.py --recompute 重算。")
    if runtime_fp != store.config_fingerprint:
        if relaxed:
            log.warning(
                "[auto_learning] ⚠️ baseline 的 config 指纹与本次运行配置不一致；"
                "当前是 GMean ratio 候选池模式（候选池与 PASS 判定都走 "
                "pass_thresholds_file 的每任务参考线，NMSE 只是 legacy 显示量）"
                "⇒ 只 warning 不拒绝启动。\n"
                f"  baseline 文件里 : {store.config_fingerprint}\n"
                f"  本次运行算出来 : {runtime_fp}\n"
                "  若要 NMSE 重新可信，请重算："
                "python -m lingbotvla.auto_learning.tools.compute_task_baseline "
                "--manifest <manifest.json> --config <lingbotvla_cli.yaml> --recompute")
            return
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
