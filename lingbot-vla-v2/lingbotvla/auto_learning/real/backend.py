"""用**真实组件**组装 `ports.Backend`（Stage B1）。

关键约定（见 `docs/auto_learning/stage_b1_integration_design.md` §3）
----------------------------------------------------
* 训练数据集**只建一次**，白名单 = 所有参与任务的 **train 回合并集**
  ⇒ 索引空间全程不变；`sample_id` == dataset local_idx
* `TaskCatalog.attach_samples(resolver)` 把回合展开成 local_idx
* `Evaluator` 走 `EvaluatorAdapter`（内含 `safe_eval_context`）
* `HardnessScorer` 按 `(task, sample_ids)` 打分 —— **用训练数据集**取 item，
  否则 local_idx 对不上（子集数据集的索引是局部的）
* `Trainer` 在**延迟训练模式**下**永不被调用**（训练由外层真实循环跑），
  但 `ports.Backend.missing()` 仍要求它有 `train_steps` ⇒ 给一个显式报错的桩

⚠️ 本模块只 import `ports` / `types` / `testing`（假件），**torch 相关一律懒加载**。
"""

from __future__ import annotations

import time
import os
import dataclasses
from typing import Any, Dict, List, Optional, Sequence

from ..ports import Backend, TrajectoryMetrics
from .determinism import deterministic_sampling
#: 分片切分规则的**唯一事实来源**（2026-10-12：原先本文件自己实现一份，
#: 而多卡评测一致性预检看不到它 ⇒ 预检必然误报，真机 al_v47 实测）。
from lingbotvla.utils.open_loop_validation import eval_shard_indices, set_eval_shard_context


# --------------------------------------------------------------------------- #
# 评测结果合并（confirm 去重复用；2026-10-11）
# --------------------------------------------------------------------------- #
def _traj_frames_for(entry: Any, episode_ids: Sequence[int]) -> Dict[int, int]:
    """取"每条回合的帧数"（用于把 mse 合并成帧加权量）。

    帧数来自 `TaskEntry.samples_by_traj`（= 该回合的 sample 数）。
    取不到就返回空 dict —— 调用方退回等权平均，不报错（mse 不参与 PASS 判定）。
    """
    frames_map: Dict[int, int] = {}
    try:
        sbt = getattr(entry, 'samples_by_traj', None) or {}
        for ep in episode_ids:
            frames_map[int(ep)] = int(len(sbt.get(int(ep), ()) or ()))
    except Exception:  # noqa: BLE001 —— 帧数只是显示口径，失败不该打断评测
        return {}
    return frames_map


def _dist_world_rank() -> tuple:
    """取运行期 (world_size, rank)；无 torch.distributed / 未初始化 / world==1 ⇒ (1, 0)。"""
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            ws = int(_dist.get_world_size())
            if ws > 1:
                return ws, int(_dist.get_rank())
    except Exception:  # noqa: BLE001 —— 单机单卡/未初始化都走这条路
        pass
    return 1, 0


def _eval_shard_indices(*, n: int, world: int, rank: int) -> List[int]:
    """每卡固定 `k = ceil(n/world)` 条的分片索引（不足处按 `% n` 回绕）。

    🔴 2026-10-12：**实现已迁到 `lingbotvla/utils/open_loop_validation.py::eval_shard_indices`**
    （唯一事实来源）。原先分片规则只在本文件里，而**多卡评测一致性预检**
    （`open_loop_validation.py`）看不到它 ⇒ 预检拿"分片前"的 ids 比对，判成
    "各 rank 不一致" 而 fail-closed 拒绝启动（真机 al_v47 实测：n_starts=6/7 混杂）。
    此处保留同名薄包装只做转发，避免两处漂移。

    为什么必须"固定 k 条"而不是"交错均分"：集合通信次数必须逐 rank 相等，
    否则先到的 rank 永久等在 all-gather 里（真机表现 = NCCL 看门狗 600 s 后 abort）。
    回绕重复的那几条在汇总时按 episode id 去重，不影响数值。
    """
    return eval_shard_indices(n=n, world=world, rank=rank)


def _all_gather_object(payload: Any) -> List[Any]:
    """`all_gather_object` 的薄封装；不可用/失败时返回 `[payload]`（单卡语义）。"""
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
            out: List[Any] = [None] * int(_dist.get_world_size())
            _dist.all_gather_object(out, payload)
            return [x for x in out if x is not None]
    except Exception:  # noqa: BLE001 —— 汇总失败不该让评测崩（退回本 rank 结果）
        pass
    return [payload]


def _weighted_mse(parts: Sequence[TrajectoryMetrics], frames_map: Dict[int, int]) -> float:
    """把多组评测的 mse 合并：按各回合帧数加权；无帧数则等权；全空则 0.0。"""
    num = den = 0.0
    for part in parts:
        for ep, val in (part.per_traj_mse or {}).items():
            w = frames_map.get(int(ep), 0)
            if w <= 0:
                continue
            num += float(val) * w
            den += w
    if den > 0:
        return num / den
    vals = [float(v) for part in parts for v in (part.per_traj_mse or {}).values()]
    return (sum(vals) / len(vals)) if vals else 0.0


def merge_trajectory_metrics(base: TrajectoryMetrics, extra: TrajectoryMetrics, *,
                             split: Any, task: Optional[str] = None,
                             entry: Any = None,
                             baseline_mse: Optional[float] = None,
                             metric_valid: Optional[bool] = None,
                             note: Optional[str] = None) -> TrajectoryMetrics:
    """把两组开环评测合成一组（逐轨迹并集）—— confirm 复用 scout 值的核心。

    为什么可以合并（数值等价性）
    ---------------------------
    ``gmean_mse`` 是 ``per_traj_mse`` 的 property（逐轨迹 MSE 的等权几何平均），
    而几何平均**对分组可结合**：``geomean(A ∪ B) == geomean([geomean(A), geomean(B)])``
    ⇒ "合并两组"与"一次性评并集"结果一致。
    前提是**逐轨迹结果与批大小/卡数无关**：实测 v43（dp8）与 v44（dp7）的
    ``scout_nmse`` 逐位相同、相对差 ``0.00e+00``（2026-10-11 取证）。

    冲突处理
    --------
    同一 episode 出现在两组时**以 extra（新算的）为准** ⇒ 天然防止同一条被计两次，
    也让"重复评测"不可能污染判决值。

    口径说明（消歧义）
    -----------------
    * ``gmean_mse``：合并后自动变成"并集条数"的几何平均 ⇒ **判定口径**；
    * ``mse``：帧加权量，合并时按各回合帧数加权重算（取不到帧数则等权）；
    * ``nmse``：``mse / baseline_mse``，ratio 模式下**只是 legacy 显示量**，
      缺 baseline 时为 ``None``（判定不看它）。
    """
    per: Dict[int, float] = {int(k): float(v) for k, v in (base.per_traj_mse or {}).items()}
    per.update({int(k): float(v) for k, v in (extra.per_traj_mse or {}).items()})
    ids: List[int] = [int(i) for i in (base.episode_ids or [])]
    _seen = set(ids)
    for i in (extra.episode_ids or []):
        if int(i) not in _seen:
            ids.append(int(i))
            _seen.add(int(i))

    frames_map = _traj_frames_for(entry, ids) if entry is not None else {}
    mse = _weighted_mse([base, extra], frames_map)
    base_line = baseline_mse if baseline_mse is not None else extra.baseline_mse
    nmse = (mse / float(base_line)) if base_line else None
    # 有效性沿用"新算那条"的判据（scout 若无效根本不会走到 confirm）；
    # 分片汇总时可显式传入（各 rank 的有效性取交集/或由调用方决定）
    valid = bool(extra.metric_valid) if metric_valid is None else bool(metric_valid)

    return TrajectoryMetrics(
        task=task or base.task,
        split=str(getattr(split, 'value', split)),
        episode_ids=ids,
        per_traj_mse=per,
        mse=float(mse),
        baseline_mse=float(base_line or 0.0),
        nmse=nmse,
        r2=None,
        metric_valid=valid,
        n_trajs=len(per),
        wall_time_s=float(getattr(base, 'wall_time_s', 0.0))
        + float(getattr(extra, 'wall_time_s', 0.0)),
        note=(note if note is not None else
              f"merged: reused {len(base.per_traj_mse or {})} + fresh "
              f"{len(extra.per_traj_mse or {})} = {len(per)} trajs"),
    )


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
class RealEvaluator:
    """把 `EvaluatorAdapter` 包成 `ports.Evaluator`（Scheduler 只认 `evaluate`）。"""

    def __init__(self, adapter: Any, catalog: Any, baseline_store: Any = None, *,
                 scout_cache=None, logger: Any = None):
        self.adapter = adapter
        self.catalog = catalog
        self.baseline_store = baseline_store
        self.scout_cache = scout_cache
        #: 仅用于"合并式 confirm"的可见性日志（`info_rank0`）；未注入则静默。
        self.logger = logger

    def _emit(self, msg: str) -> None:
        """打一条 rank0 可见日志；logger 缺失/异常都不该打断评测。"""
        for cand in (self.logger, getattr(self.adapter, 'logger', None)):
            fn = getattr(cand, 'info_rank0', None)
            if fn is None:
                continue
            try:
                fn(msg)
                return
            except Exception:  # noqa: BLE001
                continue

    def evaluate_bootstrap_scout(self, task: str, split: str, episode_ids: Sequence[int]):
        """Only Bootstrap may use a previous Step500 Scout. Rescan bypasses."""
        cache = self.scout_cache
        if cache is not None and split == "scout":
            raw = cache.load(task, episode_ids)
            if raw is not None:
                from ..types import TrajectoryMetrics
                raw["per_traj_mse"] = {int(k): float(v) for k, v in raw["per_traj_mse"].items()}
                hit = TrajectoryMetrics(**raw)
                if hit.metric_valid and hit.gmean_mse is not None:
                    return hit
        result = self.evaluate(task, split, episode_ids)
        if cache is not None and split == "scout" and result.metric_valid and result.gmean_mse is not None:
            cache.store(task, episode_ids, dataclasses.asdict(result))
        return result

    def evaluate(self, task: str, split: str,
                 episode_ids: Sequence[int]) -> TrajectoryMetrics:
        """`split` 用 Stage A 的 `EvalSplit` 值（scout/confirm/active_val/train_monitor/review）。

        🔴 **不做 scout→confirm 的自动扩张**：Stage A 的 Scheduler 已经按
        `EvalSplit` 传了**它认为该评的回合集合**，这里只负责「评这个集合」。

        ---- 轨迹分片（2026-10-11，**默认关闭**）----
        背景：现状是"每个 rank 都把整批轨迹各跑一遍"（实测 v43 8 卡 / v44 7 卡：
        每卡都跑 12/20 条），因为 FSDP 的 all-gather 是**集合通信**、必须全员参与。
        但**轨迹之间互不依赖**、逐轨迹 MSE 与卡数无关（dp8 与 dp7 逐位相同）、
        聚合是**几何平均（可分组）** ⇒ 可以把轨迹切给各卡、`all_gather_object` 汇总。

        ⚠️ **安全前提（必须遵守，否则静默死锁）**：每卡**前向次数必须相同**。
        本实现用 `k = ceil(n / W)`、每卡恒定 k 条（不足处按 `idx % n` 回绕重复，
        汇总时**按 episode id 去重**）⇒ 次数恒等。这比"交错均分"（次数 2/1/1/…）
        安全得多；代价是多算 `k*W - n` 条冗余前向（n=8,W=8 ⇒ 0 条；n=8,W=7 ⇒ 6 条）。

        开启方式：环境变量 `AL_EVAL_SHARD=1`（默认 `0` ⇒ 与改造前**逐字一致**）。
        为什么默认关：分片是否安全**只能真机验证**（CPU 单测测不出集合通信错配），
        而当前有长跑训练占用显卡 ⇒ 先按"零行为变更"落地，验证后再考虑默认开。
        """
        ids = [int(i) for i in episode_ids]
        if not self._shard_enabled() or len(ids) <= 1:
            # 非分片：清掉可能残留的分片上下文（同一进程可能先后走两条路）
            set_eval_shard_context(None)
            res = self.adapter.evaluate_task(task, _split_name(split), ids)
            return res.to_trajectory_metrics()

        world, rank = _dist_world_rank()
        my_idx = _eval_shard_indices(n=len(ids), world=world, rank=rank)
        my_ids = [ids[i] for i in my_idx]
        # 🔴 2026-10-12：把**分片上下文**公告给下面的验证层（同进程、同一线程栈）。
        #   为什么必须公告：`my_ids` 只是"本卡那份"，验证层看到的数据集也只有这几条轨迹
        #   ⇒ 它既算不出"每条轨迹该有几个窗口"（需要全局 k），也判不出"并集是否覆盖全集"
        #   （需要全局 n）。不公告就会出现真机实测的两个错判：
        #     · `_shard_k_for(本卡 ep_map)` 把 k 算成 1（全局 k=2）⇒ 抽帧步长偏大、欠采样；
        #     · `n_ids=len(ids)` 把全集当成 1 条 ⇒ 假报"分片并集 4 条 ≠ 全集 1 条（越界）"。
        _k = max(1, -(-len(ids) // max(1, int(world))))        # k = ceil(n/W)
        _my_n = len(my_ids)
        set_eval_shard_context({
            'k': _k,                                           # 每条轨迹的窗口数
            'n': len(ids),                                     # 全集轨迹条数
            'world': int(world), 'rank': int(rank),
            # 本卡应跑的次数 = k × 本卡轨迹条数；验证层据此核对"实际 == 期望"
            # （只比"各卡相等"防不住"大家一起少算"）。
            'expected_steps': _k * _my_n,
        })
        res = self.adapter.evaluate_task(task, _split_name(split), my_ids)
        local = res.to_trajectory_metrics()
        return self._gather_and_merge_shards(
            task=task, split=split, ids=ids, world=world, rank=rank,
            local=local, local_ids=my_ids)

    # ---- 轨迹分片：辅助 -------------------------------------------------- #
    @staticmethod
    def _shard_enabled() -> bool:
        """分片开关（默认关）。⚠️ 关掉时 `evaluate()` 与改造前逐字一致。"""
        return os.environ.get('AL_EVAL_SHARD', '0').strip() == '1'

    def _gather_and_merge_shards(self, *, task: str, split: str, ids: List[int],
                                 world: int, rank: int, local: TrajectoryMetrics,
                                 local_ids: List[int]) -> TrajectoryMetrics:
        """把各 rank 的"自己那份"汇总成全集结果（按 episode id 去重）。

        汇总只交换**标量**（逐轨迹 MSE / 帧数 / baseline / 有效性），体量可忽略；
        合并后 `gmean_mse` 自动变回"全集 n 条的几何平均" ⇒ **判定口径不变**。
        """
        payload = {
            'rank': int(rank),
            'ids': [int(i) for i in local_ids],
            'per_traj': {int(k): float(v) for k, v in (local.per_traj_mse or {}).items()},
            'baseline_mse': float(local.baseline_mse or 0.0),
            'metric_valid': bool(local.metric_valid),
        }
        gathered = _all_gather_object(payload)

        per: Dict[int, float] = {}
        for item in gathered:
            per.update({int(k): float(v) for k, v in (item.get('per_traj') or {}).items()})
        base_line = next((float(it.get('baseline_mse') or 0.0)
                          for it in gathered if it.get('baseline_mse')), 0.0)
        valid = all(bool(it.get('metric_valid')) for it in gathered)

        # 组装"全集"度量：逐轨迹取并集、帧加权重算 mse、baseline/valid 取各 rank 共识
        union_ids = sorted(per) or list(ids)
        agg = TrajectoryMetrics(
            task=task, split=str(split), episode_ids=union_ids,
            per_traj_mse=dict(per), mse=0.0,
            baseline_mse=base_line, nmse=None, r2=None,
            metric_valid=valid, n_trajs=len(per))
        # 借 merge 只做"帧加权 mse 重算"（base 为空 ⇒ 结果就是并集的帧加权 mse）
        empty = TrajectoryMetrics(task=task, split=str(split), episode_ids=[],
                                  per_traj_mse={}, mse=0.0,
                                  baseline_mse=base_line, nmse=None, r2=None,
                                  metric_valid=valid, n_trajs=0)
        merged = merge_trajectory_metrics(
            empty, agg, split=split, task=task,
            entry=self.catalog.entry(task), baseline_mse=base_line,
            metric_valid=valid,
            note=f"sharded: world={world} local={len(local_ids)} merged={len(per)}/{len(ids)}")
        # merge 只保证"并集"；这里把全集 id 顺序与条数显式写回（n_trajs 必须是全集条数）
        merged.episode_ids = list(ids)
        merged.n_trajs = len(per)
        self._emit(
            f"[open_loop][shard] task={task} split={split} 共 {len(ids)} 条 ⇒ world={world}、"
            f"本 rank 处理 {len(local_ids)} 条（每卡固定 ceil(n/W)）、汇总去重后 {len(per)} 条")
        return merged

    def evaluate_confirm_merged(self, task: str, scout_ids: Sequence[int],
                                confirm_ids: Sequence[int], *, scout: Any = None,
                                split: str = "confirm",
                                baseline_mse: Optional[float] = None) -> TrajectoryMetrics:
        """confirm 门：**复用 scout 已测的回合 + 只新评"scout 之外"的回合**，再合并。

        2026-10-11 用户定（A 方案）。动机与依据：
        * 现状 `confirm_val_ids = val_ids[:4]` 是 `scout_val_ids = val_ids[:2]` 的**超集**
          ⇒ 那 2 条 scout 回合会被**重跑一遍**（白花 confirm 一半算力），
          而且"复核"用了初筛看过的样本 ⇒ 复核不独立；
        * 改后：scout 那部分直接复用（bootstrap 场景来自预扫缓存、rescan 场景来自当场评测），
          只对剩余回合真推理，最后 `merge_trajectory_metrics` 合成并集判决值
          ⇒ 判决仍建立在 4 条上（样本量不变、噪声不增），但**省掉 2 条推理**。

        调用方必须保证 `scout` 与本次 confirm 处于**同一模型状态**（两个调用点都满足：
        bootstrap 内 scout→confirm 相邻；rescan 内 scout 在同一个循环体里刚算完）。

        日志：打印"复用几条 / 新评几条 / 合计几条判决"，便于从日志核对是否生效。
        """
        scout_metric = scout
        if scout_metric is None:
            # 兜底：没传 scout 就按 bootstrap 规则取（可能命中 scout 缓存）
            scout_metric = self.evaluate_bootstrap_scout(task, "scout", list(scout_ids))
        _scout_set = {int(i) for i in (scout_metric.episode_ids or scout_ids)}
        fresh_ids = [int(i) for i in confirm_ids if int(i) not in _scout_set]
        if not fresh_ids:
            # 全部命中 scout（配置成重叠时才会发生）⇒ 无需新推理，直接用 scout 值
            self._emit(f"[open_loop][confirm] task={task} 全部 {len(_scout_set)} 条已由 scout "
                       f"测过 ⇒ 无需新推理（复用 scout 值作判决）")
            return merge_trajectory_metrics(
                scout_metric,
                TrajectoryMetrics(task=task, split=str(split), episode_ids=[],
                                  per_traj_mse={}, mse=0.0,
                                  baseline_mse=float(baseline_mse or 0.0),
                                  nmse=None, r2=None, metric_valid=True, n_trajs=0),
                split=split, task=task, entry=self.catalog.entry(task),
                baseline_mse=baseline_mse)
        fresh = self.evaluate(task, split, fresh_ids)
        merged = merge_trajectory_metrics(
            scout_metric, fresh, split=split, task=task,
            entry=self.catalog.entry(task), baseline_mse=baseline_mse)
        self._emit(f"[open_loop][confirm] task={task} 复用 scout "
                   f"{len(scout_metric.per_traj_mse or {})} 条 + 新评 {len(fresh_ids)} 条 "
                   f"⇒ 判决用 {merged.n_trajs} 条（省 {len(scout_metric.per_traj_mse or {})} 条推理）")
        return merged

    def baseline_mse(self, task: str) -> float:
        if self.baseline_store is None:
            return 0.0
        entry = self.catalog.entry(task)
        b = self.baseline_store.get(task, entry.sha256_train)
        return float(b.mse) if b is not None else 0.0


def _split_name(split: Any) -> str:
    """把 `EvalSplit` / 字符串统一成 evaluator 认的 split 名。

    ⚠️ `EvaluatorAdapter.evaluate_task` 的 `split` 必须落在 manifest 的
    `train`/`val` 里（它要校验「不跨 split 泄漏」）；Stage A 的 `EvalSplit` 更细，
    这里按「train_monitor → train，其余 → val」映射。
    """
    name = getattr(split, "value", split)
    return "train" if str(name) == "train_monitor" else "val"


# --------------------------------------------------------------------------- #
# Hardness
# --------------------------------------------------------------------------- #
class RealHardnessScorer:
    """把真实 `HardnessScorer`（per-sample L1_fm）包成 `ports.HardnessScorer`。

    ``dataset`` 必须是**训练数据集**（与 `sample_id` 同一索引空间）。
    """

    def __init__(self, scorer: Any, dataset: Any, *, max_batch: int = 8, logger: Any = None,
                 cache: Any = None, cache_write: bool = True):
        self.scorer = scorer
        self.dataset = dataset
        # 验收用：fixed 模式下可用 AL_HARDNESS_FIXED_BATCH 覆盖每批样本数（默认仍 8）。
        self.max_batch = int(os.environ.get('AL_HARDNESS_FIXED_BATCH', max_batch))
        self.logger = logger
        #: 逐样本 loss 的**磁盘缓存**（跨 run 复用；None ⇒ 行为与改造前完全一致）
        self.cache = cache
        self._cache_write = bool(cache_write)
        self._aug_warned = False
        self._auto_batch = None
        if os.environ.get('AL_HARDNESS_BATCH_MODE', 'fixed') == 'auto':
            if os.environ.get('AL_HARDNESS_BATCH_APPROVED') != '1':
                raise RuntimeError('Hardness auto requires GPU per-sample numerical parity approval')
            import inspect
            if 'sample_ids' not in inspect.signature(scorer.score).parameters:
                raise RuntimeError('Hardness auto requires per-sample-ID batch-invariant RNG scorer')
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    raise RuntimeError('Hardness auto not yet supported for multi-rank FSDP2')
            except ImportError:
                pass
            from ..scan_accel import HardnessAutoBatch
            self._auto_batch = HardnessAutoBatch(
                current=max_batch, maximum=int(os.environ.get('AL_HARDNESS_BATCH_MAX', '16')),
                reserve_gib=float(os.environ.get('AL_HARDNESS_RESERVE_GIB', '10')))

    def score(self, task: str, sample_ids: Sequence[int]) -> Dict[int, float]:
        ids_all = [int(s) for s in sample_ids]
        # 阶段名走实例属性 ⇒ 递归子调用也能带上正确标签（局部变量会被重置为 main）。
        _hr_phase = getattr(self, '_hr_phase_override', None) or 'main'
        if not ids_all:
            return {}
        #: 本次**请求**的样本全集（分片/缓存都会改变 `ids`，但日志与统计要对着全集说）
        ids_requested = list(ids_all)

        # ---------------- 多卡按 rank 分片（2026-10-10）----------------
        # 🔴 现状问题：hardness 扫描在**每个 rank 上各算一遍完全相同的全量样本** ⇒
        #    7 卡时算力浪费 7×、墙钟零收益；而瓶颈恰恰是逐样本读帧（实测 ~5.0 s/样本，
        #    打分只有 ~0.2–0.3 s/样本）。⇒ 按 `ids[rank::world_size]` 交错切片，
        #    每 rank 只读自己那份，读完 `all_gather_object` 汇总。
        # 🔴 **必须同时绑定 sample_ids**：分片后同一样本会与不同邻居凑批，而默认路径的
        #    噪声是"整批一个 generator"（`fixed_noise(B, …)`）⇒ 换批即换噪声、结果不可比。
        #    绑定后逐样本噪声只由 (seed, sid) 决定，分片/批大小都不再影响数值。
        # `AL_HARDNESS_SHARD=1` 开启；未初始化分布式或 world_size==1 ⇒ 自动退回单卡全量。
        _shard_on = os.environ.get('AL_HARDNESS_SHARD', '') == '1'
        _rank, _world = 0, 1
        if _shard_on:
            try:
                import torch.distributed as _dist
                if _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
                    _rank = int(_dist.get_rank())
                    _world = int(_dist.get_world_size())
                else:
                    _shard_on = False
            except Exception:  # noqa: BLE001 —— 分片不可用时退回单卡全量
                _shard_on = False
        ids = ids_all[_rank::_world] if _shard_on else list(ids_all)

        # ---------------- 磁盘缓存命中（2026-10-10）----------------
        # 🔴 分数只依赖「权重 + 逐样本噪声口径 + 数据帧 + 打分代码」（见 hardness_cache 文档），
        #    而且逐样本 loss 是**批无关**的 ⇒ 已缓存样本可直接复用，只扫缺失的那部分。
        #    缓存是"全 rank 相同输入、确定性输出"，所以各 rank 各自读、各自合并，
        #    合并结果天然一致（与 all_gather 汇总后的结果相同），不需要额外通信。
        #    注意：命中的样本**仍参与本 rank 的 ids 切片**，所以分片与缓存正交。
        _cached: Dict[int, float] = {}
        if self.cache is not None:
            try:
                _cached, _miss = self.cache.load(task, ids)
            except Exception as _exc:  # noqa: BLE001 —— 缓存失败绝不打断扫描
                _cached, _miss = {}, list(ids)
                if self.logger is not None:
                    try:
                        self.logger.warning(f'[hardness_cache] load 失败（{_exc!r}）⇒ 全部重扫')
                    except Exception:  # noqa: BLE001
                        pass
            if _cached:
                _msg = (f'[hardness] 缓存命中 {len(_cached)}/{len(ids)} 个样本'
                        f'（本 rank；仍需扫 {len(_miss)} 个）')
                try:
                    if self.logger is not None:
                        (self.logger.info if _shard_on else self.logger.info_rank0)(_msg)
                except Exception:  # noqa: BLE001
                    pass
            ids = _miss

        out: Dict[int, float] = {int(k): float(v) for k, v in _cached.items()}
        data_seconds = 0.0
        score_seconds = 0.0
        batches = 0
        i = 0
        # 🔴 进度可见性（2026-10-10 加）：这段 while 是 AL 里**唯一没有任何输出**的长循环，
        #    单任务上千样本 × 每批 8 个 ⇒ 数百批、几十分钟静默。真机实测因此把"在扫难度"
        #    误判成"卡死"，白花约 40 分钟排查。⇒ 按**时间节流**打点（默认最多每 15s 一行），
        #    首尾各强制一行，保证"日志不动"这件事不再能被误读。
        #    分片时各 rank 内容不同 ⇒ **每 rank 都打进度**（带 rank 标记）；未分片时只有
        #    rank0 有意义，仍走 info_rank0 以免 7 份重复淹日志。
        _log_n = len(ids)
        _log_all = len(ids_requested)
        _log_every = float(os.environ.get('AL_HARDNESS_LOG_SEC', '15') or 15)
        _log_t0 = time.perf_counter()
        _log_last = 0.0
        # 首行用的批大小：auto 模式下会动态变，这里按"起始批大小"给个量级即可。
        _log_bs = (int(self._auto_batch.current) if self._auto_batch is not None
                   else max(1, int(self.max_batch)))
        _tag = f'rank{_rank}/{_world} ' if _shard_on else ''

        def _hr_log(msg: str) -> None:
            if self.logger is None:
                return
            try:
                if _shard_on:
                    self.logger.info(msg)          # 各 rank 进度不同 ⇒ 都保留
                else:
                    self.logger.info_rank0(msg)    # 未分片 ⇒ 内容相同，只留 rank0
            except Exception:  # noqa: BLE001 —— 日志失败绝不该打断扫描
                pass

        _hr_log(f'[hardness] {_tag}开始打分：task={task} phase={_hr_phase} '
                f'本 rank {_log_n}/{_log_all} 样本'
                + (f'（{_world} 卡分片，交错切片 ids[{_rank}::{_world}]）' if _shard_on else '')
                + f' batch={_log_bs}{"(auto)" if self._auto_batch is not None else ""}'
                f' ⇒ 约 {(_log_n + _log_bs - 1) // max(1, _log_bs)} 批'
                f'（每 {_log_every:.0f}s 报一次进度；AL_HARDNESS_LOG_SEC 可调）')
        # 🔴 `_hr_out` 必须在循环**之前**取值：把样本全命中缓存时循环体一次都不执行，
        #    而循环后的"验收阶段"要用它 ⇒ 否则 `UnboundLocalError`（缓存命中即崩）。
        _hr_out = os.environ.get('AL_HARDNESS_REPORT_OUT')
        # 🔴 全部命中缓存时 `ids` 为空：**不能**提前 return，否则会丢掉已缓存的样本与收尾日志。
        # 循环前 `out` 里已有的键 = 缓存命中项 ⇒ 后续新增的即为"本次新扫到"，只写这些。
        _cached_before = set(out.keys())
        while i < len(ids):
            batches += 1
            data_started = time.perf_counter()
            batch_size = self._auto_batch.current if self._auto_batch is not None else self.max_batch
            chunk = ids[i:i + batch_size]
            i += len(chunk)
            gpu_before = gpu_reserved_before = None
            if self._auto_batch is not None or _hr_out:
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_before = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                        gpu_reserved_before = torch.cuda.memory_reserved() / (1024 ** 3)
                        # This is explicitly opt-in. Records peak reserved GPU
                        # memory during THIS scorer batch, not just idle memory.
                        torch.cuda.reset_peak_memory_stats()
                except Exception:
                    gpu_before = None
            # 🔴 review v0.2 #8：取 item **必须**在确定性上下文里 ——
            #    否则 `image_augment=true` 会让同一 sample_id 两次扫描得到不同图，
            #    并消耗全局 RNG（污染训练随机流）。
            with deterministic_sampling(self.dataset) as rep:
                if rep["n_ft"] == 0 and not self._aug_warned:
                    self._aug_warned = True
                    if self.logger is not None:
                        try:
                            self.logger.warning(
                                "[auto_learning][hardness] 没能定位到 feature_transform ⇒ "
                                "无法临时关闭图像增强；若数据集开了 image_augment，"
                                "hardness 分数将不可复现。")
                        except Exception:  # noqa: BLE001
                            pass
                items = [self.dataset[j] for j in chunk]
            data_seconds += time.perf_counter() - data_started
            score_started = time.perf_counter()
            # 🔴 要求5：fixed 与 auto 必须使用**同一套逐样本噪声语义**。
            #  - 报告模式（AL_HARDNESS_REPORT_OUT）或 auto 模式：传 sample_ids ⇒ per-sample-id RNG；
            #  - 允许用 AL_HARDNESS_UNIFY_RNG=1 在 production fixed 路径也启用（默认关，行为不变）；
            #  - **分片模式强制启用**：分片会改变"谁和谁凑一批"，逐样本噪声才保证可比（见上方说明）。
            _bind_ids = (self._auto_batch is not None or bool(_hr_out) or _shard_on
                          or os.environ.get('AL_HARDNESS_UNIFY_RNG') == '1')
            vals = (self.scorer.score(items, sample_ids=chunk) if _bind_ids
                    else self.scorer.score(items))
            score_seconds += time.perf_counter() - score_started
            if self._auto_batch is not None and gpu_before is not None:
                torch.cuda.synchronize()
                gpu_after = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                peak_extra = max(0.0, torch.cuda.max_memory_reserved() / (1024 ** 3)
                                 - gpu_reserved_before)
                peak_free = min(gpu_after, gpu_before - peak_extra)
                self._auto_batch.observe(free_before_gib=gpu_before,
                                         peak_free_gib=peak_free,
                                         free_after_gib=gpu_after, processed=len(chunk))
            for sid, v in zip(chunk, vals):
                out[sid] = float(v)
                if _hr_out:
                    try:
                        from ..scan_accel import append_json_record
                        _pf = None
                        if gpu_before is not None:
                            try:
                                import torch as _torch
                                _extra = max(0.0, _torch.cuda.max_memory_reserved() / (1024 ** 3) - gpu_reserved_before)
                                _pf = min(_torch.cuda.mem_get_info()[0] / (1024 ** 3), gpu_before - _extra)
                            except Exception:
                                _pf = gpu_before
                        append_json_record(_hr_out, {
                            'kind': 'hardness_scan', 'task': str(task),
                            'phase': _hr_phase, 'rng_binding': ('per_sample_id' if _bind_ids else 'scorer_default'),
                            'batch': int(len(chunk)),
                            'sample_ids': [int(s) for s in chunk],
                            'losses': {int(s): float(out[int(s)]) for s in chunk},
                            'free_before_gib': (None if gpu_before is None else float(gpu_before)),
                            'peak_free_gib': (None if _pf is None else float(_pf)),
                        })
                    except Exception as _exc:  # noqa: BLE001
                        if self.logger is not None:
                            try:
                                self.logger.warning(f'[hardness] report write failed: {_exc!r}')
                            except Exception:
                                pass
            # ---- 进度打点（时间节流；首批与末批强制）----
            _now = time.perf_counter()
            _is_last = i >= len(ids)
            if _is_last or batches == 1 or (_now - _log_last) >= _log_every:
                _log_last = _now
                _elapsed = _now - _log_t0
                _done = min(i, len(ids))
                _rate = _done / _elapsed if _elapsed > 0 else 0.0
                _eta = ((len(ids) - _done) / _rate) if _rate > 0 else float('inf')
                _hr_log(f'[hardness] {_tag}进度 {_done}/{len(ids)}（批 {batches}）'
                        f' 用时 {_elapsed:.0f}s 速度 {_rate:.1f} 样本/s'
                        f' ETA {("?" if _eta == float("inf") else f"{_eta:.0f}s")}'
                        f' | data {data_seconds:.1f}s score {score_seconds:.1f}s'
                        + (' [最后一批]' if _is_last else ''))

        # ---------------- 分片汇总 ----------------
        if _shard_on:
            import torch.distributed as _dist
            _gathered: List[Any] = [None] * _world
            _dist.all_gather_object(_gathered, out)
            merged: Dict[int, float] = {}
            for piece in _gathered:
                if not piece:
                    continue
                for _sid, _val in piece.items():
                    _sid = int(_sid)
                    if _sid in merged:
                        raise RuntimeError(
                            f'[hardness] 分片汇总出现重复样本 sid={_sid}'
                            f'（rank={_rank}/{_world}）⇒ 分片逻辑有 bug，拒绝静默继续')
                    merged[_sid] = float(_val)
            _missing = [s for s in ids_requested if s not in merged]
            if _missing:
                raise RuntimeError(
                    f'[hardness] 分片汇总缺 {len(_missing)} 个样本'
                    f'（例：{_missing[:8]}）⇒ 有 rank 少算或多算，拒绝静默继续')
            out = merged

        # ---- 写回缓存：只写**本次新扫到**的样本（命中项无需重复写盘）----
        if self.cache is not None and self._cache_write:
            _fresh = {k: v for k, v in out.items() if k not in _cached_before}
            if _fresh:
                try:
                    self.cache.store(task, _fresh)
                except Exception as _exc:  # noqa: BLE001 —— 写缓存失败绝不影响训练
                    try:
                        if self.logger is not None:
                            self.logger.warning(f'[hardness_cache] store 失败（{_exc!r}）⇒ 忽略')
                    except Exception:  # noqa: BLE001
                        pass

        _hr_log(f'[hardness] {_tag}打分完成：task={task} phase={_hr_phase} '
                f'本 rank {len(ids)} 个样本、汇总后 {len(out)} 个'
                f' 批次={batches} 总用时 {time.perf_counter() - _log_t0:.1f}s'
                f'（data {data_seconds:.1f}s / score {score_seconds:.1f}s）')
        self.last_timing = {"data_wall_seconds": round(data_seconds, 5),
                            "score_submit_seconds": round(score_seconds, 5),
                            "batches": batches, "samples": len(ids),
                            "sharded": bool(_shard_on), "rank": int(_rank),
                            "world_size": int(_world),
                            "final_batch": (self._auto_batch.current if self._auto_batch else self.max_batch),
                            "auto_batch_mode": self._auto_batch is not None}
        # 单进程验收阶段（要求4/7）：main(Batch8) → replay(Batch1) → repeat(Batch8)。
        # 递归守卫避免阶段内再次触发；不额外加载模型、不改全局配置。
        if _hr_out and not getattr(self, '_hr_phases_running', False):
            _replay_batch = int(os.environ.get('AL_HARDNESS_REPLAY_BATCH', '0') or 0)
            _do_repeat = os.environ.get('AL_HARDNESS_REPEAT') == '1'
            _saved_batch = self.max_batch
            if _replay_batch > 0 or _do_repeat:
                self._hr_phases_running = True
                try:
                    if _replay_batch > 0:
                        self.max_batch = int(_replay_batch)
                        self._hr_phase_override = 'replay_batch%d' % int(_replay_batch)
                        _hr_phase = 'replay_batch%d' % int(_replay_batch)
                        self.score(task, ids)
                    if _do_repeat:
                        self.max_batch = int(_saved_batch)
                        self._hr_phase_override = 'repeat'
                        _hr_phase = 'repeat'
                        self.score(task, ids)
                finally:
                    self.max_batch = int(_saved_batch)
                    self._hr_phase_override = None
                    self._hr_phases_running = False
        return out


# --------------------------------------------------------------------------- #
# Trainer（延迟模式下不会被调用）
# --------------------------------------------------------------------------- #
class RealTrainerStub:
    """`defer_train=True` 时 Scheduler **不会**调它；这里显式报错而不是静默。"""

    def train_steps(self, request, num_steps):
        raise RuntimeError(
            "RealTrainer.train_steps() 不该被调用：Auto Learning 走**延迟训练模式**"
            "（`scheduler.defer_train=True`），训练由外层真实循环跑，"
            "结果经 `AutoLearnLoopHook` 回填。")


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #
def build_real_backend(
    *,
    catalog: Any,
    resolver: Any,
    adapter: Any,
    hardness: Any,
    dataset: Any = None,
    baseline_store: Any = None,
) -> Backend:
    """组装真实 `Backend`。

    ``catalog`` 必须已经 `attach_samples(resolver)` 过（否则 `train_sample_ids` 为空）。
    """
    if not getattr(catalog.entry(catalog.task_names()[0]), "train_sample_ids", None):
        raise ValueError(
            "catalog 还没 attach_samples()：TaskEntry.train_sample_ids 为空，"
            "Sampler 无法工作。先调 catalog.attach_samples(resolver)。")
    return Backend(
        catalog=catalog,
        resolver=resolver,
        evaluator=RealEvaluator(adapter, catalog, baseline_store),
        scorer=hardness,
        trainer=RealTrainerStub(),
        baseline=baseline_store,
    )


def build_real_hardness(model: Any, dataset: Any, *, logger: Any = None,
                        cache: Any = None, cache_write: bool = True, **kw) -> RealHardnessScorer:
    """便捷构造：真实 `HardnessScorer` + 训练数据集。

    ``cache``：逐样本 loss 的磁盘缓存（见 `auto_learning/hardness_cache.py`）；
    **不要**塞进 ``**kw``（那是给 `HardnessScorer` 构造函数的）。
    """
    from ..hardness import HardnessScorer

    return RealHardnessScorer(HardnessScorer(model, **kw), dataset, logger=logger,
                              cache=cache, cache_write=cache_write)


__all__ = ["RealEvaluator", "RealHardnessScorer", "RealTrainerStub",
           "build_real_backend", "build_real_hardness"]
