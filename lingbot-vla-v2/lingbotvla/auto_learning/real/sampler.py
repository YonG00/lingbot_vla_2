"""`AutoLearnSampler` —— 把 Stage A 的 `BatchSampler` 接到真实 DataLoader 上。

设计（见 `docs/stage_b1_integration_design.md` §2 / §4）
-----------------------------------------------------
* 真实循环里 **一次 `next(data_iterator)` = 一个 optimizer step**
  （`dataloader_batch_size = gbs // dp`、`num_micro_batch = gbs // (micro×dp)`，单卡下都为 1）。
* 所以「每步独立采样」== 「sampler 每产出 `batch_size` 个 index 就换一批」。
* DataLoader 在**主进程**按 `batch_size` 消费 sampler ⇒ `BatchSampler.build()` 恰好一次/step。
* 真正的采样逻辑**一行都不重写**：`prepare(req)` 一个 unit 一次，`build()` 每步一次。

🔴 **unit 边界必须重建 DataLoader 迭代器**（丢弃 prefetch），否则旧 unit 的样本会混进新 unit。
   见设计文档 §5；本模块只负责「产出正确的 index」，不管预取。
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

try:  # 真实训练环境
    from torch.utils.data import Sampler as _SamplerBase
except Exception:  # noqa: BLE001 —— 无 torch 的单测环境（Stage B1 的「无卡测试」要求可跑）
    class _SamplerBase:  # type: ignore[no-redef]
        """最小替身：只为让本模块在**没有 torch** 时也能 import 与单测。

        真实训练路径上用的是 `torch.utils.data.Sampler`（`DataLoader` 只要求
        `__iter__`，本类的实现与之兼容）。
        """


from ..ports import ReplayPlan, TrainRequest  # noqa: E402
from ..sampling.sampler import BatchSampler, PreparedRequest  # noqa: E402


@dataclass
class UnitStats:
    """一个 learning unit（若干 optimizer step）的采样统计（测试方案 §5.4）。"""

    steps: int = 0
    n_new: int = 0
    n_old: int = 0
    samples_seen: int = 0
    new_slot_counts_by_task: Counter = field(default_factory=Counter)
    old_slot_counts_by_task: Counter = field(default_factory=Counter)
    unique_replay_tasks_per_batch: List[int] = field(default_factory=list)
    per_step_losses: List[float] = field(default_factory=list)

    @property
    def unique_batches(self) -> int:
        """不同 batch 组合数（用「每步的 (new, old) 元组」去重近似）。"""
        return len(set(self._step_keys)) if hasattr(self, "_step_keys") else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps, "n_new": self.n_new, "n_old": self.n_old,
            "samples_seen": self.samples_seen,
            "new_slot_counts_by_task": dict(self.new_slot_counts_by_task),
            "old_slot_counts_by_task": dict(self.old_slot_counts_by_task),
            "unique_replay_tasks_per_batch": list(self.unique_replay_tasks_per_batch),
            "mean_unique_replay_tasks": (
                sum(self.unique_replay_tasks_per_batch) / len(self.unique_replay_tasks_per_batch)
                if self.unique_replay_tasks_per_batch else 0.0),
        }

    def check_invariants(self) -> List[str]:
        """返回违反的不变量（空 = 通过）。**每步与 unit 级都要查**。"""
        problems: List[str] = []
        if self.n_new + self.n_old != self.samples_seen:
            problems.append(
                f"n_new({self.n_new}) + n_old({self.n_old}) != samples_seen({self.samples_seen})")
        if self.steps and self.samples_seen % self.steps != 0:
            problems.append(f"samples_seen({self.samples_seen}) 不是 steps({self.steps}) 的整数倍")
        return problems


class AutoLearnSampler(_SamplerBase):
    """产出「7 NEW + 3 Replay」的 dataset local_idx（`SampleRef.sample_id`）。

    生命周期::

        sampler = AutoLearnSampler(batch_sampler)
        sampler.set_request(req)        # ← loop hook 在 unit 开始时调用
        for i in sampler: ...           # DataLoader 每 batch_size 个消费一次
        stats = sampler.take_stats()    # ← unit 结束时取统计
    """

    def __init__(self, batch_sampler: BatchSampler, *, batch_size: Optional[int] = None):
        self.batch_sampler = batch_sampler
        self._prepared: Optional[PreparedRequest] = None
        self._batch_size = int(batch_size or batch_sampler.cfg.batch_size)
        self.stats = UnitStats()
        self._last_comp = None
        self._step_keys: set = set()

    # -- unit 生命周期 -------------------------------------------------------
    def set_request(self, req: TrainRequest) -> None:
        """发布新的 `TrainRequest`（**必须在重建迭代器之前**调用）。"""
        if req.batch_size and req.batch_size != self._batch_size:
            raise ValueError(
                f"TrainRequest.batch_size({req.batch_size}) != sampler batch_size"
                f"({self._batch_size}) —— 必须等于 dataloader_batch_size")
        self._prepared = self.batch_sampler.prepare(req)
        self._last_comp = None

    def take_stats(self) -> UnitStats:
        """取走并重置本 unit 的统计。"""
        s, self.stats = self.stats, UnitStats()
        self._step_keys = set()
        return s

    @property
    def last_composition(self):
        """最近一次 `build()` 的 `BatchComposition`（供 provenance 审计）。"""
        return self._last_comp

    # -- Sampler 接口 -------------------------------------------------------
    def __iter__(self) -> Iterator[int]:
        while True:
            if self._prepared is None:
                raise RuntimeError(
                    "AutoLearnSampler 还没收到 TrainRequest —— 上层必须先 set_request(req) "
                    "再重建 DataLoader 迭代器")
            comp = self.batch_sampler.build(self._prepared)
            self._record(comp)
            yield from (int(r.sample_id) for r in comp.refs)

    def __len__(self) -> int:  # pragma: no cover - 无限流
        raise TypeError("AutoLearnSampler 是无限流（步数由 train_steps 控制），没有 len()")

    def _record(self, comp) -> None:
        st = self.stats
        st.steps += 1
        st.n_new += comp.n_new
        st.n_old += comp.n_old
        st.samples_seen += len(comp.refs)
        for r in comp.new:
            st.new_slot_counts_by_task[r.task] += 1
        for r in comp.old:
            st.old_slot_counts_by_task[r.task] += 1
        st.unique_replay_tasks_per_batch.append(len(comp.old_tasks))
        self._step_keys.add((tuple(r.sample_id for r in comp.new),
                             tuple(r.sample_id for r in comp.old)))
        self._last_comp = comp

    # -- resume -------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """`StatefulDataLoader` 依赖它做 resume。

        ⚠️ **不**存 RNG 状态 —— RNG 是 scheduler 的**单一权威**（Stage A 的教训），
        由 `extra_state["auto_learning"]["sampler_rng"]` 统一保存/恢复。
        """
        return {
            "version": 1,
            "batch_size": self._batch_size,
            "batches_built": self.batch_sampler.n_batches,
            "unit_steps_done": self.stats.steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError(f"sampler state 必须是 dict，实际 {type(state).__name__}")
        if int(state.get("version", 0)) != 1:
            raise ValueError(f"sampler state 版本不符: {state.get('version')}")
        self.batch_sampler.n_batches = int(state.get("batches_built", 0))
        # unit 内位置：由 loop hook 决定是否从 unit 开头重放
        self.stats.steps = int(state.get("unit_steps_done", 0))


__all__ = ["AutoLearnSampler", "UnitStats"]
