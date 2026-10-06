"""Batch 组装：`7 NEW + 3 OLD`（文档 §27–§29）。

分工：
  * `prepare(req)` —— 把一个 learning unit 要用的分布**编译一次**
    （NEW 的 alias table、replay 各任务的 snapshot table、校验 sample_id 归属）；
  * `build(prepared)` —— 抽**一个** optimizer step 的 batch。

一个 unit 会调用 `build` 50 次（每步重新采样），但 `prepare` 只调一次 ——
这就是 50 步语义下还能跑得快的原因。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..config import AutoLearningConfig
from ..ports import ReplayPlan, SampleResolver, TaskCatalog, TrainRequest
from ..types import BatchComposition, SampleRef
from .replay import ReplaySlot, prepare_slots, sample_replay_refs
from .rng import WeightedSampler


@dataclass
class PreparedRequest:
    """一个 learning unit 内复用的预编译结果。"""

    request: TrainRequest
    new_table: WeightedSampler
    replay_slots: List[ReplaySlot] = field(default_factory=list)


class BatchSampler:
    def __init__(
        self,
        cfg: AutoLearningConfig,
        resolver: SampleResolver,
        catalog: TaskCatalog,
        rng: random.Random,
    ) -> None:
        self.cfg = cfg
        self.resolver = resolver
        self.catalog = catalog
        self.rng = rng
        self.n_batches = 0

    # ---------------------------------------------------------------- #
    def prepare(self, req: TrainRequest) -> PreparedRequest:
        cfg = self.cfg
        # 测试方案 §I06：非法概率当场报错
        if not req.probs:
            raise ValueError(f"任务 {req.task} 没有采样概率，不能开训")
        known = set(self.catalog.entry(req.task).train_sample_ids)
        unknown = [s for s in req.probs if s not in known]
        if unknown:
            raise RuntimeError(
                f"任务 {req.task} 的采样概率含 {len(unknown)} 个不属于它的 sample_id"
                f"（例如 {sorted(unknown)[:3]}）—— 概率表与数据切分对不上"
            )
        new_table = WeightedSampler(list(req.probs.keys()), list(req.probs.values()))
        slots = (
            prepare_slots(req.replay, req.replay_slots, policy=cfg.replay_sample_policy)
            if req.replay_slots > 0 and req.replay.tasks
            else []
        )
        return PreparedRequest(request=req, new_table=new_table, replay_slots=slots)

    # ---------------------------------------------------------------- #
    def build(self, prepared: PreparedRequest) -> BatchComposition:
        """抽**一个 optimizer step** 的 batch。"""
        req = prepared.request
        want_old = req.replay_slots if prepared.replay_slots else 0
        want_new = req.batch_size - want_old  # 没有 PASS task ⇒ 全 NEW（文档 §27）

        new_refs: List[SampleRef] = [
            self.resolver.resolve(req.task, sid)
            for sid in prepared.new_table.draw_without_replacement(want_new, self.rng)
        ]
        old_refs: List[SampleRef] = (
            sample_replay_refs(prepared.replay_slots, want_old, self.resolver, self.rng)
            if want_old
            else []
        )
        if len(old_refs) != want_old:  # pragma: no cover - 正常路径不会走到
            raise RuntimeError(
                f"replay 只组出 {len(old_refs)}/{want_old} 个 slot，batch 组成不完整"
            )

        comp = BatchComposition(new=new_refs, old=old_refs)
        self._assert_train_only(comp)
        self.n_batches += 1
        return comp

    # ---------------------------------------------------------------- #
    @staticmethod
    def _assert_train_only(comp: BatchComposition) -> None:
        """val 样本永远不能被 sampler 抽中（文档 §44）。"""
        bad = [r for r in comp.refs if r.is_val]
        if bad:
            raise AssertionError(f"val sample 混进了训练 batch: {bad[:3]}")
