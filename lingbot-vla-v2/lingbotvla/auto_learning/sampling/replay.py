"""OLD Replay 采样（文档 §28–§30）。

两条硬要求：
  1. **零散复习** —— 每个 replay slot 独立采样，同一个 batch 尽量来自不同 PASS 任务；
  2. **两级采样** —— P(old sample) = P(task) × P(sample | task)，
     task 级 uniform（否则帧多的任务天然获得更多 replay），
     sample 级默认沿用「该任务 PASS 时的采样分布」（pass snapshot）。

这里只放**纯函数**：PASS 池 = registry 里 status 为 PASS 的任务，
样本分布来自 `rec.pass_sampling_snapshot`，都不需要额外的池对象。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..ports import ReplayPlan, SampleResolver
from ..types import SampleRef
from .rng import WeightedSampler, shuffled


@dataclass(frozen=True)
class ReplaySlot:
    """一个 replay slot 的解析结果：属于哪个任务、用哪张分布表。"""

    task: str
    table: Optional[WeightedSampler]  # None ⇒ 均匀
    sample_ids: Sequence[int] = field(default_factory=list)


def assign_slots(tasks: Sequence[str], k: int, rng: random.Random) -> List[str]:
    """先决定每个 replay slot 属于哪个旧任务 —— 尽量互不相同（文档 §29.1）。"""
    if not tasks or k <= 0:
        return []
    order = shuffled(tasks, rng)
    return [order[i % len(order)] for i in range(k)]


def prepare_slots(
    plan: ReplayPlan,
    k: int,
    *,
    policy: str = "pass_snapshot",
) -> List[ReplaySlot]:
    """把 replay plan 编译成 slot 表（分布只建一次，供整个 learning unit 复用）。

    这里做**一次**校验：snapshot 里的 sample_id 必须都属于当前数据集。
    对不上就报错 —— 真实仓库里「数据切分变了但 snapshot 还是旧的」是最危险的情形，
    静默跳过会让你以为在 replay，其实一个旧样本都没复习到。
    """
    slots: List[ReplaySlot] = []
    for task in plan.tasks:
        ids = list(plan.sample_ids.get(task) or [])
        if not ids:
            raise RuntimeError(f"PASS 任务 {task} 没有 sample_ids，无法组 replay slot")
        raw = plan.probs.get(task) if policy == "pass_snapshot" else None
        if raw:
            valid = set(ids)
            unknown = [s for s in raw if s not in valid]
            if unknown:
                raise RuntimeError(
                    f"任务 {task} 的 PASS snapshot 含 {len(unknown)} 个当前数据集里不存在的 "
                    f"sample_id（例如 {sorted(unknown)[:3]}）—— 数据切分变了，"
                    "不能静默跳过，请重建 snapshot 或改 replay_sample_policy=uniform"
                )
            table: Optional[WeightedSampler] = WeightedSampler(
                list(raw.keys()), [raw[s] for s in raw.keys()]
            )
        else:
            table = None  # 均匀
        slots.append(ReplaySlot(task=task, table=table, sample_ids=ids))
    return slots


def sample_replay_refs(
    slots: Sequence[ReplaySlot],
    k: int,
    resolver: SampleResolver,
    rng: random.Random,
) -> List[SampleRef]:
    """按 slot 表抽 `k` 个旧样本（每个 slot 独立抽，任务级 uniform）。"""
    if not slots or k <= 0:
        return []
    tasks = [s.task for s in slots]
    by_task = {s.task: s for s in slots}
    refs: List[SampleRef] = []
    for task in assign_slots(tasks, k, rng):
        slot = by_task[task]
        if slot.table is not None:
            sid = slot.table.draw(rng)
        else:
            sid = slot.sample_ids[rng.randrange(len(slot.sample_ids))]
        refs.append(resolver.resolve(task, sid))
    return refs
