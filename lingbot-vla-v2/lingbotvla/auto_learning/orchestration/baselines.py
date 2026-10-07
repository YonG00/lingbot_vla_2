"""Baseline A：uniform multi-task（文档 §45）。

真正要证明的不是「自动系统最终能训成功」，而是：
**在同等 optimizer step 预算下，它是否比 uniform 更快获得更多可用技能。**

两种对照：
  * `mixed`       每个 batch 从所有任务均匀混合抽（标准 multi-task baseline）
  * `round_robin` 每个 unit 专注一个任务、轮流（每任务 step 数相同，无优先级）

两者都复用同一套假世界 / 评测器 / 数据切分，所以和自主模式是同一把尺子。
同样遵循「一个 unit = `eval_interval_steps` 个 optimizer step，**每步重新采样**」。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import DemoConfig
from ..decision.metrics import median
from ..decision.thresholds import is_pass
from ..testing.sim import build_backend
from ..state.registry import TaskRegistry
from ..types import BatchComposition, EvalSplit, SampleRef


@dataclass
class BaselineResult:
    mode: str
    units: int
    global_step: int
    global_samples_seen: int
    per_task_nmse: Dict[str, float] = field(default_factory=dict)
    per_task_steps: Dict[str, int] = field(default_factory=dict)
    pass_tasks: List[str] = field(default_factory=list)
    coverage: float = 0.0
    median_nmse: Optional[float] = None
    worst_nmse: Optional[float] = None
    nmse_history: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "units": self.units,
            "global_step": self.global_step,
            "samples_seen": self.global_samples_seen,
            "coverage": round(self.coverage, 4),
            "pass": len(self.pass_tasks),
            "median_nmse": self.median_nmse,
            "worst_nmse": self.worst_nmse,
        }


def _sample_ref(resolver, task: str, sample_ids, rng: random.Random) -> SampleRef:
    return resolver.resolve(task, sample_ids[rng.randrange(len(sample_ids))])


def run_uniform(
    cfg: DemoConfig,
    units: int,
    mode: str = "mixed",
    eval_every_unit: bool = True,
) -> BaselineResult:
    al = cfg.auto_learning
    backend = build_backend(cfg)
    world = backend.extra_state
    catalog = backend.catalog
    resolver = backend.resolver
    registry = TaskRegistry.from_catalog(catalog, al)
    rng = random.Random(cfg.sim.seed + 1000)
    names = registry.names()
    batch = al.batch_size
    steps_per_unit = al.eval_interval_steps

    per_task_steps: Dict[str, int] = {n: 0 for n in names}
    result = BaselineResult(mode=mode, units=units, global_step=0, global_samples_seen=0)

    for u in range(units):
        comps: List[BatchComposition] = []
        for _ in range(steps_per_unit):
            if mode == "mixed":
                refs = []
                for _ in range(batch):
                    task = names[rng.randrange(len(names))]
                    refs.append(_sample_ref(resolver, task, registry.get(task).sample_ids, rng))
                    per_task_steps[task] += 1
                comps.append(BatchComposition(new=refs, old=[]))
            elif mode == "round_robin":
                task = names[u % len(names)]
                ids = registry.get(task).sample_ids
                refs = [_sample_ref(resolver, task, ids, rng) for _ in range(batch)]
                per_task_steps[task] += batch
                comps.append(BatchComposition(new=refs, old=[]))
            else:
                raise ValueError(f"unknown baseline mode: {mode}")

        info = world.apply_steps(comps)
        result.global_step += steps_per_unit
        result.global_samples_seen += info["samples"]

        if eval_every_unit or u == units - 1:
            for name in names:
                rec = registry.get(name)
                m = backend.evaluator.evaluate(
                    name, EvalSplit.ACTIVE_VAL.value, rec.active_val_ids
                )
                result.nmse_history.append(
                    {"unit": u + 1, "step": result.global_step, "task": name, "nmse": m.nmse}
                )

    for name in names:
        rec = registry.get(name)
        m = backend.evaluator.evaluate(name, EvalSplit.ACTIVE_VAL.value, rec.active_val_ids)
        if m.nmse is not None:
            result.per_task_nmse[name] = m.nmse
            # 🔴 必须与 AL 实验组走**同一个**判定入口（thresholds.is_pass），
            # 否则两组的"通过"口径不一致 ⇒ 对比结论静默失效（且不会报错）。
            if is_pass(al, name, nmse=m.nmse, mse=m.mse):
                result.pass_tasks.append(name)

    vals = list(result.per_task_nmse.values())
    if vals:
        result.median_nmse = round(median(vals) or 0.0, 5)
        result.worst_nmse = round(max(vals), 5)
    result.coverage = len(result.pass_tasks) / max(1, len(names))
    result.per_task_steps = per_task_steps
    return result
