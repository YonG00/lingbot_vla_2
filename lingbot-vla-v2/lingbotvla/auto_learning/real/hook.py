"""训练循环接线（Stage B1）。

职责（见 `docs/stage_b1_integration_design.md` §1 / §10）
------------------------------------------------------
* `build_hook(cfg, ...)` —— `enabled=false` 时返回 **None**，且**一个对象都不创建**。
* `AutoLearnLoopHook` —— 训练循环在三个位置调它：

    ① `on_step_begin(global_step)`：若到了 unit 边界，驱动 Scheduler 直到
       `train_unit`（**延迟模式**：scheduler 只发布 `TrainRequest`，不自己训练），
       把 request 交给 sampler，并告诉循环**必须重建 DataLoader 迭代器**。
    ② `on_step_end(global_step, loss)`：累计本 unit 的样本/损失。
    ③ `on_unit_end(global_step)`：把 `TrainResult` 回填给 Scheduler，继续做决策。

* `extra_state()` / `load_extra_state()` —— 给 `state["extra_state"]["auto_learning"]`。

⚠️ 本模块只依赖 `ports` / `types`，**不 import torch** ⇒ 无卡可单测。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..ports import TrainResult
from .sampler import UnitStats


@dataclass
class StepDirective:
    """`on_step_begin()` 的返回 —— 告诉真实循环这一步要做什么。"""

    #: True ⇒ 刚跨过 unit 边界，**必须重建 `iter(train_dataloader)`**（丢弃 prefetch）
    rebuild_iterator: bool = False
    #: 本步要用的 `TrainRequest`（仅重建时非 None）
    request: Any = None
    #: unit 内的第几步（从 0 开始）
    step_in_unit: int = 0
    #: 本 unit 一共几步
    unit_steps: int = 0
    #: 训练已结束（scheduler 走到 finish）
    finished: bool = False


class AutoLearnLoopHook:
    """把 Scheduler 接到真实训练循环上的那层胶水。"""

    def __init__(
        self,
        *,
        scheduler: Any,
        sampler: Any,
        cfg: Any,
        logger: Any = None,
        event_logger: Any = None,
    ):
        self.scheduler = scheduler
        self.sampler = sampler
        self.cfg = cfg
        self.logger = logger
        self.event_logger = event_logger
        # 🔴 关键：让 scheduler **不要**自己调 trainer
        scheduler.defer_train = True
        self._step_in_unit = 0
        self._unit_steps = 0
        self._unit_losses: List[float] = []
        self._steps_done = 0
        self._events: List[Dict[str, Any]] = []

    # -- 生命周期 -----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return True

    def on_step_begin(self, global_step: int) -> StepDirective:
        """unit 边界 ⇒ 驱动 scheduler 到 `train_unit` 并发布 request。"""
        if self._step_in_unit < self._unit_steps:
            return StepDirective(step_in_unit=self._step_in_unit,
                                 unit_steps=self._unit_steps)

        # 跨过边界：把上一个 unit 的结果回填（若有），然后推进到下一个 train_unit
        self._drain_pending_unit(global_step)
        while self.scheduler.next_action() != "train_unit":
            if self.scheduler.state.finished:
                return StepDirective(finished=True)
            ev = self.scheduler.advance()
            self._record(ev)

        ev = self.scheduler.advance()          # 触发 _train_unit（deferred）
        self._record(ev)
        req = self.scheduler.pending_train_request
        if req is None:
            raise RuntimeError("scheduler 没有发布 pending TrainRequest（defer_train 未生效？）")
        self._unit_steps = int(self.scheduler.pending_train_steps)
        self._step_in_unit = 0
        self.sampler.set_request(req)
        return StepDirective(rebuild_iterator=True, request=req,
                             step_in_unit=0, unit_steps=self._unit_steps)

    def on_step_end(self, global_step: int, loss: float) -> None:
        self._step_in_unit += 1
        self._steps_done += 1
        if loss is not None:
            self._unit_losses.append(float(loss))

    def on_unit_end(self, global_step: int) -> Optional[Dict[str, Any]]:
        """显式结束本 unit（正常情况下 `on_step_begin` 会自动 drain）。"""
        return self._drain_pending_unit(global_step)

    # -- 内部 ---------------------------------------------------------------
    def _drain_pending_unit(self, global_step: int) -> Optional[Dict[str, Any]]:
        """把 sampler 的统计打包成 `TrainResult` 回填给 scheduler。"""
        if self.scheduler.pending_train_request is None:
            return None
        stats: UnitStats = self.sampler.take_stats()
        problems = stats.check_invariants()
        if problems:
            raise RuntimeError(
                "采样统计违反不变量（测试计划 §12 fail-fast）：\n  - " + "\n  - ".join(problems))
        if stats.steps != self._unit_steps:
            raise RuntimeError(
                f"本 unit 实际跑了 {stats.steps} 步，但 scheduler 要求 {self._unit_steps} 步")
        result = TrainResult(
            loss=(sum(self._unit_losses) / len(self._unit_losses)) if self._unit_losses else 0.0,
            steps=stats.steps,
            samples_seen=stats.samples_seen,
            old_slot_counts=dict(stats.old_slot_counts_by_task),
            new_slot_counts=dict(stats.new_slot_counts_by_task),
            per_step_losses=list(self._unit_losses),
            batches_built=stats.steps,
            unique_batches=len({(round(x, 12)) for x in self._unit_losses}),
        )
        ev = self.scheduler.complete_train_unit(result)
        self._record(ev)
        self._unit_losses = []
        self._step_in_unit = 0
        self._unit_steps = 0
        return ev

    def _record(self, ev: Optional[Dict[str, Any]]) -> None:
        if not ev:
            return
        self._events.append(ev)
        if self.event_logger is not None:
            try:
                self.event_logger.log(ev)
            except Exception:  # noqa: BLE001 —— 事件写失败不该打断训练
                pass

    # -- 持久化 -------------------------------------------------------------
    def extra_state(self) -> Dict[str, Any]:
        """给 `state["extra_state"]["auto_learning"]`。**不改 checkpointer**。"""
        return {
            "version": 1,
            "scheduler": self.scheduler.state.to_state(),
            "registry": self.scheduler.registry.to_state(),
            "sampler": self.sampler.state_dict(),
            "sampler_rng": self.scheduler.rng.getstate(),
            "step_in_unit": self._step_in_unit,
            "unit_steps": self._unit_steps,
            "steps_done": self._steps_done,
            "events_offset": len(self._events),
        }

    def load_extra_state(self, raw: Dict[str, Any]) -> None:
        """恢复。⚠️ 必须在 `scheduler.resume()` **之前**调 `load_state`，之后调 `resume()`。"""
        if not isinstance(raw, dict) or int(raw.get("version", 0)) != 1:
            raise ValueError(f"auto_learning extra_state 版本不符: {raw!r:.120}")
        self.scheduler.state.load_state(raw["scheduler"])
        self.scheduler.registry.load_state(raw["registry"])
        self.sampler.load_state_dict(raw["sampler"])
        self.scheduler.rng.setstate(tuple(raw["sampler_rng"]) if raw.get("sampler_rng") else
                                    random.getstate())
        self._step_in_unit = int(raw.get("step_in_unit", 0))
        self._unit_steps = int(raw.get("unit_steps", 0))
        self._steps_done = int(raw.get("steps_done", 0))
        self.scheduler.resume()
        self.scheduler.defer_train = True


def build_hook(
    cfg: Any,
    *,
    scheduler: Any = None,
    sampler: Any = None,
    logger: Any = None,
    event_logger: Any = None,
) -> Optional[AutoLearnLoopHook]:
    """**唯一的构造入口**。

    ``cfg.enabled=False``（或 cfg 为 None）⇒ 返回 **None**，且
    **不触碰 scheduler / sampler、不创建任何对象**（测试计划 §7.9 的 disabled 回归）。
    """
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return None
    if scheduler is None or sampler is None:
        raise ValueError(
            "auto_learning.enabled=True 时必须提供 scheduler 与 sampler"
            "（由 real/backend.py 的 build_real_backend() 组装）")
    return AutoLearnLoopHook(scheduler=scheduler, sampler=sampler, cfg=cfg,
                             logger=logger, event_logger=event_logger)


__all__ = ["AutoLearnLoopHook", "StepDirective", "build_hook"]
