"""训练循环接线（Stage B1）。

职责（见 `docs/stage_b1_integration_design.md` §1 / §10）
------------------------------------------------------
* `build_hook(cfg, ...)` —— `enabled=false` 时返回 **None**，且**一个对象都不创建**。
* `AutoLearnLoopHook` —— 训练循环在三个位置调它：

    ① `on_step_begin(global_step)`：若到了 unit 边界，驱动 Scheduler 直到
       `train_unit`（**延迟模式**：scheduler 只发布 `TrainRequest`，不自己训练），
       把 request 交给 sampler，并告诉循环**必须重建 DataLoader 迭代器**。
    ② `on_step_end(global_step, loss)`：累计本 unit 的样本/损失；
       **unit 的最后一步跑完就当场把 `TrainResult` 回填给 Scheduler**（review v0.2 #2）。
    ③ `on_unit_end(global_step)`：显式回填（幂等，正常路径已被 ② 覆盖）。
    ④ `flush_partial_unit()`：训练收尾 / epoch 边界用 —— 把没跑满的 unit
       按实际步数记账（或撤回），保证收尾存档落在 `at_safe_checkpoint_boundary`。

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
        req = self._open_next_unit()
        if req is None:
            return StepDirective(finished=True)
        return StepDirective(rebuild_iterator=True, request=req,
                             step_in_unit=0, unit_steps=self._unit_steps)

    def _open_next_unit(self):
        """推进 scheduler 直到发布下一个 `train_unit` 的 `TrainRequest`。

        返回该 request；scheduler 已经结束（`finished`）时返回 **None**。

        副作用：`_unit_steps` / `_step_in_unit` / `_unit_losses` 复位，并把 request
        交给 sampler —— 🔴 **必须在重建 `iter(train_dataloader)` 之前**完成，
        否则 sampler 的 `__iter__` 会因「还没收到 TrainRequest」直接报错。
        """
        while self.scheduler.next_action() != "train_unit":
            if self.scheduler.state.finished:
                return None
            self._record(self.scheduler.advance())

        ev = self.scheduler.advance()          # 触发 _train_unit（deferred）
        self._record(ev)
        req = self.scheduler.pending_train_request
        if req is None:
            raise RuntimeError("scheduler 没有发布 pending TrainRequest（defer_train 未生效？）")
        self._unit_steps = int(self.scheduler.pending_train_steps)
        self._step_in_unit = 0
        self._unit_losses = []
        self.sampler.set_request(req)
        return req

    def on_step_end(self, global_step: int, loss: float) -> None:
        self._step_in_unit += 1
        self._steps_done += 1
        if loss is not None:
            self._unit_losses.append(float(loss))
        # 🔴 review v0.2 #2：**unit 的最后一步跑完就当场回填**，不等下一个
        #    `on_step_begin()`。否则会出现「模型已经训到 N，Scheduler 只记到账 N-50」
        #    的窗口 —— 存档落在这里就会存成「模型领先 Scheduler 一个 unit」的状态。
        if self._unit_steps > 0 and self._step_in_unit >= self._unit_steps:
            self._drain_pending_unit(global_step)

    def on_unit_end(self, global_step: int) -> Optional[Dict[str, Any]]:
        """显式结束本 unit（正常情况下 `on_step_begin` 会自动 drain）。"""
        return self._drain_pending_unit(global_step)

    # -- 内部 ---------------------------------------------------------------
    @property
    def step_in_unit(self) -> int:
        """当前 unit 内已跑了几步（unit 边界时为 0）。"""
        return self._step_in_unit

    @property
    def unit_steps(self) -> int:
        """当前 unit 一共几步（没有在飞的 unit 时为 0）。"""
        return self._unit_steps

    @property
    def at_safe_checkpoint_boundary(self) -> bool:
        """当前这一刻能否**安全地**存 Auto Learning checkpoint。

        条件（review v0.2 §14 Gate）：没有在飞的 unit（`_unit_steps == 0`）且
        `_step_in_unit == 0` 且 Scheduler 没有 pending request
        —— 即 **Scheduler 的账与模型权重完全对齐**。

        在这种边界存档，恢复时既不会「重跑已完成的 step」，也不会出现
        「模型领先 Scheduler 一个 unit」。
        """
        return (self._unit_steps == 0
                and self._step_in_unit == 0
                and self.scheduler.pending_train_request is None)

    def flush_partial_unit(self, global_step: int = 0) -> Optional[Dict[str, Any]]:
        """训练**收尾 / epoch 边界**用：把没跑满的 unit 按实际步数记账（或撤回）。

        正常路径下 unit 在最后一步就当场回填了（见 `on_step_end`）。但
        「max_steps 到顶 / STOP_AND_SAVE / epoch 切换」可能**正好停在 unit 中途**：

        * 已跑 k 步（k > 0）⇒ 模型已被更新 k 次 ⇒ **按 k 步回填**，Scheduler 与模型对齐；
        * 一步都没跑（k == 0）⇒ 模型没动 ⇒ **撤回** request，不改账。

        调用返回后 `at_safe_checkpoint_boundary` 必为 True，收尾存档因此是干净的。
        """
        if self.scheduler.pending_train_request is None:
            self._step_in_unit = 0
            self._unit_steps = 0
            self._unit_losses = []
            return None
        if self._step_in_unit <= 0:
            self.scheduler.cancel_pending_train_unit()
            self._step_in_unit = 0
            self._unit_steps = 0
            self._unit_losses = []
            self._log(
                "[auto_learning] 收尾：有一个 unit 已发布但 0 步未跑 ⇒ 撤回"
                "（模型未被更新，不改账）")
            return None
        ev = self._drain_pending_unit(global_step, allow_partial=True)
        self._log(
            "[auto_learning] 收尾：unit 没跑满 ⇒ 按**实际步数**记账"
            "（Scheduler 与模型权重对齐，不会重复训练）")
        return ev

    def _drain_pending_unit(self, global_step: int,
                            *, allow_partial: bool = False) -> Optional[Dict[str, Any]]:
        """把 sampler 的统计打包成 `TrainResult` 回填给 scheduler。"""
        if self.scheduler.pending_train_request is None:
            return None
        # 🔴 用 `stats_upto(_step_in_unit)` 而不是 `take_stats()`：
        #    `take_stats()` 会把**预取**的 batch 也算进来（num_workers>0 时一个 unit 能记到 34 步）。
        #    DataLoader 保序 ⇒ 第 k 个被消费的 batch = 第 k 个被产出的 batch。
        consumed = int(self._step_in_unit)
        stats: UnitStats = self.sampler.stats_upto(consumed)
        problems = stats.check_invariants()
        if problems:
            raise RuntimeError(
                "采样统计违反不变量（测试计划 §12 fail-fast）：\n  - " + "\n  - ".join(problems))
        if stats.steps != self._unit_steps:
            is_partial = allow_partial and stats.steps == consumed and consumed < self._unit_steps
            if not is_partial:
                raise RuntimeError(
                    f"本 unit 实际跑了 {stats.steps} 步，但 scheduler 要求 {self._unit_steps} 步。\n"
                    f"  常见原因：unit 内步数被提前 break / 训练循环结构与 hook 不匹配 /\n"
                    f"   unit 边界没重建 DataLoader 迭代器（prefetch 跨边界）。")
        result = TrainResult(
            loss=(sum(self._unit_losses) / len(self._unit_losses)) if self._unit_losses else 0.0,
            steps=stats.steps,
            samples_seen=stats.samples_seen,
            old_slot_counts=dict(stats.old_slot_counts_by_task),
            new_slot_counts=dict(stats.new_slot_counts_by_task),
            per_step_losses=list(self._unit_losses),
            batches_built=stats.steps,
            # 🔴 review v0.2 #9：按**真实 batch 组成**去重，不是「不同 loss 值数量」
            unique_batches=stats.unique_batches,
        )
        ev = self.scheduler.complete_train_unit(result, allow_partial=allow_partial)
        self._record(ev)
        self._unit_losses = []
        self._step_in_unit = 0
        self._unit_steps = 0
        return ev

    def _log(self, msg: str) -> None:
        if self.logger is None:
            return
        try:
            self.logger.info_rank0(msg)
        except Exception:  # noqa: BLE001 —— 日志失败不该打断训练
            pass

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
            #: 存档是否落在**安全边界**（Scheduler 账 == 模型权重）。
            #: 恢复时据此判定：True ⇒ 可安全 resume；False ⇒ fail-fast 拒绝静默重跑。
            "at_boundary": self.at_safe_checkpoint_boundary,
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
        self._resume_inflight_unit()

    def _resume_inflight_unit(self) -> None:
        """resume 后检查存档是否落在 **安全边界**（review v0.2 #2）。

        🔴 旧实现（B1 原始版）在这里「把本 unit 从头重跑」——
        模型已经做过 k 次 optimizer update，resume 后又额外做完整 N 次
        ⇒ **模型轨迹 / sample exposure / LR/global-step 语义都与不中断运行不同**，
        不是「丢 k 步算力」那么轻。

        现在改成 **只接受安全存档**：
          * `unit_steps == 0` ⇒ 干净边界，什么都不用做（新代码的存档都是这样）；
          * `unit_steps > 0 且 step_in_unit == 0` ⇒ unit 已发布但**一步未跑**，
            模型没有被更新 ⇒ 重发 request（无重复训练风险）；
          * `step_in_unit > 0` ⇒ **fail-fast**，绝不静默重跑、也不静默丢账。

        ⚠️ 判据同时看 `_unit_steps` 与 `_step_in_unit`：`step_in_unit == 0` 也可能是
        「unit 已开、一步未跑」——那同样不是安全边界。
        """
        if self._unit_steps <= 0 and self._step_in_unit <= 0:
            return
        if self._step_in_unit <= 0:
            # unit 已发布但**一步都没跑** ⇒ 模型没有因为本 unit 被更新 ⇒
            # 可以安全地重发 request（scheduler 状态是确定的，request 可复现）。
            self._unit_steps = 0
            self._unit_losses = []
            self._log(
                "[auto_learning] 恢复：有一个 unit 已发布但 0 步未跑 ⇒ 重发 request"
                "（模型没有被更新，不涉及重复训练）")
            self._open_next_unit()
            return
        raise RuntimeError(
            "Auto Learning checkpoint 落在 learning unit **中途**"
            f"（step_in_unit={self._step_in_unit}/{self._unit_steps}）"
            "—— 这是修复前的旧存档格式，恢复它必然要「重跑已完成的 step」，"
            "与不中断运行的模型轨迹不等价。\n"
            "  ⛔ 拒绝静默重跑。处理办法（二选一）：\n"
            "    ① 改用**上一个 unit 边界**的 checkpoint 恢复"
            "（新代码的存档都只在边界落盘：`at_safe_checkpoint_boundary`）；\n"
            "    ② 若确实要从零开始，请关掉 --train.enable_resume。")


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
