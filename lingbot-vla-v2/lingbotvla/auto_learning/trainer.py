"""TrainerAdapter —— **B0 只声明接口**，不接 Scheduler、不改训练 sampler。

B0 的产出
---------
1. 确认真实 Trainer **能**被抽象成 ``train_steps(req, num_steps) -> TrainResult``
2. 给出以后「接收外部 NEW + Replay sample indices」的**最小改造方案**（见下）

最小改造方案（B1 执行，**不动 collator / model / 循环结构**）
------------------------------------------------------------
* 新增 ``AutoLearnSampler(torch.utils.data.Sampler[int])``：每步产出
  ``new_slots + replay_slots`` 个 **local_idx**（默认 7 NEW + 3 Replay）
* 必须实现 ``state_dict()`` / ``load_state_dict()`` —— `StatefulDataLoader` 依赖它做 resume
* `build_dataloader(..., sampler=None)` 加一个**透传参数**（默认 None = 原行为）
* 训练循环在 ``auto_learning.enabled`` 时把 sampler 传进去
* 前置条件：``auto_learning.enabled`` 时**强制 rmpad=false**
  （`DynamicBatchSizeDataLoader` 会按 token 重排装箱，打散 7+3 的 slot 语义）

影响范围（B1）
-------------
    lingbotvla/data/data_loader.py         +1 个可选参数（默认 None，零行为变化）
    tasks/vla/train_lingbotvla.py          开关分支 + extra_state 追加一个 key
    lingbotvla/auto_learning/sampler.py    新增
"""

from __future__ import annotations

from typing import Any

from .ports import TrainRequest, TrainResult


class TrainerAdapter:
    """B0 的占位实现：接口在位、调用即报错，避免被误当成已完成。"""

    NOT_IMPLEMENTED_MSG = (
        "TrainerAdapter 在 Stage B0 只声明接口，**未实现**。\n"
        "  真实实现属于 B1（AutoLearnSampler + 训练循环接线 + extra_state 持久化）。\n"
        "  见 docs/stage_b0_adapter_design.md §5.5。"
    )

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(self.NOT_IMPLEMENTED_MSG)

    def train_steps(self, req: TrainRequest, num_steps: int) -> TrainResult:
        raise NotImplementedError(self.NOT_IMPLEMENTED_MSG)

    @staticmethod
    def minimal_change_plan() -> str:
        """返回 B1 的最小改造清单（供设计文档 / review 引用）。"""
        return (
            "1) 新增 AutoLearnSampler（每步 7 NEW + 3 Replay，local_idx 级别）\n"
            "2) build_dataloader(..., sampler=None) 透传\n"
            "3) auto_learning.enabled 时强制 rmpad=false，并注入 sampler\n"
            "4) extra_state['auto_learning'] 追加 registry / round / current_task /\n"
            "   attempt / sampler_state / pass_snapshot / baseline_fingerprint\n"
            "5) 不改 checkpointer、不改 collator、不改 model forward"
        )


__all__ = ["TrainerAdapter"]
