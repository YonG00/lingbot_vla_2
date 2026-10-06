"""LingBot-VLA-v2 的 Auto Learning 适配层（Stage B0）。

设计约束（见 `docs/stage_b0_adapter_design.md`）
------------------------------------------------
* **薄**：只把真实组件包成 5 个 Protocol 的实现，不重写 preprocess /
  open-loop evaluator / checkpoint。
* **零侵入**：`auto_learning.enabled=false` 时**不改变**原训练行为。
* **不接 Scheduler**（B0）：本包只提供接口与 Fixed Baseline 工具。
* **重依赖懒加载**：`torch` / `lerobot` 只在真正调用时导入 ⇒
  `catalog` / `resolver` / `ports` 可以在**没有 GPU 环境**下单独导入与测试。

用法
----
    from lingbotvla.auto_learning.catalog import TaskCatalog
    from lingbotvla.auto_learning.resolver import SampleResolver
    from lingbotvla.auto_learning.baseline import FixedBaseline, BaselineStore
    from lingbotvla.auto_learning.evaluator import EvaluatorAdapter   # 需要 torch
    from lingbotvla.auto_learning.hardness import HardnessScorer      # 需要 torch
"""

# 只 re-export **纯 stdlib** 的 DTO/Protocol，保证 `import lingbotvla.auto_learning` 廉价。
from .ports import (
    Backend,
    EvalResult,
    SampleRef,
    TaskEntry,
)

__all__ = [
    "Backend",
    "EvalResult",
    "SampleRef",
    "TaskEntry",
]
