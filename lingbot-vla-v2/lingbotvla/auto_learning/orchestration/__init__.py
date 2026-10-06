"""编排层 —— 主循环与入口。

    scheduler.py   Scheduler / SchedulerState：`advance()` 一次只做一个原子动作
    cli.py         命令行入口（`python -m auto_learning`）
    baselines.py   uniform multi-task 对照（文档 §45）

这一层只回答「按什么顺序调用下层」，不含任何判定规则 ——
判定在 `decision/`，采样在 `sampling/`，状态在 `state/`，世界在 `env/`。
"""

from .scheduler import Scheduler, SchedulerState

__all__ = ["Scheduler", "SchedulerState"]
