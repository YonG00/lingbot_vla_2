"""状态层 —— scheduler 的全部可持久化状态。

    registry.py      TaskRecord × N：状态 / attempt / best_nmse / snapshot（文档 §38）
    persistence.py   save / restore（文档 §43.9）

为什么不能用 heatmap 代替 registry：resume 之后必须能回答
「这任务什么状态？还剩几次 attempt？历史最佳？有没有 pass snapshot？」——
少任何一项，恢复出来的路线就和原来不一样。
"""

from .persistence import load_state, save_state
from .registry import TaskRecord, TaskRegistry

__all__ = ["TaskRecord", "TaskRegistry", "save_state", "load_state"]
