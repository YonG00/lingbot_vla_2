"""采样层 —— 决定「这一步喂哪些样本」。

    hardness.py   每帧难度 → 采样概率（文档 §19–§26）
    sampler.py    组 batch：7 NEW + 3 OLD（文档 §27）
    replay.py     PASS 池 + 两级 replay 采样（纯函数，文档 §28–§30）
    rng.py        显式 `random.Random` 的抽样小工具（可存档 / 恢复）

分工：任务层「学哪个」由 scheduler 决定；样本层「喂哪些帧」由这一层决定。
"""

from .hardness_scan import HardnessScan, HardnessScanner
from .replay import ReplayPlan, assign_slots, prepare_slots, sample_replay_refs
from .rng import WeightedSampler
from .sampler import BatchSampler

__all__ = [
    "HardnessScan",
    "HardnessScanner",
    "BatchSampler",
    "prepare_slots",
    "sample_replay_refs",
    "assign_slots",
    "WeightedSampler",
]
