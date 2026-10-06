"""环境层 —— ★ Stage A 的「假世界」，**Stage B 唯一需要替换的一层**。

    catalog.py      TaskCatalog + SampleResolver（唯一知道 sample_id 编码的地方）
    sim.py          假世界 + 四个 adapter 实现（World / Evaluator / Scorer / Trainer）
    fake_tasks.py   11 个具名假任务 + 随机任务生成（测试素材，方案 §3 §14）
    scenarios.py    5 套端到端场景（测试与 CLI 共用，方案 §13）

**Stage B 的替换契约在 `auto_learning/ports.py`**（TaskCatalog / SampleResolver /
Evaluator / Trainer / HardnessScorer 五个 Protocol）—— `orchestration/scheduler.py`
只依赖它们，换一份 backend 实现即可，scheduler 一行都不用改。

接入前要先确认的三件事（文档 §43）：sample_id 是否稳定、
per-sample loss 能否拿到 reduction 之前的 `[B]`、`sample_actions()` 在 FSDP 下是否安全。
"""

from .catalog import SimSampleResolver, SimTaskCatalog
from .sim import SimulatedEvaluator, SimulatedWorld, stable_u01

__all__ = [
    "SimTaskCatalog",
    "SimSampleResolver",
    "SimulatedWorld",
    "SimulatedEvaluator",
    "stable_u01",
]
