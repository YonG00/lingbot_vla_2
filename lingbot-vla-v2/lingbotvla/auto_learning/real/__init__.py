"""真实仓库侧的适配与接线（Stage B1）。

| 模块 | 职责 |
|---|---|
| `sampler.py` | `AutoLearnSampler` —— 把 Stage A 的 `BatchSampler` 接到真实 `DataLoader`（7 NEW + 3 Replay） |
| `backend.py` | 用真实组件组装 `ports.Backend`（catalog / resolver / evaluator / scorer / trainer） |
| `hook.py` | 训练循环接线：unit 边界驱动 Scheduler + 重建迭代器 + `extra_state` |

`auto_learning.enabled=false` 时**一个对象都不创建**。
"""

__all__ = ["AutoLearnSampler", "UnitStats"]


def __getattr__(name):  # PEP 562：重模块按需加载
    if name in ("AutoLearnSampler", "UnitStats"):
        from .sampler import AutoLearnSampler, UnitStats
        return {"AutoLearnSampler": AutoLearnSampler, "UnitStats": UnitStats}[name]
    if name in ("build_real_backend", "RealTaskCatalog", "RealSampleResolver",
                "RealEvaluator", "RealHardnessScorer", "RealTrainer"):
        from . import backend
        return getattr(backend, name)
    if name in ("AutoLearnLoopHook", "StepDirective", "build_hook"):
        from . import hook
        return getattr(hook, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
