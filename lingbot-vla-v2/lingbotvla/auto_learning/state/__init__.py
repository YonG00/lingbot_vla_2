"""`state` 子包（**惰性导出**）。

⚠️ 这里刻意**不**在模块顶层 eager import 子模块 —— 子模块之间存在环
（如 `sampling.hardness_scan` → `decision.metrics` → `decision.review` →
`state.registry` → `state.persistence` → `sampling.hardness_scan`），
急切导入会触发 "partially initialized module" 的 ImportError。
用 PEP 562 的 `__getattr__` 按需解析即可。
"""

from __future__ import annotations

_SUBMODULES = ('registry', 'persistence')


def __getattr__(name):
    import importlib

    for sub in _SUBMODULES:
        mod = importlib.import_module(f".{sub}", __name__)
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = []
