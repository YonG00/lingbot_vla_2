"""薄封装：把 `open_loop_validation` 的 **safe evaluation context** 暴露给 Auto Learning。

🔴 为什么必须有这一层
--------------------
`OpenLoopValidator.validate()` 现在把「RNG / training flag / use_cache / attention /
视觉网格缓存 / 图像增强」的 **snapshot → 强制 → finally 恢复 → 审计** 打包在
`safe_eval_context` 里。Auto Learning 的 `evaluate_task()` **必须**走同一个上下文 ——
否则评测会污染训练状态（`use_cache`、attention 接口、网格缓存、RNG 全部残留）。

本模块用**懒加载**避免在无 torch 环境下 import 失败。
"""

from __future__ import annotations

from typing import Any, Dict


def safe_eval_context(**kwargs: Any):
    """见 `lingbotvla.utils.open_loop_validation.safe_eval_context`（同一个实现）。"""
    from lingbotvla.utils.open_loop_validation import safe_eval_context as _impl
    return _impl(**kwargs)


def make_logger(verbose: bool = True) -> Any:
    """取仓库的统一 logger（带 `info_rank0` / `warning`）。"""
    from lingbotvla.utils import logging as _logging
    return _logging.get_logger("auto_learning")


class _FallbackLogger:
    """兜底 logger：`info_rank0` 在非 rank0 时静默。"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def info_rank0(self, msg: str, *a: Any) -> None:
        if self.verbose:
            print(msg, *a, flush=True)

    def info(self, msg: str, *a: Any) -> None:
        if self.verbose:
            print(msg, *a, flush=True)

    def warning(self, msg: str, *a: Any) -> None:
        print(f"WARNING: {msg}", *a, flush=True)


def resolve_logger(logger: Any = None, verbose: bool = True) -> Any:
    """给一个可用的 logger：优先用调用方给的，其次仓库 logger，最后兜底。"""
    if logger is not None:
        return logger
    try:
        return make_logger(verbose)
    except Exception:  # noqa: BLE001 —— 脱离仓库环境时（单测）用兜底
        return _FallbackLogger(verbose)


__all__ = ["safe_eval_context", "resolve_logger", "make_logger"]
