"""可选的阶段缓存复用与观测；本模块不导入 torch。

环境变量在进程启动时读取。默认 clear 保留现有行为；reuse 仅供空闲GPU
对照验证，不影响启动/收尾/DCP清理，也不替代OOM回退的强制清理。
"""
from __future__ import annotations

import os
import time


CACHE_POLICY = os.environ.get("AL_CUDA_CACHE_POLICY", "clear")
CACHE_TRACE = os.environ.get("AL_CUDA_CACHE_TRACE", "0") == "1"
if CACHE_POLICY not in ("clear", "reuse"):
    raise ValueError("AL_CUDA_CACHE_POLICY must be clear or reuse")

_REUSABLE_CONTEXTS = frozenset(("eval_entry", "periodic_hf"))


def _memory(cuda):
    try:
        free, total = cuda.mem_get_info()
        return {"allocated": cuda.memory_allocated(),
                "reserved": cuda.memory_reserved(), "free": free, "total": total}
    except Exception as exc:
        # 诊断失败不能阻止既有清理动作。
        return {"error": type(exc).__name__}


def prepare_cuda_cache(cuda, *, context, logger=None, collect_garbage=None, force=False):
    """保留GC，只选择是否释放CUDA空闲块；不访问模型、权重、RNG或峰值游标。

trace 时间是调用墙钟，可能包括等待此前的GPU工作；不额外 synchronize。
收益须按完整训练/评测耗时比较，不能把这个时间当纯分配器成本。
"""
    reuse = CACHE_POLICY == "reuse" and context in _REUSABLE_CONTEXTS and not force
    if not CACHE_TRACE:
        if collect_garbage is not None:
            collect_garbage()
        if cuda.is_available() and not reuse:
            cuda.empty_cache()
        return None

    available = cuda.is_available()
    before = _memory(cuda) if available else {}
    started = time.perf_counter()
    if collect_garbage is not None:
        collect_garbage()
    gc_seconds = time.perf_counter() - started
    cuda_started = time.perf_counter()
    if available and not reuse:
        cuda.empty_cache()
    cuda_seconds = time.perf_counter() - cuda_started
    report = {"context": context, "policy": CACHE_POLICY,
              "cleared": bool(available and not reuse), "gc_seconds": gc_seconds,
              "cuda_seconds": cuda_seconds, "before": before,
              "after": _memory(cuda) if available else {}}
    if logger is not None:
        import json
        try:
            logger.info_rank0("[cuda_cache] " + json.dumps(report, sort_keys=True))
        except Exception:
            pass  # 只观测，不让日志故障改变训练行为。
    return report
