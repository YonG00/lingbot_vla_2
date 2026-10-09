"""torch.compile 运行时调优（**默认安全、可 env 关**）。

背景（2026-10-09 实测，50-task GMean200 正式跑）
------------------------------------------------
那次 31 分钟的跑里，**真正的训练只有 10.8 分钟（35%）**；任务切换处出现过
**353 秒（5 分 53 秒）无日志空窗**（该窗口 220 行 `torch/_dynamo` 警告、**0 次评测**）。
两个已知成因：

1. **评测期翻转 dyno guard 监视的状态** —— `safe_eval_context` 每次评测临时改
   `config.use_cache` / `attention_implementation` / `_use_compile_predict_velocity` / 逐模块
   `.training`，进出各一次 ⇒ 编译图 guard 失效 ⇒ 重编译（外层是**整模型**图，极贵）。
   ⇒ 对策：评测区整体包 `torch.compiler.disable()`（评测本来就走 eager 热点路径，
   本来也不需要外层编译图）—— 见 :func:`eval_compile_disabled`。
2. **dynamo 的"重编译上限"太小** —— 默认 ``cache_size_limit=8``，超限后 dynamo
   **放弃编译、整段退回 eager**（表现：警告刷屏 + 之后每步变慢）。
   ⇒ 对策：提到 :data:`DEFAULT_CACHE_LIMIT`（64）—— 见 :func:`apply_dynamo_tuning`。

⚠️ 注意：``TORCHDYNAMO_CACHE_SIZE_LIMIT`` 这个环境变量在本项目用的 torch 2.8 上**不生效**
（实测仍为 8），所以必须走代码设置。

未做（等实测决定）：内层 ``dynamic=False`` → ``True``（动态形状，影响稳态性能，需 A/B）。
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, Optional

#: ``torch._dynamo.config.cache_size_limit`` 的目标值（默认 8 太小）。
DEFAULT_CACHE_LIMIT = 64

#: 覆盖 cache_size_limit 的环境变量。
CACHE_LIMIT_ENV = "AL_DYNAMO_CACHE_SIZE_LIMIT"

#: 评测期禁用编译的开关：``1``（默认）启用；``0`` 关闭（回到改动前行为）。
EVAL_DISABLE_ENV = "AL_EVAL_DISABLE_COMPILE"


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def apply_dynamo_tuning(*, logger: Optional[Any] = None) -> Dict[str, Any]:
    """把 ``torch._dynamo.config.cache_size_limit`` 提到安全值（默认 64）。

    返回 ``{"before":…, "after":…, "changed":bool, "env":…}`` 供日志/断言使用。
    **幂等**：重复调用只改一次。``AL_DYNAMO_CACHE_SIZE_LIMIT=0`` 可跳过（保持原值）。
    """
    info: Dict[str, Any] = {"env": CACHE_LIMIT_ENV, "before": None, "after": None,
                            "changed": False, "skipped": False}
    raw = os.environ.get(CACHE_LIMIT_ENV, str(DEFAULT_CACHE_LIMIT))
    try:
        want = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{CACHE_LIMIT_ENV} 必须是整数，收到 {raw!r}") from None
    if want == 0:
        info["skipped"] = True
        _log(logger, f"[compile] {CACHE_LIMIT_ENV}=0 ⇒ 保持 dynamo 默认 cache_size_limit")
        return info
    if want < 1:
        raise ValueError(f"{CACHE_LIMIT_ENV} 必须 >= 1（0 表示跳过），收到 {want}")
    try:
        from torch._dynamo import config as dynamo_config
    except Exception as exc:  # noqa: BLE001  （无 torch / 版本差异时不致命）
        info["skipped"] = True
        _log(logger, f"[compile] 无法访问 torch._dynamo.config（{type(exc).__name__}）⇒ 跳过调优")
        return info
    info["before"] = int(getattr(dynamo_config, "cache_size_limit", 0) or 0)
    if info["before"] != want:
        dynamo_config.cache_size_limit = want
        info["changed"] = True
    info["after"] = int(getattr(dynamo_config, "cache_size_limit", 0) or 0)
    _log(logger, f"[compile] dynamo cache_size_limit {info['before']} → {info['after']}"
                 f"（防止超限后整段退回 eager；{CACHE_LIMIT_ENV} 可覆盖，0=跳过）")
    return info


#: 首选机制：``torch.compiler.set_stance("force_eager")``（torch ≥2.6，**区域级**）。
_STANCE = "force_eager"


@contextlib.contextmanager
def _config_disable_cm():
    """兜底机制：临时置 ``torch._dynamo.config.disable``（全局开关，最后手段）。"""
    from torch._dynamo import config as dcfg
    before = bool(getattr(dcfg, "disable", False))
    dcfg.disable = True
    try:
        yield
    finally:
        dcfg.disable = before


def _pick_disable_factory():
    """按可用性挑一个"区域内不编译"的机制；返回 ``(名字, 工厂)``。

    torch 2.8 实测：``torch.compiler.disable()`` **不能**当上下文管理器用
    （``RuntimeError: torch._dynamo.optimize(...) is used with a context manager``），
    而 ``torch.compiler.set_stance("force_eager")`` 可以 ⇒ 优先它。
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return ("none", None)
    compiler = getattr(torch, "compiler", None)
    set_stance = getattr(compiler, "set_stance", None)
    if set_stance is not None:
        try:
            with set_stance(_STANCE):
                pass
            return ("compiler.set_stance(force_eager)", lambda: set_stance(_STANCE))
        except Exception:  # noqa: BLE001
            pass
    disable = getattr(compiler, "disable", None)
    if disable is not None:
        try:
            with disable():
                pass
            return ("compiler.disable", disable)
        except Exception:  # noqa: BLE001
            pass
    try:
        with _config_disable_cm():
            pass
        return ("dynamo.config.disable", _config_disable_cm)
    except Exception:  # noqa: BLE001
        return ("none", None)


_DISABLE_FACTORY = None          # 进程内只探一次


@contextlib.contextmanager
def eval_compile_disabled(*, logger: Optional[Any] = None):
    """评测区专用：区间内**不让 dynamo 编译**（消除评测对编译图 guard 的扰动）。

    为什么安全：评测本来就强制走 eager 热点路径（``_use_compile_predict_velocity=False``）
    且 ``torch.inference_mode()``；关掉的只是"外层编译图在评测模式下的重编译"。
    ``AL_EVAL_DISABLE_COMPILE=0`` 可恢复旧行为。
    """
    global _DISABLE_FACTORY
    if not _env_flag(EVAL_DISABLE_ENV, True):
        yield False
        return
    if _DISABLE_FACTORY is None:
        _DISABLE_FACTORY = _pick_disable_factory()
        _log(logger, f"[compile] 评测期禁编译机制 = {_DISABLE_FACTORY[0]}")
    name, factory = _DISABLE_FACTORY
    if factory is None:
        _log(logger, "[compile] 当前 torch 无可用机制 ⇒ 评测期不禁用编译")
        yield False
        return
    with factory():
        yield True


def _log(logger: Optional[Any], msg: str) -> None:
    for name in ("info_rank0", "info", "warning"):
        fn = getattr(logger, name, None)
        if callable(fn):
            try:
                fn(msg)
                return
            except Exception:  # noqa: BLE001
                continue
    print(msg, flush=True)
