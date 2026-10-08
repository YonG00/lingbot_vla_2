"""评测精度口径（纯函数，供 open_loop_validation / 工具 / 单测共用）。

用户 2026-10-08 要求把三件事**分开**：

1. **模型权重存储 dtype**（训练产物，bf16 或 fp32）；
2. **评测推理 dtype**（`bf16` / `fp32`）：必须与权重实际 dtype 一致才会真的按该精度跑
   —— 不允许"按 fp32 推理完再 cast 结果"；
3. **指标计算 dtype**：**误差按 FP32 计算**、**聚合按 FP64**（跨帧/跨轨迹求和容易累积误差）。

另外：`action_fp32=True` 时模型自己会把动作头权重上转 fp32（`_fp32_linear`），
所以推理 dtype 必须是 fp32 —— 这是既有实现的优先级，本模块保持它。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

#: 指标口径标签（写进日志/结果，避免"配置 bf16 实际跑 fp32"这类不可见偏差）
METRIC_DTYPE_LABEL = "error=fp32,aggregate=fp64"

#: 允许的推理 dtype 取值
ALLOWED_INFERENCE_DTYPES = ("auto", "bf16", "fp32")


def _name(dtype) -> str:
    if dtype is None:
        return "unknown"
    return str(dtype).replace("torch.", "")


def resolve_inference_dtype(
    *,
    requested: str = "auto",
    action_fp32: bool = False,
    weight_dtype: Any = None,
    fallback_bf16: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """决定评测推理 dtype；返回 `(torch.dtype, 说明 dict)`。

    规则：
      * `action_fp32=True` ⇒ fp32（模型内部上转，优先级最高）；
      * `requested='auto'` ⇒ 跟随**权重实际 dtype**（拿不到就按 fp32，除非 fallback_bf16=True）；
      * `requested='bf16'|'fp32'` ⇒ 必须与"权重实际 dtype（含 action_fp32 影响）"一致，
        否则 **fail-fast**：绝不静默换精度（否则就是"配置 bf16 实际跑 fp32"）。
    """
    import torch

    req = str(requested or "auto").lower()
    if req not in ALLOWED_INFERENCE_DTYPES:
        raise ValueError(
            f"eval_inference_dtype 只支持 {ALLOWED_INFERENCE_DTYPES}，收到 {requested!r}")

    if action_fp32:
        effective = torch.float32
        reason = "action_fp32=True ⇒ 动作头在 fp32 上转，推理必须 fp32"
    elif weight_dtype is not None and getattr(weight_dtype, "is_floating_point", False):
        effective = weight_dtype
        reason = "跟随权重实际 dtype"
    else:
        effective = torch.bfloat16 if fallback_bf16 else torch.float32
        reason = "拿不到权重 dtype ⇒ 用保守默认"

    if req == "auto":
        resolved = effective
    else:
        want = torch.bfloat16 if req == "bf16" else torch.float32
        if want != effective:
            raise ValueError(
                f"评测推理 dtype 请求 {req}，但权重实际是 {_name(effective)}"
                f"（action_fp32={action_fp32}）⇒ 拒绝静默换精度："
                "要么让权重精度与请求一致，要么用 eval_inference_dtype=auto")
        resolved = want
        reason = f"显式请求 {req} 且与权重一致"

    return resolved, {
        "requested": req,
        "resolved": _name(resolved),
        "weight_dtype": _name(weight_dtype),
        "action_fp32": bool(action_fp32),
        "reason": reason,
        "metric": METRIC_DTYPE_LABEL,
    }


def describe_precision(info: Dict[str, Any]) -> str:
    """一行日志用文案。"""
    return (f"权重={info.get('weight_dtype')} | 推理={info.get('resolved')}"
            f"（requested={info.get('requested')}，reason={info.get('reason')}）"
            f" | 指标={info.get('metric', METRIC_DTYPE_LABEL)}")


def fp32_error_fp64_aggregation(pred, gt):
    """Compute physical-action error in FP32, then promote for FP64 reduction.

    Cast *before subtraction*: evaluating two float64 inputs and merely
    casting the resulting difference to float64 would silently violate the
    advertised `error=fp32` metric contract.
    """
    import numpy as np

    pr32 = np.asarray(pred, dtype=np.float32)
    gt32 = np.asarray(gt, dtype=np.float32)
    return np.asarray(pr32 - gt32, dtype=np.float64), np.asarray(gt32, dtype=np.float64)


def metric_arrays_for_aggregation(err, gt):
    """误差按 fp32 计算、聚合用 fp64：返回 (err64, gt64)。"""
    import numpy as np

    err64 = np.asarray(err, dtype=np.float64)
    gt64 = np.asarray(gt, dtype=np.float64)
    return err64, gt64
