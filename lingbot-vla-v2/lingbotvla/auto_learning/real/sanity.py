"""训练关键字段的**运行时审计**与 **sanity fail-fast**（规范 §39 / §40）。

§39 —— 前几个 step / 第一次 task transition / resume 后第一步各打一次 schema：
    actions / state / joint_mask / action_is_pad 的 shape、dtype、valid count
    ⇒ 快速抓「某字段丢了 / mask 全 0 / padding 全 True / action 归一化错了」这类静默 bug。

§40 —— 基本 sanity，遇到就 fail-fast（**不静默继续训练**）：
    loss 非有限 / mask 全 0 / baseline≈0 / GT 与 pred 帧数不一致

⚠️ 只依赖 numpy（shape/dtype/取值），**不 import torch** ⇒ 无卡可测。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

#: 审计哪些键（缺失不报错，只记录 absent）
AUDIT_KEYS = (
    "actions", "action", "state", "states", "joint_mask", "action_is_pad",
    "input_ids", "lang_tokens", "lang_masks", "img_masks", "images", "image_grid_thw",
)


def _to_np(v: Any) -> Optional[np.ndarray]:
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v
    try:
        return np.asarray(v)
    except Exception:  # noqa: BLE001
        return None


def _describe(v: Any) -> Dict[str, Any]:
    """一个张量的 shape / dtype / 有效值个数。"""
    a = _to_np(v)
    if a is None:
        return {"present": True, "kind": type(v).__name__}
    info: Dict[str, Any] = {
        "present": True,
        "shape": tuple(int(x) for x in np.shape(a)),
        "dtype": str(getattr(a, "dtype", "?")),
        "numel": int(np.size(a)),
    }
    if a.size and np.issubdtype(a.dtype, np.number):
        finite = np.isfinite(a)
        info["finite"] = int(finite.sum())
        info["nonzero"] = int(np.count_nonzero(a))
        info["min"] = float(np.nanmin(a))
        info["max"] = float(np.nanmax(a))
    return info


def audit_batch_schema(batch: Dict[str, Any]) -> Dict[str, Any]:
    """§39：对一批 batch 做 schema 审计；返回可 JSON 序列化的报告。

    * mask 类（`joint_mask` / `action_is_pad` / `lang_masks` / `img_masks`）额外给 `valid`
      （非 0 个数）—— 全 0 是最危险的静默 bug。
    * 数值类额外给 `finite` / `min` / `max` —— 一眼看出 NaN/Inf 或「未归一化」。
    """
    rep: Dict[str, Any] = {}
    for key in AUDIT_KEYS:
        if key not in batch:
            continue
        info = _describe(batch[key])
        if "nonzero" in info and ("mask" in key or key == "action_is_pad"):
            info["valid"] = info["nonzero"]
        rep[key] = info
    rep["_keys_present"] = sorted(k for k in batch if not str(k).startswith("_"))
    return rep


def format_schema_audit(rep: Dict[str, Any], *, indent: str = "    ") -> str:
    """把审计报告格式化成一行行的人类可读日志。"""
    lines = [f"batch schema audit（{len(rep.get('_keys_present', []))} 个键）"]
    for key, info in rep.items():
        if key.startswith("_"):
            continue
        bits = [f"shape={info.get('shape')}", f"dtype={info.get('dtype')}"]
        if "valid" in info:
            bits.append(f"valid={info['valid']}/{info.get('numel')}")
        if "min" in info:
            bits.append(f"range=[{info['min']:.4g},{info['max']:.4g}]")
        if info.get("finite") is not None and info["finite"] != info.get("numel"):
            bits.append(f"⚠️ nonfinite={info['numel'] - info['finite']}")
        lines.append(indent + f"{key}: " + "  ".join(bits))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# §40 sanity（fail-fast）
# --------------------------------------------------------------------------- #
def assert_batch_sane(batch: Dict[str, Any], *, mask_key: str = "joint_mask") -> None:
    """训练关键字段 sanity：mask 不能全 0、数值不能 NaN/Inf。"""
    if mask_key in batch:
        a = _to_np(batch[mask_key])
        if a is None:
            raise RuntimeError(f"[sanity] `{mask_key}` 无法转成数组（类型 {type(batch[mask_key])}）")
        if a.size == 0:
            raise RuntimeError(f"[sanity] `{mask_key}` 为空张量")
        if int(np.count_nonzero(a)) == 0:
            raise RuntimeError(
                f"[sanity] `{mask_key}` 全 0 ⇒ 训练目标为空（有效 action 数 = 0）。"
                "这通常意味着 padding/mask 算错了，**不要继续训练**。")
    for key in ("actions", "action", "state", "states"):
        if key not in batch:
            continue
        a = _to_np(batch[key])
        if a is None or not np.issubdtype(a.dtype, np.floating):
            continue
        if not np.isfinite(a).all():
            bad = int(np.size(a) - np.isfinite(a).sum())
            raise RuntimeError(f"[sanity] `{key}` 含 {bad} 个 NaN/Inf")


def assert_loss_sane(loss: Any, loss_log: Optional[Dict[str, Any]] = None) -> None:
    """loss 必须有限；`batch_mean_losses` 若存在也必须有限。"""
    def _f(v):
        try:
            return float(v)
        except Exception:  # noqa: BLE001
            return None

    lv = _f(loss)
    if lv is None:
        return
    if not math.isfinite(lv):
        raise RuntimeError(f"[sanity] loss 非有限: {lv}")
    if loss_log and "batch_mean_losses" in loss_log:
        a = _to_np(loss_log["batch_mean_losses"])
        if a is not None and a.size and not np.isfinite(a).all():
            raise RuntimeError(
                f"[sanity] batch_mean_losses 含 NaN/Inf（{a}）—— hardness/统计不可信")


def assert_metrics_sane(*, mse: float, baseline_mse: float,
                        gt_frames: Optional[int] = None,
                        pred_frames: Optional[int] = None,
                        eps: float = 1e-8) -> None:
    """评测指标 sanity：分母不能≈0、GT/pred 帧数必须一致、mse 必须有限。"""
    if not math.isfinite(float(mse)):
        raise RuntimeError(f"[sanity] mse 非有限: {mse}")
    if not math.isfinite(float(baseline_mse)) or float(baseline_mse) <= eps:
        raise RuntimeError(
            f"[sanity] Fixed BaselineMSE≈0（{baseline_mse}）⇒ NMSE 会是 Inf/NaN。"
            "检查 baseline 是否算在空/常量 GT 上。")
    if gt_frames is not None and pred_frames is not None and gt_frames != pred_frames:
        raise RuntimeError(
            f"[sanity] GT/pred 帧数不一致：gt={gt_frames} vs pred={pred_frames} —— "
            "**不要静默截断**（旧实现就是这么错的）。")


__all__ = ["AUDIT_KEYS", "audit_batch_schema", "format_schema_audit",
           "assert_batch_sane", "assert_loss_sane", "assert_metrics_sane"]
