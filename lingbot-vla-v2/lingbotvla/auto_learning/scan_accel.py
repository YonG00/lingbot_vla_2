"""Fail-closed, opt-in scan acceleration decisions (no implicit GPU probe)."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


@dataclass
class HardnessAutoBatch:
    """Conservative stateful upper-bound, starting from the existing batch 8.

    Never catches CUDA OOM and never changes scoring/randomness.  Runtime
    free/peak measurements determine whether trying one higher batch is safe.
    """
    current: int = 8
    maximum: int = 16
    reserve_gib: float = 10.0
    increments: tuple[int, ...] = (1, 2, 4, 8, 12, 16)
    last_extra_gib: float = 0.0

    def __post_init__(self):
        if (self.current < 1 or self.maximum < self.current
                or not math.isfinite(self.reserve_gib) or self.reserve_gib < 8):
            raise ValueError('invalid hardness GPU batch limits')

    def observe(self, *, free_before_gib: float, peak_free_gib: float,
                free_after_gib: float, processed: int) -> int:
        if not all(math.isfinite(v) and v >= 0 for v in
                   (free_before_gib, peak_free_gib, free_after_gib)):
            self.current = 1
            return self.current
        # Peak free must be sampled during the forward (not only after).
        measured_extra = max(0.0, free_before_gib - peak_free_gib)
        self.last_extra_gib = measured_extra
        safe_free = min(peak_free_gib, free_after_gib)
        if safe_free < self.reserve_gib:
            smaller = [v for v in self.increments if v < self.current]
            self.current = max(smaller) if smaller else 1
            return self.current
        larger = [v for v in self.increments if self.current < v <= self.maximum]
        if larger and processed >= self.current:
            nxt = min(larger)
            # Conservative estimate of incremental allocation for the next size.
            projected = max(2.0, measured_extra * (nxt / self.current - 1.0) * 2.0)
            if safe_free >= self.reserve_gib + projected:
                self.current = nxt
        return self.current


def identical_tensor_shapes(items: Sequence[dict], keys: Sequence[str]) -> bool:
    """No padding guessed: batch only homogeneous preprocessed observations."""
    if len(items) < 2:
        return False
    for key in keys:
        vals = [it.get(key) for it in items]
        if any(v is None or not hasattr(v, 'shape') for v in vals):
            return False
        if len({tuple(v.shape) for v in vals}) != 1:
            return False
    grids = [it.get('image_grid_thw') for it in items]
    if any(g is None for g in grids):
        return all(g is None for g in grids)
    return len({tuple(g.shape) for g in grids}) == 1


def normalized_action_predictions(predictions: Sequence[dict], action_keys: Sequence[str]):
    """Numerical-parity payload; rejects missing keys and nonnumeric values."""
    out = []
    for pred in predictions:
        item = {}
        for key in action_keys:
            if key not in pred:
                raise ValueError(f'missing action prediction: {key}')
            value = pred[key]
            if hasattr(value, 'detach'):
                value = value.detach().float().cpu().numpy()
            if hasattr(value, 'tolist'):
                value = value.tolist()
            item[key] = value
        out.append(item)
    return out


def action_diffs(ref, cand):
    """逐元素比较两组已归一化动作，返回 (max_abs_diff, mean_abs_diff)；不可比 ⇒ (None, None)。

    结构不匹配（key 集合/层级不同）⇒ 直接 (None, None)：数值相同也不算可比。
    """
    try:
        from .eval_batch_policy import same_structure as _ss
        if not _ss(ref, cand):
            return (None, None)
        a = [float(x) for x in _flatten(ref)]
        b = [float(x) for x in _flatten(cand)]
    except Exception:
        return (None, None)
    if not a or len(a) != len(b):
        return (None, None)
    d = [abs(x - y) for x, y in zip(a, b)]
    return (max(d), sum(d) / len(d))


def _flatten(obj):
    import numpy as _np
    if obj is None:
        return []
    if isinstance(obj, dict):
        # normalized_action_predictions 返回 [ {action_key: array} ] ⇒ 必须按 key 排序打平，
        # 否则 np.asarray(list_of_dicts) 得到 object 数组、float() 抛错 ⇒ 差值恒为 None
        # （2026-10-09 GPU 诊断实测：max|Δ| 显示为哨兵 -1e0）。
        out = []
        for _k in sorted(obj):
            out.extend(_flatten(obj[_k]))
        return out
    if isinstance(obj, (list, tuple)):
        # 关键：normalized_action_predictions 返回 **list[dict]** ⇒ 必须逐元素递归，
        # 否则 np.asarray(list_of_dicts) 得到 object 数组、float() 抛错 ⇒ 差值恒 None。
        out = []
        for _v in obj:
            out.extend(_flatten(_v))
        return out
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu().numpy()
    arr = _np.asarray(obj).reshape(-1)
    return arr.tolist()


def append_json_record(path, record) -> str:
    """把一条记录并入 JSON 报告文件（不存在则创建；顶层为 {"records": [...]}）。"""
    import json as _json
    import os as _os
    data = {"records": []}
    if _os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = _json.load(fh)
            if isinstance(loaded, dict) and isinstance(loaded.get("records"), list):
                data = loaded
        except Exception:
            data = {"records": []}
    data["records"].append(record)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        _json.dump(data, fh, ensure_ascii=False, sort_keys=True, indent=1)
    _os.replace(tmp, path)
    return path
