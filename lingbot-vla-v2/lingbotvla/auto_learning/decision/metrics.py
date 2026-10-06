"""指标与判据（纯函数，无依赖）。

对应文档：
  §9   NMSE / R²
  §9.1 baseline≈0 的守卫
  §14.1 LP50（Learning Progress）
  §17.1 Overfit flag
  §23  difficulty → sampling weight
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EPS = 1e-12

#: baseline MSE 低于这个值就认为「不可用于 NMSE 排序」（文档 §9.1）
MIN_BASELINE_MSE = 1e-6


# --------------------------------------------------------------------------- #
# 基础指标
# --------------------------------------------------------------------------- #
def nmse(mse: float, baseline_mse: float) -> Optional[float]:
    """NMSE = MSE / BaselineMSE。baseline≈0 时返回 None（不可比）。"""
    if baseline_mse is None or baseline_mse < MIN_BASELINE_MSE:
        return None
    return float(mse) / float(baseline_mse)


def r2_from_nmse(value: Optional[float]) -> Optional[float]:
    """R² = 1 − NMSE。"""
    if value is None:
        return None
    return 1.0 - float(value)


def baseline_is_valid(baseline_mse: Optional[float]) -> bool:
    return baseline_mse is not None and baseline_mse >= MIN_BASELINE_MSE


def learning_progress(prev: Optional[float], cur: Optional[float]) -> Optional[float]:
    """LP = (V_prev − V_cur) / (V_prev + eps)。

    正数 = 变好；0 附近 = 停滞；负数 = 退步。
    """
    if prev is None or cur is None:
        return None
    return (float(prev) - float(cur)) / (float(prev) + EPS)


def forget_ratio(best: Optional[float], cur: Optional[float]) -> Optional[float]:
    """相对退化率 = (current − best) / (best + eps)。文档 §33。"""
    if best is None or cur is None:
        return None
    return (float(cur) - float(best)) / (float(best) + EPS)


def is_forgotten(
    current_nmse: Optional[float],
    best_nmse: Optional[float],
    pass_nmse: float,
    forget_relative_threshold: float,
) -> bool:
    """文档 §34：掉出及格线 **或** 相对退化超过阈值。"""
    if current_nmse is None:
        return False
    if current_nmse > pass_nmse:
        return True
    ratio = forget_ratio(best_nmse, current_nmse)
    return ratio is not None and ratio > forget_relative_threshold


def is_overfit(
    lp_train: Optional[float],
    lp_val: Optional[float],
    gap: Optional[float],
    gap_prev: Optional[float],
    *,
    train_lp_min: float,
    val_lp_max: float,
    gap_growth_threshold: float,
) -> bool:
    """文档 §17.1：

        Overfit = 1[ LP_T ≥ 5% ∧ LP_V ≤ 0 ∧ G_t ≥ 1.1·G_prev ]

    其中 G = (V + eps) / (T + eps)，即 train/val 差距。
    `gap_growth_threshold` 是**相对增长**阈值（默认 0.10 = 10%）。
    """
    if lp_train is None or lp_val is None or gap is None or gap_prev is None:
        return False
    if gap_prev <= EPS:
        gap_grew = gap > 0
    else:
        gap_grew = (gap - gap_prev) / gap_prev >= gap_growth_threshold
    return bool(lp_train >= train_lp_min and lp_val <= val_lp_max and gap_grew)


def train_val_gap(val_nmse: Optional[float], train_nmse: Optional[float]) -> Optional[float]:
    """G = (V + eps) / (T + eps)。"""
    if val_nmse is None or train_nmse is None:
        return None
    return (float(val_nmse) + EPS) / (float(train_nmse) + EPS)


# --------------------------------------------------------------------------- #
# Hardness → sampling probability（文档 §23–§24）
# --------------------------------------------------------------------------- #
def percentile_rank(values: Sequence[float]) -> List[float]:
    """把一组 loss 映射到 [0, 1] 的百分位（并列取平均秩）。

    这一步是关键：它让 `5000` 不会比 `3` 拿到高几千倍的采样概率。
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return [r / (n - 1) for r in ranks]


def difficulty_to_weight(difficulty: float, w_min: float, w_max: float, alpha: float) -> float:
    """w = w_min + (w_max − w_min)·d^α（文档 §23）。"""
    d = min(max(float(difficulty), 0.0), 1.0)
    return float(w_min) + (float(w_max) - float(w_min)) * (d ** float(alpha))


def normalize_weights(weights: Dict[int, float]) -> Dict[int, float]:
    """权重 → 概率。

    **非法输入一律报错，不静默兜底**（测试方案 §I06）：负权重 / NaN / 全 0
    都会让采样分布失去意义，静默兜底只会把问题推到更晚才暴露。
    """
    for key, value in weights.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"采样权重含 NaN/Inf（key={key}, value={value!r}）")
        if value < 0:
            raise ValueError(f"采样权重为负（key={key}, value={value}）")
    total = sum(weights.values())
    if not weights:
        raise ValueError("采样权重为空")
    if total <= EPS:
        raise ValueError("采样权重全为 0，无法构成分布")
    return {k: v / total for k, v in weights.items()}


# --------------------------------------------------------------------------- #
# 小统计工具（避免引入 numpy）
# --------------------------------------------------------------------------- #
def mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def median(values: Iterable[float]) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def stdev(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < EPS:
        return default
    return a / b


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def is_finite_metric(value: Optional[float]) -> bool:
    """指标是否可用于排序。

    NaN / Inf **绝不能进 `argmin`**（测试方案 §I01）—— 拿它比大小的结果是未定义的。
    非数值类型（字符串 / bool / None）同样判为不可用：指标就该是数字。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def spearman_rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """秩相关（Spearman）。

    用来验证「样本难度 → 经验采样频率」是否真的单调（测试方案 §D04）——
    比只看 `sum(prob)==1` 有意义得多。
    """
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    rx = percentile_rank(list(xs[:n]))
    ry = percentile_rank(list(ys[:n]))
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den < EPS:
        return 0.0
    return num / den


def cohen_h(delta_mean: float, s1: float, s2: float) -> float:
    """粗略效应量，用于报告 A/B 差异是否只是噪声。"""
    pooled = math.sqrt(max((s1 * s1 + s2 * s2) / 2.0, EPS))
    return delta_mean / pooled


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """成功率的小样本置信区间（Wilson）。用于「覆盖率」这种小 n 指标。"""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))
