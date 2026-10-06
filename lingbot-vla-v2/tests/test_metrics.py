"""指标与判据的单元测试。"""

from __future__ import annotations

import math

import pytest

from lingbotvla.auto_learning.decision.metrics import (
    MIN_BASELINE_MSE,
    difficulty_to_weight,
    forget_ratio,
    is_forgotten,
    is_overfit,
    learning_progress,
    nmse,
    normalize_weights,
    percentile_rank,
    r2_from_nmse,
    train_val_gap,
    wilson_interval,
)


def test_nmse_basic():
    assert nmse(0.05, 0.25) == pytest.approx(0.2)
    assert r2_from_nmse(0.2) == pytest.approx(0.8)


def test_nmse_invalid_baseline():
    """文档 §9.1：baseline≈0 不能拿 epsilon 硬修，要判为不可用。"""
    assert nmse(0.05, 0.0) is None
    assert nmse(0.05, MIN_BASELINE_MSE / 10) is None
    assert r2_from_nmse(None) is None


def test_learning_progress_sign():
    assert learning_progress(0.30, 0.24) > 0  # 变好
    assert learning_progress(0.30, 0.30) == pytest.approx(0.0)
    assert learning_progress(0.30, 0.33) < 0  # 退步
    assert learning_progress(None, 0.3) is None


def test_percentile_rank_kills_outliers():
    """文档 §24：raw loss 里一个 5000 不应该拿到几千倍的采样概率。"""
    ranks = percentile_rank([0.1, 0.2, 0.5, 3.0, 5000.0])
    assert ranks == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_percentile_rank_handles_ties():
    ranks = percentile_rank([1.0, 1.0, 1.0])
    assert ranks == pytest.approx([0.5, 0.5, 0.5])


def test_difficulty_to_weight_range():
    assert difficulty_to_weight(0.0, 1.0, 3.0, 2.0) == pytest.approx(1.0)
    assert difficulty_to_weight(1.0, 1.0, 3.0, 2.0) == pytest.approx(3.0)
    assert difficulty_to_weight(0.5, 1.0, 3.0, 2.0) == pytest.approx(1.5)
    # 越界也夹住
    assert difficulty_to_weight(9.0, 1.0, 3.0, 2.0) == pytest.approx(3.0)


def test_normalize_weights_rejects_illegal_input():
    """测试方案 §I06：非法概率要报错，不静默兜底。"""
    with pytest.raises(ValueError):
        normalize_weights({1: 0.0, 2: 0.0})  # 全 0
    with pytest.raises(ValueError):
        normalize_weights({1: -1.0, 2: 2.0})  # 负值
    with pytest.raises(ValueError):
        normalize_weights({1: float("nan"), 2: 1.0})  # NaN
    with pytest.raises(ValueError):
        normalize_weights({})  # 空


# --------------------------------------------------------------------------- #
# 测试方案 §4 Level A —— 逐条对齐
# --------------------------------------------------------------------------- #
def test_A01_nmse_is_not_clamped():
    """NMSE 可以 > 1、R² 可以 < 0 —— 绝不能 clamp 到 [0,1]。"""
    assert nmse(0.0, 1.0) == pytest.approx(0.0)
    assert nmse(1.0, 1.0) == pytest.approx(1.0)
    assert nmse(2.0, 1.0) == pytest.approx(2.0)
    assert nmse(2.0, 1.0) > 1.0
    assert r2_from_nmse(nmse(2.0, 1.0)) == pytest.approx(-1.0)
    assert r2_from_nmse(nmse(2.0, 1.0)) < 0


def test_A03_lp50_exact_values():
    """方案 §A03 的四个数值。方向写反是最容易犯的错。"""
    assert learning_progress(0.60, 0.48) == pytest.approx(0.20, abs=1e-4)
    assert learning_progress(0.60, 0.58) == pytest.approx(0.0333, abs=1e-4)
    assert learning_progress(0.60, 0.60) == pytest.approx(0.0, abs=1e-9)
    assert learning_progress(0.60, 0.66) == pytest.approx(-0.10, abs=1e-4)


def test_A05_overfit_requires_train_better_and_val_worse():
    """方案 §A05 的反例：train 变差时**不能**自动判 overfit。"""
    kw = dict(train_lp_min=0.05, val_lp_max=0.0, gap_growth_threshold=0.10)
    # train .60→.40（LP=+33%）、val .62→.68（LP=-10%）、gap 扩大 ⇒ overfit
    assert is_overfit(0.33, -0.10, 0.68 / 0.40, 0.62 / 0.60, **kw)
    # train .60→.65（LP=-8%）⇒ 不能因为 train 变差就判 overfit
    assert not is_overfit(-0.08, 0.19, 0.50 / 0.65, 0.62 / 0.60, **kw)


def test_A04_forgotten_three_cases():
    """方案 §A04 的三个例子（与 §F03–F05 同源）。"""
    assert is_forgotten(0.34, 0.20, 0.30, 0.30)  # 掉出线
    assert is_forgotten(0.20, 0.08, 0.30, 0.30)  # 仍及格但退化 150%
    assert not is_forgotten(0.22, 0.20, 0.30, 0.30)  # 小幅波动


def test_forget_ratio_and_is_forgotten():
    assert forget_ratio(0.10, 0.20) == pytest.approx(1.0)
    # 掉出及格线
    assert is_forgotten(0.34, 0.20, 0.30, 0.30)
    # 仍及格但相对退化明显
    assert is_forgotten(0.20, 0.08, 0.30, 0.30)
    # 都还好
    assert not is_forgotten(0.18, 0.17, 0.30, 0.30)


def test_overfit_flag():
    # train 在进步、val 没进步、gap 扩大 ⇒ overfit
    assert is_overfit(
        0.10, -0.01, 1.30, 1.00,
        train_lp_min=0.05, val_lp_max=0.0, gap_growth_threshold=0.10,
    )
    # val 也在进步 ⇒ 不算
    assert not is_overfit(
        0.10, 0.05, 1.30, 1.00,
        train_lp_min=0.05, val_lp_max=0.0, gap_growth_threshold=0.10,
    )
    # gap 没扩大 ⇒ 不算
    assert not is_overfit(
        0.10, -0.01, 1.02, 1.00,
        train_lp_min=0.05, val_lp_max=0.0, gap_growth_threshold=0.10,
    )


def test_train_val_gap():
    assert train_val_gap(0.2, 0.1) == pytest.approx(2.0, rel=1e-6)


def test_wilson_interval_small_n_is_wide():
    lo, hi = wilson_interval(5, 5)
    assert lo < 0.60 and hi == pytest.approx(1.0)
    lo2, hi2 = wilson_interval(50, 50)
    assert lo2 > lo
    assert hi2 - lo2 < hi - lo
