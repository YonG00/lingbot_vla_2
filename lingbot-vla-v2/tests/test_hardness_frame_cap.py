"""帧数上限压缩契约（2026-10-10 用户定）。

规则原话：「判断轨迹超出 300，就按这个 `np.linspace(0, n - 1, k)` 就小于等于 300，进行约束。」

⇒ `if n > cap: idx = np.linspace(0, n - 1, cap).astype(int)`：
   * 结果长度 = `min(n, cap)`；
   * **含首尾**（第 0 帧与最后一帧必被保留）；
   * **严格递增**（保序、无重复）；
   * `cap ≤ 0` 或 `n ≤ cap` ⇒ 原样返回（不改动）。
"""

from __future__ import annotations

import numpy as np
import pytest

from lingbotvla.auto_learning.sampling.hardness_scan import uniform_cap_frames

CAP = 300


@pytest.mark.parametrize('n', [1, 2, 10, 299, 300])
def test_below_or_equal_cap_is_untouched(n):
    src = list(range(n))
    out = uniform_cap_frames(src, CAP)
    assert out == src, '未超限不得改动（含恰好等于上限）'


@pytest.mark.parametrize('n', [301, 302, 459, 606, 928, 963, 1315, 10000])
def test_over_cap_is_compressed_to_cap(n):
    src = list(range(n))
    out = uniform_cap_frames(src, CAP)
    assert len(out) == CAP, f'n={n} 应压到 {CAP}，实际 {len(out)}'
    assert out[0] == src[0] and out[-1] == src[-1], '必须含首尾帧'
    assert all(out[i] < out[i + 1] for i in range(len(out) - 1)), '必须严格递增（保序无重复）'
    assert len(set(out)) == CAP, '不得有重复帧'
    assert all(0 <= i < n for i in out), '下标必须落在原范围内'


def test_uniform_spacing_is_close_to_ideal():
    """间隔应接近 `(n-1)/(cap-1)`（均匀抽样，而不是截断或只取前 300 帧）。"""
    n = 1315
    out = uniform_cap_frames(list(range(n)), CAP)
    gaps = np.diff(out)
    ideal = (n - 1) / (CAP - 1)
    assert abs(float(gaps.mean()) - ideal) < 1e-9
    assert int(gaps.max()) - int(gaps.min()) <= 1, f'间隔应只差至多 1，实际 {gaps.min()}~{gaps.max()}'


@pytest.mark.parametrize('cap', [0, -1, -300])
def test_zero_or_negative_cap_disables_compression(cap):
    src = list(range(1315))
    assert uniform_cap_frames(src, cap) == src, 'cap ≤ 0 表示关闭压缩'


def test_returns_plain_python_ints():
    """返回值必须是原生 int（下游要对 sample_id 做 json/键操作，不能是 np.int64）。"""
    out = uniform_cap_frames(list(range(1315)), CAP)
    assert all(type(i) is int for i in out)


def test_preserves_arbitrary_sample_ids_not_just_range():
    """真实 sample_id 不是连续下标（跨轨迹拼接），压缩必须按位置取样、返回原值。"""
    src = [1000 + 7 * i for i in range(500)]        # 步长 7 的假 sample_id
    out = uniform_cap_frames(src, CAP)
    assert len(out) == CAP
    assert out[0] == src[0] and out[-1] == src[-1]
    assert out == [src[i] for i in np.linspace(0, len(src) - 1, CAP).astype(int)]


def test_cap_two_keeps_only_ends():
    src = list(range(1000))
    assert uniform_cap_frames(src, 2) == [0, 999]


def test_empty_input_stays_empty():
    assert uniform_cap_frames([], CAP) == []
