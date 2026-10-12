"""合并式 confirm（A 方案）的 CPU 契约测试 —— 2026-10-11 用户定。

背景（真机实测）
--------------
`confirm_val_ids = val_ids[:4]` 是 `scout_val_ids = val_ids[:2]` 的**超集** ⇒ 原先 confirm
把 4 条全重跑，其中 2 条与 scout 完全重复（白花一半算力），且"复核"用了初筛看过的样本。
改后：复用 scout 已测回合 + 只新评"scout 之外"的回合，再合并成并集判决值。

本文件锁三件事
-------------
1. **数值等价**：分两组合并的 `gmean_mse` == 一次性评并集的 `gmean_mse`（几何平均可分组）；
2. **只新评该评的**：`evaluate_confirm_merged` 只对非 scout 回合真推理，且判决条数 = 并集条数；
3. **边界**：空输入 / 含 0 值 / 无 baseline / key 冲突，都不崩且口径正确。
"""
from __future__ import annotations

import math

from lingbotvla.auto_learning.ports import TrajectoryMetrics
from lingbotvla.auto_learning.real.backend import RealEvaluator, merge_trajectory_metrics


def _tm(split, per, ids, *, mse=None, base=1.0, valid=True):
    per = {int(k): float(v) for k, v in per.items()}
    return TrajectoryMetrics(
        task='t', split=split, episode_ids=[int(i) for i in ids],
        per_traj_mse=per,
        mse=float(mse if mse is not None else (sum(per.values()) / len(per) if per else 0.0)),
        baseline_mse=base, nmse=None, r2=None, metric_valid=valid, n_trajs=len(per))


def test_merge_equals_single_shot_gmean():
    """分两组合并 == 一次性 4 条（几何平均可分组）。"""
    per = {101: 0.02, 102: 0.05, 103: 0.03, 104: 0.09}
    whole = _tm('confirm', per, [101, 102, 103, 104])
    a = _tm('scout', {101: 0.02, 102: 0.05}, [101, 102])
    b = _tm('confirm', {103: 0.03, 104: 0.09}, [103, 104])
    merged = merge_trajectory_metrics(a, b, split='confirm')
    assert merged.n_trajs == 4
    assert math.isclose(merged.gmean_mse, whole.gmean_mse, rel_tol=1e-12), (
        merged.gmean_mse, whole.gmean_mse)
    assert sorted(merged.episode_ids) == [101, 102, 103, 104]
    assert merged.split == 'confirm'


def test_merge_prefers_fresh_on_key_conflict():
    """同一回合出现在两组 ⇒ 以 extra（新算）为准，且不会被计两次。"""
    a = _tm('scout', {1: 0.5, 2: 0.5}, [1, 2])
    b = _tm('confirm', {2: 0.1, 3: 0.1}, [2, 3])
    merged = merge_trajectory_metrics(a, b, split='confirm')
    assert merged.per_traj_mse == {1: 0.5, 2: 0.1, 3: 0.1}
    assert merged.n_trajs == 3
    assert merged.episode_ids == [1, 2, 3]


def test_merge_edge_cases():
    """空输入 / 含 0 值 / 无 baseline 都不崩，且 nmse 口径正确。"""
    empty = _tm('confirm', {}, [])
    a = _tm('scout', {1: 0.2}, [1])
    m = merge_trajectory_metrics(a, empty, split='confirm')
    assert m.n_trajs == 1 and m.mse == 0.2

    zero = merge_trajectory_metrics(
        _tm('scout', {1: 0.0}, [1]), _tm('confirm', {2: 0.2}, [2]), split='confirm')
    assert zero.gmean_mse == 0.0, '含 0 值 ⇒ geometric_mse 返回 0.0（不是 None）'

    nobase = merge_trajectory_metrics(
        _tm('scout', {1: 0.2}, [1]), _tm('confirm', {2: 0.2}, [2], base=0.0),
        split='confirm', baseline_mse=0.0)
    assert nobase.nmse is None, '无 baseline ⇒ nmse=None（ratio 模式下不参与判定）'


class _Cat:
    def __init__(self, entry):
        self._e = entry

    def entry(self, task):
        return self._e


class _Entry:
    samples_by_traj = {101: [0] * 4, 102: [0] * 6, 103: [0] * 10, 104: [0] * 10}


class _Adapter:
    logger = None

    def __init__(self):
        self.calls = []
        self.logger = None

    def evaluate_task(self, task, split, ids):
        self.calls.append(list(ids))

        class _R:
            pass
        return _R()


class _SpyEvaluator(RealEvaluator):
    """只覆盖 evaluate()：记录被真评的回合并返回固定 MSE。"""

    def __init__(self, catalog):
        super().__init__(_Adapter(), catalog, None)
        self.seen = []

    def evaluate(self, task, split, episode_ids):
        ids = [int(i) for i in episode_ids]
        self.seen.append(ids)
        return _tm(split, {i: 0.01 * i for i in ids}, ids)


def test_evaluate_confirm_merged_only_evaluates_fresh_ids():
    """confirm 只对"非 scout"的回合真推理；判决条数 = 并集条数（4）。"""
    ev = _SpyEvaluator(_Cat(_Entry()))
    scout = _tm('scout', {101: 0.1, 102: 0.2}, [101, 102])
    merged = ev.evaluate_confirm_merged('t', [101, 102], [101, 102, 103, 104],
                                        scout=scout, split='confirm', baseline_mse=1.0)
    assert ev.seen == [[103, 104]], f'只该新评 103/104，实评 {ev.seen}'
    assert merged.n_trajs == 4
    assert sorted(merged.episode_ids) == [101, 102, 103, 104]
    # 帧加权 mse：4/6/10/10 帧
    exp = (0.1 * 4 + 0.2 * 6 + 1.03 * 10 + 1.04 * 10) / 30
    assert math.isclose(merged.mse, exp, rel_tol=1e-12), (merged.mse, exp)
    assert 'merged' in merged.note


def test_evaluate_confirm_merged_all_reused_skips_inference():
    """若 confirm 回合全在 scout 内 ⇒ 零推理，直接复用 scout 值。"""
    ev = _SpyEvaluator(_Cat(_Entry()))
    scout = _tm('scout', {101: 0.1, 102: 0.2}, [101, 102])
    merged = ev.evaluate_confirm_merged('t', [101, 102], [101, 102],
                                        scout=scout, split='confirm', baseline_mse=1.0)
    assert ev.seen == [], '不应有任何真推理'
    assert merged.n_trajs == 2
