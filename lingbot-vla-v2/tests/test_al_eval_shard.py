"""评测"轨迹分片 + 汇总"的 CPU 契约测试 —— 2026-10-11。

背景
----
训练态开环评测现状是"**每个 rank 都把整批轨迹各跑一遍**"（实测 v43 8 卡 / v44 7 卡：
每卡都跑 12/20 条），因为 FSDP 的 all-gather 是集合通信、必须全员参与。

但轨迹之间互不依赖、逐轨迹 MSE 与卡数无关（dp8 与 dp7 逐位相同）、聚合是**几何平均（可分组）**
⇒ 可以把轨迹切给各卡、`all_gather_object` 汇总。

🔴 本文件只能锁"**数学与索引**"，**锁不了集合通信安全**（真实 FSDP 错配会静默死锁）：
   那必须真机验证（见 `docs/auto_learning/AL_PENDING_CHANGES.md` 台账）。

锁定四件事
---------
1. `_eval_shard_indices`：**每卡恒 k = ceil(n/W) 条**（次数恒等是安全前提）、可越界回绕、覆盖全集；
2. `evaluate()` 分片后汇总的结果 == **单卡全量**结果（几何平均逐位一致，1e-12）；
3. 汇总**按 episode id 去重**（回绕重复的条目只算一次，不污染几何平均）；
4. 开关语义：`AL_EVAL_SHARD` 未设/0 ⇒ **原路径逐字不变**；=1 才分片。
"""
from __future__ import annotations

import math

import pytest

from lingbotvla.auto_learning.ports import EvalResult, TrajectoryMetrics
from lingbotvla.auto_learning.real import backend as B
from lingbotvla.auto_learning.real.backend import (RealEvaluator, _eval_shard_indices,
                                                   merge_trajectory_metrics)


# --------------------------------------------------------------------------- #
# 1) 分片索引规则
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('n,world', [(4, 7), (8, 7), (8, 8), (2, 8), (12, 7), (1, 8), (4, 4)])
def test_shard_indices_counts_and_coverage(n, world):
    k = -(-n // world)
    idxs = [_eval_shard_indices(n=n, world=world, rank=r) for r in range(world)]
    # ① 每卡恒定 k 条（次数恒等 ⇒ 集合通信对齐）
    assert all(len(x) == k for x in idxs), [len(x) for x in idxs]
    # ② 所有索引合法（回绕后仍在 [0, n)）
    assert all(0 <= i < n for x in idxs for i in x)
    # ③ 并集覆盖全集（否则有轨迹永远没被评）
    assert set().union(*idxs) == set(range(n)), sorted(set().union(*idxs))


def test_shard_indices_zero_and_guard():
    assert _eval_shard_indices(n=0, world=7, rank=0) == []
    assert _eval_shard_indices(n=4, world=0, rank=0) == []


# --------------------------------------------------------------------------- #
# 2/3) 汇总 == 单卡全量（含去重）
# --------------------------------------------------------------------------- #
class _Cat:
    def __init__(self, entry):
        self._e = entry

    def entry(self, task):
        return self._e


class _Entry:
    """每条回合的帧数不同 ⇒ 顺带覆盖"帧加权 mse"的合并路径。"""

    def __init__(self, n):
        self.samples_by_traj = {i: [0] * (10 + i) for i in range(n)}


class _FakeAdapter:
    """per-episode MSE 固定为 0.01*(ep+1)，与批大小无关（模拟真实确定性）。"""

    logger = None

    def __init__(self, n):
        self.n = n
        self.calls = []

    def evaluate_task(self, task, split, ids):
        self.calls.append([int(i) for i in ids])
        per_ids = [int(i) for i in ids]
        per = [0.01 * (i + 1) for i in per_ids]
        return EvalResult(task=task, split=split, episode_ids=per_ids,
                          mse=sum(per) / len(per) if per else 0.0,
                          per_traj_mse=per, per_traj_ids=per_ids,
                          n_traj=len(per_ids), baseline_mse=1.0)


def _make(n, world, monkeypatch, shard=True):
    ev = RealEvaluator(_FakeAdapter(n), _Cat(_Entry(n)), None)
    monkeypatch.setenv('AL_EVAL_SHARD', '1' if shard else '0')
    monkeypatch.setattr(B, '_dist_world_rank', lambda: (world, 0), raising=True)
    return ev


def test_single_rank_matches_sharded(monkeypatch):
    """分片汇总 == 单卡全量（n=8, W=8 与 W=7 两种；W=7 含回绕重复 ⇒ 必须去重）。"""
    for world in (8, 7):
        full = _make(8, 1, monkeypatch, shard=False)
        whole = full.evaluate('t', 'active_val', list(range(8)))

        # 模拟各 rank 的结果并 in-process 汇总（不打桩 dist：直接喂 payload）
        ev = RealEvaluator(_FakeAdapter(8), _Cat(_Entry(8)), None)
        payloads = []
        for r in range(world):
            idx = _eval_shard_indices(n=8, world=world, rank=r)
            local_ids = [idx_ for idx_ in idx]
            per = {i: 0.01 * (i + 1) for i in local_ids}
            payloads.append({'rank': r, 'ids': local_ids, 'per_traj': per,
                             'baseline_mse': 1.0, 'metric_valid': True})
        monkeypatch.setattr(B, '_all_gather_object', lambda p: payloads, raising=True)
        monkeypatch.setattr(B, '_dist_world_rank', lambda: (world, 0), raising=True)
        monkeypatch.setenv('AL_EVAL_SHARD', '1')
        got = ev.evaluate('t', 'active_val', list(range(8)))

        assert got.n_trajs == whole.n_trajs == 8, (got.n_trajs, world)
        assert sorted(got.episode_ids) == list(range(8))
        # 几何平均是小量（~0.045）⇒ 用"相对小量"的绝对容差，别用 rel_tol（近零量会失真）
        assert abs(got.gmean_mse - whole.gmean_mse) <= 1e-12 * max(1.0, abs(whole.gmean_mse))
        # mse 是**帧加权**量：合并路径按各回合帧数（10+i）加权 ⇒ 与 adapter 的算术平均不同，
        # 这是既有口径（`_weighted_mse`），不是分片引入的差异。这里按帧加权口径对照。
        fw = sum(0.01 * (i + 1) * (10 + i) for i in range(8)) / sum(10 + i for i in range(8))
        assert abs(got.mse - fw) <= 1e-12 * max(1.0, abs(fw)), (got.mse, fw)
        assert got.note.startswith('sharded:')


def test_switch_off_uses_original_path(monkeypatch):
    """未开开关 ⇒ 只评一次全量（原路径逐字不变）。"""
    ev = _make(4, 8, monkeypatch, shard=False)
    got = ev.evaluate('t', 'active_val', [0, 1, 2, 3])
    assert got.n_trajs == 4
    assert ev.adapter.calls == [[0, 1, 2, 3]], ev.adapter.calls


def test_switch_on_only_evaluates_local_shard(monkeypatch):
    """开开关 ⇒ 单 rank 视角只评自己那 k 条（不是全量）。"""
    ev = _make(4, 8, monkeypatch, shard=True)
    monkeypatch.setattr(B, '_dist_world_rank', lambda: (8, 2), raising=True)
    monkeypatch.setattr(B, '_all_gather_object', lambda p: [p], raising=True)
    ev.evaluate('t', 'active_val', [0, 1, 2, 3])
    assert ev.adapter.calls == [[2]], ev.adapter.calls   # k=ceil(4/8)=1 ⇒ rank2 评 idx2
