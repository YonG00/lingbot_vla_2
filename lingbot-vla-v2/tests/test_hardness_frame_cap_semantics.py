"""帧数压缩的**端到端语义**测试（2026-10-10 用户要求：「检查选中与没选中的概率值」）。

压缩只应改变「**被扫到哪些帧**」，因此必须守住三条不变量：

  I1. **未被选中的样本**（不在抽样轨迹里的、以及在轨迹里但被压缩掉尾部的）
      权重**恰好等于** `difficulty_unscored_default` 折算的默认权重；
  I2. **被选中的样本**权重要与"未压缩时按同一套规则算出的"一致
      （即：压缩不改变"被扫到的帧"的权重公式，只改变集合）；
  I3. **概率归一化后**：所有概率之和 = 1；未选中样本的概率 = `default_w / Σweights`；
      选中样本的概率严格按各自难度排序（loss 越大 → 概率越大）。

另外守住"关闭压缩（cap=0）时与旧行为逐值一致"（回归保护）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import pytest

from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.decision.metrics import difficulty_to_weight
from lingbotvla.auto_learning.ports import TaskEntry
from lingbotvla.auto_learning.sampling.hardness_scan import HardnessScanner
from lingbotvla.auto_learning.state.registry import TaskRecord

CAP = 300
FRACTION = 0.2


# --------------------------------------------------------------------------- #
# 桩：记录 -> 样本；scorer 返回"越靠后 loss 越大"的确定性值
# --------------------------------------------------------------------------- #
class _StubScorer:
    """per-sample loss 由 sample_id 决定（单调递增 ⇒ 难度可预测）。"""

    def __init__(self):
        self.logger = None
        self.calls: List[List[int]] = []

    def score(self, task: str, sample_ids):  # noqa: ANN001
        ids = [int(s) for s in sample_ids]
        self.calls.append(ids)
        return {s: 0.01 * i for i, s in enumerate(ids)}


def _make_world(*, traj_lens: List[int], cap: int, n_traj: int | None = None):
    """构造 (record, entry, scanner)；轨迹的 sample_id 连续编号。"""
    cfg = AutoLearningConfig(
        hardness_probe_fraction=FRACTION,
        hardness_max_frames_per_traj=cap,
        hardness_weight_min=1.0,
        hardness_weight_max=3.0,
        hardness_alpha=2.0,
        difficulty_unscored_default=0.5,
    )
    by_traj: Dict[int, List[int]] = {}
    samples: List[int] = []
    sid = 0
    for t, n in enumerate(traj_lens):
        ids = list(range(sid, sid + n))
        sid += n
        by_traj[t] = ids
        samples.extend(ids)
    rec = TaskRecord(task_name='toy', train_traj_ids=list(range(len(traj_lens))),
                     sample_ids=samples, hardness_version=0)
    entry = TaskEntry(name='toy', train_traj_ids=list(range(len(traj_lens))),
                      train_sample_ids=samples, samples_by_traj=by_traj)
    return rec, entry, HardnessScanner(cfg, _StubScorer())


def _default_weight(cfg: AutoLearningConfig) -> float:
    return difficulty_to_weight(cfg.difficulty_unscored_default, cfg.hardness_weight_min,
                                cfg.hardness_weight_max, cfg.hardness_alpha)


# --------------------------------------------------------------------------- #
# 核心：选中 vs 未选中
# --------------------------------------------------------------------------- #
def test_unselected_samples_get_default_weight_and_probability():
    """未选中样本：权重 = 默认权重；概率 = default_w / Σweights（且三者必然相等）。"""
    rec, entry, scanner = _make_world(traj_lens=[100, 120, 90, 110, 95], cap=CAP)
    scan = scanner.scan(rec, entry)
    cfg = scanner.cfg
    dw = _default_weight(cfg)

    selected = set(scan.scanned_sample_ids)
    unselected = [s for s in rec.sample_ids if s not in selected]
    assert selected and unselected, '本用例应同时存在选中与未选中样本'

    for s in unselected:
        assert scan.weights[s] == pytest.approx(dw), f'未选中样本 {s} 权重应等于默认值'
    total = sum(scan.weights.values())
    probs_unsel = {scan.probs[s] for s in unselected}
    assert len(probs_unsel) == 1, '所有未选中样本概率必须相同'
    assert probs_unsel.pop() == pytest.approx(dw / total, rel=1e-12)


def test_selected_samples_keep_difficulty_ordering():
    """选中样本：loss 越大 ⇒ 权重/概率越大（压缩不得打乱这条）。"""
    rec, entry, scanner = _make_world(traj_lens=[80, 100, 60, 90, 70], cap=CAP)
    scan = scanner.scan(rec, entry)
    pairs = sorted((scan.losses[s], scan.probs[s]) for s in scan.scanned_sample_ids)
    losses = [p[0] for p in pairs]
    probs = [p[1] for p in pairs]
    assert losses == sorted(losses) and all(probs[i] <= probs[i + 1] for i in range(len(probs) - 1)), \
        '概率必须随 loss 单调不减（压缩不得破坏排序）'


def test_probs_sum_to_one_and_are_positive():
    rec, entry, scanner = _make_world(traj_lens=[250, 400, 310, 120, 500], cap=CAP)
    scan = scanner.scan(rec, entry)
    assert set(scan.probs) == set(rec.sample_ids), '每个样本都必须有权重/概率'
    assert all(p > 0 for p in scan.probs.values())
    assert sum(scan.probs.values()) == pytest.approx(1.0, rel=1e-12, abs=1e-12)


# --------------------------------------------------------------------------- #
# 压缩本身的语义
# --------------------------------------------------------------------------- #
def test_compression_only_drops_tail_within_selected_trajectories():
    """被压缩掉的帧 = 选中轨迹中"未被 linspace 取到"的那些 ⇒ 它们必须是未选中（默认权重）。"""
    rec, entry, scanner = _make_world(traj_lens=[1000, 50, 50, 50, 50], cap=CAP)
    scan = scanner.scan(rec, entry)
    chosen = set(scan.scanned_traj_ids)
    assert 0 in chosen, 'fraction=0.2 应至少抽到一条轨迹'

    long_traj = entry.samples_by_traj[0]
    dropped = [s for s in long_traj if s not in set(scan.scanned_sample_ids)]
    assert len(dropped) > 0, '长轨迹必须被压缩（1000 > 300）'
    # 期望样本数 = Σ(选中轨迹的 min(帧数, cap))——按实际抽中哪些轨迹算，不写死
    expected = sum(min(len(entry.samples_by_traj[t]), CAP) for t in scan.scanned_traj_ids)
    assert len(scan.scanned_sample_ids) == expected
    dw = _default_weight(scanner.cfg)
    for s in dropped:
        assert scan.weights[s] == pytest.approx(dw), '被压缩掉的帧应按未评分默认权重处理'


def test_scanned_count_equals_cap_for_long_trajectory():
    rec, entry, scanner = _make_world(traj_lens=[1000, 60, 60, 60, 60], cap=CAP)
    scan = scanner.scan(rec, entry)
    long_len = sum(1 for s in scan.scanned_sample_ids if s in set(entry.samples_by_traj[0]))
    assert long_len == CAP, f'长轨迹应恰好扫 {CAP} 帧，实际 {long_len}'
    assert scan.n_scanned == len(scan.scanned_sample_ids)


def test_compression_does_not_affect_short_trajectories():
    """全部轨迹 ≤ cap ⇒ 与关闭压缩（cap=0）**逐值一致**（回归保护）。"""
    lens = [100, 120, 90, 110, 95]
    rec1, entry1, s1 = _make_world(traj_lens=lens, cap=CAP)
    rec2, entry2, s2 = _make_world(traj_lens=lens, cap=0)
    a, b = s1.scan(rec1, entry1), s2.scan(rec2, entry2)
    assert a.scanned_sample_ids == b.scanned_sample_ids
    assert a.weights == b.weights, '未触发压缩时权重必须逐值相同'
    assert a.probs == b.probs


def test_disabled_cap_keeps_all_frames_of_selected_trajectories():
    rec, entry, scanner = _make_world(traj_lens=[1000, 50, 50, 50, 50], cap=0)
    scan = scanner.scan(rec, entry)
    expected = sum(len(entry.samples_by_traj[t]) for t in scan.scanned_traj_ids)
    assert len(scan.scanned_sample_ids) == expected, 'cap=0 时必须扫全部帧'


# --------------------------------------------------------------------------- #
# 与配置交互
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('cap', [2, 5, 300])
def test_tiny_and_normal_caps_are_respected(cap):
    rec, entry, scanner = _make_world(traj_lens=[1000, 40, 40, 40, 40], cap=cap)
    scan = scanner.scan(rec, entry)
    long_len = sum(1 for s in scan.scanned_sample_ids if s in set(entry.samples_by_traj[0]))
    assert long_len == min(1000, cap)
    assert sum(scan.probs.values()) == pytest.approx(1.0, rel=1e-12)


def test_all_samples_accounted_for_in_weights():
    """权重表必须覆盖 `record.sample_ids` 全集（不重不漏）。"""
    rec, entry, scanner = _make_world(traj_lens=[500, 300, 301, 60], cap=CAP)
    scan = scanner.scan(rec, entry)
    assert sorted(scan.weights) == sorted(rec.sample_ids)
    assert sorted(scan.probs) == sorted(rec.sample_ids)
    assert math.isclose(sum(scan.probs.values()), 1.0, rel_tol=1e-12)
