"""hardness 扫描的**进度可见性**回归（CPU；2026-10-10 加）。

背景
----
`RealHardnessScorer.score()` 的批循环是 AL 里唯一没有输出的长循环：
单任务上千样本、每批 8 个 ⇒ 数百批、几十分钟静默。真机实测因此把"正在扫难度"
误判成"卡死"，白花约 40 分钟排查。

本用例用**桩 scorer + 桩数据集**（不碰 torch / 模型）验证：
  1. 扫描外层 `HardnessScanner.scan()` 打「扫描开始」与「扫描完成」；
  2. 内层逐批打「进度 x/y」与「打分完成」；
  3. 批次与样本数自洽（进度不丢样本、不重复计数）。

只依赖 `numpy`（仓库既有依赖）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lingbotvla.auto_learning.real.backend import RealHardnessScorer
from lingbotvla.auto_learning.sampling.hardness_scan import HardnessScanner


class RecLogger:
    """收集 `info_rank0` 消息（模拟仓库 logger 的 rank0 接口）。"""

    def __init__(self):
        self.lines: list[str] = []

    def info_rank0(self, msg: str) -> None:
        self.lines.append(str(msg))

    def warning(self, msg: str) -> None:  # noqa: D102
        self.lines.append('WARN ' + str(msg))

    def info(self, msg: str) -> None:  # noqa: D102
        self.lines.append(str(msg))


class StubCore:
    """桩：`score(items) -> np.ndarray(B,)`，只记录调用次数。"""

    def __init__(self):
        self.calls = 0
        self.seen = 0

    def score(self, items):  # noqa: ANN001
        self.calls += 1
        self.seen += len(items)
        return np.arange(len(items), dtype=float)


class StubDataset:
    """桩数据集：`dataset[j]` 返回带 joint_mask 的最小 item。"""

    def __getitem__(self, idx):  # noqa: ANN001
        return {'joint_mask': True, 'idx': int(idx)}


def _record(n_samples: int, trajs=None):
    trajs = trajs if trajs is not None else list(range(4))
    per = max(1, n_samples // max(1, len(trajs)))
    samples_by_traj = {}
    sid = 0
    for t in trajs:
        ids = []
        for _ in range(per):
            ids.append(sid)
            sid += 1
        samples_by_traj[t] = ids
    return SimpleNamespace(
        task_name='stub_task',
        train_traj_ids=list(trajs),
        sample_ids=list(range(sid)),
        hardness_version=0,
        samples_by_traj=samples_by_traj,
    )


def _cfg(fraction=1.0):
    return SimpleNamespace(hardness_probe_fraction=fraction,
                           difficulty_unscored_default=0.5,
                           hardness_weight_min=1.0,
                           hardness_weight_max=3.0,
                           hardness_alpha=2.0)


def test_score_emits_progress_and_completion():
    log = RecLogger()
    core = StubCore()
    # max_batch=4 ⇒ 40 样本 = 10 批
    scorer = RealHardnessScorer(core, StubDataset(), max_batch=4, logger=log)
    out = scorer.score('stub_task', list(range(40)))

    assert len(out) == 40, '打分结果必须覆盖全部样本'
    assert core.calls == 10, f'批数应为 10，实际 {core.calls}'
    assert core.seen == 40, '每批样本数之和必须等于总数（不重不漏）'

    joined = '\n'.join(log.lines)
    assert '[hardness] 开始打分' in joined, joined
    assert 'samples=40' in joined, joined
    assert '约 10 批' in joined, joined
    assert '[hardness] 打分完成' in joined, joined
    # 时间节流下仍必须有"首批"与"最后一批"的进度行
    assert '[hardness] 进度 4/40' in joined, joined
    assert '[hardness] 进度 40/40' in joined, joined
    assert '[最后一批]' in joined, joined
    # 完成行必须报告批次与总用时
    assert '批次=10' in joined, joined
    assert 'data ' in joined and 'score ' in joined, joined


def test_score_progress_is_monotonic_and_covers_all():
    log = RecLogger()
    scorer = RealHardnessScorer(StubCore(), StubDataset(), max_batch=3, logger=log)
    scorer.score('stub_task', list(range(30)))

    progress = [ln for ln in log.lines if ln.startswith('[hardness] 进度 ')]
    assert progress, '必须有进度行'
    # 形如 `[hardness] 进度 12/30（批 4）...`
    seen_vals = []
    for ln in progress:
        seg = ln.split('进度 ', 1)[1].split('/', 1)[0]
        seen_vals.append(int(seg))
    assert seen_vals == sorted(seen_vals), f'进度必须单调递增: {seen_vals}'
    assert seen_vals[-1] == 30, f'最后一行必须报满: {seen_vals}'


def test_scan_emits_start_and_done_with_counts():
    log = RecLogger()
    scorer = RealHardnessScorer(StubCore(), StubDataset(), max_batch=8, logger=log)
    scanner = HardnessScanner(_cfg(fraction=0.5), scorer)
    record = _record(n_samples=64, trajs=list(range(4)))

    scan = scanner.scan(record, SimpleNamespace(samples_by_traj=record.samples_by_traj))

    joined = '\n'.join(log.lines)
    assert '[hardness] 扫描开始' in joined, joined
    assert 'task=stub_task' in joined, joined
    assert '扫描完成' in joined, joined
    assert scan.scanned_sample_ids == sorted(scan.losses.keys()), 'scanned 与 losses 必须一一对应'
    assert len(scan.losses) == len(set(scan.losses)), '样本不得重复计入'
    assert 0 < scan.n_scanned <= len(record.sample_ids)
    # 外层"待打分样本 x/y 帧"与内层"开始打分 samples=" 必须一致
    outer = [ln for ln in log.lines if '待打分样本' in ln][0]
    inner = [ln for ln in log.lines if '[hardness] 开始打分' in ln][0]
    n_outer = int(outer.split('待打分样本 ', 1)[1].split('/', 1)[0])
    n_inner = int(inner.split('samples=', 1)[1].split(' ', 1)[0])
    assert n_outer == n_inner, f'外层/内层样本数不一致: {n_outer} vs {n_inner}'
