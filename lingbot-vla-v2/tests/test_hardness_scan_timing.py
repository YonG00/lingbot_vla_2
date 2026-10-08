"""Hardness Scan 计时埋点的回归测试（不改变扫描行为 / 不新增常驻张量）。

设计要点（对应审查要求）：
* **首次 vs 稳态分离**：首次扫描含 train-mode 的 torch.compile/预热（首跑实测该窗口 369 s，
  大部分是编译）⇒ 分别写 `..._seconds_warmup` / `..._seconds_steady`，避免把编译算进难度扫描成本。
* **CUDA 异步**：只在扫描**前后各同步一次**（`_cuda_sync_for_timing`），绝不逐样本同步。
* **行为不变**：计时打开/关闭时，扫描结果（probs/losses/样本 ID）、调度决策、RNG 状态必须一致。
* **不新增常驻张量**：scheduler.py 不在模块级 import torch；同步助手在无 CUDA 时是 no-op。
"""
from __future__ import annotations

import inspect
import random
import re

import pytest

from al_fixtures import scheduler_of
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.decision.thresholds import PassThresholds
from lingbotvla.auto_learning.orchestration import scheduler as sch
from lingbotvla.auto_learning.orchestration.scheduler import _cuda_sync_for_timing
from lingbotvla.auto_learning.real.build import SchedulerLoggerAdapter
from lingbotvla.auto_learning.testing.fake_tasks import cfg_with


class RepoLog:
    def info_rank0(self, *a, **k): pass
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


class Writer:
    def __init__(self): self.points = []
    def add_scalar(self, name, value, step): self.points.append((name, float(value), int(step)))


def _run(monkeypatch=None):
    cfg = AutoLearningConfig(seed=3, eval_interval_steps=5, min_steps_before_defer=5,
                             defer_retry_steps=5, pass_metric="mse",
                             pass_thresholds_file="x", max_attempts_per_task=2)
    cfg.pass_thresholds = PassThresholds(tasks={"easy_pass": 0.10, "unlearnable": 0.10})
    w = Writer()
    lg = SchedulerLoggerAdapter(RepoLog(), writer=w)
    lg.set_tb_step_offset(train_global_step=500, al_global_step=0)
    sched = scheduler_of(cfg_with(["easy_pass", "unlearnable"], al=cfg), logger=lg)
    sched.run(max_actions=60)
    return sched, w.points


TIMING_TAGS = ("auto_learning/hardness_scan_seconds",
               "auto_learning/hardness_scan_seconds_warmup",
               "auto_learning/hardness_scan_seconds_steady",
               "auto_learning/hardness_scan_samples",
               "auto_learning/hardness_scan_trajs")


def test_warmup_and_steady_are_separated():
    sched, points = _run()
    warm = [p for p in points if p[0].endswith("hardness_scan_seconds_warmup")]
    steady = [p for p in points if p[0].endswith("hardness_scan_seconds_steady")]
    raw = [p for p in points if p[0].endswith("hardness_scan_seconds")]
    assert len(warm) == 1, f"首次（含 compile）应恰好一条，实际 {len(warm)}"
    assert len(steady) == len(sched.scans) or len(steady) >= 1, "后续扫描应记 steady"
    assert len(raw) == len(warm) + len(steady)
    # 首次一定排在 steady 之前（step 不晚于）
    assert warm[0][2] <= min(p[2] for p in steady)


def test_samples_and_trajs_are_logged():
    _sched, points = _run()
    samples = [p for p in points if p[0].endswith("hardness_scan_samples")]
    trajs = [p for p in points if p[0].endswith("hardness_scan_trajs")]
    assert samples and trajs
    assert all(v > 0 for _t, v, _s in samples)
    assert all(v >= 1 for _t, v, _s in trajs)


def test_timing_does_not_change_scan_results_or_axis(monkeypatch):
    """把 `perf_counter` 换成常量（等价于"计时关闭"）后，除时间值外一切必须一致。"""
    sched_a, pts_a = _run()
    real_pc = sch.time.perf_counter
    monkeypatch.setattr(sch.time, "perf_counter", lambda: 12345.0)
    sched_b, pts_b = _run()
    monkeypatch.setattr(sch.time, "perf_counter", real_pc)

    def strip(pts):
        return [p for p in pts if p[0] not in TIMING_TAGS]

    assert strip(pts_a) == strip(pts_b), "计时不得改变其它任何日志/横轴"

    for name in sched_a.scans:
        a, b = sched_a.scans[name], sched_b.scans[name]
        assert a.scanned_sample_ids == b.scanned_sample_ids
        assert a.probs == pytest.approx(b.probs)
        assert a.losses == pytest.approx(b.losses)
        assert a.difficulties == pytest.approx(b.difficulties)
        assert a.scanned_traj_ids == b.scanned_traj_ids


def test_timing_helpers_do_not_touch_random_state():
    st0 = random.getstate()
    st = st0
    for _ in range(5):
        _cuda_sync_for_timing()
        sch.time.perf_counter()
        st = random.getstate()
    assert st == st0, "计时路径不得消耗全局 RNG"


def test_no_warmup_steady_double_label_on_same_scan():
    _sched, points = _run()
    seen = {}
    for tag, _v, step in points:
        if tag in TIMING_TAGS:
            seen.setdefault(step, set()).add(tag)
    for step, tags in seen.items():
        assert not ({"auto_learning/hardness_scan_seconds_warmup",
                     "auto_learning/hardness_scan_seconds_steady"} <= tags), \
            f"step {step} 同时被标为预热与稳态"


def test_module_level_has_no_torch_import_and_sync_is_guarded():
    src = inspect.getsource(sch)
    assert not re.search(r"^import torch", src, re.M), "scheduler.py 不得在模块级 import torch"
    assert not re.search(r"^from torch", src, re.M)
    # torch 只允许出现在惰性同步助手里（缩进块内）
    for line in src.splitlines():
        if re.match(r"\s*import torch", line):
            assert line.startswith("        "), f"`import torch` 必须在函数体内: {line!r}"
    _cuda_sync_for_timing()      # 无 CUDA 时必须是 no-op，且不得抛异常
