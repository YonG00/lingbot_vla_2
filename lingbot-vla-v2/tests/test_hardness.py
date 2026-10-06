"""Hardness 扫描（文档 §19–§26）。"""

from __future__ import annotations

import random

import pytest

from lingbotvla.auto_learning.sampling.hardness_scan import HardnessScanner
from lingbotvla.auto_learning.state.registry import TaskRegistry
from lingbotvla.auto_learning.testing.sim import SimulatedWorld
from al_fixtures import make_cfg
from al_fixtures import catalog_of, registry_of
from lingbotvla.auto_learning.testing.sim import SimulatedHardnessScorer


def _setup(probe_fraction=0.33):
    cfg = make_cfg(1)
    cfg.auto_learning.hardness_probe_fraction = probe_fraction
    cfg.sim.hardness_latent_skew = 2.0
    world = SimulatedWorld(cfg.sim)
    reg = registry_of(cfg)
    rec = reg.get("t0")
    scanner = HardnessScanner(cfg.auto_learning, SimulatedHardnessScorer(world))
    return cfg, world, rec, scanner


def test_probe_scans_whole_trajectories_and_about_one_third():
    cfg, world, rec, scanner = _setup(0.33)
    scan = scanner.scan(rec, catalog_of(cfg).entry(rec.task_name))
    assert len(scan.scanned_traj_ids) == pytest.approx(14, abs=1)  # ceil(0.33 × 40)
    # 抽到的是**完整轨迹**：每条轨迹的样本数 = samples_per_traj
    assert scan.n_scanned == len(scan.scanned_traj_ids) * cfg.sim.tasks[0].samples_per_traj
    assert scan.n_total == len(rec.sample_ids)


def test_difficulties_and_probs_are_well_formed():
    cfg, world, rec, scanner = _setup(0.5)
    scan = scanner.scan(rec, catalog_of(cfg).entry(rec.task_name))
    assert all(0.0 <= d <= 1.0 for d in scan.difficulties.values())
    assert sum(scan.probs.values()) == pytest.approx(1.0)
    # 权重必须在设计范围内（文档 §23）
    lo = cfg.auto_learning.hardness_weight_min
    hi = cfg.auto_learning.hardness_weight_max
    assert all(lo - 1e-9 <= w <= hi + 1e-9 for w in scan.weights.values())


def test_unscored_samples_still_get_probability():
    """文档 §21：没扫到的样本不能被判死刑。"""
    cfg, world, rec, scanner = _setup(0.25)
    scan = scanner.scan(rec, catalog_of(cfg).entry(rec.task_name))
    unscored = [s for s in rec.sample_ids if s not in scan.difficulties]
    assert unscored, "本测试需要存在未扫描样本"
    assert all(scan.probs[s] > 0 for s in unscored)
    expected_w = 1.0 + (3.0 - 1.0) * (0.5 ** 2.0)
    assert scan.weights[unscored[0]] == pytest.approx(expected_w)


def test_version_increments_on_rescan():
    cfg, world, rec, scanner = _setup()
    s1 = scanner.scan(rec, catalog_of(cfg).entry(rec.task_name))
    rec.hardness_version = s1.version
    s2 = scanner.scan(rec, catalog_of(cfg).entry(rec.task_name))
    assert s2.version == s1.version + 1


def test_scan_roundtrip_serialization():
    cfg, world, rec, scanner = _setup()
    scan = scanner.scan(rec, catalog_of(cfg).entry(rec.task_name))
    again = type(scan).from_state(scan.to_state())
    assert again.probs == pytest.approx(scan.probs)
    assert again.version == scan.version
