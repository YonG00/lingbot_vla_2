"""Sampling / Hardness（测试方案 §7 Level D）。

解耦后：
  * `BatchSampler.prepare(req)` 编译分布（一个 learning unit 只做一次）
  * `BatchSampler.build(prepared)` 抽**一个** optimizer step 的 batch
  * replay 由 `ReplayPlan`（从 registry 派生）描述，没有单独的池对象
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from lingbotvla.auto_learning.testing import fake_tasks as ft
from lingbotvla.auto_learning.testing.fake_tasks import make_cfg as ft_make_cfg
from lingbotvla.auto_learning.testing.sim import SimulatedHardnessScorer, SimulatedWorld
from lingbotvla.auto_learning.ports import ReplayPlan
from lingbotvla.auto_learning.sampling.hardness_scan import HardnessScanner
from lingbotvla.auto_learning.sampling.sampler import BatchSampler
from lingbotvla.auto_learning.state.registry import TaskRegistry
from al_fixtures import (
    catalog_of,
    default_al,
    make_cfg,
    resolver_of,
    scheduler_of,
    train_request,
)


def _setup(n_tasks=2, probe=1.0):
    cfg = make_cfg(n_tasks)
    cfg.auto_learning = default_al(hardness_probe_fraction=probe)
    catalog = catalog_of(cfg)
    resolver = resolver_of(cfg)
    world = SimulatedWorld(cfg.sim)
    rng = random.Random(1)
    scanner = HardnessScanner(cfg.auto_learning, SimulatedHardnessScorer(world))
    sampler = BatchSampler(cfg.auto_learning, resolver, catalog, rng)
    return cfg, world, catalog, resolver, scanner, sampler


def _rec(cfg, catalog, task):
    return TaskRegistry.from_catalog(catalog, cfg.auto_learning).get(task)


def _probs(cfg, catalog, task):
    return {sid: 1.0 for sid in catalog.entry(task).train_sample_ids}


# =========================================================================== #
# batch 组成
# =========================================================================== #
def test_batch_shape_without_replay():
    cfg, world, catalog, resolver, scanner, sampler = _setup(2)
    comp = sampler.build(sampler.prepare(train_request(cfg, "t0", _probs(cfg, catalog, "t0"))))
    assert comp.n_old == 0
    assert comp.n_new == cfg.auto_learning.batch_size  # 没有 PASS ⇒ 10 NEW


def test_batch_shape_with_replay():
    cfg, world, catalog, resolver, scanner, sampler = _setup(2)
    req = train_request(cfg, "t0", _probs(cfg, catalog, "t0"), replay_tasks=["t1"])
    comp = sampler.build(sampler.prepare(req))
    assert comp.n_new == cfg.auto_learning.new_slots
    assert comp.n_old == cfg.auto_learning.replay_slots
    assert comp.old_tasks == ["t1"]


def test_val_samples_never_sampled():
    cfg, world, catalog, resolver, scanner, sampler = _setup(2)
    val_trajs = set(_rec(cfg, catalog, "t0").val_traj_ids)
    req = train_request(cfg, "t0", _probs(cfg, catalog, "t0"), replay_tasks=["t1"])
    for _ in range(50):
        comp = sampler.build(sampler.prepare(req))
        for ref in comp.refs:
            assert not ref.is_val
            assert ref.episode_id not in val_trajs


def test_batch_has_no_duplicate_samples():
    """同一个 batch 内不放回 —— 重复样本会让同一帧被重复加权。"""
    cfg, world, catalog, resolver, scanner, sampler = _setup(1, probe=1.0)
    for _ in range(50):
        comp = sampler.build(
            sampler.prepare(train_request(cfg, "t0", _probs(cfg, catalog, "t0")))
        )
        ids = [r.sample_id for r in comp.refs]
        assert len(set(ids)) == len(ids)


def test_replay_slots_prefer_distinct_tasks():
    cfg, world, catalog, resolver, scanner, sampler = _setup(4)
    req = train_request(cfg, "t3", _probs(cfg, catalog, "t3"), replay_tasks=["t0", "t1", "t2"])
    multi = 0
    for _ in range(60):
        comp = sampler.build(sampler.prepare(req))
        if len(comp.old_tasks) == cfg.auto_learning.replay_slots:
            multi += 1
    assert multi >= 55


# =========================================================================== #
# §7 Level D —— Hardness
# =========================================================================== #
def test_D02_weights_stay_in_range():
    cfg, world, catalog, resolver, scanner, sampler = _setup(1, probe=0.5)
    scan = scanner.scan(_rec(cfg, catalog, "t0"), catalog.entry("t0"))
    lo = cfg.auto_learning.hardness_weight_min
    hi = cfg.auto_learning.hardness_weight_max
    assert all(lo - 1e-9 <= w <= hi + 1e-9 for w in scan.weights.values())
    assert sum(scan.probs.values()) == pytest.approx(1.0)


def test_D03_unscored_samples_get_default_difficulty():
    cfg, world, catalog, resolver, scanner, sampler = _setup(1, probe=0.25)
    scan = scanner.scan(_rec(cfg, catalog, "t0"), catalog.entry("t0"))
    rec = _rec(cfg, catalog, "t0")
    unscored = [s for s in rec.sample_ids if s not in scan.difficulties]
    assert unscored, "本测试需要存在未扫描样本"
    assert all(scan.probs[s] > 0 for s in unscored)
    expected_w = 1.0 + (3.0 - 1.0) * (0.5 ** 2.0)
    assert scan.weights[unscored[0]] == pytest.approx(expected_w)


def test_D04_empirical_frequency_matches_theoretical_probs():
    from lingbotvla.auto_learning.decision.metrics import (
        difficulty_to_weight,
        normalize_weights,
        spearman_rank_correlation,
    )
    from lingbotvla.auto_learning.sampling.rng import WeightedSampler

    n = 50
    weights = {i: difficulty_to_weight(i / (n - 1), 1.0, 3.0, 2.0) for i in range(n)}
    probs = normalize_weights(weights)
    keys = list(probs.keys())
    table = WeightedSampler(keys, [probs[k] for k in keys])

    rng = random.Random(11)
    draws = 50_000
    counts = Counter(table.draw(rng) for _ in range(draws))
    empirical = [counts.get(k, 0) / draws for k in keys]
    theoretical = [probs[k] for k in keys]
    rho = spearman_rank_correlation(theoretical, empirical)
    assert rho > 0.95, f"秩相关只有 {rho}"
    assert counts[keys[-1]] > counts[keys[0]] * 2
    assert all(counts.get(k, 0) > 0 for k in keys)


def test_D05_hardest_sample_does_not_monopolize_batches():
    cfg = ft_make_cfg([ft._spec("t0", curve=[0.9, 0.5, 0.25], n_train_trajs=4)])
    cfg.auto_learning = default_al(hardness_probe_fraction=1.0)
    catalog = catalog_of(cfg)
    resolver = resolver_of(cfg)
    world = SimulatedWorld(cfg.sim)
    rng = random.Random(5)
    scanner = HardnessScanner(cfg.auto_learning, SimulatedHardnessScorer(world))
    sampler = BatchSampler(cfg.auto_learning, resolver, catalog, rng)

    rec = _rec(cfg, catalog, "t0")
    scan = scanner.scan(rec, catalog.entry("t0"))
    hardest = max(scan.losses, key=lambda k: scan.losses[k])

    batches = 400
    counts = Counter()
    req = train_request(cfg, "t0", scan.probs)
    for _ in range(batches):
        comp = sampler.build(sampler.prepare(req))
        ids = [r.sample_id for r in comp.new]
        assert len(set(ids)) == len(ids), "同一个 batch 内不允许放回重复样本"
        for sid in ids:
            counts[sid] += 1

    total = batches * cfg.auto_learning.new_slots
    uniform = 1.0 / scan.n_total
    share = counts[hardest] / total
    assert share > uniform * 1.3, f"最难样本没有被倾斜：{share} vs 均匀 {uniform}"
    assert share < uniform * 4.0, f"最难样本份额 {share:.4%} 超出权重上限允许的范围"

    ranked = sorted(scan.difficulties.items(), key=lambda kv: kv[1])
    n = len(ranked)
    top = [k for k, _ in ranked[int(0.9 * n):]]
    bottom = [k for k, _ in ranked[: int(0.1 * n)]]
    top_freq = sum(counts[s] for s in top) / (len(top) * total)
    bottom_freq = sum(counts[s] for s in bottom) / (len(bottom) * total)
    assert top_freq > bottom_freq * 2.0, (top_freq, bottom_freq)


def test_D06_probe_subset_rotates_across_attempts():
    cfg, world, catalog, resolver, scanner, sampler = _setup(1, probe=0.33)
    rec = _rec(cfg, catalog, "t0")
    scans = []
    for _ in range(3):
        s = scanner.scan(rec, catalog.entry("t0"))
        rec.hardness_version = s.version
        scans.append(s)
    assert not (set(scans[0].scanned_traj_ids) & set(scans[1].scanned_traj_ids))
    assert not (set(scans[1].scanned_traj_ids) & set(scans[2].scanned_traj_ids))
    covered = set().union(*[set(s.scanned_traj_ids) for s in scans])
    assert covered == set(rec.train_traj_ids), "3 轮之后应当覆盖全部 train 轨迹"


def test_D06b_probe_fraction_matches_the_configured_ratio():
    """`hardness_probe_fraction` 必须接近真实比例。

    旧实现用 `round(1/fraction)`，在 0.4 / 0.6 这类取值上会偏成 0.5。
    现在 `k = ceil(n_traj × fraction)`。
    """
    expected = {0.20: 8, 0.25: 10, 0.33: 14, 0.40: 16, 0.60: 24, 0.80: 32, 1.00: 40}
    for frac, k in expected.items():
        cfg = make_cfg(1)
        cfg.auto_learning = default_al(hardness_probe_fraction=frac)
        catalog = catalog_of(cfg)
        scanner = HardnessScanner(
            cfg.auto_learning, SimulatedHardnessScorer(SimulatedWorld(cfg.sim))
        )
        rec = _rec(cfg, catalog, "t0")
        scan = scanner.scan(rec, catalog.entry("t0"))
        assert len(scan.scanned_traj_ids) == k, f"fraction={frac} 应抽 {k} 条"
        ratio = len(scan.scanned_traj_ids) / 40
        assert abs(ratio - frac) <= 0.05, f"fraction={frac} 实际 {ratio:.2f}"


def test_D07_rescue_switch_off_defers_immediately():
    cfg = make_cfg(1, curves={"t0": [0.90, 0.70, 0.66, 0.64, 0.63, 0.62]})
    cfg.auto_learning = default_al(defer_resample_retry=False, max_attempts_per_task=1)
    sched = scheduler_of(cfg)
    sched.run(max_actions=200)
    assert sched.registry.get("t0").status == "EXHAUSTED"
    assert all("RESCUE" not in str(e.get("decision", "")) for e in sched.events)
    assert max(r.hardness_version for r in sched.registry) == 1


def test_D07b_defer_retry_steps_is_actually_executed():
    """`defer_retry_steps` 要真的按配置执行，而不是写多少都只多跑一个单元。"""
    steps = {}
    for retry in (50, 100):
        cfg = make_cfg(1, curves={"t0": [0.90, 0.80, 0.79, 0.785, 0.782, 0.780]})
        cfg.auto_learning = default_al(
            defer_resample_retry=True, max_attempts_per_task=1, defer_retry_steps=retry
        )
        sched = scheduler_of(cfg)
        sched.run(max_actions=300)
        assert any(e.get("decision") == "RESCUE" for e in sched.events), "应当触发 rescue"
        steps[retry] = sched.registry.get("t0").total_task_steps
    assert steps[100] == steps[50] + 50, f"defer_retry_steps 没被执行：{steps}"
