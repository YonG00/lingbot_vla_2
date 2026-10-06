"""测试方案 §8 Level E —— Replay。

Replay 是最容易「代码看起来对、长期分布其实不对」的部分，所以这里以统计断言为主。

解耦后 replay 是一组**纯函数**（`prepare_slots` / `sample_replay_refs` / `assign_slots`），
PASS 池直接从 registry 派生，不再有单独的 ReplayPool 对象。
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.ports import ReplayPlan
from lingbotvla.auto_learning.sampling.replay import assign_slots, prepare_slots, sample_replay_refs
from lingbotvla.auto_learning.sampling.sampler import BatchSampler
from al_fixtures import catalog_of, resolver_of, scheduler_of, train_request
from lingbotvla.auto_learning.testing.fake_tasks import cfg_with, make_cfg
from lingbotvla.auto_learning.types import SampleRef, TaskStatus


def _al(**kw) -> AutoLearningConfig:
    base = dict(
        seed=3,
        eval_interval_steps=50,
        min_steps_before_defer=100,
        pass_nmse=0.30,
        batch_size=10,
        new_slots=7,
        replay_slots=3,
    )
    base.update(kw)
    return AutoLearningConfig(**base)


class _PlainResolver:
    """最朴素的 resolver：sample_id 就是它自己（不依赖任何编码约定）。"""

    def resolve(self, task: str, sample_id: int) -> SampleRef:
        return SampleRef(task=task, sample_id=int(sample_id), episode_id=0, frame_id=int(sample_id))


def _plan(tasks, sample_ids, probs=None) -> ReplayPlan:
    return ReplayPlan(
        tasks=list(tasks),
        probs=dict(probs or {t: None for t in tasks}),
        sample_ids={t: list(ids) for t, ids in sample_ids.items()},
    )


def _two_tasks():
    from lingbotvla.auto_learning.testing import fake_tasks as ft

    return [ft._spec("t0", curve=[0.5, 0.2]), ft._spec("t1", curve=[0.5, 0.2])]


# =========================================================================== #
# E01 NEW / OLD slot 比例
# =========================================================================== #
def test_E01_slot_ratio_is_exact():
    cfg = make_cfg(_two_tasks(), al=_al())
    rng = random.Random(0)
    sampler = BatchSampler(cfg.auto_learning, resolver_of(cfg), catalog_of(cfg), rng)
    probs = {sid: 1.0 for sid in catalog_of(cfg).entry("t0").train_sample_ids}

    for _ in range(20):  # PASS pool 空 ⇒ 10 NEW
        comp = sampler.build(sampler.prepare(train_request(cfg, "t0", probs)))
        assert (comp.n_new, comp.n_old) == (10, 0)

    for _ in range(20):  # 有 PASS ⇒ 严格 7 NEW + 3 OLD
        req = train_request(cfg, "t0", probs, replay_tasks=["t1"])
        comp = sampler.build(sampler.prepare(req))
        assert (comp.n_new, comp.n_old) == (7, 3)


# =========================================================================== #
# E02 OLD 零散化
# =========================================================================== #
def test_E02_old_slots_spread_across_tasks():
    from lingbotvla.auto_learning.testing import fake_tasks as ft

    cfg = make_cfg([ft._spec(f"p{i}", curve=[0.5, 0.2]) for i in range(5)], al=_al())
    rng = random.Random(0)
    sampler = BatchSampler(cfg.auto_learning, resolver_of(cfg), catalog_of(cfg), rng)
    probs = {sid: 1.0 for sid in catalog_of(cfg).entry("p0").train_sample_ids}

    uniq = Counter()
    for _ in range(200):
        req = train_request(cfg, "p0", probs, replay_tasks=[f"p{i}" for i in range(5)])
        comp = sampler.build(sampler.prepare(req))
        uniq[len(set(comp.old_tasks))] += 1
    assert uniq[3] >= 190, f"3 个 replay slot 应尽量来自 3 个不同任务，实测 {dict(uniq)}"


# =========================================================================== #
# E03 Task-level replay 公平性（不按数据量加权）
# =========================================================================== #
def test_E03_replay_task_frequency_is_uniform_not_size_weighted():
    rng = random.Random(7)
    plan = _plan(["big", "small"], {"big": list(range(5000)), "small": list(range(10000, 11000))})
    counts = Counter()
    for _ in range(4000):
        for task in assign_slots(list(plan.tasks), 1, rng):
            counts[task] += 1
    ratio = counts["big"] / counts["small"]
    assert 0.9 < ratio < 1.1, f"数据量 5:1 不能变成 replay 频率 5:1，实测 {dict(counts)}"


# =========================================================================== #
# E04 PASS Snapshot Replay
# =========================================================================== #
def test_E04_replay_uses_own_pass_snapshot():
    plan = _plan(
        ["A", "B"],
        {"A": list(range(1000)), "B": list(range(5000, 6000))},
        {"A": {111: 1.0}, "B": {5555: 1.0}},
    )
    slots = prepare_slots(plan, 2)
    rng = random.Random(3)
    got = Counter()
    for _ in range(200):
        for ref in sample_replay_refs(slots, 2, _PlainResolver(), rng):
            got[(ref.task, ref.sample_id)] += 1
    assert got[("A", 111)] == 200
    assert got[("B", 5555)] == 200
    assert ("A", 5555) not in got and ("B", 111) not in got, "不能串任务的权重"


def test_E04_snapshot_policy_uniform_is_the_control_arm():
    plan = _plan(["A"], {"A": list(range(500))}, {"A": {7: 1.0}})
    slots = prepare_slots(plan, 1, policy="uniform")
    rng = random.Random(3)
    seen = {
        ref.sample_id
        for _ in range(200)
        for ref in sample_replay_refs(slots, 1, _PlainResolver(), rng)
    }
    assert len(seen) > 50, "uniform 策略下不该被 snapshot 锁死在一个样本上"


# =========================================================================== #
# E05 bootstrap 自动 PASS 没有 snapshot ⇒ 回落 uniform，且不能崩
# =========================================================================== #
def test_E05_auto_pass_has_no_snapshot_and_falls_back_to_uniform():
    cfg = cfg_with(["already_known", "easy_pass"], al=_al())
    sched = scheduler_of(cfg)
    while sched.state.bootstrap_queue:
        sched.advance()

    rec = sched.registry.get("already_known")
    assert rec.status == TaskStatus.PASS.value
    assert rec.pass_sampling_snapshot is None, "没训过就不该有 snapshot"

    plan = sched.replay_plan()
    assert "already_known" in plan.tasks
    assert plan.probs["already_known"] is None, "没有 snapshot ⇒ 回落 uniform"

    sched.run(max_actions=400)
    assert sched.state.finished
    assert any(row["n_old"] > 0 for row in sched.metrics_rows), "应当发生过 replay"


# =========================================================================== #
# E06 Re-PASS 覆盖 snapshot（版本号 +1）
# =========================================================================== #
def test_E06_repass_overwrites_snapshot_and_bumps_version():
    cfg = cfg_with(["forgotten", "unlearnable"], al=_al(max_attempts_per_task=3))
    cfg.auto_learning.review_after_task_transitions = 1
    sched = scheduler_of(cfg)
    sched.run(max_actions=800)

    rec = sched.registry.get("forgotten")
    assert rec.reopen_count >= 1
    assert rec.pass_sampling_version >= 2, "回炉后再次 PASS 必须覆盖旧 snapshot 并 +1"
    assert rec.pass_sampling_snapshot is not None
    plan = sched.replay_plan()
    assert plan.probs["forgotten"] == pytest.approx(rec.pass_sampling_snapshot)
