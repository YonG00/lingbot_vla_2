"""专门验证「一个 learning unit = num_steps 次**独立** batch sampling」。

这是本轮最关键的语义修正：旧实现抽**一个** batch 就当成 50 个 optimizer step，
导致 replay 零散度、加权采样效果、`samples_seen` 统计全部失真。
"""

from __future__ import annotations

import pytest

from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.testing import fake_tasks as ft
from lingbotvla.auto_learning.ports import ReplayPlan, TrainRequest
from lingbotvla.auto_learning.testing.fake_tasks import make_cfg
from al_fixtures import (
    catalog_of,
    registry_of,
    scheduler_of,
    train_request,
    world_of,
)


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


def _backend(cfg):
    from lingbotvla.auto_learning.testing.sim import build_backend

    return build_backend(cfg)


def _probs(cfg, task="t0"):
    return {sid: 1.0 for sid in catalog_of(cfg).entry(task).train_sample_ids}


# =========================================================================== #
def test_train_steps_builds_one_batch_per_optimizer_step():
    cfg = make_cfg([ft._spec("t0", curve=[0.9, 0.5, 0.25])], al=_al())
    b = _backend(cfg)
    req = train_request(cfg, "t0", _probs(cfg))
    res = b.trainer.train_steps(req, 50)

    assert res.steps == 50
    assert res.batches_built == 50, "50 个 optimizer step 必须抽 50 个 batch"
    assert len(res.per_step_losses) == 50
    assert res.unique_batches > 1, "50 次抽样不该全部抽到同一个 batch"


def test_samples_seen_is_steps_times_batch_size():
    cfg = make_cfg([ft._spec("t0", curve=[0.9, 0.5, 0.25])], al=_al())
    b = _backend(cfg)
    req = train_request(cfg, "t0", _probs(cfg))
    res = b.trainer.train_steps(req, 50)
    assert res.samples_seen == 50 * 10, f"应消费 500 个样本，实际 {res.samples_seen}"


def test_consecutive_units_use_different_batches():
    """两个 unit 之间 RNG 继续前进 ⇒ batch 组合应当不同。"""
    cfg = make_cfg([ft._spec("t0", curve=[0.9, 0.5, 0.25])], al=_al())
    b = _backend(cfg)
    req = train_request(cfg, "t0", _probs(cfg))
    a = b.trainer.train_steps(req, 50)
    c = b.trainer.train_steps(req, 50)
    assert a.new_slot_counts != c.new_slot_counts


def test_learning_unit_increases_world_by_exactly_one_unit():
    """50 个 step = 1 个 unit 的训练量（不是 50 个 unit）。"""
    cfg = make_cfg([ft._spec("t0", curve=[0.9, 0.5, 0.25])], al=_al())
    b = _backend(cfg)
    world = b.extra_state
    before = world.true_units("t0")
    b.trainer.train_steps(train_request(cfg, "t0", _probs(cfg)), 50)
    gained = world.true_units("t0") - before
    # 全 batch 都是 t0 ⇒ 应当正好 +1.0 unit
    assert gained == pytest.approx(1.0, abs=1e-9), f"实际增加 {gained}"


def test_replay_slots_are_spread_over_the_window_not_one_batch():
    """10 个 batch 的窗口里，replay 应当**分散**发生，而不是集中在一个 batch。"""
    cfg = make_cfg(
        [ft._spec("t0", curve=[0.9, 0.5, 0.25]), ft._spec("t1", curve=[0.5, 0.2])],
        al=_al(),
    )
    b = _backend(cfg)
    req = train_request(cfg, "t0", _probs(cfg), replay_tasks=["t1"])
    res = b.trainer.train_steps(req, 50)
    assert res.old_slot_counts.get("t1", 0) == 50 * 3, "每个 step 都应有 3 个 replay slot"


def test_scheduler_global_samples_seen_counts_real_samples():
    cfg = make_cfg(
        [ft._spec("t0", curve=[0.9, 0.5, 0.25]), ft._spec("t1", curve=[0.5, 0.2])], al=_al()
    )
    sched = scheduler_of(cfg)
    sched.run(max_actions=400)

    expected = sched.state.units_run * cfg.auto_learning.eval_interval_steps * cfg.auto_learning.batch_size
    assert sched.state.global_samples_seen == expected, (
        f"units={sched.state.units_run} ⇒ 应消费 {expected} 个样本，"
        f"实际 {sched.state.global_samples_seen}"
    )
    assert sched.state.global_samples_seen > 0
    # 每个 unit 行都要记下真实样本数
    for row in sched.metrics_rows:
        assert row["samples_seen"] == cfg.auto_learning.eval_interval_steps * cfg.auto_learning.batch_size
        assert row["batches_built"] == cfg.auto_learning.eval_interval_steps


def test_uniform_baseline_also_counts_real_samples():
    from lingbotvla.auto_learning.orchestration.baselines import run_uniform

    cfg = make_cfg(
        [ft._spec(f"t{i}", curve=[0.6, 0.3]) for i in range(4)], al=_al()
    )
    res = run_uniform(cfg, 4, mode="mixed")
    assert res.global_samples_seen == 4 * 50 * 10
    assert sum(res.per_task_steps.values()) == res.global_samples_seen
