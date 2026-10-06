"""LingBot Regression（文档 §44）：关掉 auto_learning 时，原训练行为必须保持原样。

Stage A 里没有真 Trainer，所以这里证明的是**等价命题**：
  * 关闭开关 ⇒ scheduler 不产生任何副作用（不评测、不训练、不改 registry）
  * 相同 seed ⇒ 逐位可复现（所以 Stage B 里「开/关」的 diff 才有意义）
"""

from __future__ import annotations

from lingbotvla.auto_learning.orchestration.baselines import run_uniform
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from al_fixtures import make_cfg
from al_fixtures import scheduler_of


def test_disabled_scheduler_has_no_side_effects():
    cfg = make_cfg(2)
    cfg.auto_learning.enabled = False
    sched = scheduler_of(cfg)
    sched.run(max_actions=50)

    assert sched.state.finished
    assert sched.state.stop_reason == "auto_learning_disabled"
    assert sched.state.eval_events == 0
    assert sched.state.units_run == 0
    assert sched.state.global_step == 0
    assert sched.metrics_rows == []
    assert sched.heatmap_rows == []
    assert all(r.status == "CANDIDATE" for r in sched.registry)
    assert len(sched.replay_plan().tasks) == 0


def test_two_identical_runs_are_bit_identical():
    s1 = scheduler_of(make_cfg(3))
    s2 = scheduler_of(make_cfg(3))
    s1.run(max_actions=300)
    s2.run(max_actions=300)
    assert s1.metrics_rows == s2.metrics_rows
    assert s1.registry.table() == s2.registry.table()
    assert s1.state.stop_reason == s2.state.stop_reason


def test_uniform_baseline_is_deterministic():
    cfg = make_cfg(4)
    a = run_uniform(cfg, 10, mode="mixed")
    b = run_uniform(cfg, 10, mode="mixed")
    assert a.per_task_nmse == b.per_task_nmse
    assert a.global_step == b.global_step == 500


def test_uniform_modes_differ():
    """两种 baseline 必须是**不同**的东西，否则对照没有区分力。"""
    cfg = make_cfg(4)
    mixed = run_uniform(cfg, 8, mode="mixed")
    rr = run_uniform(cfg, 8, mode="round_robin")
    assert mixed.per_task_steps != rr.per_task_steps
    assert sum(mixed.per_task_steps.values()) == 4000
    assert sum(rr.per_task_steps.values()) == 4000
    assert max(rr.per_task_steps.values()) == 1000  # round robin 每任务一样多
