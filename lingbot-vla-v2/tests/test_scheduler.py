"""端到端：跑真配置文件，确认整条闭环真的能走通。"""

from __future__ import annotations

import os

import pytest

from lingbotvla.auto_learning.config import load_config
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from al_fixtures import make_cfg
from al_fixtures import scheduler_of

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "lingbotvla", "auto_learning", "configs")


def _cfg(name: str):
    return load_config(os.path.join(CONFIG_DIR, name))


def test_demo_4task_exercises_every_state_machine_path():
    cfg = _cfg("demo_4task.yaml")
    sched = scheduler_of(cfg)
    sched.run(max_actions=400)

    assert sched.state.finished, "状态机必须能自己停下来，不能无限循环"
    counts = sched.registry.counts()
    assert counts["PASS"] >= 2, counts
    assert counts["EXHAUSTED"] >= 1, counts
    assert any(r.reopen_count > 0 for r in sched.registry), "应当触发过 遗忘 → 回炉"
    assert any(r.attempt_count >= 2 for r in sched.registry), "应当有任务用到第二次 attempt"
    # 预算别失控
    assert sched.state.units_run <= 80
    # 每个 PASS 的任务都在 replay pool 里（除非它被回炉了）
    for r in sched.registry:
        if r.status == "PASS":
            assert r.task_name in sched.replay_plan().tasks


def test_demo_12task_respects_step_budget_and_beats_uniform():
    cfg = _cfg("demo_12task.yaml")
    sched = scheduler_of(cfg)
    sched.run(max_actions=3000)
    assert sched.state.finished
    assert sched.state.global_step <= 2000
    summary = sched.registry.summary()
    assert summary["pass"] >= 3

    from lingbotvla.auto_learning.orchestration.baselines import run_uniform

    rr = run_uniform(cfg, sched.state.units_run, mode="round_robin")
    # 自主模式在同等 step 预算下不应该比「每任务平均分配」差
    assert summary["pass"] >= len(rr.pass_tasks), (summary, rr.pass_tasks)


def test_max_new_tasks_attempted_cap():
    """口径 = 主动尝试过多少个新任务（不是 newly PASS 数）。"""
    cfg = make_cfg(6)
    cfg.auto_learning.max_new_tasks_attempted_this_run = 2
    cfg.auto_learning.max_global_steps = 100000
    sched = scheduler_of(cfg)
    sched.run(max_actions=400)
    assert sched.state.finished
    assert len(sched.state.trained_tasks) == 2
    assert sched.state.stop_reason == "max_new_tasks_attempted_reached(2)"
    # 免费 PASS 不占这个预算
    assert set(sched.state.auto_passed).isdisjoint(set(sched.state.trained_tasks))


def test_hardness_refresh_on_continue_changes_nothing_catastrophic():
    cfg = make_cfg(2)
    cfg.auto_learning.refresh_hardness_on_continue = True
    sched = scheduler_of(cfg)
    sched.run(max_actions=200)
    assert sched.state.finished
    assert all(r.hardness_version >= 1 for r in sched.registry if r.total_task_steps > 0)


def test_defer_resample_rescue_path():
    cfg = make_cfg(1, curves={"t0": [0.90, 0.70, 0.66, 0.64, 0.63, 0.62]})
    cfg.auto_learning.defer_resample_retry = True
    cfg.auto_learning.min_steps_before_defer = 100
    sched = scheduler_of(cfg)
    sched.run(max_actions=200)
    assert sched.state.finished
    # 至少有一个 attempt 触发过 rescue（hardness 版本被刷新过两次以上）
    versions = [r.hardness_version for r in sched.registry]
    assert max(versions) >= 2, versions


def test_overfit_early_defer_path():
    """构造一个「train 一直降、val 卡住不动」的任务，必须触发 early defer（文档 §17.1）。"""
    cfg = make_cfg(1)
    spec = cfg.sim.tasks[0]
    spec.curve = [0.45] * 10  # val 永远卡在 0.45，过不了线
    spec.overfit_rate = 0.40  # train-monitor 因为「背训练集」而持续变好
    sched = scheduler_of(cfg)
    sched.run(max_actions=200)
    assert sched.state.finished
    assert any(row["overfit"] for row in sched.metrics_rows), "应当触发过 overfit 判据"
    kinds = [t["kind"] for t in sched.state.transitions]
    assert "DEFER" in kinds, kinds
