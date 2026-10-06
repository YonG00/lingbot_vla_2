"""测试方案 §13 —— 五套一键端到端场景。

场景定义在 `auto_learning/env/scenarios.py`（与 CLI 工具共用同一份），
这里只负责跑起来 + 断言。
"""

from __future__ import annotations

import pytest

from lingbotvla.auto_learning.testing.scenarios import SCENARIOS, build
from al_fixtures import scheduler_of
from lingbotvla.auto_learning.types import TaskStatus


def _run(scn, limit: int = 3000) -> Scheduler:
    sched = scheduler_of(scn.cfg)
    sched.run(max_actions=limit)
    assert sched.state.finished, f"[{scn.name}] 场景必须能自己停下来"
    return sched


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_expected_final_states(name):
    scn = build(name)
    sched = _run(scn)
    for task, status in scn.expect.items():
        got = sched.registry.get(task).status
        assert got == status, f"[{scn.name}] {task} 期望 {status}，实际 {got}"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_expected_transition_kinds(name):
    scn = build(name)
    sched = _run(scn)
    kinds = {t["kind"] for t in sched.state.transitions}
    for want in scn.expect_kinds:
        assert want in kinds, f"[{scn.name}] 期望出现过 {want}，实际 {sorted(kinds)}"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_untrained_tasks(name):
    scn = build(name)
    sched = _run(scn)
    for task in scn.expect_untrained:
        assert sched.registry.get(task).total_task_steps == 0, (
            f"[{scn.name}] {task} 期望一次都没训过"
        )


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_is_bounded(name):
    """所有场景都必须有界收敛（不允许失控）。"""
    scn = build(name)
    sched = _run(scn)
    assert sched.state.units_run <= 60, f"[{scn.name}] 单元数失控"
    for rec in sched.registry:
        assert rec.attempt_count <= scn.cfg.auto_learning.max_attempts_per_task
        assert rec.reopen_count <= scn.cfg.auto_learning.max_attempts_per_task


# --------------------------------------------------------------------------- #
# 各场景特有的细节断言
# --------------------------------------------------------------------------- #
def test_scenario1_step_counts_and_rounds():
    sched = _run(build("1-ideal"))
    assert sched.registry.get("B").total_task_steps == 50
    assert sched.registry.get("C").total_task_steps == 100
    d = sched.registry.get("D")
    assert d.attempt_count == 2
    assert d.total_task_steps == 200
    assert sched.state.round >= 2
    assert sched.state.stop_reason == "all_tasks_resolved"


def test_scenario3_target_never_trained():
    sched = _run(build("3-transfer"))
    b = sched.registry.get("B")
    assert b.total_task_steps == 0
    assert b.attempt_count == 0
    assert "B" in sched.state.auto_passed


def test_scenario4_replay_throughout_and_snapshot_updated():
    sched = _run(build("4-forgetting"))
    a = sched.registry.get("A")
    assert a.reopen_count >= 1
    assert a.pass_sampling_version >= 2, "回炉后再次 PASS 必须更新 snapshot"
    units_with_replay = sum(1 for row in sched.metrics_rows if row["n_old"] > 0)
    assert units_with_replay >= len(sched.metrics_rows) - 2
    reopens = [t for t in sched.state.transitions if t["kind"] == "REOPEN"]
    assert reopens and reopens[0]["code"]


def test_scenario5_budget_terminates_the_pingpong():
    sched = _run(build("5-conflict"))
    a, b = sched.registry.get("A"), sched.registry.get("B")
    assert a.attempt_count <= 2 and b.attempt_count <= 2
    assert a.reopen_count >= 1 or b.reopen_count >= 1
    assert a.status == TaskStatus.EXHAUSTED.value or b.status == TaskStatus.EXHAUSTED.value
    assert sched.state.units_run <= 40
