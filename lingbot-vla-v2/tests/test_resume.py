"""Resume（文档 §44 「Resume」那一组）。

核心断言：**中途存档再恢复，必须走出与不中断完全相同的路线**。
只恢复模型 checkpoint 是不够的（文档 §43.9）——current task / attempt_count /
round / pass snapshot / hardness version / RNG state 缺一不可。
"""

from __future__ import annotations

import json
import os

import pytest

from lingbotvla.auto_learning.state import persistence
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from al_fixtures import local_tmpdir, make_cfg
from al_fixtures import scheduler_of


def _snapshot(sched: Scheduler) -> dict:
    return {
        "stop_reason": sched.state.stop_reason,
        "global_step": sched.state.global_step,
        "units": sched.state.units_run,
        "round": sched.state.round,
        "transitions": sched.state.transition_count,
        "current_task": sched.state.current_task,
        "trained": list(sched.state.trained_tasks),
        "newly_passed": list(sched.state.newly_passed),
        "registry": sched.registry.table(),
        "world": sched.extra_state.snapshot_table(),
        "metrics_rows": sched.metrics_rows,
        "transitions_log": sched.state.transitions,
        "rng_state": list(sched.rng.getstate()[1]),
        "pool": sched.replay_plan(),
    }


def _run_pair(tmpdir: str, prefix_actions: int = 10, budget_steps: int = 800):
    """跑两遍：第一遍中途存档后跑完；第二遍跑到同一断点后从存档恢复再跑完。"""
    cfg1 = make_cfg(3)
    cfg1.auto_learning.max_global_steps = budget_steps
    s1 = scheduler_of(cfg1)
    for _ in range(prefix_actions):
        s1.advance()
    state_path = os.path.join(tmpdir, "state.json")
    persistence.save_state(s1, state_path)
    s1.run(max_actions=500)

    cfg2 = make_cfg(3)
    cfg2.auto_learning.max_global_steps = budget_steps
    s2 = scheduler_of(cfg2)
    for _ in range(prefix_actions):
        s2.advance()
    persistence.load_state(s2, state_path)
    s2.run(max_actions=500)
    return s1, s2


def test_resume_reproduces_identical_route():
    with local_tmpdir() as d:
        s1, s2 = _run_pair(d)
        a, b = _snapshot(s1), _snapshot(s2)
        for key in (
            "stop_reason",
            "global_step",
            "units",
            "registry",
            "world",
            "metrics_rows",
            "transitions_log",
            "rng_state",
            "pool",
        ):
            assert a[key] == b[key], f"{key} 在 resume 后不一致"


def test_resume_next_candidate_identical():
    """文档 §44 明确要求：恢复后「下一个 candidate」必须一致。"""
    with local_tmpdir() as d:
        s1, s2 = _run_pair(d, prefix_actions=6)
        assert s1.state.finished and s2.state.finished
        assert s1.state.trained_tasks == s2.state.trained_tasks
        assert s1.state.newly_passed == s2.state.newly_passed
        assert [t["task"] for t in s1.state.transitions] == [
            t["task"] for t in s2.state.transitions
        ]


def test_resume_mid_attempt_keeps_attempt_step():
    """在 attempt 中途存档：attempt_step / 采样概率 / hardness 版本都要活下来。"""
    with local_tmpdir() as d:
        cfg = make_cfg(2)
        cfg.auto_learning.max_global_steps = 10000
        s = scheduler_of(cfg)
        while s.state.current_task is None:
            s.advance()
        s.advance()  # 至少跑一个 unit
        path = os.path.join(d, "s.json")
        persistence.save_state(s, path)

        cfg2 = make_cfg(2)
        cfg2.auto_learning.max_global_steps = 10000
        s2 = scheduler_of(cfg2)
        persistence.load_state(s2, path)

        assert s2.state.current_task == s.state.current_task
        assert s2.state.attempt_step == s.state.attempt_step
        rec, rec2 = s.registry.get(s.state.current_task), s2.registry.get(s2.state.current_task)
        assert rec2.hardness_version == rec.hardness_version
        assert rec2.sample_probs == pytest.approx(rec.sample_probs)
        assert rec2.attempt_count == rec.attempt_count


def test_resume_after_finished_continues():
    """存档时已经 finished，恢复后必须能继续跑（否则「恢复」变成「立刻结束」）。"""
    with local_tmpdir() as d:
        cfg = make_cfg(3)
        cfg.auto_learning.max_transitions = 1
        s = scheduler_of(cfg)
        s.run(max_actions=200)
        assert s.state.finished and s.state.transition_count == 1
        path = os.path.join(d, "s.json")
        persistence.save_state(s, path)

        cfg2 = make_cfg(3)
        cfg2.auto_learning.max_transitions = 1
        s2 = scheduler_of(cfg2)
        persistence.load_state(s2, path)
        assert s2.state.finished, "存档里就该是 finished"

        s2.resume()
        s2.al.max_transitions = None  # 放宽预算后再继续
        s2.run(max_actions=200)
        assert s2.state.finished
        assert s2.state.transition_count > 1
        assert s2.state.units_run > s.state.units_run


def test_state_version_mismatch_is_rejected():
    with local_tmpdir() as d:
        path = os.path.join(d, "bad.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": 999}, fh)
        s = scheduler_of(make_cfg(1))
        with pytest.raises(ValueError):
            persistence.load_state(s, path)
