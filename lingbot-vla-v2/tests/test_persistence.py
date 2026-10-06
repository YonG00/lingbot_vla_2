"""测试方案 §10 Level G —— Persistence / Resume。"""

from __future__ import annotations

import os

import pytest

from lingbotvla.auto_learning.state import persistence
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from lingbotvla.auto_learning.testing import fake_tasks as ft
from al_fixtures import local_tmpdir
from lingbotvla.auto_learning.testing.fake_tasks import cfg_with, make_cfg
from lingbotvla.auto_learning.types import TaskStatus
from al_fixtures import scheduler_of

# 方案 §G01 点名要逐字段核对的清单
G01_FIELDS = (
    "status",
    "attempt_count",
    "attempt_step",
    "best_nmse",
    "current_val_nmse",
    "scout_nmse",
    "reopen_count",
    "forgotten",
    "last_transition_reason",
    "last_transition_code",
    "ever_passed",
    "pass_sampling_version",
)


def _al(**kw) -> AutoLearningConfig:
    base = dict(seed=3, eval_interval_steps=50, min_steps_before_defer=100, pass_nmse=0.30)
    base.update(kw)
    return AutoLearningConfig(**base)


# =========================================================================== #
# G01 Registry Save / Load
# =========================================================================== #
def test_G01_registry_fields_survive_roundtrip():
    with local_tmpdir() as d:
        cfg = cfg_with(["easy_pass", "unlearnable", "already_known"], al=_al())
        sched = scheduler_of(cfg)
        sched.run(max_actions=200)
        path = os.path.join(d, "state.json")
        persistence.save_state(sched, path)

        fresh = scheduler_of(cfg)
        persistence.load_state(fresh, path)

        for name in sched.registry.names():
            a, b = sched.registry.get(name), fresh.registry.get(name)
            for field in G01_FIELDS:
                assert getattr(a, field) == getattr(b, field), f"{name}.{field}"
            assert a.sample_probs == pytest.approx(b.sample_probs)
        assert sched.state.round == fresh.state.round
        assert sched.state.global_step == fresh.state.global_step
        assert sched.state.transition_count == fresh.state.transition_count


# =========================================================================== #
# G02 PASS Sampling Snapshot 恢复
# =========================================================================== #
def test_G02_pass_snapshot_survives_restart_and_is_still_used():
    with local_tmpdir() as d:
        cfg = cfg_with(["easy_pass", "unlearnable"], al=_al())
        sched = scheduler_of(cfg)
        sched.run(max_actions=200)
        plan = sched.replay_plan()
        assert "easy_pass" in plan.tasks
        snapshot = dict(sched.registry.get("easy_pass").pass_sampling_snapshot)
        assert snapshot, "应当有一份非空 snapshot"

        path = os.path.join(d, "state.json")
        persistence.save_state(sched, path)
        fresh = scheduler_of(cfg)
        persistence.load_state(fresh, path)

        restored = fresh.replay_plan()
        assert restored.probs["easy_pass"] == pytest.approx(snapshot)
        assert "easy_pass" in restored.tasks


# =========================================================================== #
# G03 Resume Equivalence（详细版在 test_resume.py，这里做方案点名的四项）
# =========================================================================== #
def test_G03_resume_equivalence_on_registry_trace_and_selection():
    with local_tmpdir() as d:
        cfg = cfg_with(["easy_pass", "unlearnable", "slow_learn"], al=_al())
        cfg.auto_learning.max_global_steps = 1200

        run_a = scheduler_of(cfg)
        run_a.run(max_actions=500)

        cfg_b = cfg_with(["easy_pass", "unlearnable", "slow_learn"], al=_al())
        cfg_b.auto_learning.max_global_steps = 1200
        run_b = scheduler_of(cfg_b)
        for _ in range(6):
            run_b.advance()
        path = os.path.join(d, "mid.json")
        persistence.save_state(run_b, path)

        cfg_c = cfg_with(["easy_pass", "unlearnable", "slow_learn"], al=_al())
        cfg_c.auto_learning.max_global_steps = 1200
        run_c = scheduler_of(cfg_c)
        persistence.load_state(run_c, path)
        run_c.resume()
        run_c.run(max_actions=500)

        assert run_a.registry.table() == run_c.registry.table()
        assert [t["task"] for t in run_a.state.transitions] == [
            t["task"] for t in run_c.state.transitions
        ]
        assert [r["task"] for r in run_a.metrics_rows] == [r["task"] for r in run_c.metrics_rows]
        assert run_a.state.stop_reason == run_c.state.stop_reason


# =========================================================================== #
# G04 Config Compatibility
# =========================================================================== #
def test_G04_resume_rejects_changed_semantic_config():
    with local_tmpdir() as d:
        cfg = cfg_with(["easy_pass"], al=_al(pass_nmse=0.30))
        sched = scheduler_of(cfg)
        sched.run(max_actions=100)
        path = os.path.join(d, "state.json")
        persistence.save_state(sched, path)

        changed = cfg_with(["easy_pass"], al=_al(pass_nmse=0.25))
        fresh = scheduler_of(changed)
        with pytest.raises(ValueError) as err:
            persistence.load_state(fresh, path)
        assert "pass_nmse" in str(err.value)


def test_G04_resume_rejects_changed_task_list():
    with local_tmpdir() as d:
        cfg = cfg_with(["easy_pass"], al=_al())
        sched = scheduler_of(cfg)
        sched.run(max_actions=100)
        path = os.path.join(d, "state.json")
        persistence.save_state(sched, path)

        fresh = scheduler_of(cfg_with(["easy_pass", "unlearnable"], al=_al()))
        with pytest.raises(ValueError):
            persistence.load_state(fresh, path)


def test_G04_explicit_override_allows_config_change():
    with local_tmpdir() as d:
        cfg = cfg_with(["easy_pass"], al=_al(pass_nmse=0.30))
        sched = scheduler_of(cfg)
        sched.run(max_actions=100)
        path = os.path.join(d, "state.json")
        persistence.save_state(sched, path)

        fresh = scheduler_of(cfg_with(["easy_pass"], al=_al(pass_nmse=0.25)))
        persistence.load_state(fresh, path, allow_config_change=True)  # 不抛
        assert fresh.state.global_step == sched.state.global_step


def test_G04_non_semantic_change_is_allowed():
    """`max_global_steps` 这类不影响历史含义的参数，改了不该拦。"""
    with local_tmpdir() as d:
        cfg = cfg_with(["easy_pass"], al=_al())
        sched = scheduler_of(cfg)
        sched.run(max_actions=100)
        path = os.path.join(d, "state.json")
        persistence.save_state(sched, path)

        other = cfg_with(["easy_pass"], al=_al())
        other.auto_learning.max_global_steps = 99999
        fresh = scheduler_of(other)
        persistence.load_state(fresh, path)  # 不抛
        assert fresh.state.global_step == sched.state.global_step
