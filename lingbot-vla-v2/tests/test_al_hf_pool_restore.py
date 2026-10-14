"""HF 池导入 → 首次调度的纯 CPU 契约；不加载模型或数据集。"""

from copy import deepcopy

import pytest

from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from lingbotvla.auto_learning.testing import fake_tasks as ft
from lingbotvla.auto_learning.testing.sim import build_backend
from lingbotvla.auto_learning.types import TaskStatus, TrajectoryMetrics


@pytest.fixture
def make_scheduler(monkeypatch):
    # 假 backend 不需要 CUDA 计时；显式隔离 GPU/torch。
    monkeypatch.setattr(
        "lingbotvla.auto_learning.orchestration.scheduler._cuda_sync_for_timing",
        lambda: None,
    )

    def make(names=("A", "B", "C", "D")):
        al = AutoLearningConfig(
            batch_size=10, new_slots=7, replay_slots=3,
            rescan_candidates_after_transition=False,
        )
        cfg = ft.make_cfg([ft._spec(name, curve=[0.9, 0.6]) for name in names], al=al)
        sched = Scheduler(build_backend(cfg), al, seed=1)
        for rec in sched.registry:
            rec.scout_nmse = 0.9
        return sched

    return make


def test_imported_pool_survives_first_dispatch(make_scheduler):
    old = make_scheduler()
    statuses = (TaskStatus.PASS, TaskStatus.DEFER, TaskStatus.EXHAUSTED, TaskStatus.CANDIDATE)
    for rec, status in zip(old.registry, statuses):
        rec.set_status(status, "saved")
        rec.attempt_count = 2 if status == TaskStatus.EXHAUSTED else 1
    raw = old.registry.to_state()
    sched = make_scheduler()
    sched.import_pool_registry(raw)

    assert sched.registry.counts() == old.registry.counts()
    assert sched.state.bootstrap_queue == []
    assert sched.next_action() == "select"
    sched.defer_train = True
    event = sched.advance()
    assert event["action"] == "select"
    assert event["task"] == "D"
    for name, status in zip(("A", "B", "C"), statuses):
        assert sched.registry.get(name).status == status.value
    assert sched.registry.get("D").attempt_count == 2


def test_only_new_manifest_tasks_bootstrap(make_scheduler):
    old = make_scheduler(("A",))
    old.registry.get("A").set_status(TaskStatus.DEFER)
    raw = old.registry.to_state()
    raw["records"]["removed"] = deepcopy(raw["records"]["A"])
    sched = make_scheduler(("A", "B"))
    sched.import_pool_registry(raw)
    assert sched.state.bootstrap_queue == ["B"]
    assert sched.advance()["task"] == "B"
    assert sched.registry.get("A").status == TaskStatus.DEFER.value
    assert "removed" not in sched.registry.records


@pytest.mark.parametrize("import_empty", [False, True])
def test_no_usable_pool_keeps_fresh_bootstrap(make_scheduler, import_empty):
    sched = make_scheduler()
    if import_empty:
        sched.import_pool_registry({})
    assert sched.state.bootstrap_queue == ["A", "B", "C", "D"]
    assert sched.next_action() == "bootstrap"


@pytest.mark.parametrize("attempts,expected", [(1, TaskStatus.CANDIDATE), (2, TaskStatus.EXHAUSTED)])
def test_hf_does_not_resume_in_progress_attempt(make_scheduler, attempts, expected):
    old = make_scheduler(("A",))
    rec = old.registry.get("A")
    rec.set_status(TaskStatus.TRAINING)
    rec.attempt_count = attempts
    sched = make_scheduler(("A",))
    sched.import_pool_registry(old.registry.to_state())
    assert sched.registry.get("A").status == expected.value
    assert sched.registry.get("A").attempt_count == attempts
    assert sched.state.current_task is None


def test_stale_candidate_cannot_start_third_attempt(make_scheduler):
    sched = make_scheduler(("A", "B"))
    sched.state.bootstrap_queue.clear()
    sched.registry.get("A").attempt_count = 2
    sched.registry.get("A").scout_nmse = 0.4  # 排序本来优先 A
    sched.defer_train = True
    assert sched.advance()["task"] == "B"
    assert sched.registry.get("A").status == TaskStatus.EXHAUSTED.value
    assert sched.registry.get("A").attempt_count == 2


def test_defer_gets_retry_only_at_normal_rollover(make_scheduler):
    old = make_scheduler(("A",))
    old.registry.get("A").set_status(TaskStatus.DEFER)
    old.registry.get("A").attempt_count = 1
    sched = make_scheduler(("A",))
    sched.import_pool_registry(old.registry.to_state())
    assert sched.registry.get("A").status == TaskStatus.DEFER.value
    event = sched.advance()
    assert event["action"] == "round_rollover"
    assert sched.registry.get("A").attempt_count == 1
    sched.defer_train = True
    assert sched.advance()["task"] == "A"
    assert sched.registry.get("A").attempt_count == 2


def test_import_retains_sample_id_validation(make_scheduler):
    old = make_scheduler(("A",))
    old.registry.get("A").sample_probs = {999999999: 1.0}
    old.registry.get("A").pass_sampling_snapshot = {999999999: 1.0}
    old.registry.get("A").set_status(TaskStatus.DEFER)
    sched = make_scheduler(("A",))
    issues = sched.import_pool_registry(old.registry.to_state())
    assert len(issues) == 2
    assert sched.registry.get("A").sample_probs == {}
    assert sched.registry.get("A").pass_sampling_snapshot is None
    assert sched.registry.get("A").status == TaskStatus.DEFER.value


def test_exhausted_can_still_pass_free_rescan(make_scheduler, monkeypatch):
    sched = make_scheduler(("A",))
    sched.al.scout_confirm_enabled = False
    rec = sched.registry.get("A")
    rec.set_status(TaskStatus.EXHAUSTED)
    rec.attempt_count = 2
    monkeypatch.setattr(sched, "_timed_eval", lambda *a, **kw: TrajectoryMetrics(
        task="A", split="scout", episode_ids=[0, 1],
        per_traj_mse={0: 0.01, 1: 0.01}, baseline_mse=0.1,
        mse=0.01, nmse=0.1, r2=None, n_trajs=2, metric_valid=True,
    ))
    sched._rescan()
    assert rec.status == TaskStatus.PASS.value
    assert rec.attempt_count == 2


def test_newer_hf_refreshes_only_imported_candidates(make_scheduler):
    old = make_scheduler()
    for name, status in zip(("A", "B", "C", "D"), (
        TaskStatus.PASS, TaskStatus.DEFER, TaskStatus.EXHAUSTED, TaskStatus.CANDIDATE,
    )):
        old.registry.get(name).set_status(status)
    sched = make_scheduler()
    sched.import_pool_registry(old.registry.to_state(), refresh_candidates=True)
    assert sched.state.bootstrap_queue == ["D"]
    assert sched.advance()["task"] == "D"
    assert sched.state.bootstrap_queue == []
    for name in ("A", "B", "C"):
        assert sched.registry.get(name).status == old.registry.get(name).status
