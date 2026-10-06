"""测试方案 §5 Level B —— 状态机（在**真 Scheduler** 上验证，不是只测纯函数）。

方案要求「所有状态迁移都做成确定性测试」。这里用 `tests/fake_tasks.py` 的具名假任务
主动造出每一种迁移，而不是等随机模拟碰巧触发。
"""

from __future__ import annotations

import pytest

from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.decision.state_machine import decide_after_unit, rollover_round
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from lingbotvla.auto_learning.state.registry import TaskRecord, TaskRegistry
from lingbotvla.auto_learning.testing import fake_tasks as ft
from al_fixtures import local_tmpdir
from lingbotvla.auto_learning.testing.fake_tasks import ALL, cfg_with, make_cfg
from lingbotvla.auto_learning.types import Decision, ReasonCode, TaskStatus
from al_fixtures import scheduler_of


# --------------------------------------------------------------------------- #
def _al(**kw) -> AutoLearningConfig:
    base = dict(seed=3, eval_interval_steps=50, min_steps_before_defer=100, pass_nmse=0.30)
    base.update(kw)
    return AutoLearningConfig(**base)


def _advance_until(sched: Scheduler, action: str, limit: int = 400) -> dict:
    for _ in range(limit):
        ev = sched.advance()
        if ev.get("action") == action:
            return ev
    raise AssertionError(f"{limit} 个动作内没等到 action={action}")


def _bootstrap_all(sched: Scheduler) -> None:
    """把全局 2-val 扫描跑完。"""
    while sched.state.bootstrap_queue:
        sched.advance()


# =========================================================================== #
# B01 Candidate → Training
# =========================================================================== #
def test_B01_selects_lowest_scout_and_enters_training():
    cfg = make_cfg(
        [
            ft._spec("A", curve=[0.55, 0.20]),
            ft._spec("B", curve=[0.42, 0.20]),
            ft._spec("C", curve=[0.70, 0.20]),
        ],
        al=_al(),
    )
    sched = scheduler_of(cfg)
    _bootstrap_all(sched)

    scouts = {r.task_name: r.scout_nmse for r in sched.registry}
    assert scouts["B"] < scouts["A"] < scouts["C"], scouts  # 造得对

    ev = sched.advance()
    assert ev["action"] == "select"
    assert ev["task"] == "B", "必须选 scout NMSE 最低的（离通关最近）"
    rec = sched.registry.get("B")
    assert rec.status == TaskStatus.TRAINING.value
    assert rec.attempt_count == 1
    # 不是最难、也不是字典序第一
    assert ev["task"] != "A" and ev["task"] != "C"


# =========================================================================== #
# B02 Training → PASS
# =========================================================================== #
def test_B02_pass_ends_attempt_saves_snapshot_and_leaves_candidate_pool():
    cfg = cfg_with(["easy_pass"], al=_al())
    sched = scheduler_of(cfg)
    _bootstrap_all(sched)
    sched.advance()  # select

    ev = _advance_until(sched, "train_unit")
    while ev["decision"] == "CONTINUE":
        ev = _advance_until(sched, "train_unit")

    assert ev["decision"] == "PASS"
    rec = sched.registry.get("easy_pass")
    assert rec.status == TaskStatus.PASS.value
    assert sched.state.current_task is None, "attempt 必须结束"
    assert rec.pass_sampling_snapshot is not None, "PASS 时必须存下采样分布"
    assert "easy_pass" in sched.replay_plan().tasks
    assert rec.pass_sampling_version == 1
    # 不再出现在候选选择里
    assert "easy_pass" not in [r.task_name for r in sched.registry.candidate_records()]


# =========================================================================== #
# B03 PASS 后继续训练开关
# =========================================================================== #
def test_B03_continue_after_pass_switch_changes_control_flow():
    rec = TaskRecord(task_name="t")
    rec.current_val_nmse = 0.28
    rec.lp50 = 0.10

    off = decide_after_unit(rec, _al(continue_after_pass=False), attempt_step=50)
    assert off.decision == Decision.PASS
    assert off.code == ReasonCode.VAL_BELOW_THRESHOLD.value

    on = decide_after_unit(
        rec,
        _al(continue_after_pass=True, post_pass_max_steps=200, post_pass_min_lp=0.03),
        attempt_step=50,
    )
    assert on.decision == Decision.CONTINUE
    assert on.code == ReasonCode.CONTINUE_AFTER_PASS.value


# =========================================================================== #
# B04 / B05 —— 50 step 不 DEFER、100 step 才 DEFER
# =========================================================================== #
def test_B04_B05_defer_only_after_min_steps():
    rec = TaskRecord(task_name="t")
    rec.current_val_nmse = 0.59
    rec.lp50 = 0.01
    cfg = _al(min_steps_before_defer=100)

    v50 = decide_after_unit(rec, cfg, attempt_step=50)
    assert v50.decision == Decision.CONTINUE
    assert v50.code == ReasonCode.BELOW_MIN_STEPS.value

    v100 = decide_after_unit(rec, cfg, attempt_step=100)
    assert v100.decision == Decision.DEFER
    assert v100.code == ReasonCode.LP_LOW.value


# =========================================================================== #
# B06 Overfit 提前 DEFER（含「关掉开关」的反面）
# =========================================================================== #
def test_B06_overfit_early_defer_and_switch_off():
    cfg = cfg_with(["overfit"], al=_al())
    sched = scheduler_of(cfg)
    _bootstrap_all(sched)
    sched.advance()  # select
    ev = _advance_until(sched, "train_unit")
    assert ev["decision"] == "DEFER_OVERFIT", ev
    assert any(row["overfit"] for row in sched.metrics_rows)

    # 关掉开关：同样数据不应在 step 50 就终止
    cfg2 = cfg_with(["overfit"], al=_al(early_defer_on_overfit=False))
    sched2 = scheduler_of(cfg2)
    _bootstrap_all(sched2)
    sched2.advance()
    ev2 = _advance_until(sched2, "train_unit")
    assert ev2["decision"] == "CONTINUE", "关掉开关后不应提前 DEFER"
    assert ev2["overfit"] is True, "overfit 判据本身仍然命中，只是不据此终止"


# =========================================================================== #
# B07 DEFER 在本轮内不可再被选中
# =========================================================================== #
def test_B07_deferred_task_not_selectable_in_same_round():
    cfg = cfg_with(["plateau", "slow_learn"], al=_al())
    sched = scheduler_of(cfg)
    _bootstrap_all(sched)

    # plateau 的 scout 更低 ⇒ 先被选；它练不动 ⇒ DEFER
    ev = sched.advance()
    assert ev["task"] == "plateau"
    while sched.state.current_task is not None:
        sched.advance()
    assert sched.registry.get("plateau").status == TaskStatus.DEFER.value

    # 本轮还没结束（slow_learn 仍是 candidate）⇒ plateau 不可再选，哪怕它 NMSE 更低
    cands = [r.task_name for r in sched.registry.candidate_records()]
    assert "plateau" not in cands
    ev2 = sched.advance()
    assert ev2["task"] == "slow_learn", "本轮内必须跳过 DEFER 的任务"


# =========================================================================== #
# B08 新 round 重新激活 DEFER
# =========================================================================== #
def test_B08_new_round_reactivates_defer():
    cfg = _al(max_attempts_per_task=2)
    a = TaskRecord(task_name="a")
    a.set_status(TaskStatus.DEFER)
    a.attempt_count = 1
    b = TaskRecord(task_name="b")
    b.set_status(TaskStatus.DEFER)
    b.attempt_count = 2  # 预算用尽 ⇒ 不该被激活
    reg = TaskRegistry({"a": a, "b": b})

    promoted = rollover_round(reg, cfg)
    assert promoted == ["a"]
    assert a.status == TaskStatus.CANDIDATE.value
    assert a.last_transition_code == ReasonCode.ROUND_ROLLOVER.value
    assert b.status == TaskStatus.DEFER.value


# =========================================================================== #
# B09 Attempt exhaustion
# =========================================================================== #
def test_B09_attempt_exhaustion_after_two_failures():
    cfg = cfg_with(["unlearnable"], al=_al(max_attempts_per_task=2))
    sched = scheduler_of(cfg)
    sched.run(max_actions=400)

    rec = sched.registry.get("unlearnable")
    assert rec.attempt_count == 2
    assert rec.status == TaskStatus.EXHAUSTED.value
    assert rec.last_transition_code == ReasonCode.ATTEMPT_BUDGET_EXHAUSTED.value
    assert sched.state.finished


# =========================================================================== #
# B10 PASS → Forgotten → Candidate（还有预算）
# =========================================================================== #
def test_B10_forgotten_reopens_when_budget_left():
    cfg = cfg_with(["forgotten", "unlearnable"], al=_al(max_attempts_per_task=3))
    cfg.auto_learning.review_after_task_transitions = 1
    sched = scheduler_of(cfg)
    sched.run(max_actions=600)

    rec = sched.registry.get("forgotten")
    assert rec.reopen_count >= 1, "应当被 review 抓出遗忘并回炉"
    assert rec.ever_passed


# =========================================================================== #
# B11 Forgotten 但没预算 ⇒ EXHAUSTED，不能无限回炉
# =========================================================================== #
def test_B11_forgotten_without_budget_goes_exhausted():
    cfg = cfg_with(["forgotten", "unlearnable"], al=_al(max_attempts_per_task=1))
    cfg.auto_learning.review_after_task_transitions = 1
    sched = scheduler_of(cfg)
    sched.run(max_actions=600)

    rec = sched.registry.get("forgotten")
    assert rec.status == TaskStatus.EXHAUSTED.value, "预算用尽后不能再回炉"
    assert rec.last_transition_code == ReasonCode.ATTEMPT_BUDGET_EXHAUSTED.value
    # 被闸门拒掉的回炉不计入 reopen_count（它数的是「批准的回炉次数」）
    assert rec.reopen_count == 0


# =========================================================================== #
# 附带：11 个具名假任务都必须能真的跑起来（否则库本身是坏的）
# =========================================================================== #
@pytest.mark.parametrize("name", sorted(ALL))
def test_every_fake_task_runs(name):
    cfg = cfg_with([name], al=_al())
    sched = scheduler_of(cfg)
    sched.run(max_actions=400)
    assert sched.state.finished, f"{name} 没能跑到停止"
    assert sched.state.units_run <= 60
