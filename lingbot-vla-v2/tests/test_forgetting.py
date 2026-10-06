"""测试方案 §9 Level F —— Review 与遗忘。"""

from __future__ import annotations

import pytest

from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.decision.metrics import is_forgotten
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from lingbotvla.auto_learning.testing import fake_tasks as ft
from lingbotvla.auto_learning.testing.fake_tasks import cfg_with, make_cfg
from lingbotvla.auto_learning.types import ReasonCode, TaskStatus
from al_fixtures import scheduler_of


def _al(**kw) -> AutoLearningConfig:
    base = dict(seed=3, eval_interval_steps=50, min_steps_before_defer=100, pass_nmse=0.30)
    base.update(kw)
    return AutoLearningConfig(**base)


# =========================================================================== #
# F01 Review 节奏
# =========================================================================== #
def test_F01_review_fires_on_every_nth_transition_only():
    cfg = make_cfg(
        [ft.easy_pass(f"T{i}") for i in range(6)],
        al=_al(review_after_task_transitions=2, max_new_tasks_attempted_this_run=None),
    )
    sched = scheduler_of(cfg)
    sched.run(max_actions=800)

    review_at = [
        row["transition"]
        for row in sched.state.transitions
        if row["kind"] == "REOPEN" or row.get("review")
    ]
    counts = [t for t in sched.state.transitions if t["kind"] in ("PASS", "DEFER", "EXHAUSTED")]
    assert len(counts) >= 4
    # 从事件流里数 review 次数
    n_reviews = sum(1 for e in sched.events if e.get("action") == "review")
    assert n_reviews == len(counts) // 2, (n_reviews, len(counts), review_at)


def test_F01_rescan_is_not_a_transition():
    """纯 candidate rescan 不能算 transition，否则 review 节奏会被打乱。"""
    cfg = make_cfg(
        [ft.easy_pass("A"), ft.easy_pass("B"), ft.easy_pass("C")], al=_al()
    )
    sched = scheduler_of(cfg)
    sched.run(max_actions=800)
    kinds = {t["kind"] for t in sched.state.transitions}
    assert kinds <= {"PASS", "DEFER", "DEFER_OVERFIT", "EXHAUSTED", "REOPEN", "REOPEN_EXHAUSTED"}
    # transition_count 必须等于 transitions 条数
    assert sched.state.transition_count == len(sched.state.transitions)


# =========================================================================== #
# F02 2-traj 疑似 → 4-traj 确认 → 恢复正常则保持 PASS
# =========================================================================== #
def test_F02_confirm_can_rescue_a_false_alarm():
    """2 条 val 看着退化、补到 4 条正常 ⇒ 不能误回炉（文档 §32）。"""
    spec = ft.already_known("A")
    # 让 review 的 2-traj scout 偏高（疑似退化），4-traj confirm 正常
    spec.curve = [0.18]
    spec.forget_rate = 0.0
    spec.degrade_slope = 0.0
    cfg = make_cfg([spec, ft.unlearnable("B")], al=_al(review_after_task_transitions=1))
    sched = scheduler_of(cfg)
    sched.run(max_actions=600)

    rec = sched.registry.get("A")
    assert rec.status == TaskStatus.PASS.value
    assert rec.reopen_count == 0, "A 从未退化，不该被回炉"
    outcomes = [
        o
        for e in sched.events
        if e.get("action") == "review"
        for o in e["outcomes"]
        if o["task"] == "A"
    ]
    assert outcomes and all(o["action"] in ("ok", "ok_after_confirm") for o in outcomes)


# =========================================================================== #
# F03 / F04 / F05 —— 遗忘判定的三种情形
# =========================================================================== #
def test_F03_below_pass_line_always_forgotten():
    """best=.25 current=.31 ⇒ 哪怕只退化 24%，掉线就得回炉。"""
    assert is_forgotten(0.31, 0.25, pass_nmse=0.30, forget_relative_threshold=0.30)


def test_F04_severe_relative_drop_forgotten_even_if_still_passing():
    """best=.08 current=.20 ⇒ 仍低于 0.30，但退化 150%。"""
    assert is_forgotten(0.20, 0.08, pass_nmse=0.30, forget_relative_threshold=0.30)


def test_F05_small_wobble_not_forgotten():
    """best=.20 current=.22 ⇒ 保持 PASS。"""
    assert not is_forgotten(0.22, 0.20, pass_nmse=0.30, forget_relative_threshold=0.30)


def test_F05b_mild_drop_task_stays_pass_in_a_real_run():
    cfg = cfg_with(["mild_drop", "unlearnable"], al=_al(review_after_task_transitions=1))
    sched = scheduler_of(cfg)
    sched.run(max_actions=600)
    rec = sched.registry.get("mild_drop")
    assert rec.status == TaskStatus.PASS.value
    assert rec.reopen_count == 0


# =========================================================================== #
# F06 回炉不能无限发生
# =========================================================================== #
def test_F06_reopen_is_bounded_by_attempt_budget():
    cfg = cfg_with(["forgotten", "unlearnable"], al=_al(max_attempts_per_task=1))
    cfg.auto_learning.review_after_task_transitions = 1
    sched = scheduler_of(cfg)
    sched.run(max_actions=800)

    rec = sched.registry.get("forgotten")
    assert rec.status == TaskStatus.EXHAUSTED.value
    assert rec.last_transition_code == ReasonCode.ATTEMPT_BUDGET_EXHAUSTED.value
    # `reopen_count` 数的是「**批准**的回炉次数」—— 被预算闸门拒掉的不计
    assert rec.reopen_count == 0, "预算用尽 ⇒ 回炉被拒 ⇒ 不该记一次" 


def test_F06b_reopen_reason_code_distinguishes_two_cases():
    """掉出线 vs 相对退化 —— 两种原因必须能分辨（文档 §34）。"""
    cfg = cfg_with(["forgotten", "unlearnable"], al=_al(max_attempts_per_task=3))
    cfg.auto_learning.review_after_task_transitions = 1
    sched = scheduler_of(cfg)
    sched.run(max_actions=800)

    codes = {t["code"] for t in sched.state.transitions if t["kind"] == "REOPEN"}
    assert codes, "应当发生过 REOPEN"
    assert codes <= {
        ReasonCode.FORGOTTEN_BELOW_PASS_LINE.value,
        ReasonCode.FORGOTTEN_RELATIVE_DEGRADATION.value,
    }
