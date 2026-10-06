"""状态机 / 判定逻辑（文档 §44 「状态机」那一组）。"""

from __future__ import annotations

from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.state.registry import TaskRecord
from lingbotvla.auto_learning.decision.state_machine import (
    apply_defer,
    apply_pass,
    apply_reopen,
    decide_after_unit,
    rollover_round,
    should_rescue,
    start_attempt,
)
from lingbotvla.auto_learning.state.registry import TaskRegistry
from lingbotvla.auto_learning.types import Decision, TaskStatus


def _cfg(**kw) -> AutoLearningConfig:
    base = dict(pass_nmse=0.30, min_lp50=0.05, min_steps_before_defer=100, eval_interval_steps=50)
    base.update(kw)
    return AutoLearningConfig(**base)


def _rec(name="t", **kw) -> TaskRecord:
    return TaskRecord(task_name=name, **kw)


# --------------------------------------------------------------------------- #
def test_pass_immediately_when_under_threshold():
    """50 step 即达标能立刻 PASS（文档 §44）。"""
    rec = _rec()
    rec.current_val_nmse = 0.28
    rec.lp50 = 0.01
    d, reason = decide_after_unit(rec, _cfg(), attempt_step=50)
    assert d == Decision.PASS, reason


def test_continue_before_min_steps_even_with_low_lp():
    rec = _rec()
    rec.current_val_nmse = 0.60
    rec.lp50 = -0.02
    d, _ = decide_after_unit(rec, _cfg(), attempt_step=50)
    assert d == Decision.CONTINUE


def test_defer_after_budget_with_low_lp():
    rec = _rec()
    rec.current_val_nmse = 0.60
    rec.lp50 = 0.01
    d, reason = decide_after_unit(rec, _cfg(), attempt_step=100)
    assert d == Decision.DEFER
    assert "lp50" in reason


def test_continue_when_lp_high():
    rec = _rec()
    rec.current_val_nmse = 0.60
    rec.lp50 = 0.20
    d, _ = decide_after_unit(rec, _cfg(), attempt_step=100)
    assert d == Decision.CONTINUE


def test_overfit_defers_early():
    rec = _rec()
    rec.current_val_nmse = 0.50
    rec.lp50 = -0.01
    rec.overfit = True
    d, _ = decide_after_unit(rec, _cfg(), attempt_step=50)
    assert d == Decision.DEFER_OVERFIT


def test_continue_after_pass_keeps_training():
    cfg = _cfg(continue_after_pass=True, post_pass_max_steps=200, post_pass_min_lp=0.03)
    rec = _rec()
    rec.current_val_nmse = 0.25
    rec.lp50 = 0.10
    d, _ = decide_after_unit(rec, cfg, attempt_step=50)
    assert d == Decision.CONTINUE
    rec.lp50 = 0.01
    d, _ = decide_after_unit(rec, cfg, attempt_step=50)
    assert d == Decision.PASS


# --------------------------------------------------------------------------- #
def test_first_failure_defers_second_exhausts():
    """第二次仍然失败 ⇒ EXHAUSTED（文档 §16）。"""
    cfg = _cfg(max_attempts_per_task=2)
    rec = _rec()
    start_attempt(rec)
    assert apply_defer(rec, cfg, "x") == TaskStatus.DEFER
    assert rec.status == TaskStatus.DEFER.value

    start_attempt(rec)
    assert apply_defer(rec, cfg, "x") == TaskStatus.EXHAUSTED
    assert rec.status == TaskStatus.EXHAUSTED.value


def test_rollover_promotes_only_with_budget():
    cfg = _cfg(max_attempts_per_task=2)
    a = _rec("a")
    b = _rec("b")
    a.set_status(TaskStatus.DEFER)
    a.attempt_count = 1
    b.set_status(TaskStatus.DEFER)
    b.attempt_count = 2
    reg = TaskRegistry({"a": a, "b": b})
    promoted = rollover_round(reg, cfg)
    assert promoted == ["a"]
    assert a.status == TaskStatus.CANDIDATE.value
    assert b.status == TaskStatus.DEFER.value


def test_reopen_respects_attempt_cap():
    """文档 §36：遗忘回炉也不能无限循环。"""
    cfg = _cfg(max_attempts_per_task=2)
    ok = _rec("ok")
    ok.attempt_count = 1
    apply_pass(ok, cfg, "pass")
    assert apply_reopen(ok, cfg, "forgot") == TaskStatus.CANDIDATE
    assert ok.reopen_count == 1
    assert ok.forgotten

    capped = _rec("capped")
    capped.attempt_count = 2
    apply_pass(capped, cfg, "pass")
    assert apply_reopen(capped, cfg, "forgot") == TaskStatus.EXHAUSTED


def test_pass_updates_best_monotonically():
    cfg = _cfg()
    rec = _rec()
    rec.current_val_nmse = 0.20
    apply_pass(rec, cfg, "p")
    rec.current_val_nmse = 0.26
    apply_pass(rec, cfg, "p")
    assert rec.best_nmse == pytest_approx(0.20)


def pytest_approx(v):
    import pytest

    return pytest.approx(v)


def test_should_rescue_gate():
    cfg = _cfg(defer_resample_retry=True)
    assert should_rescue(cfg, False) is True
    assert should_rescue(cfg, True) is False
    assert should_rescue(_cfg(defer_resample_retry=False), False) is False
