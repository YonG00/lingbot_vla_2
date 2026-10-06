"""测试方案 §6 Level C —— 全局调度逻辑（多任务共同存在时的行为）。"""

from __future__ import annotations

from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from lingbotvla.auto_learning.testing import fake_tasks as ft
from lingbotvla.auto_learning.testing.fake_tasks import cfg_with, make_cfg
from lingbotvla.auto_learning.types import ReasonCode, TaskStatus
from al_fixtures import scheduler_of


def _al(**kw) -> AutoLearningConfig:
    base = dict(seed=3, eval_interval_steps=50, min_steps_before_defer=100, pass_nmse=0.30)
    base.update(kw)
    return AutoLearningConfig(**base)


def _bootstrap_all(sched: Scheduler) -> None:
    while sched.state.bootstrap_queue:
        sched.advance()


# =========================================================================== #
# C01 Initial 2→4 Auto-PASS
# =========================================================================== #
def test_C01_two_traj_pass_must_survive_four_traj_confirm():
    """2 条 val 看着达标、补到 4 条却不行 ⇒ 不能 PASS（文档 §7.2）。"""
    cfg = make_cfg(
        [
            ft._spec("A", curve=[0.30, 0.22], scout_nmse_override=0.20, confirm_nmse_override=0.22),
            ft._spec("B", curve=[0.45, 0.25], scout_nmse_override=0.25, confirm_nmse_override=0.42),
            ft._spec("C", curve=[0.60, 0.25]),
        ],
        al=_al(),
    )
    sched = scheduler_of(cfg)
    _bootstrap_all(sched)

    assert sched.registry.get("A").status == TaskStatus.PASS.value
    assert sched.registry.get("A").last_transition_code == ReasonCode.BOOTSTRAP_PASS.value
    assert sched.registry.get("B").status == TaskStatus.CANDIDATE.value
    assert sched.registry.get("C").status == TaskStatus.CANDIDATE.value
    # B 的 scout 估计被 4-val 的更可信值覆盖
    assert sched.registry.get("B").scout_nmse > 0.30
    # 免费的 PASS 不占「新增技能」预算
    assert sched.state.auto_passed == ["A"]
    assert sched.state.newly_passed == []


# =========================================================================== #
# C02 自动挑「最近的一关」
# =========================================================================== #
def test_C02_picks_nearest_task_not_hardest_not_alphabetical():
    cfg = make_cfg(
        [
            ft._spec("A", curve=[0.80, 0.20]),
            ft._spec("B", curve=[0.46, 0.20]),
            ft._spec("C", curve=[0.62, 0.20]),
        ],
        al=_al(),
    )
    sched = scheduler_of(cfg)
    _bootstrap_all(sched)
    ev = sched.advance()
    assert ev["action"] == "select"
    assert ev["task"] == "B"
    assert ev["task"] != "A"  # 不是最难
    assert ev["task"] != "C"  # 不是字典序第二


# =========================================================================== #
# C03 Transfer 后重新排序 ⇒ 未训练的任务被 rescan 自动 PASS
# =========================================================================== #
def test_C03_transfer_reranks_and_auto_passes_untrained_task():
    """自动学习最有价值的一条路径：训一个技能，别的技能自己变好（文档 §46）。"""
    a = ft._spec("source", curve=[0.40, 0.30, 0.24], group="pair", transfer=0.90)
    b = ft._spec("target", curve=[0.50, 0.28], group="pair", transfer=0.90)
    cfg = make_cfg([a, b], al=_al())
    sched = scheduler_of(cfg)
    _bootstrap_all(sched)

    assert sched.registry.get("target").status == TaskStatus.CANDIDATE.value
    # source 先被选（scout 更低）
    assert sched.advance()["task"] == "source"
    sched.run(max_actions=400)

    rec = sched.registry.get("target")
    assert rec.status == TaskStatus.PASS.value
    assert rec.total_task_steps == 0, "target 一次都没训过"
    assert rec.last_transition_code == ReasonCode.RESCAN_PASS.value
    assert "target" in sched.state.auto_passed
    assert "target" not in sched.state.newly_passed


# =========================================================================== #
# C04 一个任务不得无限霸占
# =========================================================================== #
def test_C04_unlearnable_does_not_block_the_rest():
    cfg = make_cfg(
        [
            ft.easy_pass("A"),
            ft.unlearnable("B"),
            ft.easy_pass("C"),
            ft.easy_pass("D"),
        ],
        al=_al(max_attempts_per_task=2),
    )
    sched = scheduler_of(cfg)
    sched.run(max_actions=800)

    assert sched.registry.get("B").status == TaskStatus.EXHAUSTED.value
    for name in ("A", "C", "D"):
        assert sched.registry.get(name).status == TaskStatus.PASS.value, name
    assert sched.state.finished
    # 学不会的任务确实被尝试过两次，而不是一次就放弃
    assert sched.registry.get("B").attempt_count == 2


def test_C04b_slow_learn_keeps_continuing_while_lp_is_high():
    """`slow_learn` 的意义是「持续进步」：LP 一直够高 ⇒ 不能被提前 DEFER。"""
    cfg = cfg_with(["slow_learn"], al=_al())
    sched = scheduler_of(cfg)
    while sched.state.bootstrap_queue:
        sched.advance()
    sched.advance()  # select

    decisions = []
    for _ in range(3):
        ev = sched.advance()
        assert ev["action"] == "train_unit"
        decisions.append((ev["decision"], ev["lp50"]))
    assert all(d == "CONTINUE" for d, _ in decisions), decisions
    assert all(lp is not None and lp >= cfg.auto_learning.min_lp50 for _, lp in decisions)
    # 它最终因为曲线走平而 DEFER（不是 PASS —— 曲线最低只到 0.42）
    sched.run(max_actions=400)
    assert sched.registry.get("slow_learn").status == TaskStatus.EXHAUSTED.value


# =========================================================================== #
# C05 全局停止
# =========================================================================== #
def test_C05_global_stop_is_clean():
    cfg = make_cfg([ft.easy_pass("A"), ft.already_known("B")], al=_al())
    sched = scheduler_of(cfg)
    sched.run(max_actions=400)

    assert sched.state.finished
    assert sched.state.stop_reason == "all_tasks_resolved"
    assert sched.registry.counts()[TaskStatus.CANDIDATE.value] == 0
    assert sched.registry.counts()[TaskStatus.DEFER.value] == 0
    # 再 advance 一次不应该改变任何东西（不能空转 / 反复 rescan）
    before = (sched.state.units_run, sched.state.transition_count, sched.state.eval_events)
    sched.advance()
    after = (sched.state.units_run, sched.state.transition_count, sched.state.eval_events)
    assert before == after


# =========================================================================== #
# C06 max_new_tasks_attempted_this_run / max_new_tasks_passed_this_run
# =========================================================================== #
def test_C06_attempted_cap_counts_attempts_not_passes():
    """口径 = 「本次主动**尝试**过多少个新任务」，**不是**「newly PASS 多少个」。

    5 个都学不会的任务 + cap=2 ⇒ 只尝试 2 个（旧口径会跑满 5 个）。
    """
    cfg = make_cfg(
        [ft.unlearnable(f"U{i}") for i in range(5)],
        al=_al(max_new_tasks_attempted_this_run=2, max_global_steps=100000),
    )
    sched = scheduler_of(cfg)
    sched.run(max_actions=800)

    assert sched.state.finished
    assert len(sched.state.trained_tasks) == 2, "只应主动尝试 2 个新任务"
    assert sched.state.stop_reason == "max_new_tasks_attempted_reached(2)"
    assert sched.state.newly_passed == [], "一个都没通过，但这不影响 attempted 口径"


def test_C06b_passed_cap_is_optional():
    """可选的 `max_new_tasks_passed_this_run`：按 newly PASS 数收工。"""
    cfg = make_cfg(
        [ft.easy_pass(f"L{i}") for i in range(6)],
        al=_al(max_new_tasks_passed_this_run=2, max_global_steps=100000),
    )
    sched = scheduler_of(cfg)
    sched.run(max_actions=800)

    assert sched.state.finished
    assert sched.state.stop_reason == "max_new_tasks_passed_reached(2)"
    assert len(sched.state.newly_passed) == 2
    assert len(sched.state.trained_tasks) == 2


def test_C06c_bootstrap_auto_pass_consumes_neither_budget():
    cfg = make_cfg(
        [ft.already_known("K1"), ft.already_known("K2")]
        + [ft.easy_pass(f"L{i}") for i in range(4)],
        al=_al(max_new_tasks_attempted_this_run=2, max_global_steps=100000),
    )
    sched = scheduler_of(cfg)
    sched.run(max_actions=800)

    assert set(sched.state.auto_passed) == {"K1", "K2"}
    assert len(sched.state.trained_tasks) == 2, "免费 PASS 不能吃掉「尝试新任务」的预算"
    assert sched.state.stop_reason == "max_new_tasks_attempted_reached(2)"


# =========================================================================== #
# C06d / C06e —— 新增额度只管「真正的新任务」，维护不吃额度
# （review v0.2 P1 指出的语义）
# =========================================================================== #
def _cap_scenario(cap_new=None, cap_passed=None):
    """A 在 bootstrap 免费 PASS、且极易被忘；B 需要训 4 个 unit。"""
    a = ft._spec("A", curve=[0.18], forget_rate=1.8, degrade_slope=0.20)
    b = ft._spec("B", curve=[0.60, 0.50, 0.40, 0.28], forget_rate=0.01, degrade_slope=0.01)
    return make_cfg(
        [a, b],
        al=_al(
            review_after_task_transitions=1,
            max_attempts_per_task=2,
            max_new_tasks_attempted_this_run=cap_new,
            max_new_tasks_passed_this_run=cap_passed,
            max_global_steps=100000,
        ),
    )


def test_C06d_maintenance_reopen_does_not_consume_new_task_quota():
    """review v0.2 的场景：启动 auto-PASS 的任务被忘掉后，**额度用完了也要能回炉**。

    旧口径用 `attempt_count > 0` 筛「可维护的任务」，而 auto-PASS 的任务
    `attempt_count == 0` ⇒ 会被排除 ⇒ scheduler 直接停止、A 再也修不回来。
    """
    sched = scheduler_of(_cap_scenario(cap_new=1))
    sched.run(max_actions=800)

    assert sched.state.finished
    a = sched.registry.get("A")
    assert a.ever_passed is True, "A 应当在 bootstrap 免费 PASS 过"
    assert a.reopen_count >= 1, "A 应当被忘掉并回炉"
    # 关键：额度用完之后，A **仍然被当作 maintenance 训练了**（而不是直接停跑）
    assert a.attempt_count >= 1, "A 应当作为维护任务真的被训过"
    assert sched.state.repassed == ["A"], "A 的再次通过算「恢复」，不算首次通过"
    # 额度只被真正的新任务消耗
    assert sched.state.new_tasks_attempted == ["B"]
    assert sched.state.newly_passed == ["B"]
    assert "max_new_tasks_attempted_reached" not in sched.state.stop_reason


def test_C06e_repass_does_not_consume_passed_quota():
    """旧任务恢复通过**不占** `max_new_tasks_passed_this_run`。

    `cap_passed=2`：如果 A 的「恢复通过」被误算成 newly PASS，`newly_passed` 就会是
    `['B','A']` 两条 —— 用它当断言就能抓到。
    （cap=1 时 B 一通过就该收工，A 根本没机会被维护，那个场景测不出这条。）
    """
    sched = scheduler_of(_cap_scenario(cap_passed=2))
    sched.run(max_actions=800)

    assert sched.state.finished
    assert sched.registry.get("A").attempt_count >= 1, "A 被当作维护任务训练过"
    assert sched.state.newly_passed == ["B"], "只有 B 是首次通过（A 的恢复不算）"
    assert sched.state.repassed == ["A"]
    assert "max_new_tasks_passed_reached" not in sched.state.stop_reason


def test_C06f_new_task_quota_still_blocks_brand_new_tasks():
    """额度仍然要拦住**真正的新任务**（别把闸门拆了）。"""
    cfg = make_cfg(
        [ft.easy_pass(f"L{i}") for i in range(6)],
        al=_al(max_new_tasks_attempted_this_run=2, max_global_steps=100000),
    )
    sched = scheduler_of(cfg)
    sched.run(max_actions=800)
    assert sched.state.stop_reason == "max_new_tasks_attempted_reached(2)"
    assert len(sched.state.new_tasks_attempted) == 2
    assert len(sched.state.trained_tasks) == 2


def test_C06g_trained_tasks_is_superset_of_new_tasks():
    """`trained_tasks` 是「训练过的全部」；`new_tasks_attempted` 只是其中真新的那些。"""
    sched = scheduler_of(_cap_scenario(cap_new=1))
    sched.run(max_actions=800)
    assert set(sched.state.new_tasks_attempted) <= set(sched.state.trained_tasks)
    assert "A" in sched.state.trained_tasks  # A 回炉时被训过
    assert "A" not in sched.state.new_tasks_attempted  # 但它不是新任务


