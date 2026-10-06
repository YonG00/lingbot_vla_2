"""测试方案 §12 Level I —— 错误注入 / 防御性测试。

Demo 阶段很适合主动制造错误：一个坏指标、一个空池子、一个对不上的 sample_id。
这些在真实仓库里会以「跑了几小时才发现」的形式出现。
"""

from __future__ import annotations

import math
import random

import pytest

from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.decision.state_machine import decide_after_unit
from lingbotvla.auto_learning.testing.sim import SimulatedEvaluator
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from lingbotvla.auto_learning.sampling.rng import validate_probs
from lingbotvla.auto_learning.sampling.sampler import BatchSampler
from lingbotvla.auto_learning.ports import ReplayPlan
from lingbotvla.auto_learning.state.registry import TaskRecord, TaskRegistry
from lingbotvla.auto_learning.testing import fake_tasks as ft
from lingbotvla.auto_learning.testing.fake_tasks import cfg_with, make_cfg
from lingbotvla.auto_learning.types import Decision, ReasonCode, TaskStatus
from al_fixtures import catalog_of, default_al, replay_plan_of, resolver_of, scheduler_of, train_request


def _al(**kw) -> AutoLearningConfig:
    base = dict(seed=3, eval_interval_steps=50, min_steps_before_defer=100, pass_nmse=0.30)
    base.update(kw)
    return AutoLearningConfig(**base)


class _CorruptEvaluator(SimulatedEvaluator):
    """把指定任务的 NMSE 换成 NaN / Inf，模拟坏指标。"""

    def __init__(self, world, cfg, bad, value=float("nan")):
        super().__init__(world, cfg)
        self.bad = set(bad)
        self.value = value

    def evaluate(self, task, split, episode_ids):
        m = super().evaluate(task, split, episode_ids)
        if task in self.bad:
            m.nmse = self.value
            m.mse = self.value
        return m


# =========================================================================== #
# I01 NaN / Inf Metric
# =========================================================================== #
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_I01_non_finite_metric_is_excluded_never_enters_argmin(bad_value):
    cfg = make_cfg(
        [ft._spec("good", curve=[0.45, 0.22]), ft._spec("bad", curve=[0.10])],
        al=_al(),
    )
    sched = scheduler_of(cfg)
    sched.evaluator = _CorruptEvaluator(sched.extra_state, cfg.sim, {"bad"}, bad_value)
    sched.reviewer.evaluator = sched.evaluator
    sched.run(max_actions=400)

    bad = sched.registry.get("bad")
    assert bad.status == TaskStatus.CANDIDATE.value
    assert bad.metric_valid is False
    assert bad.last_transition_code == ReasonCode.METRIC_INVALID.value
    # 从没被选中过（NaN 的 scout 值比任何数都「低」的话，就会抢走选择权）
    assert bad.attempt_count == 0
    assert "bad" not in sched.state.trained_tasks
    assert "bad" not in [r.task_name for r in sched.registry.candidate_records()]
    assert sched.registry.get("good").status == TaskStatus.PASS.value


def test_I01_non_finite_val_during_attempt_is_bounded():
    """attempt 中途拿到 NaN：先 CONTINUE，用满预算后 DEFER（不能无限转）。"""
    rec = TaskRecord(task_name="t")
    rec.current_val_nmse = float("nan")
    cfg = _al()
    v50 = decide_after_unit(rec, cfg, attempt_step=50)
    assert v50.decision == Decision.CONTINUE
    assert v50.code == ReasonCode.METRIC_INVALID.value
    v100 = decide_after_unit(rec, cfg, attempt_step=100)
    assert v100.decision == Decision.DEFER


def test_I01_is_finite_metric_helper():
    from lingbotvla.auto_learning.decision.metrics import is_finite_metric

    assert is_finite_metric(0.3)
    assert not is_finite_metric(float("nan"))
    assert not is_finite_metric(float("inf"))
    assert not is_finite_metric(None)
    assert not is_finite_metric("0.3")


# =========================================================================== #
# I02 空 Candidate Pool
# =========================================================================== #
def test_I02_empty_candidate_with_defer_triggers_new_round():
    cfg = cfg_with(["unlearnable"], al=_al(max_attempts_per_task=2))
    sched = scheduler_of(cfg)
    sched.run(max_actions=400)
    kinds = [t["kind"] for t in sched.state.transitions]
    assert kinds.count("DEFER") == 1
    assert kinds.count("EXHAUSTED") == 1
    assert sched.state.round == 2, "第一轮结束后必须开第二轮"
    assert sched.state.stop_reason == "all_tasks_resolved"


def test_I02_empty_candidate_and_no_defer_stops_cleanly():
    cfg = cfg_with(["already_known"], al=_al())
    sched = scheduler_of(cfg)
    sched.run(max_actions=100)
    assert sched.state.stop_reason == "all_tasks_resolved"
    assert sched.state.units_run == 0
    assert not sched.state.transitions


# =========================================================================== #
# I03 空 PASS Pool
# =========================================================================== #
def _req(cfg, task, probs, replay_tasks=()):
    return train_request(cfg, task, probs, replay_tasks=list(replay_tasks))


def _al_cfg(*specs, **kw):
    from al_fixtures import default_al

    return make_cfg(list(specs), al=default_al(**kw))


# =========================================================================== #
# I03 空 PASS Pool
# =========================================================================== #
def test_I03_empty_pass_pool_degrades_to_all_new():
    cfg = _al_cfg(ft._spec("t0", curve=[0.5, 0.2]))
    catalog = catalog_of(cfg)
    rng = random.Random(0)
    sampler = BatchSampler(cfg.auto_learning, resolver_of(cfg), catalog, rng)
    probs = {sid: 1.0 for sid in catalog.entry("t0").train_sample_ids}
    comp = sampler.build(sampler.prepare(_req(cfg, "t0", probs)))
    assert (comp.n_new, comp.n_old) == (10, 0)


# =========================================================================== #
# I04 Replay slots > PASS task 数量
# =========================================================================== #
def test_I04_replay_slots_may_exceed_pass_tasks():
    from lingbotvla.auto_learning.sampling.replay import sample_replay_refs
    from al_fixtures import replay_plan_of

    cfg = _al_cfg(ft._spec("A", curve=[0.5, 0.2]))
    plan = replay_plan_of(cfg, ["A"])
    slots = __import__("lingbotvla.auto_learning.sampling.replay", fromlist=["prepare_slots"]).prepare_slots(plan, 3)
    refs = sample_replay_refs(slots, 3, resolver_of(cfg), random.Random(0))
    assert len(refs) == 3, "只有一个 PASS 任务时也要能组满 3 个 slot"
    assert {r.task for r in refs} == {"A"}


# =========================================================================== #
# I05 Sample ID 不存在 —— 必须报错，不能 silently skip
# =========================================================================== #
def test_I05_stale_snapshot_ids_raise_instead_of_silent_skip():
    from lingbotvla.auto_learning.sampling.replay import prepare_slots
    from lingbotvla.auto_learning.ports import ReplayPlan

    plan = ReplayPlan(
        tasks=["A"],
        probs={"A": {999: 1.0, 1000: 1.0}},  # 这些 id 不在数据集里
        sample_ids={"A": [1, 2, 3]},
    )
    with pytest.raises(RuntimeError) as err:
        prepare_slots(plan, 1)
    assert "sample_id" in str(err.value)


def test_I05_pass_task_without_samples_raises():
    from lingbotvla.auto_learning.sampling.replay import prepare_slots
    from lingbotvla.auto_learning.ports import ReplayPlan

    plan = ReplayPlan(tasks=["A"], probs={"A": None}, sample_ids={"A": []})
    with pytest.raises(RuntimeError):
        prepare_slots(plan, 1)


def test_I05_new_probs_must_belong_to_the_task():
    cfg = _al_cfg(ft._spec("A", curve=[0.5, 0.2]), ft._spec("B", curve=[0.5, 0.2]))
    catalog = catalog_of(cfg)
    sampler = BatchSampler(cfg.auto_learning, resolver_of(cfg), catalog, random.Random(0))
    foreign = catalog.entry("B").train_sample_ids[0]
    with pytest.raises(RuntimeError):
        sampler.prepare(_req(cfg, "A", {foreign: 1.0}))


# =========================================================================== #
# I06 Sampling Probability 非法
# =========================================================================== #
def test_I06_validate_probs_rejects_bad_values():
    with pytest.raises(ValueError):
        validate_probs({1: -0.5, 2: 1.0})
    with pytest.raises(ValueError):
        validate_probs({1: float("nan")})
    with pytest.raises(ValueError):
        validate_probs({1: 0.0, 2: 0.0})
    with pytest.raises(ValueError):
        validate_probs({})
    validate_probs({1: 0.0, 2: 1.0})  # 合法：允许个别 0 权重


def test_I06_sampler_refuses_illegal_probs():
    cfg = _al_cfg(ft._spec("A", curve=[0.5, 0.2]))
    catalog = catalog_of(cfg)
    sampler = BatchSampler(cfg.auto_learning, resolver_of(cfg), catalog, random.Random(0))
    sid = catalog.entry("A").train_sample_ids[0]
    for bad in ({sid: -1.0}, {sid: float("nan")}, {sid: 0.0}):
        with pytest.raises(ValueError):
            sampler.prepare(_req(cfg, "A", bad))


# =========================================================================== #
# I07 / I08 —— probe 不足 fail-fast、旧配置字段名
# =========================================================================== #
def test_I07_probe_shortfall_fails_fast():
    """配置要求 4 条 val / 4 条 train 探针，数据不够就必须报错 —— 不能静默少用几条。

    静默降级会让「4-val PASS 判断」悄悄退化成 2-val。
    """
    cfg = make_cfg([ft._spec("t0", curve=[0.5, 0.2], n_train_trajs=2)], al=_al())
    with pytest.raises(ValueError) as err:
        TaskRegistry.from_catalog(catalog_of(cfg), cfg.auto_learning)
    assert "train" in str(err.value)

    cfg2 = make_cfg(
        [ft._spec("t0", curve=[0.5, 0.2], n_train_trajs=10, n_val_trajs=2)], al=_al()
    )
    with pytest.raises(ValueError) as err2:
        TaskRegistry.from_catalog(catalog_of(cfg2), cfg2.auto_learning)
    assert "val" in str(err2.value)


def test_I07b_sufficient_probes_build_fine():
    cfg = make_cfg(
        [ft._spec("t0", curve=[0.5, 0.2], n_train_trajs=4, n_val_trajs=4)], al=_al()
    )
    rec = TaskRegistry.from_catalog(catalog_of(cfg), cfg.auto_learning).get("t0")
    assert len(rec.active_val_ids) == 4
    assert len(rec.train_monitor_ids) == 4
    assert len(rec.scout_val_ids) == 2


def test_I08_old_config_key_gets_a_clear_rename_error():
    """旧字段名要给明确的改名提示，而不是「未知字段」。"""
    from lingbotvla.auto_learning.config import AutoLearningConfig

    with pytest.raises(ValueError) as err:
        AutoLearningConfig.from_dict({"max_new_skills_this_run": 3})
    msg = str(err.value)
    assert "max_new_tasks_attempted_this_run" in msg
    assert "改名" in msg
