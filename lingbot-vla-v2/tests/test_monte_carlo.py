"""测试方案 §14 —— Monte Carlo 压力测试。

这是 Demo 阶段很值得做、真实模型阶段反而不方便做的测试：
**随机造 50 个任务 × 多个 seed，专找边界 bug**。

它不证明算法效果，只证明「这台自动学习机器本身是可信的」：

    * 永不死循环
    * attempt 不超限
    * 同一时刻最多一个 TRAINING
    * 最终可收敛（不会停在半个状态上）
    * candidate selection 永远合法
    * 所有状态转换都有 reason + code
    * resume 不破坏状态

方案点名这种测试特别容易发现：round 计数错位 / DEFER 永远无法重新激活 /
PASS 被忘记后 attempt 计数异常 / 最后一个 candidate 被错误跳过 /
review 正好在 transition 时触发造成重复事件 / state mutation 顺序导致列表迭代错误。
"""

from __future__ import annotations

import os
import random
from typing import List

import pytest

from lingbotvla.auto_learning.config import AutoLearningConfig, TaskSpecConfig
from lingbotvla.auto_learning.decision.metrics import is_finite_metric
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from lingbotvla.auto_learning.state import persistence
from al_fixtures import local_tmpdir
from lingbotvla.auto_learning.testing.fake_tasks import make_cfg
from lingbotvla.auto_learning.types import TaskStatus
from al_fixtures import scheduler_of

#: seed 数可用环境变量放大（CI 上跑几百个，本地跑几十个）
N_SEEDS = int(os.environ.get("AL_MC_SEEDS", "20"))
N_TASKS = int(os.environ.get("AL_MC_TASKS", "50"))
SAMPLES_PER_TRAJ = 20  # 小一点，保证 20 个 seed 也能秒级跑完
KINDS = ["easy", "learn", "plateau", "unlearnable", "known", "fragile"]


def random_task(rng: random.Random, idx: int) -> TaskSpecConfig:
    """按「行为种类」随机造一个任务（不是随机数字，而是随机的**行为模式**）。"""
    kind = rng.choice(KINDS)
    name = f"{kind}_{idx:02d}"
    common = dict(
        name=name,
        baseline_mse=0.25,
        eval_noise=0.03,
        n_train_trajs=6,
        n_val_trajs=10,
        samples_per_traj=SAMPLES_PER_TRAJ,
    )
    if kind == "easy":
        return TaskSpecConfig(curve=[rng.uniform(0.35, 0.50), rng.uniform(0.15, 0.28)], **common)
    if kind == "learn":
        a = rng.uniform(0.45, 0.75)
        steps = rng.randint(2, 4)
        curve = [a] + [a - (a - 0.20) * (i + 1) / steps for i in range(steps)]
        return TaskSpecConfig(curve=curve, **common)
    if kind == "plateau":
        a = rng.uniform(0.60, 0.80)
        return TaskSpecConfig(curve=[a, a - 0.02, a - 0.03], **common)
    if kind == "unlearnable":
        return TaskSpecConfig(nmse0=0.85, nmse_floor=0.82, learn_k=0.3, **common)
    if kind == "known":
        return TaskSpecConfig(curve=[rng.uniform(0.08, 0.26)], **common)
    # fragile：学得快、忘得快 ⇒ 制造 reopen / 冲突
    return TaskSpecConfig(
        curve=[rng.uniform(0.30, 0.45), rng.uniform(0.14, 0.22)],
        forget_rate=rng.uniform(0.6, 1.6),
        degrade_slope=rng.uniform(0.15, 0.35),
        **common,
    )


def random_cfg(seed: int) -> object:
    rng = random.Random(seed)
    tasks = [random_task(rng, i) for i in range(N_TASKS)]
    al = AutoLearningConfig(
        seed=seed,
        eval_interval_steps=50,
        min_steps_before_defer=100,
        pass_nmse=0.30,
        max_attempts_per_task=rng.choice([1, 2, 3]),
        max_reopens_per_task=rng.choice([1, 2, 3]),
        review_after_task_transitions=rng.choice([0, 1, 2, 3]),
        defer_resample_retry=rng.random() < 0.3,
        defer_retry_steps=rng.choice([50, 100]),
        continue_after_pass=rng.random() < 0.15,
        post_pass_max_steps=100,
        early_defer_on_overfit=rng.random() < 0.7,
        max_global_steps=rng.choice([None, 2000, 5000]),
    )
    return make_cfg(tasks, al=al, seed=seed)


# --------------------------------------------------------------------------- #
def check_invariants(sched: Scheduler, cfg, where: str) -> None:
    al = cfg.auto_learning
    training = [r for r in sched.registry if r.status == TaskStatus.TRAINING.value]
    assert len(training) <= 1, f"[{where}] 同时有 {len(training)} 个 TRAINING"

    if sched.state.current_task is not None:
        assert training and training[0].task_name == sched.state.current_task, where
    else:
        assert not training, f"[{where}] 没有 current_task 却有 TRAINING"

    for rec in sched.registry:
        assert rec.attempt_count <= al.max_attempts_per_task, (
            f"[{where}] {rec.task_name} attempt={rec.attempt_count} 超限"
        )
        # reopen 由**独立的 churn guard** 限，不占 attempt 预算
        if al.max_reopens_per_task is not None:
            assert rec.reopen_count <= al.max_reopens_per_task, (
                f"[{where}] {rec.task_name} reopen={rec.reopen_count} "
                f"超过 churn guard {al.max_reopens_per_task}"
            )
        if rec.scout_nmse is not None and rec.metric_valid:
            # 「已扫描 + 标为有效」的值必须是有限的。还没扫到的任务 scout_nmse=None
            # 是合法的中间态（bootstrap 队列还没轮到它）。
            assert is_finite_metric(rec.scout_nmse), (
                f"[{where}] {rec.task_name} 标为有效但 scout_nmse={rec.scout_nmse!r}"
            )
        if rec.status == TaskStatus.PASS.value:
            assert rec.current_val_nmse is not None, where

    for row in sched.state.transitions:
        assert row["reason"], f"[{where}] 迁移没有 reason：{row}"
        assert row["code"], f"[{where}] 迁移没有 code：{row}"
        assert row["task"] in sched.registry.names(), where

    # 被选中的任务必须在当时确实可被选中（曾经 PASS/EXHAUSTED 的不该再进 TRAINING）
    for ev in sched.events:
        if ev.get("action") == "select":
            rec = sched.registry.get(ev["task"])
            assert rec.attempt_count >= 1, f"[{where}] select 之后 attempt 没加"


def drive_with_invariants(sched: Scheduler, cfg, limit: int = 4000) -> Scheduler:
    steps = 0
    while not sched.state.finished and steps < limit:
        sched.advance()
        steps += 1
        check_invariants(sched, cfg, f"action#{steps}")
    assert sched.state.finished, f"{limit} 个动作内没有停止（疑似死循环）"
    return sched


def run_with_invariants(cfg, limit: int = 4000) -> Scheduler:
    return drive_with_invariants(scheduler_of(cfg), cfg, limit)


# =========================================================================== #
@pytest.mark.parametrize("seed", range(N_SEEDS))
def test_monte_carlo_invariants(seed):
    cfg = random_cfg(seed)
    sched = run_with_invariants(cfg)

    assert sched.state.stop_reason, "停止必须有原因"
    counts = sched.registry.counts()
    assert sum(counts.values()) == N_TASKS
    assert counts[TaskStatus.TRAINING.value] == 0, counts
    # 收敛：正常跑完（不是被预算截断）时不该还有 DEFER 挂着
    if sched.state.stop_reason == "all_tasks_resolved":
        assert counts[TaskStatus.DEFER.value] == 0, counts
        assert counts[TaskStatus.CANDIDATE.value] == 0, counts


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_monte_carlo_resume_is_transparent(seed):
    """随机场景下 resume 也不能破坏状态（方案 §14 的「resume 不破坏状态」）。"""
    cfg = random_cfg(seed)
    cfg.auto_learning.max_global_steps = 1500
    reference = run_with_invariants(cfg)

    with local_tmpdir() as d:
        cfg2 = random_cfg(seed)
        cfg2.auto_learning.max_global_steps = 1500
        sched = scheduler_of(cfg2)
        for _ in range(12):
            if sched.state.finished:
                break
            sched.advance()
        path = os.path.join(d, "state.json")
        persistence.save_state(sched, path)

        cfg3 = random_cfg(seed)
        cfg3.auto_learning.max_global_steps = 1500
        resumed = scheduler_of(cfg3)
        persistence.load_state(resumed, path)
        resumed.resume()
        drive_with_invariants(resumed, cfg3)

    assert resumed.registry.table() == reference.registry.table()
    assert resumed.state.stop_reason == reference.state.stop_reason


def test_monte_carlo_random_configs_are_valid():
    """生成的配置本身要合法（否则上面的断言可能是在测一个坏输入）。"""
    for seed in range(N_SEEDS):
        cfg = random_cfg(seed)
        cfg.auto_learning.validate()
        assert len(cfg.sim.tasks) == N_TASKS
        assert len({t.name for t in cfg.sim.tasks}) == N_TASKS, "任务名必须唯一"
