"""具名假任务库（测试方案 §3）。

不写「一个随机 Fake Evaluator」，而是准备**行为明确的**虚拟任务，
这样各种状态是**主动造出来**的，而不是等随机模拟碰巧触发。

| 任务 | 行为 | 主要用途 |
|---|---|---|
| `easy_pass` | 0.45 → 0.25 | 快速 PASS |
| `slow_learn` | 0.70 → 0.60 → 0.50 → 0.42 | 持续进步 |
| `plateau` | 0.70 → 0.68 → 0.67 | 低 LP → DEFER |
| `overfit` | train 0.60→0.30，val 0.62→0.70 | Overfit 判断 |
| `already_known` | 初始 0.18 | bootstrap 自动 PASS |
| `false_scout_pass` | 2-traj=0.25，4-traj=0.38 | 2→4 PASS 确认 |
| `forgotten` | PASS 时 0.18，之后退化 | Reopen |
| `mild_drop` | 轻微退化 | **不应**误判 Forgotten |
| `relative_forgetting` | 仍及格但明显退化 | 相对退化回炉 |
| `unlearnable` | 始终约 0.9 | attempt exhaustion |
| `transfer_target` | 未训练，随 sibling 训练自动下降 | rescan / transfer |
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..config import AutoLearningConfig, DemoConfig, SimConfig, TaskSpecConfig

DEFAULT_SAMPLES = 77
DEFAULT_N_TRAIN = 40


def _spec(name: str, **kw) -> TaskSpecConfig:
    base = dict(
        name=name,
        baseline_mse=0.25,
        eval_noise=0.03,
        n_train_trajs=DEFAULT_N_TRAIN,
        n_val_trajs=10,
        samples_per_traj=DEFAULT_SAMPLES,
        forget_rate=0.05,
        degrade_slope=0.08,
    )
    base.update(kw)
    return TaskSpecConfig(**base)


# --------------------------------------------------------------------------- #
def easy_pass(name: str = "easy_pass", **kw) -> TaskSpecConfig:
    return _spec(name, curve=[0.45, 0.25], **kw)


def slow_learn(name: str = "slow_learn", **kw) -> TaskSpecConfig:
    return _spec(name, curve=[0.70, 0.60, 0.50, 0.42], **kw)


def plateau(name: str = "plateau", **kw) -> TaskSpecConfig:
    return _spec(name, curve=[0.70, 0.68, 0.67], **kw)


def overfit(name: str = "overfit", **kw) -> TaskSpecConfig:
    """train 变好、val 变差 ⇒ 命中 §17.1 的 overfit 判据。"""
    return _spec(name, curve=[0.62, 0.70], train_curve=[0.60, 0.30], **kw)


def already_known(name: str = "already_known", **kw) -> TaskSpecConfig:
    return _spec(name, curve=[0.18], degrade_slope=0.03, **kw)


def false_scout_pass(name: str = "false_scout_pass", **kw) -> TaskSpecConfig:
    """2 条 val 看着达标（0.25），补到 4 条就露馅（0.38）。"""
    return _spec(
        name,
        curve=[0.38, 0.24],
        scout_nmse_override=0.25,
        confirm_nmse_override=0.38,
        **kw,
    )


def forgotten(name: str = "forgotten", **kw) -> TaskSpecConfig:
    """学得快、忘得更快 ⇒ PASS 之后会被 review 抓出来回炉。

    `forget_rate` 刻意调到「即使 replay 一直覆盖它，净效果仍然是退化」：
    单任务场景下它一个人占满 3 个 replay slot（presence=0.3），
    净变化 = +0.3（被 replay 训到）− 1.8×0.7×0.4（遗忘打折后）< 0。
    """
    return _spec(name, curve=[0.32, 0.18], forget_rate=1.8, degrade_slope=0.20, **kw)


def mild_drop(name: str = "mild_drop", **kw) -> TaskSpecConfig:
    """只掉一点点 ⇒ 不应误判 Forgotten。"""
    return _spec(name, curve=[0.14, 0.10], forget_rate=0.02, degrade_slope=0.02, **kw)


def relative_forgetting(name: str = "relative_forgetting", **kw) -> TaskSpecConfig:
    """仍低于及格线，但相对最佳值退化很多 ⇒ 仍应回炉（文档 §34 情况 B）。"""
    return _spec(name, curve=[0.08, 0.20], forget_rate=0.30, degrade_slope=0.25, **kw)


def unlearnable(name: str = "unlearnable", **kw) -> TaskSpecConfig:
    """怎么练都过不了 ⇒ 两次 attempt 后 EXHAUSTED，且不能霸占调度。"""
    return _spec(name, nmse0=0.90, nmse_floor=0.88, learn_k=0.30, **kw)


def transfer_target(
    name: str = "transfer_target",
    group: str = "pair",
    source: Optional[str] = None,
    transfer: float = 0.75,
    **kw,
) -> TaskSpecConfig:
    """自己没被训过，靠同 group 的 sibling 被训而自动变好。"""
    spec = _spec(name, curve=[0.45, 0.28], group=group, transfer=transfer, **kw)
    if source:
        spec.note = f"transfer from {source}"
    return spec


ALL = {
    "easy_pass": easy_pass,
    "slow_learn": slow_learn,
    "plateau": plateau,
    "overfit": overfit,
    "already_known": already_known,
    "false_scout_pass": false_scout_pass,
    "forgotten": forgotten,
    "mild_drop": mild_drop,
    "relative_forgetting": relative_forgetting,
    "unlearnable": unlearnable,
    "transfer_target": transfer_target,
}


# --------------------------------------------------------------------------- #
def make_cfg(
    specs: List[TaskSpecConfig],
    al: Optional[AutoLearningConfig] = None,
    seed: int = 3,
    **sim_kw,
) -> DemoConfig:
    """用一组假任务拼一个 DemoConfig。"""
    if al is None:
        al = AutoLearningConfig(seed=seed)
    sim = SimConfig(seed=seed, steps_per_unit=al.eval_interval_steps, tasks=list(specs), **sim_kw)
    return DemoConfig(auto_learning=al, sim=sim, description="scripted fake tasks")


def cfg_with(names: List[str], al: Optional[AutoLearningConfig] = None, **kw) -> DemoConfig:
    """按名字取假任务拼配置，例如 `cfg_with(["easy_pass", "unlearnable"])`。"""
    specs = [ALL[n]() for n in names]
    return make_cfg(specs, al=al, **kw)


# --------------------------------------------------------------------------- #
# 随机任务生成（测试方案 §14 Monte Carlo 压力测试用；测试与 CLI 共用）
# --------------------------------------------------------------------------- #
KINDS = ["easy", "learn", "plateau", "unlearnable", "known", "fragile"]


def random_task(
    rng: random.Random,
    idx: int,
    *,
    n_train_trajs: int = 6,
    samples_per_traj: int = 20,
) -> TaskSpecConfig:
    """按「行为种类」随机造一个任务（随机的**行为模式**，不是随机数字）。"""
    kind = rng.choice(KINDS)
    common = dict(
        name=f"{kind}_{idx:02d}",
        baseline_mse=0.25,
        eval_noise=0.03,
        n_train_trajs=n_train_trajs,
        n_val_trajs=10,
        samples_per_traj=samples_per_traj,
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
    return TaskSpecConfig(  # fragile：学得快、忘得快 ⇒ 制造 reopen / 冲突
        curve=[rng.uniform(0.30, 0.45), rng.uniform(0.14, 0.22)],
        forget_rate=rng.uniform(0.6, 1.6),
        degrade_slope=rng.uniform(0.15, 0.35),
        **common,
    )


def random_cfg(
    seed: int,
    *,
    n_tasks: int = 50,
    n_train_trajs: int = 6,
    samples_per_traj: int = 20,
) -> DemoConfig:
    """随机造一整套配置：任务行为随机，**调度参数也随机**（专找边界 bug）。"""
    rng = random.Random(seed)
    tasks = [random_task(rng, i, n_train_trajs=n_train_trajs, samples_per_traj=samples_per_traj)
             for i in range(n_tasks)]
    al = AutoLearningConfig(
        seed=seed,
        eval_interval_steps=50,
        min_steps_before_defer=100,
        pass_nmse=0.30,
        max_attempts_per_task=rng.choice([1, 2, 3]),
        review_after_task_transitions=rng.choice([0, 1, 2, 3]),
        defer_resample_retry=rng.random() < 0.3,
        continue_after_pass=rng.random() < 0.15,
        post_pass_max_steps=100,
        early_defer_on_overfit=rng.random() < 0.7,
        max_global_steps=rng.choice([None, 2000, 5000]),
    )
    return make_cfg(tasks, al=al, seed=seed)
