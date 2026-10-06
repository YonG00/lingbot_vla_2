"""测试用的小配置工厂。"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Dict, List, Optional

from lingbotvla.auto_learning.config import AutoLearningConfig, DemoConfig, SimConfig, TaskSpecConfig


@contextmanager
def local_tmpdir(prefix: str = "al_test_"):
    """在工作目录下开临时目录。

    刻意不用 pytest 的 `tmp_path`：它落在系统临时目录，在受限沙箱里可能没有写权限。
    """
    base = os.path.join(os.getcwd(), ".tmp_auto_learning")
    os.makedirs(base, exist_ok=True)
    path = tempfile.mkdtemp(prefix=prefix, dir=base)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def make_cfg(
    n_tasks: int = 2,
    curves: Optional[Dict[str, List[float]]] = None,
    al: Optional[AutoLearningConfig] = None,
    n_train_trajs: int = 40,
    samples_per_traj: int = 77,
    **sim_kw,
) -> DemoConfig:
    forget_rate = float(sim_kw.pop("forget_rate", 0.05))
    degrade_slope = float(sim_kw.pop("degrade_slope", 0.10))
    tasks = []
    for i in range(n_tasks):
        name = f"t{i}"
        curve = (curves or {}).get(name)
        tasks.append(
            TaskSpecConfig(
                name=name,
                curve=curve if curve else [0.90, 0.50, 0.25],
                nmse0=0.90,
                nmse_floor=0.20,
                learn_k=1.0,
                forget_rate=forget_rate,
                degrade_slope=degrade_slope,
                eval_noise=0.03,
                baseline_mse=0.25,
                n_train_trajs=n_train_trajs,
                n_val_trajs=10,
                samples_per_traj=samples_per_traj,
            )
        )
    sim = SimConfig(seed=3, steps_per_unit=50, tasks=tasks, **sim_kw)
    if al is None:
        al = AutoLearningConfig(seed=3)
    return DemoConfig(auto_learning=al, sim=sim)


def default_al(**kw) -> AutoLearningConfig:
    base = dict(
        seed=3,
        eval_interval_steps=50,
        min_steps_before_defer=100,
        pass_nmse=0.30,
        min_lp50=0.05,
        max_attempts_per_task=2,
        batch_size=10,
        new_slots=7,
        replay_slots=3,
    )
    base.update(kw)
    return AutoLearningConfig(**base)


# --------------------------------------------------------------------------- #
# 解耦后的构造 helper（测试里少写样板）
# --------------------------------------------------------------------------- #
from lingbotvla.auto_learning.testing.catalog import SimSampleResolver, SimTaskCatalog  # noqa: E402
from lingbotvla.auto_learning.testing.sim import (  # noqa: E402
    SimulatedHardnessScorer,
    SimulatedWorld,
    build_scheduler,
)
from lingbotvla.auto_learning.sampling.hardness_scan import HardnessScanner  # noqa: E402
from lingbotvla.auto_learning.state.registry import TaskRegistry  # noqa: E402


def catalog_of(cfg) -> SimTaskCatalog:
    return SimTaskCatalog(cfg.sim.tasks)


def registry_of(cfg) -> TaskRegistry:
    return TaskRegistry.from_catalog(catalog_of(cfg), cfg.auto_learning)


def resolver_of(cfg) -> SimSampleResolver:
    return SimSampleResolver(cfg.sim.task_map(), [t.name for t in cfg.sim.tasks])


def world_of(cfg) -> SimulatedWorld:
    return SimulatedWorld(cfg.sim)


def scanner_of(cfg, world=None) -> HardnessScanner:
    return HardnessScanner(cfg.auto_learning, SimulatedHardnessScorer(world or world_of(cfg)))


def scheduler_of(cfg, logger=None):
    return build_scheduler(cfg, logger=logger)


def train_request(cfg, task, probs, replay_tasks=(), replay_probs=None, sample_ids=None):
    """构造一个 `TrainRequest`（测试用）。"""
    from lingbotvla.auto_learning.ports import ReplayPlan, TrainRequest

    al = cfg.auto_learning
    tasks = list(replay_tasks)
    cat = catalog_of(cfg)
    ids = dict(sample_ids or {})
    for t in tasks:
        ids.setdefault(t, list(cat.entry(t).train_sample_ids))
    return TrainRequest(
        task=task,
        probs=probs,
        replay=ReplayPlan(
            tasks=tasks,
            probs=dict(replay_probs or {t: None for t in tasks}),
            sample_ids=ids,
        ),
        start_step=0,
        batch_size=al.batch_size,
        new_slots=al.new_slots,
        replay_slots=al.replay_slots,
    )


def replay_plan_of(cfg, tasks, probs=None, sample_ids=None):
    """构造一个 `ReplayPlan`（测试用）。"""
    from lingbotvla.auto_learning.ports import ReplayPlan

    cat = catalog_of(cfg)
    ids = dict(sample_ids or {})
    for t in tasks:
        ids.setdefault(t, list(cat.entry(t).train_sample_ids))
    return ReplayPlan(
        tasks=list(tasks),
        probs=dict(probs or {t: None for t in tasks}),
        sample_ids=ids,
    )
