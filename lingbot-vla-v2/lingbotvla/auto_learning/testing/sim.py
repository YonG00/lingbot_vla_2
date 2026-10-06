"""★ Stage A 的「假世界」+ 它的四个 adapter 实现。

把真模型 / 真数据集换成可配置的 NMSE 动力学，让整个 scheduler 能在**秒级**被演练。

本文件提供：
    SimulatedWorld           世界动力学（学习 / 遗忘 / transfer）
    SimulatedEvaluator       open-loop 评测（文档 §51）
    SimulatedHardnessScorer  per-sample flow loss（难度代理）
    SimulatedTrainer         跑 num_steps 个 optimizer step（**每步重新采样**）
    build_backend(cfg)       把上面这些打包成 `ports.Backend`
    build_scheduler(cfg)     便捷构造（测试 / CLI 用）

⚠️ 边界（文档 §4 已声明）
  * 能测：state machine / scheduler / attempt / replay / review / resume / logging
  * 测不了：数值正确性；也**测不了** hard-sampling 的收益（假世界的学习增益
    只取决于 task 样本占比，不取决于抽中哪些样本）—— 那要留到 Stage B 的真 A/B。
"""

from __future__ import annotations

import hashlib
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from ..config import DemoConfig, SimConfig, TaskSpecConfig
from ..decision.metrics import MIN_BASELINE_MSE, mean
from ..ports import (
    Backend,
    SampleResolver,
    TaskCatalog,
    TrainRequest,
    TrainResult,
)
from ..types import BatchComposition, TrajectoryMetrics
from .catalog import (
    SimSampleResolver,
    SimTaskCatalog,
    frame_of_sample,
    sample_ids_of,
    traj_of_sample,
)

__all__ = [
    "stable_u01",
    "sample_ids_of",
    "traj_of_sample",
    "frame_of_sample",
    "FakeTask",
    "SimulatedWorld",
    "SimulatedEvaluator",
    "SimulatedHardnessScorer",
    "SimulatedTrainer",
    "build_backend",
    "build_scheduler",
    "FAKE_SECONDS_PER_TRAJ",
]

#: 假评测的「单轨迹耗时」（秒）—— 只为让 System panel 有量纲，不代表真实性能
FAKE_SECONDS_PER_TRAJ = 0.35


@lru_cache(maxsize=1 << 20)
def _latent(task: str, sample_id: int, skew: float) -> float:
    """样本的固有难度（与训练进度无关）—— 缓存，因为一个 run 里会被问几十万次。"""
    key = f"latent|{task}|{sample_id}".encode("utf-8")
    u = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / float(1 << 64)
    return u ** skew


def stable_u01(*parts: Any) -> float:
    """跨进程稳定的 [0,1) 伪随机数。

    刻意**不用** Python 的 `hash()`（PYTHONHASHSEED 会变），也不用全局 RNG，
    这样「同一个任务 + 同一组轨迹」的评测噪声在任何时候都完全一致 ——
    这就是文档 §8.1 要求的 explicit / reproducible noise。
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


# --------------------------------------------------------------------------- #
class FakeTask:
    """一个假任务的运行时状态。

    唯一的隐藏状态是 `eff_units` —— 「有效训练 unit 数」。
    训练让它变大；别的任务在训练时它会衰减（遗忘）；replay 会减慢衰减。
    它可以是**负数**：说明忘得比 base 还差。
    """

    def __init__(self, spec: TaskSpecConfig) -> None:
        self.spec = spec
        self.eff_units: float = 0.0
        self.total_steps: int = 0
        self.peak_eff_units: float = 0.0
        self.units_trained: float = 0.0

    @property
    def name(self) -> str:
        return self.spec.name

    def true_nmse(self, which: str = "val") -> float:
        return self.spec.nmse_at(self.eff_units, which)

    def true_mse(self, which: str = "val") -> float:
        return self.true_nmse(which) * self.spec.baseline_mse

    def to_state(self) -> Dict[str, float]:
        return {
            "eff_units": self.eff_units,
            "total_steps": self.total_steps,
            "peak_eff_units": self.peak_eff_units,
            "units_trained": self.units_trained,
        }

    def load_state(self, raw: Dict[str, float]) -> None:
        self.eff_units = float(raw.get("eff_units", 0.0))
        self.total_steps = int(raw.get("total_steps", 0))
        self.peak_eff_units = float(raw.get("peak_eff_units", 0.0))
        self.units_trained = float(raw.get("units_trained", 0.0))


# --------------------------------------------------------------------------- #
class SimulatedWorld:
    """假世界的动力学。

    `apply_steps(comps)` 消费**一个 learning unit 的全部 batch**
    （`comps` 里每个元素是一个 optimizer step 的 composition）：

      1. 按整个窗口的**样本占比**给每个任务学习量
      2. 其余任务按 `forget_rate` 遗忘（被 replay 覆盖的比例打折）
      3. 同 group 的兄弟任务吃到 `transfer`
      4. 返回窗口内的平均假 loss 与逐 batch loss

    为什么一次吃一批 composition 而不是循环 50 次：语义等价（占比按整个窗口算），
    但快几十倍 —— Monte Carlo 压力测试要跑 20~500 个 seed。
    """

    def __init__(self, sim_cfg: SimConfig) -> None:
        self.cfg = sim_cfg
        self.tasks: Dict[str, FakeTask] = {t.name: FakeTask(t) for t in sim_cfg.tasks}
        self.unit_counter: int = 0
        self.step_counter: int = 0

    # ---------------------------------------------------------------- #
    def baseline_mse(self, task: str) -> float:
        return self.tasks[task].spec.baseline_mse

    def true_nmse(self, task: str) -> float:
        return self.tasks[task].true_nmse()

    def true_units(self, task: str) -> float:
        return self.tasks[task].eff_units

    # ---------------------------------------------------------------- #
    def sample_loss(self, task: str, sample_id: int) -> float:
        """单个训练样本上的假 flow-matching loss（hardness 的 cheap proxy）。"""
        ft = self.tasks[task]
        latent = _latent(task, int(sample_id), self.cfg.hardness_latent_skew)
        scale = 1.0 / (1.0 + 0.9 * max(ft.eff_units, 0.0))
        return self.cfg.loss_floor + latent * scale

    # ---------------------------------------------------------------- #
    def apply_steps(self, comps: Sequence[BatchComposition]) -> Dict[str, Any]:
        """消费一个 learning unit 的全部 batch（每个元素 = 一个 optimizer step）。"""
        cfg = self.cfg
        n_steps = len(comps)
        if n_steps == 0:
            return {"loss": 0.0, "samples": 0, "per_batch_losses": []}

        self.unit_counter += 1
        self.step_counter += n_steps
        units = n_steps / float(cfg.steps_per_unit)

        total = sum(len(c.refs) for c in comps) or 1
        presence: Counter = Counter()
        old_count: Counter = Counter()
        per_batch_losses: List[float] = []
        all_losses: List[float] = []
        for comp in comps:
            batch_losses = []
            for ref in comp.refs:
                presence[ref.task] += 1
                loss = self.sample_loss(ref.task, ref.sample_id)
                batch_losses.append(loss)
                all_losses.append(loss)
            for ref in comp.old:
                old_count[ref.task] += 1
            per_batch_losses.append(mean(batch_losses) or 0.0)
        presence_frac = {k: v / total for k, v in presence.items()}
        replay_denom = max(1, sum(old_count.values()))

        for name, ft in self.tasks.items():
            p = presence_frac.get(name, 0.0)
            ft.eff_units += units * p
            if p > 0.0:
                ft.units_trained += units * p
                ft.peak_eff_units = max(ft.peak_eff_units, ft.eff_units)
            ft.total_steps += int(round(n_steps * p))
            if p < 1.0:
                old_frac = old_count.get(name, 0) / replay_denom
                forget = ft.spec.forget_rate * units * (1.0 - p)
                forget *= 1.0 - cfg.replay_forget_protection * old_frac
                ft.eff_units -= forget

        # --- sibling transfer：训练一个任务顺便带起同 group 的兄弟 ---
        for name, p in presence_frac.items():
            spec = self.tasks[name].spec
            if spec.transfer <= 0.0 or not spec.group:
                continue
            for other, ft in self.tasks.items():
                if other == name or ft.spec.group != spec.group:
                    continue
                ft.eff_units += spec.transfer * units * p

        return {
            "loss": mean(all_losses) or 0.0,
            "samples": total,
            "per_batch_losses": per_batch_losses,
            "presence": presence_frac,
        }

    def apply_batch(self, comp: BatchComposition, steps: int = 1) -> Dict[str, Any]:
        """兼容旧接口：`steps` 个**完全相同**的 batch。"""
        return self.apply_steps([comp] * max(1, steps))

    # ---------------------------------------------------------------- #
    def to_state(self) -> Dict[str, Any]:
        return {
            "unit_counter": self.unit_counter,
            "step_counter": self.step_counter,
            "tasks": {name: ft.to_state() for name, ft in self.tasks.items()},
        }

    def load_state(self, raw: Dict[str, Any]) -> None:
        self.unit_counter = int(raw.get("unit_counter", 0))
        self.step_counter = int(raw.get("step_counter", 0))
        for name, st in (raw.get("tasks") or {}).items():
            if name in self.tasks:
                self.tasks[name].load_state(st)

    def snapshot_table(self) -> List[Dict[str, Any]]:
        rows = []
        for name, ft in self.tasks.items():
            rows.append(
                {
                    "task": name,
                    "true_nmse": round(ft.true_nmse(), 5),
                    "eff_units": round(ft.eff_units, 4),
                    "units_trained": round(ft.units_trained, 4),
                    "total_steps": ft.total_steps,
                }
            )
        return rows


# --------------------------------------------------------------------------- #
class SimulatedEvaluator:
    """`evaluate(task, split, episode_ids) -> TrajectoryMetrics` 的假实现。"""

    def __init__(self, world: SimulatedWorld, sim_cfg: Optional[SimConfig] = None) -> None:
        self.world = world
        self.cfg = sim_cfg or world.cfg

    def baseline_mse(self, task: str) -> float:
        return self.world.baseline_mse(task)

    def _traj_bias(self, task: str, episode_id: int, sigma: float) -> float:
        return sigma * (2.0 * stable_u01("traj", task, episode_id) - 1.0)

    def _aggregate_jitter(self, task: str, episode_ids: Sequence[int], sigma: float) -> float:
        u = stable_u01("agg", task, tuple(sorted(episode_ids)), self.world.unit_counter)
        return 0.5 * sigma * (2.0 * u - 1.0)

    def evaluate(self, task: str, split: str, episode_ids: Sequence[int]) -> TrajectoryMetrics:
        ft = self.world.tasks[task]
        spec = ft.spec
        base = spec.baseline_mse
        valid = base >= MIN_BASELINE_MSE
        sigma = spec.eval_noise * self.cfg.eval_noise_scale
        ids = list(episode_ids)

        override = spec.override_for(split)
        if override is not None:
            true_mse = override * base
            sigma = 0.0
        else:
            which = "train" if split == "train_monitor" else "val"
            true_mse = ft.true_mse(which)
            if which == "train" and spec.train_curve is None and spec.overfit_rate > 0.0:
                true_mse *= max(0.05, 1.0 - spec.overfit_rate * min(ft.eff_units, 3.0))

        per_traj: Dict[int, float] = {}
        # 抖动分两部分：逐轨迹的（子集偏差）+ 窗口级的（时间抖动）。
        # 🔴 窗口级抖动必须**作用到每条轨迹**上，否则
        # `mse != mean(per_traj_mse)` —— 聚合口径就不自洽了
        # （`test_evaluator_return_shape_and_fields` 抓出来的）。
        agg_jitter = self._aggregate_jitter(task, ids, sigma)
        for ep in ids:
            noisy = true_mse * (1.0 + self._traj_bias(task, ep, sigma) + agg_jitter)
            per_traj[ep] = max(noisy, 0.0)

        agg = mean(per_traj.values()) or 0.0
        agg = max(agg, 0.0)
        value = (agg / base) if valid else None
        note = "" if valid else f"baseline_mse={base:.3g} ≈ 0 ⇒ metric invalid（文档 §9.1）"

        return TrajectoryMetrics(
            task=task,
            split=split,
            episode_ids=ids,
            per_traj_mse={k: round(v, 8) for k, v in per_traj.items()},
            mse=round(agg, 8),
            baseline_mse=base,
            nmse=None if value is None else round(value, 6),
            r2=None if value is None else round(1.0 - value, 6),
            metric_valid=valid,
            n_trajs=len(ids),
            wall_time_s=round(FAKE_SECONDS_PER_TRAJ * len(ids), 4),
            note=note,
        )


# --------------------------------------------------------------------------- #
class SimulatedHardnessScorer:
    """`HardnessScorer`（Stage A：直接用假 world 的 per-sample loss）。"""

    def __init__(self, world: SimulatedWorld) -> None:
        self.world = world

    def score(self, task: str, sample_ids: Sequence[int]) -> Dict[int, float]:
        return {int(s): float(self.world.sample_loss(task, int(s))) for s in sample_ids}


# --------------------------------------------------------------------------- #
class SimulatedTrainer:
    """`Trainer`（Stage A）：跑 num_steps 个 optimizer step，**每步重新采样**。

    这是本轮最关键的一条语义修正：一个 learning unit = `num_steps` 次独立 batch 抽样，
    不是「抽一个 batch 代表 50 步」。`samples_seen` 因此是 `num_steps × batch_size`。
    """

    def __init__(
        self,
        world: SimulatedWorld,
        sampler,
        resolver: SampleResolver,
        catalog: TaskCatalog,
        cfg: SimConfig,
    ) -> None:
        self.world = world
        self.sampler = sampler
        self.resolver = resolver
        self.catalog = catalog
        self.cfg = cfg

    def train_steps(self, request: TrainRequest, num_steps: int) -> TrainResult:
        prepared = self.sampler.prepare(request)
        comps: List[BatchComposition] = [self.sampler.build(prepared) for _ in range(num_steps)]
        info = self.world.apply_steps(comps)

        old_counts: Counter = Counter()
        new_counts: Counter = Counter()
        signatures = set()
        for comp in comps:
            for ref in comp.old:
                old_counts[ref.task] += 1
            for ref in comp.new:
                new_counts[ref.sample_id] += 1
            signatures.add(tuple(sorted(r.sample_id for r in comp.refs)))

        return TrainResult(
            loss=info["loss"],
            steps=num_steps,
            samples_seen=info["samples"],
            old_slot_counts=dict(old_counts),
            new_slot_counts=dict(new_counts),
            per_step_losses=list(info["per_batch_losses"]),
            batches_built=len(comps),
            unique_batches=len(signatures),
        )


# --------------------------------------------------------------------------- #
def build_backend(cfg: DemoConfig, rng=None) -> Backend:
    """把 Stage A 的全部实现打包成 `ports.Backend`。"""
    import random

    from ..sampling.sampler import BatchSampler

    world = SimulatedWorld(cfg.sim)
    catalog = SimTaskCatalog(cfg.sim.tasks)
    resolver = SimSampleResolver(cfg.sim.task_map(), catalog.task_names())
    rng = rng or random.Random(cfg.sim.seed)
    sampler = BatchSampler(cfg.auto_learning, resolver, catalog, rng)
    return Backend(
        catalog=catalog,
        evaluator=SimulatedEvaluator(world, cfg.sim),
        trainer=SimulatedTrainer(world, sampler, resolver, catalog, cfg.sim),
        scorer=SimulatedHardnessScorer(world),
        resolver=resolver,
        extra_state=world,
    )


def build_scheduler(cfg: DemoConfig, logger=None):
    """便捷构造：等价于 `Scheduler(build_backend(cfg), cfg.auto_learning)`。"""
    from ..orchestration.scheduler import Scheduler

    return Scheduler(
        build_backend(cfg), cfg.auto_learning, seed=cfg.sim.seed, logger=logger
    )
