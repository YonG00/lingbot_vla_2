"""配置：文档 §41 的全部参数 + Stage A 假世界的任务规格。

配置格式：YAML（需 PyYAML）或 JSON（零依赖）。两种都支持，`load_config` 自动识别。
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:  # 可选依赖
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover
    _yaml = None


# --------------------------------------------------------------------------- #
# 文档 §41 的参数表
# --------------------------------------------------------------------------- #
@dataclass
class AutoLearningConfig:
    """自主学习的全部可调参数。默认值 = 文档 §41 的推荐初值。"""

    enabled: bool = True

    # ----- task pool -----
    task_names: Optional[List[str]] = None
    #: 「本次最多**主动尝试**多少个新任务」—— 这是第一阶段的主开关。
    #: 统计口径 = 曾经进入 TRAINING（attempt_count 0→1）的不同任务数。
    max_new_tasks_attempted_this_run: Optional[int] = None
    #: 可选：「本次最多让多少个任务 newly PASS」后再收工（None = 不限）。
    max_new_tasks_passed_this_run: Optional[int] = None
    #: churn guard：同一任务最多被「遗忘 → 回炉」几次。
    #: **不占用 attempt 预算** —— attempt 只数真正的主训练机会（见下方 attempts 段）。
    max_reopens_per_task: Optional[int] = 3

    # ----- task evaluation -----
    global_scout_val_trajs: int = 2
    active_train_probe_trajs: int = 4
    active_val_probe_trajs: int = 4
    pass_nmse: float = 0.30

    # ----- train / eval cadence -----
    eval_interval_steps: int = 50
    min_steps_before_defer: int = 100

    # ----- after PASS -----
    continue_after_pass: bool = False
    post_pass_max_steps: int = 0
    post_pass_min_lp: float = 0.03

    # ----- learning progress -----
    min_lp50: float = 0.05

    # ----- attempts -----
    #: 一个任务最多获得几次**真正的主训练机会**（进入 TRAINING 才算一次）。
    #: 回炉（reopen）**不**消耗它 —— 那是 churn guard 的事。
    max_attempts_per_task: int = 2

    # ----- optional rescue -----
    defer_resample_retry: bool = False
    #: 触发 rescue 后**额外**给多少 step（必须是 eval_interval_steps 的整数倍）
    defer_retry_steps: int = 50

    # ----- overfit -----
    early_defer_on_overfit: bool = True
    overfit_train_lp_min: float = 0.05
    overfit_val_lp_max: float = 0.00
    overfit_gap_growth_threshold: float = 0.10

    # ----- batch -----
    batch_size: int = 10
    new_slots: int = 7
    replay_slots: int = 3

    # ----- hardness -----
    hardness_probe_fraction: float = 0.33
    hardness_weight_min: float = 1.0
    hardness_weight_max: float = 3.0
    hardness_alpha: float = 2.0
    difficulty_unscored_default: float = 0.5
    refresh_hardness_on_continue: bool = False

    # ----- replay -----
    replay_task_policy: str = "uniform"
    replay_sample_policy: str = "pass_snapshot"

    # ----- review -----
    review_after_task_transitions: int = 2
    forget_relative_threshold: float = 0.30

    # ----- scheduler -----
    rescan_candidates_after_transition: bool = True
    max_global_steps: Optional[int] = None
    max_transitions: Optional[int] = None

    # ----- misc -----
    seed: int = 0
    #: 🔴 review v0.2 #8：v1 要求训练数据集 `image_augment=false`。
    #: True 时**不 fail-fast**，而是打印强警告（hardness 侧仍会临时关增强 + 还原 RNG，
    #: 但「同一个 sample_id 的难度」在不同扫描之间仍可能漂移）。默认 False。
    allow_image_augment: bool = False
    #: 🔴 review v0.2 #6：`enabled=true` 时 baseline store **默认必填**。
    #: 仅 smoke / 单测允许设 True 跳过（此时 NMSE=None，所有任务会被排除出候选池）。
    allow_missing_baseline: bool = False

    # ---------------------------------------------------------------- #
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.new_slots + self.replay_slots != self.batch_size:
            raise ValueError(
                f"new_slots({self.new_slots}) + replay_slots({self.replay_slots}) "
                f"!= batch_size({self.batch_size})"
            )
        if self.eval_interval_steps <= 0:
            raise ValueError("eval_interval_steps must be > 0")
        if self.min_steps_before_defer % self.eval_interval_steps != 0:
            raise ValueError(
                "min_steps_before_defer 必须是 eval_interval_steps 的整数倍，"
                "否则 DEFER 判定点会落在两次评测之间"
            )
        if self.defer_retry_steps % self.eval_interval_steps != 0:
            raise ValueError(
                f"defer_retry_steps({self.defer_retry_steps}) 必须是 "
                f"eval_interval_steps({self.eval_interval_steps}) 的整数倍，"
                "否则 rescue 的额外步数落不到整单元上"
            )
        if self.defer_retry_steps <= 0:
            raise ValueError("defer_retry_steps must be > 0")
        if self.max_attempts_per_task < 1:
            raise ValueError("max_attempts_per_task must be >= 1")
        if self.max_reopens_per_task is not None and self.max_reopens_per_task < 0:
            raise ValueError("max_reopens_per_task must be >= 0 or None")
        for name in ("max_new_tasks_attempted_this_run", "max_new_tasks_passed_this_run"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None")
        if not 0.0 < self.hardness_probe_fraction <= 1.0:
            raise ValueError("hardness_probe_fraction must be in (0, 1]")
        if self.replay_task_policy not in ("uniform",):
            raise ValueError(f"unsupported replay_task_policy: {self.replay_task_policy}")
        if self.replay_sample_policy not in ("pass_snapshot", "uniform"):
            raise ValueError(f"unsupported replay_sample_policy: {self.replay_sample_policy}")
        if self.hardness_weight_max < self.hardness_weight_min:
            raise ValueError("hardness_weight_max must be >= hardness_weight_min")

    # ---------------------------------------------------------------- #
    @property
    def retry_budget_steps(self) -> int:
        """触发 rescue 后，本 attempt 的 DEFER 阈值往后推多少步。"""
        return self.defer_retry_steps

    # ---------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AutoLearningConfig":
        return _filtered(cls, raw)

    @property
    def units_before_defer(self) -> int:
        return self.min_steps_before_defer // self.eval_interval_steps


# --------------------------------------------------------------------------- #
# Stage A 假世界：单个任务的动力学规格
# --------------------------------------------------------------------------- #
@dataclass
class TaskSpecConfig:
    """一个假任务的「真实」动力学。

    两种曲线二选一：
      * `curve`:  显式 NMSE 序列，`curve[i]` = 累计训练 i 个 unit 后的 NMSE。
                  文档 Stage A 举例用的就是这种（Task A: 0.90 → 0.55 → 0.28）。
      * 解析式:   `nmse(u) = nmse_floor + (nmse0 − nmse_floor)·exp(−learn_k·u)`

    `u` = 该任务累计**有效**训练 unit 数（被遗忘会往下掉，甚至掉成负数）。
    `u < 0` 时 NMSE 沿 `degrade_slope` 继续恶化 —— 这是「遗忘到比 base 还差」，
    对应文档里 Task D「0.30 → 0.18 → 退化到 0.38」。
    """

    name: str
    nmse0: float = 0.8
    nmse_floor: float = 0.15
    learn_k: float = 1.2
    curve: Optional[List[float]] = None
    degrade_slope: float = 0.0
    forget_rate: float = 0.05
    group: str = "default"
    transfer: float = 0.0
    eval_noise: float = 0.03
    baseline_mse: float = 0.25
    n_train_trajs: int = 40
    n_val_trajs: int = 10
    samples_per_traj: int = 77
    hardness_spread: float = 1.0
    overfit_rate: float = 0.0
    train_curve: Optional[List[float]] = None
    scout_nmse_override: Optional[float] = None
    confirm_nmse_override: Optional[float] = None
    note: str = ""

    def nmse_at(self, u: float, which: str = "val") -> float:
        """`which="val"` 用 `curve`；`which="train"` 优先用 `train_curve`。

        两者分开是为了能造出「train 变好、val 变差」的 overfit 假任务
        （测试方案 §3 的 `overfit`），而不用靠 `overfit_rate` 那个近似旋钮。
        """
        curve = self.train_curve if (which == "train" and self.train_curve) else self.curve
        if u < 0.0:
            return self._nmse_nonneg(0.0, curve) + self.degrade_slope * (-u)
        return self._nmse_nonneg(u, curve)

    def _nmse_nonneg(self, u: float, curve: Optional[List[float]]) -> float:
        if curve:
            c = curve
            if u <= 0:
                return float(c[0])
            if u >= len(c) - 1:
                return float(c[-1])
            i = int(u)
            f = u - i
            return float(c[i]) * (1 - f) + float(c[i + 1]) * f
        import math

        p = 1.0 - math.exp(-self.learn_k * u)
        return self.nmse_floor + (self.nmse0 - self.nmse_floor) * (1.0 - p)

    def override_for(self, split: str) -> Optional[float]:
        """`scout` / `confirm` 允许直接指定观测值（造 `false_scout_pass` 用）。

        指定后**完全绕过曲线与噪声**，所以测试里可以断言精确数值。
        """
        if split == "scout":
            return self.scout_nmse_override
        if split == "confirm":
            return self.confirm_nmse_override
        return None

    @property
    def train_traj_ids(self) -> List[int]:
        return list(range(self.n_train_trajs))

    @property
    def val_traj_ids(self) -> List[int]:
        return list(range(100, 100 + self.n_val_trajs))

    def sample_id(self, traj_id: int, frame_id: int) -> int:
        return traj_id * 1000 + frame_id


@dataclass
class SimConfig:
    """Stage A 假世界的全局参数（只影响模拟，不影响 scheduler 语义）。"""

    seed: int = 7
    steps_per_unit: int = 50
    eval_noise_scale: float = 1.0
    replay_forget_protection: float = 0.60
    hardness_latent_skew: float = 2.0
    loss_floor: float = 0.02
    tasks: List[TaskSpecConfig] = field(default_factory=list)

    def task_map(self) -> Dict[str, TaskSpecConfig]:
        return {t.name: t for t in self.tasks}


# --------------------------------------------------------------------------- #
@dataclass
class DemoConfig:
    auto_learning: AutoLearningConfig = field(default_factory=AutoLearningConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    description: str = ""

    def task_names(self) -> List[str]:
        names = self.auto_learning.task_names
        if names:
            return list(names)
        return [t.name for t in self.sim.tasks]


# --------------------------------------------------------------------------- #
#: 旧字段名 → 新字段名。用旧名会得到一条明确的改名提示，而不是「未知字段」。
RENAMED_KEYS = {
    "max_new_skills_this_run": "max_new_tasks_attempted_this_run",
}


def _filtered(cls: type, raw: Optional[Dict[str, Any]]):
    if not raw:
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    renamed = {k: RENAMED_KEYS[k] for k in unknown if k in RENAMED_KEYS}
    if renamed:
        pairs = "、".join(f"`{k}` → `{v}`" for k, v in sorted(renamed.items()))
        raise ValueError(
            f"{cls.__name__} 的字段已改名：{pairs}。"
            "语义也变了：新口径统计的是「主动尝试的新任务数」，不是「newly PASS 数」。"
        )
    if unknown:
        raise ValueError(f"{cls.__name__} 收到未知字段: {sorted(unknown)}")
    return cls(**raw)


def _read_raw(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        if _yaml is None:
            raise RuntimeError(
                f"{path} 是 YAML，但当前环境没有 PyYAML。"
                "要么 `pip install pyyaml`，要么改用同名的 .json 配置。"
            )
        return _yaml.safe_load(text) or {}
    return json.loads(text)


def load_config(path: str) -> DemoConfig:
    """读取配置（YAML 或 JSON）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    raw = _read_raw(path)
    al = AutoLearningConfig.from_dict(raw.get("auto_learning"))
    sim_raw = raw.get("sim") or {}
    tasks = [_filtered(TaskSpecConfig, t) for t in sim_raw.get("tasks", [])]
    sim = SimConfig(
        seed=sim_raw.get("seed", 7),
        steps_per_unit=sim_raw.get("steps_per_unit", al.eval_interval_steps),
        eval_noise_scale=sim_raw.get("eval_noise_scale", 1.0),
        replay_forget_protection=sim_raw.get("replay_forget_protection", 0.60),
        hardness_latent_skew=sim_raw.get("hardness_latent_skew", 2.0),
        loss_floor=sim_raw.get("loss_floor", 0.02),
        tasks=tasks,
    )
    return DemoConfig(
        auto_learning=al,
        sim=sim,
        description=raw.get("description", ""),
    )


def dump_config(cfg: DemoConfig, path: str) -> None:
    payload = {
        "description": cfg.description,
        "auto_learning": cfg.auto_learning.to_dict(),
        "sim": {
            "seed": cfg.sim.seed,
            "steps_per_unit": cfg.sim.steps_per_unit,
            "eval_noise_scale": cfg.sim.eval_noise_scale,
            "replay_forget_protection": cfg.sim.replay_forget_protection,
            "hardness_latent_skew": cfg.sim.hardness_latent_skew,
            "loss_floor": cfg.sim.loss_floor,
            "tasks": [dataclasses.asdict(t) for t in cfg.sim.tasks],
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
