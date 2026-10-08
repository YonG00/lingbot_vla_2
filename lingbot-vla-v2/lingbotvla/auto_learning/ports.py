"""Auto Learning 的**统一契约层**（Stage B0 + B1 合并）。

**Scheduler 只依赖这里的 Protocol** —— 不 import 任何模拟实现、不认识 sample_id 编码。

合并说明（B1）
-------------
Stage B0 先落了「真实仓库侧」的 DTO（`TaskEntry` 用**回合 id**、`SampleRef(task, episode, frame)`、
`EvalResult`），Stage B1 又把 Stage A 的 Scheduler/Registry/Sampler 移植进来
（它们要 `train_sample_ids` / `TrajectoryMetrics` / `ReplayPlan`）。
这里**统一成一份**，关键约定：

* ``SampleRef.sample_id`` = **数据集 local_idx**（稳定全量索引空间里的位置）
* ``TaskEntry.train_sample_ids`` = 该 task 全部 train 样本的 local_idx
* ``TaskEntry.train_traj_ids`` = 该 task 的 train **回合号**（hardness 按整条轨迹抽子集）
* ``Evaluator.evaluate`` 返回 `TrajectoryMetrics`；`evaluate_task` 是真实侧便利接口
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from .types import BatchComposition, SampleRef, TrajectoryMetrics  # noqa: F401  (re-export)

# --------------------------------------------------------------------------- #
# DTO
# --------------------------------------------------------------------------- #


@dataclass
class TaskEntry:
    """`TaskCatalog` 对单个任务的描述 —— scheduler 需要的**全部静态信息**。

    ``train_sample_ids`` / ``samples_by_traj`` 是 Scheduler/Sampler 的主键；
    真实仓库里 **sample_id == dataset local_idx**（§3 稳定索引空间）。
    """

    name: str
    #: 回合号（真实仓库 = episode id）
    train_traj_ids: List[int] = field(default_factory=list)
    val_traj_ids: List[int] = field(default_factory=list)
    #: 该任务全部训练样本 id（真实仓库 = dataset local_idx）
    train_sample_ids: List[int] = field(default_factory=list)
    #: 轨迹 → 该轨迹的样本 id（hardness 按**整条轨迹**抽子集）
    samples_by_traj: Dict[int, List[int]] = field(default_factory=dict)
    #: 该任务 val 样本的 local_idx（评测用；真实仓库才有）
    val_sample_ids: List[int] = field(default_factory=list)
    #: 固定分母（FixedBaselineMSE）；0 表示尚未算
    baseline_mse: float = 0.0
    #: 审计用（真实仓库的 task_split manifest）
    n_total: int = 0
    sha256_train: Optional[str] = None
    sha256_val: Optional[str] = None
    val_ratio: Optional[float] = None
    strategy: Optional[str] = None

    @property
    def n_train_trajs(self) -> int:
        return len(self.train_traj_ids)

    @property
    def n_val_trajs(self) -> int:
        return len(self.val_traj_ids)

    # -- B0 兼容接口 ---------------------------------------------------------
    @property
    def n_train(self) -> int:
        return len(self.train_traj_ids)

    @property
    def n_val(self) -> int:
        return len(self.val_traj_ids)

    def ids_for(self, split: str) -> List[int]:
        """**回合号**列表（B0 的便利接口）。"""
        if split == "train":
            return list(self.train_traj_ids)
        if split == "val":
            return list(self.val_traj_ids)
        raise ValueError(f"未知 split: {split!r}（只能是 'train' / 'val'）")


@dataclass
class ReplayPlan:
    """replay 池：PASS 任务 + 各自 PASS 时的采样分布（None ⇒ 均匀）。"""

    tasks: List[str] = field(default_factory=list)
    probs: Dict[str, Optional[Dict[int, float]]] = field(default_factory=dict)
    sample_ids: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class TrainRequest:
    """一次 learning unit 交给 trainer 的全部信息。"""

    task: str
    probs: Dict[int, float] = field(default_factory=dict)
    replay: ReplayPlan = field(default_factory=ReplayPlan)
    start_step: int = 0
    batch_size: int = 0
    new_slots: int = 0
    replay_slots: int = 0


@dataclass
class TrainResult:
    """一次 learning unit 的结果。"""

    loss: float = 0.0
    steps: int = 0
    #: **真实消费的样本数** = steps × batch_size（不是 1 个 batch）
    samples_seen: int = 0
    old_slot_counts: Dict[str, int] = field(default_factory=dict)
    new_slot_counts: Dict[int, int] = field(default_factory=dict)
    per_step_losses: List[float] = field(default_factory=list)
    batches_built: int = 0
    unique_batches: int = 0


@dataclass
class EvalResult:
    """`EvaluatorAdapter.evaluate_task()` 的返回（真实仓库侧便利接口）。

    口径说明
    --------
    * ``mse``  = 官方 open-loop 口径：**按完整 trajectory 聚合后再对轨迹平均**
    * ``nmse`` = ``mse / baseline_mse``，分母是**该 task 的固定 baseline**
    * ⚠️ ``nmse`` 与旧的 ``1 - r2``（分母用评测集自身 pooled 方差）**不可直接比较**
    """

    task: str
    split: str
    episode_ids: List[int]
    mse: float
    nmse: Optional[float] = None
    baseline_mse: Optional[float] = None
    mae: Optional[float] = None
    per_traj_mse: List[float] = field(default_factory=list)
    per_traj_ids: List[Any] = field(default_factory=list)
    per_traj_frames: List[int] = field(default_factory=list)
    n_traj: int = 0
    n_chunks: int = 0
    frames: int = 0
    dims: int = 0
    eval_seconds: float = 0.0
    action_keys: List[str] = field(default_factory=list)
    baseline_fingerprint: Optional[str] = None

    @property
    def gmean_mse(self) -> Optional[float]:
        from .decision.gmean import geometric_mse
        # Do not accept a complete-looking but wrong trajectory set from a cached
        # validator evaluation (e.g. scout 2 IDs accidentally reused for confirm 4).
        if (len(self.per_traj_ids) != len(self.episode_ids)
                or set(map(str, self.per_traj_ids)) != set(map(str, self.episode_ids))):
            return None
        return geometric_mse(self.per_traj_mse, expected_count=self.n_traj, ids=self.per_traj_ids)

    def to_trajectory_metrics(self, *, metric_valid: Optional[bool] = None) -> TrajectoryMetrics:
        """转成 Scheduler 认识的 `TrajectoryMetrics`（字段语义逐一对齐）。"""
        import math

        mse = float(self.mse)
        base = float(self.baseline_mse) if self.baseline_mse is not None else 0.0
        valid = (metric_valid if metric_valid is not None
                 else bool(math.isfinite(mse) and base > 0))
        per_traj = {}
        for i, v in zip(self.per_traj_ids, self.per_traj_mse):
            try:
                per_traj[int(i)] = float(v)
            except (TypeError, ValueError):
                continue
        return TrajectoryMetrics(
            task=self.task, split=self.split, episode_ids=list(self.episode_ids),
            per_traj_mse=per_traj, mse=mse, baseline_mse=base,
            nmse=self.nmse, r2=None, metric_valid=valid, n_trajs=int(self.n_traj),
            wall_time_s=float(self.eval_seconds),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task, "split": self.split, "episode_ids": list(self.episode_ids),
            "mse": self.mse, "gmean_mse": self.gmean_mse, "nmse": self.nmse, "baseline_mse": self.baseline_mse,
            "mae": self.mae, "per_traj_mse": list(self.per_traj_mse),
            "per_traj_ids": list(self.per_traj_ids),
            "per_traj_frames": list(self.per_traj_frames),
            "n_traj": self.n_traj, "n_chunks": self.n_chunks,
            "frames": self.frames, "dims": self.dims,
            "eval_seconds": self.eval_seconds, "action_keys": list(self.action_keys),
            "baseline_fingerprint": self.baseline_fingerprint,
        }


# --------------------------------------------------------------------------- #
# Protocols（Scheduler 只认这些）
# --------------------------------------------------------------------------- #


@runtime_checkable
class TaskCatalog(Protocol):
    """任务清单与静态信息。"""

    def task_names(self) -> List[str]:
        ...

    def entry(self, task: str) -> TaskEntry:
        ...


@runtime_checkable
class SampleResolver(Protocol):
    """`sample_id`（真实仓库 = dataset local_idx） → `SampleRef`。"""

    def resolve(self, task: str, sample_id: int) -> SampleRef:
        ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, task: str, split: str, episode_ids: Sequence[int]) -> TrajectoryMetrics:
        ...

    def baseline_mse(self, task: str) -> float:
        ...


@runtime_checkable
class Trainer(Protocol):
    """跑 `num_steps` 个 optimizer step。

    🔴 契约：**每一步都要重新从 NEW / Replay sampler 抽一个 batch**。
    """

    def train_steps(self, request: TrainRequest, num_steps: int) -> TrainResult:
        ...


@runtime_checkable
class HardnessScorer(Protocol):
    """给指定样本打难度分（真实仓库 = per-sample flow loss）。"""

    def score(self, task: str, sample_ids: Sequence[int]) -> Dict[int, float]:
        ...


@runtime_checkable
class ExtraState(Protocol):
    """需要随 scheduler 一起 save/restore 的额外状态（Stage A 的假世界）。"""

    def to_state(self) -> Dict[str, Any]:
        ...

    def load_state(self, raw: Dict[str, Any]) -> None:
        ...


# --------------------------------------------------------------------------- #
@dataclass
class Backend:
    """Scheduler 需要的全部外部依赖 —— 一个 bundle，便于替换与测试。"""

    catalog: Any = None
    evaluator: Any = None
    trainer: Any = None
    scorer: Any = None
    resolver: Any = None
    #: 可选：需要随状态一起持久化的额外对象（Stage A 的假世界）
    extra_state: Optional[Any] = None
    #: 可选：Fixed Baseline store（真实仓库用）
    baseline: Optional[Any] = None

    def missing(self) -> List[str]:
        """返回不满足契约的项（空 = 通过）。"""
        problems: List[str] = []
        for name, proto in (
            ("catalog", ("task_names", "entry")),
            ("evaluator", ("evaluate", "baseline_mse")),
            ("trainer", ("train_steps",)),
            ("scorer", ("score",)),
            ("resolver", ("resolve",)),
        ):
            obj = getattr(self, name, None)
            if obj is None:
                problems.append(f"{name} 缺失")
                continue
            for method in proto:
                if not callable(getattr(obj, method, None)):
                    problems.append(f"{name}.{method}() 缺失")
        return problems
