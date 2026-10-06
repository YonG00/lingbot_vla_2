"""Auto Learning 的接口契约（Stage B0）。

五个 Protocol 与 Stage A Demo 的 `auto_learning/ports.py` **同名同形**，
以便已冻结的 Scheduler 在 B1 里**不改一行**就能指过来。

本模块**只依赖 stdlib**（`typing` / `dataclasses`），可以脱离 torch/lerobot 导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

# --------------------------------------------------------------------------- #
# DTO
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskEntry:
    """一个任务在 `tools/task_split.py` 里的固定划分（**只读**）。"""

    name: str
    train_ids: tuple = ()      # 固定 train 回合号（默认 40 条）
    val_ids: tuple = ()        # 固定 val 回合号（默认 10 条）
    n_total: int = 0
    sha256_train: Optional[str] = None
    sha256_val: Optional[str] = None
    val_ratio: Optional[float] = None
    strategy: Optional[str] = None

    @property
    def n_train(self) -> int:
        return len(self.train_ids)

    @property
    def n_val(self) -> int:
        return len(self.val_ids)

    def ids_for(self, split: str) -> List[int]:
        """``split`` ∈ {"train", "val"}。"""
        if split == "train":
            return list(self.train_ids)
        if split == "val":
            return list(self.val_ids)
        raise ValueError(f"未知 split: {split!r}（只能是 'train' / 'val'）")


@dataclass(frozen=True)
class SampleRef:
    """一条样本的**稳定坐标**。

    ⚠️ 正式代码里**不重新编码 sample id** —— 坐标就是
    ``(task, episode, frame)``，其中 ``frame`` 是数据集里的**绝对帧号**
    （`hf_dataset['index']`，与 `base_dataset.py::__getitem__` 用的同一个值）。
    """

    task: str
    episode: int
    frame: int


@dataclass
class EvalResult:
    """`EvaluatorAdapter.evaluate_task()` 的返回。

    口径说明
    --------
    * ``mse``  = 官方 open-loop 口径：**按完整 trajectory 聚合后再对轨迹平均**
    * ``nmse`` = ``mse / baseline_mse``，分母是**该 task 的固定 baseline**
      （task-global-mean + trajectory-balanced，只预计算一次，不随模型更新）
    * ⚠️ ``nmse`` 与旧的 ``1 - r2``（分母用评测集自身的 pooled 方差）**不可直接比较**
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task, "split": self.split, "episode_ids": list(self.episode_ids),
            "mse": self.mse, "nmse": self.nmse, "baseline_mse": self.baseline_mse,
            "mae": self.mae, "per_traj_mse": list(self.per_traj_mse),
            "per_traj_ids": list(self.per_traj_ids),
            "per_traj_frames": list(self.per_traj_frames),
            "n_traj": self.n_traj, "n_chunks": self.n_chunks,
            "frames": self.frames, "dims": self.dims,
            "eval_seconds": self.eval_seconds, "action_keys": list(self.action_keys),
            "baseline_fingerprint": self.baseline_fingerprint,
        }


@dataclass
class TrainRequest:
    """一次 `train_steps` 的输入（**B0 只声明，不驱动**）。"""

    task: str
    probs: Dict[Any, float] = field(default_factory=dict)     # 新任务样本权重
    replay: Any = None                                        # ReplayPlan
    start_step: int = 0
    batch_size: int = 0
    new_slots: int = 0
    replay_slots: int = 0


@dataclass
class TrainResult:
    """一次 `train_steps` 的返回（**B0 只声明，不驱动**）。"""

    loss: float = 0.0
    steps: int = 0
    samples_seen: int = 0
    batches_built: int = 0
    per_step_losses: List[float] = field(default_factory=list)
    new_slot_counts: Dict[Any, int] = field(default_factory=dict)
    old_slot_counts: Dict[Any, int] = field(default_factory=dict)
    unique_batches: int = 0


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class TaskCatalog(Protocol):
    """任务与其固定 train/val 划分（**复用** `tools/task_split.py` 的 manifest）。"""

    def names(self) -> List[str]:
        ...

    def entry(self, task: str) -> TaskEntry:
        ...


@runtime_checkable
class SampleResolver(Protocol):
    """`local_idx ↔ (task, episode, frame)` 的稳定双向映射。"""

    def local_to_ref(self, local_idx: int) -> SampleRef:
        ...

    def ref_to_local(self, ref: SampleRef) -> int:
        ...

    def episode_map(self):
        ...


@runtime_checkable
class Evaluator(Protocol):
    """开环评测（**复用** `OpenLoopValidator`，不重写 metric pipeline）。"""

    def evaluate_task(self, task_id: str, split: str,
                      episode_ids: Optional[Sequence[int]] = None) -> EvalResult:
        ...


@runtime_checkable
class HardnessScorer(Protocol):
    """逐样本难度（固定 noise / 固定 flow time / no_grad → per-sample L1_fm）。"""

    def score(self, items: Sequence[Dict[str, Any]]):
        ...


@runtime_checkable
class Trainer(Protocol):
    """训练步抽象（**B0 只声明**）。"""

    def train_steps(self, req: TrainRequest, num_steps: int) -> TrainResult:
        ...


# --------------------------------------------------------------------------- #
# Backend bundle
# --------------------------------------------------------------------------- #


@dataclass
class Backend:
    """把 5 个适配器打包（与 Stage A Demo 的 `ports.Backend` 对应）。"""

    catalog: Any = None
    resolver: Any = None
    evaluator: Any = None
    scorer: Any = None
    trainer: Any = None
    baseline: Any = None
    extra_state: Any = None

    def missing(self) -> List[str]:
        """返回**缺失**的必需组件名（供 fail-fast 用）。"""
        need = ("catalog", "resolver", "evaluator", "scorer", "trainer")
        return [n for n in need if getattr(self, n, None) is None]
