"""共享值类型。刻意保持「无依赖」，方便任何模块 import。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TaskStatus(str, Enum):
    """任务状态（文档 §5）。用 str 枚举，便于直接写进 JSON。"""

    CANDIDATE = "CANDIDATE"
    TRAINING = "TRAINING"
    PASS = "PASS"
    DEFER = "DEFER"
    EXHAUSTED = "EXHAUSTED"


class Decision(str, Enum):
    """一次 50-step 学习单元结束后的判定结果（文档 §14–§17）。"""

    CONTINUE = "CONTINUE"
    PASS = "PASS"
    DEFER = "DEFER"
    DEFER_OVERFIT = "DEFER_OVERFIT"


class TransitionKind(str, Enum):
    """任务状态迁移的类型。**只有这几种算一次 transition**（测试方案 §F01）。

    纯 candidate rescan **不算** transition —— 它不改变任何任务的状态。
    """

    PASS = "PASS"
    DEFER = "DEFER"
    DEFER_OVERFIT = "DEFER_OVERFIT"
    EXHAUSTED = "EXHAUSTED"
    REOPEN = "REOPEN"
    REOPEN_EXHAUSTED = "REOPEN_EXHAUSTED"


class ReasonCode(str, Enum):
    """机器可读的迁移原因（测试方案 §H01）。

    光有人类可读的句子不够 —— 「为什么 PASS / 为什么 DEFER」必须能被程序断言，
    否则长跑结束以后只能靠肉眼看日志。
    """

    # --- PASS ---
    VAL_BELOW_THRESHOLD = "val_nmse_below_threshold"
    CONTINUE_AFTER_PASS = "continue_after_pass"
    # --- DEFER ---
    OVERFIT = "overfit"
    BELOW_MIN_STEPS = "below_min_steps_before_defer"
    LP_HIGH = "lp50_above_min"
    LP_LOW = "low_progress"
    LP_UNAVAILABLE = "lp50_unavailable"
    METRIC_INVALID = "metric_invalid"
    # --- EXHAUSTED ---
    ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
    REOPEN_CHURN_GUARD = "reopen_churn_guard"
    # --- REOPEN ---
    FORGOTTEN_BELOW_PASS_LINE = "forgotten_below_pass_line"
    FORGOTTEN_RELATIVE_DEGRADATION = "forgotten_relative_degradation"
    # --- 调度 ---
    ROUND_ROLLOVER = "round_rollover"
    BOOTSTRAP_PASS = "bootstrap_auto_pass"
    RESCAN_PASS = "rescan_auto_pass"
    SELECTED = "selected_as_current_task"


@dataclass(frozen=True)
class Verdict:
    """一次判定的完整结论：决策 + 机器码 + 人话。

    故意支持解包（`decision, detail = verdict`），所以老的调用点不用改。
    """

    decision: Decision
    code: str
    detail: str

    def __iter__(self):
        return iter((self.decision, self.detail))


class EvalSplit(str, Enum):
    """评测用途（文档 §7）。"""

    SCOUT = "scout"  # 2 条 val，全局扫描
    CONFIRM = "confirm"  # 4 条 val，PASS 确认
    ACTIVE_VAL = "active_val"  # 当前任务 4 条 val
    TRAIN_MONITOR = "train_monitor"  # 当前任务 4 条 train
    REVIEW = "review"  # PASS pool 复查（2 或 4 条 val）


@dataclass(frozen=True)
class SampleRef:
    """一个训练样本的稳定身份（文档 §43.1）。

    `sample_id` 必须稳定：hardness 扫描结果、PASS snapshot、replay 采样
    全靠它做映射。绝不靠 batch 顺序猜样本身份。
    """

    task: str
    sample_id: int
    episode_id: int
    frame_id: int
    is_val: bool = False

    @property
    def uid(self) -> str:
        return f"{self.task}:{self.sample_id}"


@dataclass
class BatchComposition:
    """一个 batch 的构成：new_slots 个当前任务样本 + replay_slots 个旧任务样本。"""

    new: List[SampleRef] = field(default_factory=list)
    old: List[SampleRef] = field(default_factory=list)

    @property
    def refs(self) -> List[SampleRef]:
        return self.new + self.old

    @property
    def n_new(self) -> int:
        return len(self.new)

    @property
    def n_old(self) -> int:
        return len(self.old)

    @property
    def old_tasks(self) -> List[str]:
        seen: List[str] = []
        for r in self.old:
            if r.task not in seen:
                seen.append(r.task)
        return seen


@dataclass
class TrajectoryMetrics:
    """一次 open-loop 评测的聚合结果。

    Stage A 由 `sim.SimulatedEvaluator` 产生；Stage B 由真模型产生，
    但字段语义必须完全一致（文档 §51）。
    """

    task: str
    split: str
    episode_ids: List[int]
    per_traj_mse: Dict[int, float]
    mse: float
    baseline_mse: float
    nmse: Optional[float]
    r2: Optional[float]
    metric_valid: bool
    n_trajs: int
    wall_time_s: float = 0.0
    note: str = ""

    def to_row(self) -> Dict[str, object]:
        return {
            "task": self.task,
            "split": self.split,
            "n_trajs": self.n_trajs,
            "mse": self.mse,
            "baseline_mse": self.baseline_mse,
            "nmse": self.nmse,
            "r2": self.r2,
            "metric_valid": self.metric_valid,
            "note": self.note,
        }
