"""Task Registry —— scheduler 的「运行状态」（文档 §38）。

为什么不能用 heatmap 代替它：resume 之后要能回答
「这个任务什么状态？还剩几次 attempt？历史最佳？有没有 pass snapshot？」
少了任何一项，恢复出来的路线就会和原来不一样。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..config import AutoLearningConfig
from ..decision.metrics import is_finite_metric
from ..ports import TaskCatalog
from ..types import TaskStatus

#: eval_history 只保留最近这么多条，避免 registry.json 无限膨胀
MAX_EVAL_HISTORY = 120


def _require(available: int, needed: int, what: str) -> None:
    """探针轨迹数量校验 —— 不足就报错，不静默降级。"""
    if available < needed:
        raise ValueError(
            f"{what}只有 {available} 条，但配置要求 {needed} 条探针轨迹。"
            "静默少用几条会让「4-val PASS 判断」悄悄退化成 2-val —— "
            "请改配置（global_scout_val_trajs / active_val_probe_trajs / "
            "active_train_probe_trajs）或补数据。"
        )


def _select_task_names(requested: Sequence[str], available: Sequence[str]) -> List[str]:
    """把 `al.task_names` 解析成最终任务列表。

    * 去重（保留**首次出现**的顺序）
    * 校验存在（不存在就报错，并列出可用名字）
    * 空列表直接报错 —— 「一个任务都不跑」没有意义，想跑全部请用 `null`
    """
    requested = [str(n) for n in requested]
    if not requested:
        raise ValueError(
            "task_names 是空列表。想跑全部任务请设成 null（或不写这个字段）；"
            "空列表会让本次 run 一个任务都没有。"
        )
    seen: List[str] = []
    for name in requested:
        if name not in seen:
            seen.append(name)
    unknown = [n for n in seen if n not in set(available)]
    if unknown:
        raise ValueError(
            f"task_names 里有 catalog 不存在的任务：{unknown}；可用的是 {sorted(available)}"
        )
    return seen


def _int_keyed(raw: Optional[Dict[str, Any]]) -> Dict[int, float]:
    if not raw:
        return {}
    return {int(k): float(v) for k, v in raw.items()}


def _str_keyed(raw: Optional[Dict[int, float]]) -> Dict[str, float]:
    if not raw:
        return {}
    return {str(k): float(v) for k, v in raw.items()}


@dataclass
class TaskRecord:
    task_name: str

    # ---- 数据切分：一旦确定就永不改变（文档 §7）----
    train_traj_ids: List[int] = field(default_factory=list)
    val_traj_ids: List[int] = field(default_factory=list)
    scout_val_ids: List[int] = field(default_factory=list)
    confirm_val_ids: List[int] = field(default_factory=list)
    active_val_ids: List[int] = field(default_factory=list)
    train_monitor_ids: List[int] = field(default_factory=list)
    sample_ids: List[int] = field(default_factory=list)

    # ---- 指标 ----
    status: str = TaskStatus.CANDIDATE.value
    baseline_mse: Optional[float] = None
    metric_valid: bool = True
    #: 该任务当前判定口径下**有没有可用的通过线**。
    #: nmse 口径恒 True；mse 口径下，阈值表里该任务是 null ⇒ False（needs_calibration）。
    #: False 的任务**不进候选池、不消耗 attempt**（类比 `metric_valid=False`，但语义独立：
    #: rescan 里 `metric_valid` 会被重新洗白，而它不会 —— 通过线是配置层面的，与 scout 无关）。
    pass_line_usable: bool = True
    scout_nmse: Optional[float] = None
    #: Same Scout evaluation, persisted for metric-aligned GMean priority / resume.
    scout_gmean_mse: Optional[float] = None
    current_val_nmse: Optional[float] = None
    current_train_nmse: Optional[float] = None
    prev_val_nmse: Optional[float] = None
    prev_train_nmse: Optional[float] = None
    best_nmse: Optional[float] = None
    #: 绝对动作 MSE —— 判定口径 `pass_metric="mse"` 时参与比较；
    #: 与 nmse 同步记录（nmse 仍用于**跨任务**排序/统计，两者共用同一个 baseline）。
    #: 旧存档没有这两个键 ⇒ `from_dict` 走默认 None（向后兼容）。
    current_val_mse: Optional[float] = None
    best_mse: Optional[float] = None
    current_val_gmean_mse: Optional[float] = None
    best_gmean_mse: Optional[float] = None
    train_val_gap_ratio: Optional[float] = None
    lp50: Optional[float] = None
    lp_train: Optional[float] = None
    overfit: bool = False

    # ---- 记账 ----
    attempt_count: int = 0
    attempt_step: int = 0
    total_task_steps: int = 0
    forgotten: bool = False
    reopen_count: int = 0
    ever_passed: bool = False
    last_eval_step: Optional[int] = None
    last_transition_reason: str = ""
    last_transition_code: str = ""

    # ---- 采样 ----
    sample_probs: Dict[int, float] = field(default_factory=dict)
    hardness_version: int = 0
    pass_sampling_snapshot: Optional[Dict[int, float]] = None
    #: 每次 PASS 都会 +1（含回炉后再次 PASS）—— 用来确认新 snapshot 覆盖了旧版本（测试方案 §E06）
    pass_sampling_version: int = 0

    # ---- 历史（供 heatmap / LP 追溯）----
    eval_history: List[Dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------------- #
    @property
    def status_enum(self) -> TaskStatus:
        return TaskStatus(self.status)

    def set_status(self, status: TaskStatus, reason: str = "", code: str = "") -> None:
        self.status = status.value
        if reason:
            self.last_transition_reason = reason
        if code:
            self.last_transition_code = code

    def note_eval(self, row: Dict[str, Any]) -> None:
        self.eval_history.append(row)
        if len(self.eval_history) > MAX_EVAL_HISTORY:
            del self.eval_history[: len(self.eval_history) - MAX_EVAL_HISTORY]

    # ---------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        raw = dict(self.__dict__)
        raw["sample_probs"] = _str_keyed(self.sample_probs)
        raw["pass_sampling_snapshot"] = (
            None if self.pass_sampling_snapshot is None else _str_keyed(self.pass_sampling_snapshot)
        )
        return raw

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TaskRecord":
        data = dict(raw)
        data["sample_probs"] = _int_keyed(data.get("sample_probs"))
        snap = data.get("pass_sampling_snapshot")
        data["pass_sampling_snapshot"] = None if snap is None else _int_keyed(snap)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def table_row(self) -> Dict[str, Any]:
        return {
            "task": self.task_name,
            "status": self.status,
            "attempts": self.attempt_count,
            "attempt_step": self.attempt_step,
            "total_steps": self.total_task_steps,
            "scout_nmse": self.scout_nmse,
            "scout_gmean_mse": self.scout_gmean_mse,
            "val_nmse": self.current_val_nmse,
            "val_mse": self.current_val_mse,
            "val_gmean_mse": self.current_val_gmean_mse,
            "train_nmse": self.current_train_nmse,
            "best_nmse": self.best_nmse,
            "best_mse": self.best_mse,
            "best_gmean_mse": self.best_gmean_mse,
            "lp50": self.lp50,
            "overfit": self.overfit,
            "forgotten": self.forgotten,
            "reopen": self.reopen_count,
            "ever_passed": self.ever_passed,
            "hardness_v": self.hardness_version,
            "reason": self.last_transition_reason,
        }


# --------------------------------------------------------------------------- #
class TaskRegistry:
    """任务集合 + 状态。"""

    def __init__(self, records: Dict[str, TaskRecord], order: Optional[List[str]] = None) -> None:
        self.records = records
        self.order = order or list(records.keys())

    # ---------------------------------------------------------------- #
    @classmethod
    def from_catalog(cls, catalog: TaskCatalog, al: AutoLearningConfig) -> "TaskRegistry":
        """从 `TaskCatalog` 构建 —— **不认识 `SimConfig` / `TaskSpecConfig`**。

        `al.task_names` 在这里生效（后端无关层）：
          * `None` ⇒ 用 catalog 的全部任务；
          * 给了列表 ⇒ **严格按给定顺序**取这些任务（去重、校验存在、空列表报错）。
        """
        available = catalog.task_names()
        if al.task_names is None:
            names = list(available)
        else:
            names = _select_task_names(al.task_names, available)

        records: Dict[str, TaskRecord] = {}
        order: List[str] = []
        for name in names:
            entry = catalog.entry(name)
            val_ids = list(entry.val_traj_ids)
            train_ids = list(entry.train_traj_ids)
            # 🔴 probe 数量不足必须 fail-fast（不能静默少用几条，
            # 否则「4-val PASS 判断」会悄悄退化成 2-val）
            _require(
                len(val_ids),
                max(al.global_scout_val_trajs, al.active_val_probe_trajs),
                f"{name} 的 val 轨迹数",
            )
            _require(len(train_ids), al.active_train_probe_trajs, f"{name} 的 train 轨迹数")
            records[name] = TaskRecord(
                task_name=name,
                train_traj_ids=train_ids,
                val_traj_ids=val_ids,
                scout_val_ids=val_ids[: al.global_scout_val_trajs],
                confirm_val_ids=val_ids[: max(al.global_scout_val_trajs, al.active_val_probe_trajs)],
                active_val_ids=val_ids[: al.active_val_probe_trajs],
                train_monitor_ids=train_ids[: al.active_train_probe_trajs],
                sample_ids=list(entry.train_sample_ids),
                baseline_mse=entry.baseline_mse,
                metric_valid=entry.baseline_mse > 0.0,
            )
            order.append(name)
        return cls(records, order)

    # ---------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterable[TaskRecord]:
        for name in self.order:
            yield self.records[name]

    def get(self, name: str) -> TaskRecord:
        return self.records[name]

    def names(self) -> List[str]:
        return list(self.order)

    def by_status(self, status: TaskStatus) -> List[TaskRecord]:
        return [r for r in self if r.status == status.value]

    def counts(self) -> Dict[str, int]:
        out = {s.value: 0 for s in TaskStatus}
        for r in self:
            out[r.status] = out.get(r.status, 0) + 1
        return out

    def pass_tasks(self) -> List[str]:
        return [r.task_name for r in self.by_status(TaskStatus.PASS)]

    def candidate_records(self) -> List[TaskRecord]:
        """可被 scheduler 选中的任务。

        `metric_valid` + `nmse` 有限 + **有可用通过线**，三个条件缺一不可 ——
        否则 NaN 会进 `min(...)`，排序结果未定义（测试方案 §I01）；
        没有通过线的任务（`pass_metric="mse"` 且阈值表 null）不该被选中白烧 attempt。
        """
        return [
            r
            for r in self.by_status(TaskStatus.CANDIDATE)
            if r.metric_valid and is_finite_metric(r.scout_nmse) and r.pass_line_usable
        ]

    def coverage(self) -> float:
        n = len(self)
        if n == 0:
            return 0.0
        return len(self.by_status(TaskStatus.PASS)) / n

    def table(self) -> List[Dict[str, Any]]:
        return [r.table_row() for r in self]

    def summary(self) -> Dict[str, Any]:
        counts = self.counts()
        nm = [r.current_val_nmse for r in self if r.current_val_nmse is not None]
        from ..decision.metrics import median, mean

        return {
            "n_tasks": len(self),
            "pass": counts.get("PASS", 0),
            "candidate": counts.get("CANDIDATE", 0),
            "defer": counts.get("DEFER", 0),
            "exhausted": counts.get("EXHAUSTED", 0),
            "coverage": round(self.coverage(), 4),
            "median_val_nmse": None if not nm else round(median(nm) or 0.0, 5),
            "worst_val_nmse": None if not nm else round(max(nm), 5),
            "mean_val_nmse": None if not nm else round(mean(nm) or 0.0, 5),
        }

    # ---------------------------------------------------------------- #
    def to_state(self) -> Dict[str, Any]:
        return {
            "order": list(self.order),
            "records": {name: rec.to_dict() for name, rec in self.records.items()},
        }

    def load_state(self, raw: Dict[str, Any]) -> None:
        self.order = list(raw.get("order") or list(self.records.keys()))
        for name, data in (raw.get("records") or {}).items():
            self.records[name] = TaskRecord.from_dict(data)
