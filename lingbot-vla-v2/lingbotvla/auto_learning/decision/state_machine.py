"""状态机与判定（文档 §5–§17）。

本模块刻意写成**纯函数**：不碰 IO、不碰 RNG，只根据 registry 里的数字决定
「下一步该 PASS / 继续 / DEFER / EXHAUSTED」。这样测试方案 §5 的用例可以直接单测。

每个判定都返回 `Verdict(decision, code, detail)`：
  * `code`   机器可读（`ReasonCode`），可以写断言；
  * `detail` 人类可读，直接进日志与 CSV。
只有人话没有机器码的话，长跑结束以后「为什么这么跑」就只能靠肉眼看日志（测试方案 §H01）。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..config import AutoLearningConfig
from ..state.registry import TaskRecord, TaskRegistry
from ..types import Decision, ReasonCode, TaskStatus, Verdict
from .metrics import forget_ratio, is_finite_metric
from .thresholds import (
    PassCheck,
    check_pass,
    forget_code_ex,
    is_forgotten_ex,
    pass_line,
)


# --------------------------------------------------------------------------- #
# 一次 50-step 学习单元之后的判定（文档 §42 的菱形分支）
# --------------------------------------------------------------------------- #
def decide_after_unit(
    record: TaskRecord,
    cfg: AutoLearningConfig,
    *,
    attempt_step: int,
    defer_after_steps: Optional[int] = None,
) -> Verdict:
    """一次 learning unit 之后的判定。

    `defer_after_steps` 是本 attempt 的 DEFER 阈值 —— 默认 `min_steps_before_defer`，
    触发 rescue 后会被推后 `defer_retry_steps`（所以 `defer_retry_steps` 是**真的**在执行，
    不是写多少都只多跑一个单元）。
    """
    val = record.current_val_nmse
    threshold = cfg.min_steps_before_defer if defer_after_steps is None else defer_after_steps

    # NaN / Inf 绝不能参与「达标」判断（测试方案 §I01）。
    # 但也不能无限 CONTINUE —— 用满 min_steps 后照样 DEFER，保证有界。
    if not is_finite_metric(val):
        if attempt_step >= threshold:
            return Verdict(
                Decision.DEFER,
                ReasonCode.METRIC_INVALID.value,
                f"metric_invalid（nmse={val!r}）且已用满 {attempt_step} step",
            )
        return Verdict(
            Decision.CONTINUE,
            ReasonCode.METRIC_INVALID.value,
            f"metric_invalid（nmse={val!r}），先继续观察",
        )

    # ① 达标立刻 PASS —— 哪怕只训了 50 step（文档 §44）
    #    🔴 判定口径**统一**走 decision/thresholds：默认 `nmse`（= 旧的全局 pass_nmse），
    #       切到 `mse` 时按任务查阈值表。**不要在别处再写裸的 pass_nmse 比较**（否则口径会分叉）。
    _metric, _line = pass_line(cfg, record.task_name)
    _check = check_pass(
        cfg, record.task_name, nmse=val, mse=record.current_val_mse
    )
    if _check == PassCheck.PASS:
        _shown = record.current_val_mse if _metric == "mse" else val
        if (
            cfg.continue_after_pass
            and attempt_step < cfg.post_pass_max_steps
            and (record.lp50 is None or record.lp50 >= cfg.post_pass_min_lp)
        ):
            return Verdict(
                Decision.CONTINUE,
                ReasonCode.CONTINUE_AFTER_PASS.value,
                f"continue_after_pass: {_metric}={_shown:.6g} 已达标但仍在进步，"
                f"继续到 {cfg.post_pass_max_steps} step",
            )
        return Verdict(
            Decision.PASS,
            ReasonCode.VAL_BELOW_THRESHOLD.value,
            f"{_metric}={_shown:.6g} <= threshold={_line:.6g}（task={record.task_name}）",
        )

    # ② 过拟合提前 DEFER（文档 §17.1）
    if cfg.early_defer_on_overfit and record.overfit:
        return Verdict(
            Decision.DEFER_OVERFIT,
            ReasonCode.OVERFIT.value,
            f"overfit: lp_train={record.lp_train} lp_val={record.lp50} "
            f"gap={record.train_val_gap_ratio}",
        )

    # ③ 还没到最小判断预算，再给一个单元（文档 §14.2）
    if attempt_step < threshold:
        return Verdict(
            Decision.CONTINUE,
            ReasonCode.BELOW_MIN_STEPS.value,
            f"attempt_step={attempt_step} < defer_after_steps={threshold}",
        )

    # ④ 到预算了：看 LP
    lp = record.lp50
    if lp is None:
        return Verdict(
            Decision.CONTINUE, ReasonCode.LP_UNAVAILABLE.value, "lp50 缺失，再观察一个单元"
        )
    if lp >= cfg.min_lp50:
        return Verdict(
            Decision.CONTINUE,
            ReasonCode.LP_HIGH.value,
            f"lp50={lp:.4f} >= min_lp50={cfg.min_lp50}（还在学）",
        )
    return Verdict(
        Decision.DEFER,
        ReasonCode.LP_LOW.value,
        f"lp50={lp:.4f} < min_lp50={cfg.min_lp50} 且已用满 {attempt_step} step",
    )


def should_rescue(cfg: AutoLearningConfig, rescued: bool) -> bool:
    """文档 §15：准备 DEFER 时，可选地「换样本分布再给一次机会」。"""
    return bool(cfg.defer_resample_retry and not rescued)


def forget_code(best: Optional[float], current: Optional[float], pass_nmse: float) -> str:
    """遗忘的**原因码** —— 掉出及格线 vs 相对退化，两者处置不同（文档 §34）。

    ⚠️ 这是**旧的口径固定版**（只认 nmse 及格线），保留是为了向后兼容与既有单测。
    框架内部请走 `decision.thresholds.forget_code_ex(cfg, task, cur_nmse=…, cur_mse=…)`，
    否则切到 `pass_metric="mse"` 时原因码会与判定口径分叉。
    """
    if current is not None and current > pass_nmse:
        return ReasonCode.FORGOTTEN_BELOW_PASS_LINE.value
    return ReasonCode.FORGOTTEN_RELATIVE_DEGRADATION.value


# --------------------------------------------------------------------------- #
# 状态迁移
# --------------------------------------------------------------------------- #
def start_attempt(record: TaskRecord) -> None:
    record.attempt_count += 1
    record.attempt_step = 0
    record.prev_val_nmse = None
    record.prev_train_nmse = None
    record.lp50 = None
    record.lp_train = None
    record.overfit = False
    record.set_status(
        TaskStatus.TRAINING,
        "selected_as_current_task",
        ReasonCode.SELECTED.value,
    )


def apply_pass(
    record: TaskRecord,
    cfg: AutoLearningConfig,
    reason: str,
    code: str = ReasonCode.VAL_BELOW_THRESHOLD.value,
) -> None:
    record.set_status(TaskStatus.PASS, f"PASS: {reason}", code)
    record.forgotten = False
    record.ever_passed = True
    record.pass_sampling_version += 1
    # PASS 时把当时的采样分布存下来，供以后 replay 用（文档 §30）。
    # 从没训过的任务（bootstrap 直接 PASS）这里是 None ⇒ replay 自然回落到 uniform。
    record.pass_sampling_snapshot = dict(record.sample_probs) if record.sample_probs else None
    if record.current_val_nmse is not None:
        record.best_nmse = (
            record.current_val_nmse
            if record.best_nmse is None
            else min(record.best_nmse, record.current_val_nmse)
        )
    # 绝对 MSE 与 nmse 同步记账（判定口径可能切到 mse；跨任务排序仍用 nmse）
    if record.current_val_mse is not None:
        record.best_mse = (
            record.current_val_mse
            if record.best_mse is None
            else min(record.best_mse, record.current_val_mse)
        )


def apply_defer(
    record: TaskRecord,
    cfg: AutoLearningConfig,
    reason: str,
    code: str = ReasonCode.LP_LOW.value,
) -> TaskStatus:
    """第一次失败 → DEFER；用完 attempt 预算 → EXHAUSTED（文档 §16）。"""
    if record.attempt_count >= cfg.max_attempts_per_task:
        record.set_status(
            TaskStatus.EXHAUSTED,
            f"EXHAUSTED: attempt {record.attempt_count}/{cfg.max_attempts_per_task} 用尽（{reason}）",
            ReasonCode.ATTEMPT_BUDGET_EXHAUSTED.value,
        )
        return TaskStatus.EXHAUSTED
    record.set_status(
        TaskStatus.DEFER,
        f"DEFER: attempt {record.attempt_count}/{cfg.max_attempts_per_task}（{reason}）",
        code,
    )
    return TaskStatus.DEFER


def attempt_budget_exhausted(record: TaskRecord, cfg: AutoLearningConfig) -> bool:
    """**训练**预算是否用尽。

    只看 `attempt_count` —— 也就是「这个任务真正成为主任务并发生训练」的次数。
    `reopen_count` 是纯诊断，**不**参与这里（回炉不该吃掉训练机会）。
    """
    return record.attempt_count >= cfg.max_attempts_per_task


def churn_guard_tripped(record: TaskRecord, cfg: AutoLearningConfig) -> bool:
    """**churn guard**：同一任务被「遗忘 → 回炉」太多次就判终态。

    单独一个开关，而不是拿 `reopen_count` 去占 attempt 预算 ——
    因为「免费 PASS → 遗忘 → 回炉」这条路径一次训练都不发生，
    只靠 attempt 是拦不住的。
    """
    return cfg.max_reopens_per_task is not None and record.reopen_count >= cfg.max_reopens_per_task


def is_terminal(record: TaskRecord, cfg: AutoLearningConfig) -> bool:
    """该任务在本 run 内是否**不再被 rescan 复活**。

    只看 churn guard：`attempt_count` 用尽的任务**仍然允许**被 transfer 免费带过线
    （那是白赚的）；但如果它反复「PASS → 遗忘」，churn guard 会把它钉死，循环自然终止。
    """
    return churn_guard_tripped(record, cfg)


def apply_reopen(
    record: TaskRecord,
    cfg: AutoLearningConfig,
    reason: str,
    code: str = ReasonCode.FORGOTTEN_BELOW_PASS_LINE.value,
) -> TaskStatus:
    """遗忘 → 回炉（文档 §35–§36）。

    两条独立的闸门：
      * `attempt_count` 用尽 ⇒ 已经没有训练机会了，回去也没法修 ⇒ EXHAUSTED
      * `reopen_count` 达到 churn guard ⇒ 反复回炉没意义 ⇒ EXHAUSTED
    """
    record.forgotten = True
    # 先判闸门，**再**记数 —— 这样 `reopen_count` 数的是「**批准**的回炉次数」，
    # `max_reopens_per_task=0` 时它就是 0（而不是「尝试过 1 次」）。
    if attempt_budget_exhausted(record, cfg):
        record.set_status(
            TaskStatus.EXHAUSTED,
            f"EXHAUSTED: 已遗忘但训练预算用尽"
            f"（attempt={record.attempt_count}/{cfg.max_attempts_per_task}）（{reason}）",
            ReasonCode.ATTEMPT_BUDGET_EXHAUSTED.value,
        )
        return TaskStatus.EXHAUSTED
    if churn_guard_tripped(record, cfg):
        record.set_status(
            TaskStatus.EXHAUSTED,
            f"EXHAUSTED: 回炉次数达上限"
            f"（reopen={record.reopen_count}/{cfg.max_reopens_per_task}）（{reason}）",
            ReasonCode.REOPEN_CHURN_GUARD.value,
        )
        return TaskStatus.EXHAUSTED
    record.reopen_count += 1
    record.set_status(
        TaskStatus.CANDIDATE,
        f"REOPEN: 遗忘回炉（{reason}）",
        code,
    )
    return TaskStatus.CANDIDATE


def rollover_round(registry: TaskRegistry, cfg: AutoLearningConfig) -> List[str]:
    """本轮候选都处理完了 ⇒ 给 DEFER 且还有预算的任务第二次机会（文档 §16）。"""
    promoted: List[str] = []
    for rec in registry.by_status(TaskStatus.DEFER):
        if rec.attempt_count < cfg.max_attempts_per_task:
            rec.set_status(
                TaskStatus.CANDIDATE,
                "round_rollover: 本轮结束，重新进入候选",
                ReasonCode.ROUND_ROLLOVER.value,
            )
            promoted.append(rec.task_name)
    return promoted


def mark_forgotten_if_needed(
    record: TaskRecord,
    cfg: AutoLearningConfig,
    current_nmse: Optional[float],
    current_mse: Optional[float] = None,
) -> Tuple[bool, str]:
    """复查后更新遗忘标记（不改状态，状态由 `apply_reopen` 负责）。

    返回 `(是否遗忘, 原因码)`。

    ⚠️ 遗留 API，框架内部已不走这里（review 用 `thresholds.is_forgotten_ex`）；
    仅 nmse 时与统一口径一致，mse 判定请传 ``current_mse``。
    """
    if current_nmse is None:
        return False, ""
    task = getattr(record, "task_name", "")
    if is_forgotten_ex(
        cfg, task, cur_nmse=current_nmse, cur_mse=current_mse, best_nmse=record.best_nmse
    ):
        record.forgotten = True
        return True, forget_code_ex(cfg, task, cur_nmse=current_nmse, cur_mse=current_mse)
    return False, ""
