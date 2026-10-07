"""PASS pool 复查 / 遗忘判定（文档 §31–§36）。

为什么按「task transition 数」而不是 global step 触发：
遗忘更可能跟「又学了多少个新技能」相关，而不是单纯走了多少 step。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import AutoLearningConfig
from .metrics import forget_ratio, is_finite_metric
from ..state.registry import TaskRecord, TaskRegistry
from .state_machine import apply_reopen
from .thresholds import forget_code_ex, is_forgotten_ex, metric_value, pass_line
from ..types import EvalSplit, TaskStatus


@dataclass
class ReviewOutcome:
    task: str
    scout_nmse: Optional[float]
    confirmed: bool
    confirm_nmse: Optional[float] = None
    best_nmse: Optional[float] = None
    forget_ratio: Optional[float] = None
    forgotten: bool = False
    action: str = "ok"
    code: str = ""

    def to_row(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "scout_nmse": self.scout_nmse,
            "confirmed": self.confirmed,
            "confirm_nmse": self.confirm_nmse,
            "best_nmse": self.best_nmse,
            "forget_ratio": self.forget_ratio,
            "forgotten": self.forgotten,
            "action": self.action,
            "code": self.code,
        }


class Reviewer:
    def __init__(self, cfg: AutoLearningConfig, evaluator) -> None:
        self.cfg = cfg
        self.evaluator = evaluator

    def review(self, registry: TaskRegistry, global_step: int) -> List[ReviewOutcome]:
        cfg = self.cfg
        outcomes: List[ReviewOutcome] = []
        for rec in list(registry.by_status(TaskStatus.PASS)):
            scout = self.evaluator.evaluate(rec.task_name, EvalSplit.REVIEW.value, rec.scout_val_ids)
            rec.last_eval_step = global_step
            rec.note_eval({"step": global_step, "split": "review_scout", "nmse": scout.nmse})
            suspect = self._suspect(rec, scout.nmse, scout.mse)
            outcome = ReviewOutcome(
                task=rec.task_name,
                scout_nmse=scout.nmse,
                confirmed=False,
                best_nmse=rec.best_nmse,
                forget_ratio=forget_ratio(rec.best_nmse, scout.nmse),
            )
            if not suspect:
                outcome.action = "ok"
                outcomes.append(outcome)
                continue

            # 疑似退化 ⇒ 补到 4 条确认（文档 §32）
            confirm = self.evaluator.evaluate(
                rec.task_name, EvalSplit.REVIEW.value, rec.confirm_val_ids
            )
            outcome.confirmed = True
            outcome.confirm_nmse = confirm.nmse
            outcome.forget_ratio = forget_ratio(rec.best_nmse, confirm.nmse)
            rec.current_val_nmse = confirm.nmse
            rec.current_val_mse = confirm.mse
            rec.note_eval({"step": global_step, "split": "review_confirm", "nmse": confirm.nmse})
            rec.note_eval(
                {"step": global_step, "split": "review_confirm", "mse": confirm.mse}
            )

            # 🔴 遗忘判定统一走 thresholds（口径跟随 cfg.pass_metric；无可用阈值时只剩相对退化）
            if is_forgotten_ex(
                cfg,
                rec.task_name,
                cur_nmse=confirm.nmse,
                cur_mse=confirm.mse,
                best_nmse=rec.best_nmse,
            ):
                code = forget_code_ex(
                    cfg, rec.task_name, cur_nmse=confirm.nmse, cur_mse=confirm.mse
                )
                status = apply_reopen(
                    rec,
                    cfg,
                    f"review: current={confirm.nmse} best={rec.best_nmse}",
                    code,
                )
                outcome.forgotten = True
                outcome.code = code
                outcome.action = "reopen" if status == TaskStatus.CANDIDATE else "exhausted"
            else:
                rec.forgotten = False
                outcome.action = "ok_after_confirm"
            outcomes.append(outcome)
        return outcomes

    # ---------------------------------------------------------------- #
    def _suspect(self, rec: TaskRecord, nmse: Optional[float],
                 mse: Optional[float] = None) -> bool:
        """疑似退化 ⇒ 值得补一次 confirm。

        口径与正式判定**保持一致**：先看"掉出及格线"（跟随 `cfg.pass_metric`），
        再看相对退化。该任务没有可用阈值时，只剩相对退化这一条。
        """
        cfg = self.cfg
        if nmse is None:
            return True
        metric, line = pass_line(cfg, rec.task_name)
        if line is not None:
            val = metric_value(metric, nmse=nmse, mse=mse)
            if is_finite_metric(val) and float(val) > line:
                return True
        ratio = forget_ratio(rec.best_nmse, nmse)
        return ratio is not None and ratio > cfg.forget_relative_threshold
