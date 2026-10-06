"""PASS pool 复查 / 遗忘判定（文档 §31–§36）。

为什么按「task transition 数」而不是 global step 触发：
遗忘更可能跟「又学了多少个新技能」相关，而不是单纯走了多少 step。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import AutoLearningConfig
from .metrics import forget_ratio, is_forgotten
from ..state.registry import TaskRecord, TaskRegistry
from .state_machine import apply_reopen, forget_code
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
            suspect = self._suspect(rec, scout.nmse)
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
            rec.note_eval({"step": global_step, "split": "review_confirm", "nmse": confirm.nmse})

            if is_forgotten(confirm.nmse, rec.best_nmse, cfg.pass_nmse, cfg.forget_relative_threshold):
                code = forget_code(rec.best_nmse, confirm.nmse, cfg.pass_nmse)
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
    def _suspect(self, rec: TaskRecord, nmse: Optional[float]) -> bool:
        cfg = self.cfg
        if nmse is None:
            return True
        if nmse > cfg.pass_nmse:
            return True
        ratio = forget_ratio(rec.best_nmse, nmse)
        return ratio is not None and ratio > cfg.forget_relative_threshold
