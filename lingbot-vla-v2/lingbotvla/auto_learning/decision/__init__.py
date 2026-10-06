"""决策层 —— 纯函数，不碰 IO / RNG，可直接单测。

    metrics.py         NMSE / R² / LP50 / 遗忘率 / overfit 判据（文档 §9 §14 §17 §23）
    state_machine.py   状态迁移 + 「一次 unit 之后怎么判」（文档 §5–§17）
    review.py          PASS pool 复查 + 遗忘判定（文档 §31–§36）

这一层是整个系统的「规则书」：给同样的数字，永远得到同样的判定。
"""

from .metrics import (
    forget_ratio,
    is_forgotten,
    is_overfit,
    learning_progress,
    nmse,
    percentile_rank,
)
from .review import Reviewer, ReviewOutcome
from .state_machine import decide_after_unit

__all__ = [
    "nmse",
    "learning_progress",
    "forget_ratio",
    "is_forgotten",
    "is_overfit",
    "percentile_rank",
    "Reviewer",
    "ReviewOutcome",
    "decide_after_unit",
]
