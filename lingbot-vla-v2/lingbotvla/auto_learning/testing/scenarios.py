"""测试方案 §13 —— 五套一键端到端场景（测试与 CLI 共用同一份定义）。

每个场景都是「人工造出来的完整自动学习过程」，用来验证多机制协同时的行为。
放在包里而不是测试里，是为了让 `python -m auto_learning.tools.run_demo_scenarios`
和 pytest 用的是**同一份**场景定义（避免两处参数漂移）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import AutoLearningConfig, DemoConfig
from . import fake_tasks as ft
from .fake_tasks import make_cfg

#: 曲线按「一个 unit 只给当前任务 0.7 的有效训练量」标定（batch = 7 NEW + 3 OLD）


def _al(**kw) -> AutoLearningConfig:
    base = dict(seed=3, eval_interval_steps=50, min_steps_before_defer=100, pass_nmse=0.30)
    base.update(kw)
    return AutoLearningConfig(**base)


@dataclass
class Scenario:
    name: str
    title: str
    cfg: DemoConfig
    #: 期望的最终状态（任务名 → PASS / EXHAUSTED / CANDIDATE）
    expect: Dict[str, str] = field(default_factory=dict)
    #: 期望出现过的迁移类型
    expect_kinds: List[str] = field(default_factory=list)
    #: 期望「一次都没训过」的任务
    expect_untrained: List[str] = field(default_factory=list)
    notes: str = ""


def scenario1_ideal() -> Scenario:
    """A 初始 PASS → B 50 step PASS → C 100 step PASS → D DEFER 后第二轮 PASS。"""
    cfg = make_cfg(
        [
            ft._spec("A", curve=[0.18], forget_rate=0.01, degrade_slope=0.01),
            ft._spec("B", curve=[0.45, 0.22], forget_rate=0.01, degrade_slope=0.01),
            ft._spec("C", curve=[0.50, 0.36, 0.16], forget_rate=0.01, degrade_slope=0.01),
            ft._spec("D", curve=[0.60, 0.58, 0.56, 0.20], forget_rate=0.01, degrade_slope=0.01),
        ],
        al=_al(review_after_task_transitions=2),
    )
    return Scenario(
        name="1-ideal",
        title="理想学习路径：全部 PASS（含一次 DEFER → 第二轮修复）",
        cfg=cfg,
        expect={"A": "PASS", "B": "PASS", "C": "PASS", "D": "PASS"},
        expect_kinds=["PASS", "DEFER"],
        expect_untrained=["A"],
        notes="A 是 bootstrap 免费 PASS，不应出现在训练序列里",
    )


def scenario2_unlearnable() -> Scenario:
    """存在学不会的任务：B 不能阻塞整个训练。"""
    cfg = make_cfg(
        [
            ft.easy_pass("A"),
            ft.unlearnable("B"),
            ft._spec("C", curve=[0.55, 0.40, 0.26]),
            ft.easy_pass("D"),
        ],
        al=_al(max_attempts_per_task=2),
    )
    return Scenario(
        name="2-unlearnable",
        title="存在学不会任务：B 最终 EXHAUSTED，A/C/D 正常 PASS",
        cfg=cfg,
        expect={"A": "PASS", "B": "EXHAUSTED", "C": "PASS", "D": "PASS"},
        expect_kinds=["EXHAUSTED", "PASS"],
    )


def scenario3_transfer() -> Scenario:
    """Transfer：训 A 之后 B 自动变好，不需要再完整训练。"""
    cfg = make_cfg(
        [
            ft._spec("A", curve=[0.45, 0.32, 0.24], group="pair", transfer=0.90),
            ft._spec("B", curve=[0.55, 0.25], group="pair", transfer=0.90),
        ],
        al=_al(),
    )
    return Scenario(
        name="3-transfer",
        title="Transfer：A 训完把 B 带过线（B 一次都没训过）",
        cfg=cfg,
        expect={"A": "PASS", "B": "PASS"},
        expect_kinds=["PASS"],
        expect_untrained=["B"],
        notes="验证「一个技能顶多个任务」——rescan 后 B 应直接进 PASS confirm",
    )


def scenario4_forgetting() -> Scenario:
    """Forgetting / Reopen：A 被忘掉 → 回炉 → 修复后重新 PASS。"""
    cfg = make_cfg(
        [
            ft._spec("A", curve=[0.38, 0.16], forget_rate=0.60, degrade_slope=0.20),
            ft._spec("B", curve=[0.60, 0.50, 0.40, 0.28], forget_rate=0.01, degrade_slope=0.01),
            ft._spec("C", curve=[0.62, 0.52, 0.42, 0.28], forget_rate=0.01, degrade_slope=0.01),
        ],
        al=_al(review_after_task_transitions=1, max_attempts_per_task=3),
    )
    return Scenario(
        name="4-forgetting",
        title="遗忘与回炉：A PASS → 被忘 → reopen → 修复后再次 PASS",
        cfg=cfg,
        expect={"A": "PASS", "B": "PASS", "C": "PASS"},
        expect_kinds=["PASS", "REOPEN"],
        notes="replay 全程都要在发生；A 的 pass_sampling_version 应 ≥ 2",
    )


def scenario5_conflict() -> Scenario:
    """A/B 冲突：互相遗忘，靠 attempt budget 终止，不允许无限横跳。"""
    cfg = make_cfg(
        [
            ft._spec("A", curve=[0.35, 0.20], forget_rate=1.5, degrade_slope=0.35),
            ft._spec("B", curve=[0.36, 0.21], forget_rate=1.5, degrade_slope=0.35),
            ft._spec("C", curve=[0.60, 0.50, 0.40, 0.28], forget_rate=0.01, degrade_slope=0.01),
        ],
        al=_al(review_after_task_transitions=1, max_attempts_per_task=2),
    )
    return Scenario(
        name="5-conflict",
        title="A/B 灾难性左右横跳：attempt budget 必须终止循环",
        cfg=cfg,
        expect={"C": "PASS"},
        expect_kinds=["REOPEN", "EXHAUSTED"],
        notes="A 或 B 至少一个 EXHAUSTED；总单元数必须有界",
    )


SCENARIOS = {
    "1-ideal": scenario1_ideal,
    "2-unlearnable": scenario2_unlearnable,
    "3-transfer": scenario3_transfer,
    "4-forgetting": scenario4_forgetting,
    "5-conflict": scenario5_conflict,
}


def build(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"未知场景 {name}，可选：{sorted(SCENARIOS)}")
    return SCENARIOS[name]()


def build_all() -> List[Scenario]:
    return [f() for f in SCENARIOS.values()]
