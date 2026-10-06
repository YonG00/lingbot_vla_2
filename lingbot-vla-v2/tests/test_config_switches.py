"""测试方案 §5/§6 的补充 —— **配置开关真的改变了行为**。

`pytest` 能断言「改了之后行为确实不同」；**改多少 → 往哪个方向变** 用
`tools/sweep_params.py` 看趋势表。

每个用例都是「同一个配置，只改一个参数，跑两遍，断言结果的关系」。
这条比「跑通」重要得多 —— 一个参数如果写了不生效（或方向反了），
Demo 的结论就是错的（`defer_retry_steps` 之前就是死参数）。
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import pytest

from lingbotvla.auto_learning.config import AutoLearningConfig, DemoConfig, load_config
from lingbotvla.auto_learning.testing import fake_tasks as ft
from lingbotvla.auto_learning.testing.fake_tasks import make_cfg
from lingbotvla.auto_learning.tools.sweep_params import apply_override, run_metrics

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "lingbotvla", "auto_learning", "configs")


def _demo(name: str = "demo_4task.yaml") -> DemoConfig:
    return load_config(os.path.join(CONFIG_DIR, name))


def _al(**kw) -> AutoLearningConfig:
    base = dict(seed=3, eval_interval_steps=50, min_steps_before_defer=100, pass_nmse=0.30)
    base.update(kw)
    return AutoLearningConfig(**base)


def _overfit_cfg() -> DemoConfig:
    """一个会命中 overfit 的任务（train 变好、val 变差）。"""
    return make_cfg([ft.overfit("o0")], al=_al(early_defer_on_overfit=False))


def _rescue_cfg() -> DemoConfig:
    """一个练不动但会触发 rescue 的任务。"""
    return make_cfg(
        [ft._spec("p0", curve=[0.90, 0.80, 0.79, 0.785, 0.782, 0.780])],
        al=_al(defer_resample_retry=True, max_attempts_per_task=1, defer_retry_steps=50),
    )


@dataclass
class SwitchCase:
    name: str
    param: str
    low: Any
    high: Any
    check: Callable[[Dict[str, Any], Dict[str, Any]], bool]
    note: str
    base: Callable[[], DemoConfig] = _demo
    #: 两边都要额外设的参数（例如 continue_after_pass 要配 post_pass_max_steps）
    extra: Dict[str, Any] = field(default_factory=dict)


CASES = [
    SwitchCase(
        "pass_nmse",
        "pass_nmse",
        0.15,
        0.45,
        lambda lo, hi: hi["pass"] > lo["pass"],
        "通关线放松 ⇒ PASS 更多",
    ),
    SwitchCase(
        "min_lp50",
        "min_lp50",
        0.0,
        0.20,
        lambda lo, hi: hi["units"] < lo["units"],
        "LP 阈值提高 ⇒ 更早 DEFER ⇒ units 更少",
    ),
    SwitchCase(
        "min_steps_before_defer",
        "min_steps_before_defer",
        50,
        300,
        lambda lo, hi: hi["units"] > lo["units"],
        "最小判断预算变大 ⇒ units 更多",
    ),
    SwitchCase(
        "max_attempts_per_task",
        "max_attempts_per_task",
        1,
        3,
        lambda lo, hi: hi["units"] > lo["units"],
        "训练机会变多 ⇒ units 更多",
    ),
    SwitchCase(
        "max_reopens_per_task",
        "max_reopens_per_task",
        0,
        3,
        lambda lo, hi: lo["reopens"] == 0 and hi["reopens"] > 0,
        "churn guard=0 ⇒ 一忘就不再回炉；放开 ⇒ 有回炉",
    ),
    SwitchCase(
        "defer_resample_retry",
        "defer_resample_retry",
        False,
        True,
        lambda lo, hi: hi["units"] > lo["units"],
        "rescue 打开 ⇒ 多跑一段",
        base=_rescue_cfg,
    ),
    SwitchCase(
        "defer_retry_steps",
        "defer_retry_steps",
        50,
        150,
        lambda lo, hi: hi["units"] > lo["units"],
        "rescue 步数真的按配置执行",
        base=_rescue_cfg,
    ),
    SwitchCase(
        "review_after_task_transitions",
        "review_after_task_transitions",
        0,
        1,
        lambda lo, hi: lo["reviews"] == 0 and hi["reviews"] > 0,
        "0 ⇒ 从不复查；1 ⇒ 每次迁移都复查",
    ),
    SwitchCase(
        "forget_relative_threshold",
        "forget_relative_threshold",
        0.05,
        5.00,
        lambda lo, hi: lo["reopens"] > hi["reopens"],
        "阈值越小越敏感 ⇒ 回炉更多",
        base=lambda: _demo("demo_12task.yaml"),
    ),
    SwitchCase(
        "continue_after_pass",
        "continue_after_pass",
        False,
        True,
        lambda lo, hi: hi["units"] > lo["units"],
        "PASS 后继续训 ⇒ units 更多",
        extra={"post_pass_max_steps": 200, "post_pass_min_lp": 0.0},
    ),
    SwitchCase(
        "early_defer_on_overfit",
        "early_defer_on_overfit",
        False,
        True,
        lambda lo, hi: hi["units"] < lo["units"],
        "命中 overfit 就提前 DEFER ⇒ units 更少",
        base=_overfit_cfg,
    ),
    SwitchCase(
        "max_new_tasks_attempted_this_run",
        "max_new_tasks_attempted_this_run",
        1,
        4,
        lambda lo, hi: hi["trained"] > lo["trained"],
        "主动尝试上限变大 ⇒ 训练的任务更多",
    ),
    SwitchCase(
        "max_new_tasks_passed_this_run",
        "max_new_tasks_passed_this_run",
        1,
        None,
        lambda lo, hi: hi["units"] >= lo["units"] and hi["pass"] >= lo["pass"],
        "按 PASS 数收工：放开上限不会更差",
        base=lambda: _demo("demo_12task.yaml"),
    ),
]


def _run_case(case: SwitchCase, value: Any) -> Dict[str, Any]:
    cfg = case.base()
    for key, val in case.extra.items():
        setattr(cfg.auto_learning, key, val)
    cfg.auto_learning.validate()
    cfg = apply_override(cfg, case.param, value)
    return run_metrics(cfg, max_actions=2000)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_switch_changes_behaviour(case: SwitchCase):
    lo = _run_case(case, case.low)
    hi = _run_case(case, case.high)
    assert case.check(lo, hi), (
        f"[{case.name}] {case.note}\n"
        f"  {case.param}={case.low!r} → pass={lo['pass']} units={lo['units']} "
        f"reopens={lo['reopens']} reviews={lo['reviews']} trained={lo['trained']}\n"
        f"  {case.param}={case.high!r} → pass={hi['pass']} units={hi['units']} "
        f"reopens={hi['reopens']} reviews={hi['reviews']} trained={hi['trained']}"
    )


def test_every_preset_param_exists_in_config():
    """`sweep_params` 的预设参数名必须都真实存在（防止改名后工具静默失效）。"""
    from lingbotvla.auto_learning.tools.sweep_params import PRESETS

    fields = {f.name for f in __import__("dataclasses").fields(AutoLearningConfig)}
    for name, preset in PRESETS.items():
        assert preset.param in fields, f"预设 {name} 的参数 {preset.param} 不存在"
        assert preset.values, f"预设 {name} 没有值"


def test_sweep_values_are_all_valid():
    """预设里的每个值都要能通过 validate（否则工具会打出一堆「非法配置」）。"""
    from lingbotvla.auto_learning.tools.sweep_params import PRESETS

    base = _demo()
    for name, preset in PRESETS.items():
        for value in preset.values:
            apply_override(base, preset.param, value)  # 不抛就算通过


def test_sweep_params_tool_runs_end_to_end(capsys):
    """扫描工具本身要能跑（`--list` + 一个真实扫描）。"""
    from lingbotvla.auto_learning.tools.sweep_params import main

    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "pass_nmse" in out and "hardness_probe_fraction" in out

    assert main([
        "-c", os.path.join(CONFIG_DIR, "demo_4task.yaml"),
        "--param", "pass_nmse", "--values", "0.20,0.40",
    ]) == 0
    out = capsys.readouterr().out
    assert "pass_nmse" in out
    assert "PASS" in out
