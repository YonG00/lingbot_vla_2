"""参数扫描 / 消融：改一个参数，看**结果怎么变**。

    # 看有哪些预设扫描
    python -m auto_learning.tools.sweep_params --list

    # 扫一个参数
    python -m auto_learning.tools.sweep_params --param pass_nmse

    # 自己给值
    python -m auto_learning.tools.sweep_params --param min_lp50 --values 0.0,0.02,0.05,0.10

    # 扫全部预设（慢）
    python -m auto_learning.tools.sweep_params --all

    # 多 seed 取平均，降低噪声
    python -m auto_learning.tools.sweep_params --param pass_nmse --seeds 5

为什么要它：`pytest` 只能断言「这个开关改了之后行为确实不同」；
**改多少 → 结果往哪个方向走** 需要看趋势。这个工具就是那张趋势表，
也可以拿它做 Stage B 的消融（换 backend 之后同样能跑）。
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import random
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import AutoLearningConfig, DemoConfig, load_config
from ..decision.metrics import median
from ..obs import report as rpt
from ..types import TaskStatus

# --------------------------------------------------------------------------- #
# 一次 run 要汇报的指标
# --------------------------------------------------------------------------- #
METRIC_KEYS = [
    "pass",
    "exhausted",
    "candidate",
    "defer",
    "units",
    "step",
    "samples",
    "transitions",
    "reopens",
    "trained",
    "auto_pass",
    "newly_pass",
    "reviews",
    "median_nmse",
    "worst_nmse",
]


def run_metrics(cfg: DemoConfig, seed: Optional[int] = None, max_actions: int = 2000) -> Dict[str, Any]:
    """跑一遍配置，返回一张指标表（不含 RNG 状态，可 JSON 序列化）。"""
    from ..testing.sim import build_scheduler

    if seed is not None:
        cfg.sim.seed = seed
    sched = build_scheduler(cfg)
    sched.run(max_actions=max_actions)

    counts = sched.registry.counts()
    vals = [r.current_val_nmse for r in sched.registry if r.current_val_nmse is not None]
    return {
        "pass": counts[TaskStatus.PASS.value],
        "exhausted": counts[TaskStatus.EXHAUSTED.value],
        "candidate": counts[TaskStatus.CANDIDATE.value],
        "defer": counts[TaskStatus.DEFER.value],
        "units": sched.state.units_run,
        "step": sched.state.global_step,
        "samples": sched.state.global_samples_seen,
        "transitions": sched.state.transition_count,
        "reopens": sum(r.reopen_count for r in sched.registry),
        "trained": len(sched.state.trained_tasks),
        "auto_pass": len(sched.state.auto_passed),
        "newly_pass": len(sched.state.newly_passed),
        "reviews": sum(1 for e in sched.events if e.get("action") == "review"),
        "median_nmse": None if not vals else round(median(vals) or 0.0, 5),
        "worst_nmse": None if not vals else round(max(vals), 5),
        "stop": sched.state.stop_reason,
        "coverage": round(sched.registry.coverage(), 4),
    }


def average_metrics(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """多 seed 取平均（`stop` 取众数，`median_nmse` 等取均值）。"""
    if not runs:
        return {}
    out: Dict[str, Any] = {}
    for key in runs[0]:
        values = [r.get(key) for r in runs]
        if key == "stop":
            out[key] = max(set(values), key=values.count)
        elif len(set(values)) == 1:
            out[key] = values[0]  # 单 seed（或完全一致）时保留原始类型，别把 int 变成 3.0000
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            out[key] = round(statistics.fmean(values), 5)
        else:
            out[key] = values[0]
    out["n_seeds"] = len(runs)
    return out


# --------------------------------------------------------------------------- #
# 覆盖一个参数
# --------------------------------------------------------------------------- #
def apply_override(cfg: DemoConfig, param: str, value: Any) -> DemoConfig:
    """在配置副本上改一个参数；带联动校验（不合法会抛 ValueError）。"""
    new = copy.deepcopy(cfg)
    al = new.auto_learning
    if param == "new_slots":
        al.new_slots = int(value)
        al.replay_slots = al.batch_size - al.new_slots
    elif param == "replay_slots":
        al.replay_slots = int(value)
        al.new_slots = al.batch_size - al.replay_slots
    elif param == "batch_size":
        # batch_size 与 new/replay slots 是绑定的，改它要联动
        al.batch_size = int(value)
        al.replay_slots = min(al.replay_slots, al.batch_size - 1)
        al.new_slots = al.batch_size - al.replay_slots
    elif not hasattr(al, param):
        raise KeyError(f"AutoLearningConfig 没有字段 {param}")
    else:
        setattr(al, param, value)
    al.validate()
    return new


def _parse_values(raw: str) -> List[Any]:
    out: List[Any] = []
    for token in raw.split(","):
        token = token.strip()
        if token.lower() in ("none", "null", ""):
            out.append(None)
        elif token.lower() in ("true", "false"):
            out.append(token.lower() == "true")
        else:
            try:
                out.append(int(token))
            except ValueError:
                out.append(float(token))
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# --------------------------------------------------------------------------- #
@dataclass
class Preset:
    param: str
    values: List[Any]
    note: str
    #: 期望的趋势（只写给人看，不做断言）
    expect: str = ""


PRESETS: Dict[str, Preset] = {
    "pass_nmse": Preset(
        "pass_nmse", [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        "通关线（NMSE ≤ 此值算 PASS）",
        "越高 ⇒ PASS 越多、units 越少、EXHAUSTED 越少",
    ),
    "min_lp50": Preset(
        "min_lp50", [0.0, 0.02, 0.05, 0.10, 0.20],
        "DEFER 的 LP 阈值（低于它就算学不动）",
        "越高 ⇒ 越早 DEFER、units 越少、EXHAUSTED 越多",
    ),
    "min_steps_before_defer": Preset(
        "min_steps_before_defer", [50, 100, 150, 200, 300],
        "最少训多少步才允许 DEFER",
        "越大 ⇒ units 越多（不会过早放弃）",
    ),
    "max_attempts_per_task": Preset(
        "max_attempts_per_task", [1, 2, 3],
        "每个任务最多几次**主训练**机会",
        "越大 ⇒ EXHAUSTED 越少、units 越多",
    ),
    "max_reopens_per_task": Preset(
        "max_reopens_per_task", [0, 1, 2, 3, None],
        "churn guard：最多回炉几次（不占训练预算）",
        "越大 ⇒ reopens 越多；0 ⇒ 一忘就不再回炉",
    ),
    "defer_resample_retry": Preset(
        "defer_resample_retry", [False, True],
        "准备 DEFER 时是否换一组 hardness 分布再给一次机会",
        "True ⇒ units 更多",
    ),
    "defer_retry_steps": Preset(
        "defer_retry_steps", [50, 100, 150],
        "rescue 额外给多少 step",
        "越大 ⇒ units 越多（真的在执行）",
    ),
    "hardness_probe_fraction": Preset(
        "hardness_probe_fraction", [0.10, 0.20, 0.33, 0.50, 1.00],
        "hardness 扫描覆盖多少比例的训练轨迹",
        "越大 ⇒ 扫描帧数越多（影响采样分布，不一定改变覆盖率）",
    ),
    "hardness_weight_max": Preset(
        "hardness_weight_max", [1.0, 2.0, 3.0, 5.0],
        "最难样本的采样权重上限（1.0 = 关掉 hard-sampling）",
        "越大 ⇒ 采样越偏向难样本；⚠️ 假世界**测不出**收益（见 README §1）",
    ),
    "review_after_task_transitions": Preset(
        "review_after_task_transitions", [0, 1, 2, 5],
        "每几次任务迁移做一次 PASS pool 复查",
        "0 ⇒ 从不复查（reopens 必为 0）",
    ),
    "forget_relative_threshold": Preset(
        "forget_relative_threshold", [0.05, 0.30, 1.00, 5.00],
        "相对退化多少算遗忘",
        "越小 ⇒ 越敏感、reopens 越多",
    ),
    "continue_after_pass": Preset(
        "continue_after_pass", [False, True],
        "PASS 之后是否继续训（post-pass 阶段）",
        "True ⇒ units 更多",
    ),
    "early_defer_on_overfit": Preset(
        "early_defer_on_overfit", [False, True],
        "命中 overfit 时是否提前 DEFER",
        "True ⇒ units 更少",
    ),
    "max_new_tasks_attempted_this_run": Preset(
        "max_new_tasks_attempted_this_run", [1, 2, 4, 8, None],
        "本次最多主动尝试多少个新任务（第一阶段主开关）",
        "越小 ⇒ trained 越少、越早停",
    ),
    "max_new_tasks_passed_this_run": Preset(
        "max_new_tasks_passed_this_run", [1, 2, 4, None],
        "本次最多让多少个任务 newly PASS（可选）",
        "越小 ⇒ 越早停",
    ),
    "batch_size": Preset(
        "batch_size", [10, 20],
        "每个 optimizer step 的样本数（改它会联动 new/replay slots）",
        "越大 ⇒ samples 越多（同一 unit 消费更多样本）",
    ),
}


# --------------------------------------------------------------------------- #
def sweep(
    base_cfg: DemoConfig,
    preset: Preset,
    seeds: int = 1,
    max_actions: int = 2000,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for value in preset.values:
        try:
            cfg = apply_override(base_cfg, preset.param, value)
        except (ValueError, KeyError) as exc:
            rows.append({"value": _fmt(value), "note": f"非法配置：{exc}"})
            continue
        runs = []
        for i in range(max(1, seeds)):
            seed = None if seeds <= 1 else base_cfg.sim.seed + i
            runs.append(run_metrics(cfg, seed=seed, max_actions=max_actions))
        row = average_metrics(runs)
        row["value"] = _fmt(value)
        rows.append(row)
    return rows


def print_sweep(preset: Preset, rows: List[Dict[str, Any]], param_width: int = 26) -> None:
    print("=" * 100)
    print(f"参数：{preset.param} —— {preset.note}")
    if preset.expect:
        print(f"预期趋势：{preset.expect}")
    print("=" * 100)
    keys = ["value"] + [k for k in METRIC_KEYS if any(k in r for r in rows)]
    headers = {
        "value": preset.param,
        "pass": "PASS",
        "exhausted": "EXHST",
        "candidate": "CAND",
        "defer": "DEFER",
        "units": "units",
        "step": "step",
        "samples": "samples",
        "transitions": "trans",
        "reopens": "reopen",
        "trained": "trained",
        "auto_pass": "autoP",
        "newly_pass": "newP",
        "reviews": "review",
        "median_nmse": "中位NMSE",
        "worst_nmse": "最差NMSE",
    }
    print(rpt.format_table(rows, keys, [headers.get(k, k) for k in keys],
                           align_right=[k for k in keys if k != "value"]))
    stops = {r.get("stop") for r in rows}
    if len(stops) > 1:
        print(f"停止原因：{sorted(s for s in stops if s)}")


def _default_config() -> str:
    """默认配置 —— 用**包内相对路径**，避免依赖 cwd。"""
    from pathlib import Path
    return str(Path(__file__).resolve().parents[1] / "configs" / "demo_4task.yaml")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sweep_params")
    p.add_argument("-c", "--config", default=_default_config())
    p.add_argument("--param", default=None)
    p.add_argument("--values", default=None, help="逗号分隔；不给就用预设")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--max-actions", type=int, default=2000)
    p.add_argument("--list", action="store_true", help="列出所有预设")
    p.add_argument("--all", action="store_true", help="跑全部预设")
    p.add_argument("-o", "--out", default=None, help="把结果写成 JSON")
    args = p.parse_args(argv)

    if args.list:
        for name, preset in PRESETS.items():
            print(f"{name:34s} {_fmt_values(preset.values):28s} {preset.note}")
            if preset.expect:
                print(f"{'':34s} → {preset.expect}")
        return 0

    base = load_config(args.config)
    results: Dict[str, Any] = {}

    if args.all:
        names = list(PRESETS)
    elif args.param:
        names = [args.param]
    else:
        print("请给 --param，或用 --list / --all")
        return 2

    for name in names:
        preset = PRESETS.get(name)
        if preset is None:
            if not args.values:
                print(f"[error] {name} 没有预设，请用 --values 给一组值")
                return 2
            preset = Preset(name, _parse_values(args.values), "自定义")
        elif args.values:
            preset = Preset(name, _parse_values(args.values), preset.note, preset.expect)
        rows = sweep(base, preset, seeds=args.seeds, max_actions=args.max_actions)
        print_sweep(preset, rows)
        print()
        results[name] = {"note": preset.note, "expect": preset.expect, "rows": rows}

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"结果已写入 {args.out}")
    return 0


def _fmt_values(values: Sequence[Any]) -> str:
    return ",".join(_fmt(v) for v in values)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
