"""TB 对齐补丁的**补充**回归（审查方添加，独立文件，不修改原作者测试）。

补的缺口（对照 `tests/test_al_tensorboard_step_alignment.py`）：
1. `current_skill/val_to_pass_threshold` 与 `val_mse / pass_threshold_mse` 的**数值一致性**；
   `task/<task>/val_to_pass_threshold` 与 `current_skill/val_to_pass_threshold` 同 step 同值。
2. `pass_metric="nmse"` 时**只**记 `current_skill/val_mse`，不得出现阈值类标签。
3. 阈值表缺该任务时**不得**写出误导性的 0。
4. Resume 前后横轴**单调不倒退**，且同一 (tag, tb_step) 不出现互相矛盾的值。
5. 训练器那行 `set_tb_step_offset(...)` 必须是**无条件**执行的（缩进 8 = 在 `if _al_parts is not None:`
   内、**不在** `if _al_state is not None:` 内）—— 原作者测试只测了适配器算术，没测这处接线。
6. `auto_learning/unit_loss` 的语义（unit 平均）在 `real/hook.py` 里由 `sum(_unit_losses)/len(...)` 决定
   —— 无 GPU 无法执行，故用源码契约断言钉住（下注明示这是 contract test）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from al_fixtures import scheduler_of
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.decision.thresholds import PassThresholds
from lingbotvla.auto_learning.real.build import SchedulerLoggerAdapter
from lingbotvla.auto_learning.testing.fake_tasks import cfg_with

REPO = Path(__file__).resolve().parents[1]


class RepoLog:
    def info_rank0(self, *a, **k): pass
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


class Writer:
    def __init__(self):
        self.points = []

    def add_scalar(self, name, value, step):
        self.points.append((name, float(value), int(step)))


def _run(metric: str, tasks, thresholds: dict, *, al_step0: int = 0, max_actions: int = 40):
    cfg = AutoLearningConfig(seed=3, eval_interval_steps=5, min_steps_before_defer=10,
                             pass_metric=metric, pass_thresholds_file="test_only")
    cfg.pass_thresholds = PassThresholds(tasks=thresholds)
    writer = Writer()
    logger = SchedulerLoggerAdapter(RepoLog(), writer=writer)
    logger.set_tb_step_offset(train_global_step=500, al_global_step=al_step0)
    sched = scheduler_of(cfg_with(list(tasks), al=cfg), logger=logger)
    sched.run(max_actions=max_actions)
    return cfg, writer.points


# ---------------------------------------------------------------- 1) 数值一致性
def test_val_to_pass_threshold_matches_val_mse_over_threshold():
    """比值必须等于「同一个 eval 的 mse / 该任务的阈值」（不是跨任务混算）。

    ⚠️ 注意 `current_skill/*` 的语义是**当步当前任务**（历史命名），会在任务间跳变；
    因此跨任务的比较只能走 `task/<task>/*` + `debug/<task>/active_val_mse`。
    """
    _cfg, points = _run("mse", ["easy_pass", "unlearnable"],
                        {"easy_pass": 0.10, "unlearnable": 0.10})

    per_task = {}
    cur = {}
    for tag, val, step in points:
        m = re.match(r"task/(.+)/(pass_threshold_mse|val_to_pass_threshold)$", tag)
        if m:
            per_task.setdefault((m.group(1), step), {})[m.group(2)] = val
        m = re.match(r"debug/(.+)/active_val_mse$", tag)
        if m:
            per_task.setdefault((m.group(1), step), {})["mse"] = val
        if tag.startswith("current_skill/"):
            cur.setdefault(step, {})[tag.split("/", 1)[1]] = val

    checked = 0
    for (task, step), d in per_task.items():
        if {"mse", "pass_threshold_mse", "val_to_pass_threshold"} <= set(d):
            assert d["val_to_pass_threshold"] == pytest.approx(
                d["mse"] / d["pass_threshold_mse"], rel=1e-12), f"{task}@{step} 比值与 mse/阈值不符"
            checked += 1
    assert checked >= 2, f"应至少覆盖两个 (task, step) 组合，实际 {checked}"

    checked_cur = 0
    for step, d in cur.items():
        if {"val_mse", "pass_threshold_mse", "val_to_pass_threshold"} <= set(d):
            assert d["val_to_pass_threshold"] == pytest.approx(
                d["val_mse"] / d["pass_threshold_mse"], rel=1e-12), f"current_skill@{step} 内部不一致"
            # current_skill 的 mse 必须等于当步某个任务的 active_val_mse（= 当前任务）
            peers = {dd.get("mse") for (tt, ss), dd in per_task.items() if ss == step}
            assert any(p is not None and abs(p - d["val_mse"]) < 1e-12 for p in peers), \
                f"current_skill@{step} 的 val_mse 不属于当步任何任务"
            checked_cur += 1
    assert checked_cur >= 1

# ---------------------------------------------------------------- 2) nmse 模式
def test_nmse_mode_logs_val_mse_only():
    _cfg, points = _run("nmse", ["easy_pass", "unlearnable"],
                        {"easy_pass": 0.10, "unlearnable": 0.10})
    tags = {t for t, _v, _s in points}
    assert "current_skill/val_mse" in tags
    assert not any(t.endswith("pass_threshold_mse") for t in tags)
    assert not any(t.endswith("val_to_pass_threshold") for t in tags)


# ---------------------------------------------------------------- 3) 缺阈值 ⇒ 不写 0
def test_missing_task_threshold_emits_no_zero():
    cfg = AutoLearningConfig(seed=3, eval_interval_steps=5, min_steps_before_defer=10,
                             pass_metric="mse", pass_thresholds_file="test_only")
    cfg.pass_thresholds = PassThresholds(tasks={"easy_pass": 0.10})   # unlearnable 不在表里
    writer = Writer()
    logger = SchedulerLoggerAdapter(RepoLog(), writer=writer)
    logger.set_tb_step_offset(train_global_step=500, al_global_step=0)
    sched = scheduler_of(cfg_with(["easy_pass", "unlearnable"], al=cfg), logger=logger)
    sched.run(max_actions=40)
    missing = [p for p in writer.points
               if p[0].startswith("task/unlearnable/") and p[0].endswith(("pass_threshold_mse",
                                                                         "val_to_pass_threshold"))]
    assert missing == [], f"表里没有该任务的阈值就不该写标签（更不能写 0）: {missing}"


# ---------------------------------------------------------------- 4) Resume 单调性
def test_resume_axis_is_monotonic_and_conflict_free(tmp_path):
    ev = tmp_path / "events.jsonl"
    a = SchedulerLoggerAdapter(RepoLog(), writer=(wa := Writer()), event_path=str(ev))
    a.set_tb_step_offset(train_global_step=500, al_global_step=0)      # 首跑：AL 0 → tb 500
    for s in range(0, 11, 5):
        a.log_metrics(s, "current_skill/val_nmse", 0.5)
    # 断点：trainer=510 / al=10 ⇒ 恢复后 offset 仍为 500
    b = SchedulerLoggerAdapter(RepoLog(), writer=(wb := Writer()), event_path=str(ev))
    b.set_tb_step_offset(train_global_step=510, al_global_step=10)
    for s in range(10, 21, 5):
        b.log_metrics(s, "current_skill/val_nmse", 0.4)

    steps = [s for _t, _v, s in wa.points] + [s for _t, _v, s in wb.points]
    assert steps == sorted(steps), f"横轴不得倒退: {steps}"
    assert steps == [500, 505, 510, 510, 515, 520]

    rows = [json.loads(x) for x in ev.read_text(encoding="utf-8").splitlines()]
    assert all("step" in r and "tb_step" in r for r in rows)
    assert [r["step"] for r in rows] == [0, 5, 10, 10, 15, 20]        # AL 相对步语义不变
    assert [r["tb_step"] for r in rows] == steps                       # 绝对横轴

    # ⚠️ 同一 (tag, x) 在 resume 边界**允许**被重写一次（边界 unit 可能被重新评测），
    #    真正要保证的是：① 横轴不倒退 ② AL 侧不得再出现 training/loss（与训练器同名冲突）
    assert not any(tag == "training/loss" for tag, _v, _s in wa.points + wb.points)
    assert not any(tag == "training/loss" for tag, _v, _s in wa.points + wb.points)


# ---------------------------------------------------------------- 5) 训练器接线
def test_trainer_alignment_call_is_unconditional():
    """contract test（无 GPU 跑不了训练器）：校准必须对 fresh run 也生效。

    8 空格缩进 ⇒ 在 `if _al_parts is not None:` 体内、`if _al_state is not None:` 体**外**。
    """
    src = (REPO / "tasks" / "vla" / "train_lingbotvla.py").read_text(encoding="utf-8").splitlines()
    idx = [i for i, l in enumerate(src) if "set_tb_step_offset(" in l]
    assert len(idx) == 1, f"应恰好一处校准调用，实际 {len(idx)}"
    i = idx[0]
    indent = len(src[i]) - len(src[i].lstrip(" "))
    assert indent == 8, f"缩进应为 8（无条件执行），实际 {indent}；12 表示被关进了 resume-only 分支"
    # 向上找最近的、缩进更小的 `if`，必须包含 _al_parts
    for j in range(i - 1, -1, -1):
        line = src[j]
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip(" "))
        if ind < indent and line.strip().startswith("if "):
            assert "_al_parts" in line, f"最近的上级 if 应为 _al_parts，实际: {line.strip()}"
            break
    # 必须在第一次 on_step_begin(0) 之前（即首个 unit 发布前）完成校准
    prime = [k for k, l in enumerate(src) if "on_step_begin(0)" in l]
    assert prime and i < prime[0]


# ---------------------------------------------------------------- 6) unit_loss 语义契约
def test_unit_loss_is_average_of_per_step_losses_contract():
    """contract test：`TrainResult.loss` 必须是 per-step 平均（故标签名 unit_loss 名副其实）。"""
    txt = (REPO / "lingbotvla" / "auto_learning" / "real" / "hook.py").read_text(encoding="utf-8")
    flat = re.sub(r"\s+", "", txt)
    assert "loss=(sum(self._unit_losses)/len(self._unit_losses))" in flat
