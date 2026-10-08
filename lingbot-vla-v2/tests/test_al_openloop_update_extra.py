"""开环优化补丁的**补充**边界测试（审查方添加，独立文件，不修改原作者测试）。

补的缺口（对照 `tests/test_al_openloop_update.py`）：
1. 旧 checkpoint（**缺** task_switch_count/full_rescan_count）恢复后应从 0 开始，且之后第 3 次切换仍触发
   —— 指南 §"需要知道的行为变化" 明确承诺，但原测试只覆盖"带新字段"的恢复。
2. Rescan 节奏的 **N=1 / N=2 边界**（原测试只覆盖 N=3）。
3. 关闭 rescan 期间**计数器仍应推进**，重新打开后按累计次数决定触发点。
4. `--high-cv-policy warn` 下 **CV 恰好等于上限**（严格 `>`）与刚超过上限的差异。
5. warn 与 baseline 上限**同时命中**时，status 与 `baseline_capped` 两个信号彼此独立。
6. warn 生成的数值线在**判定层**（`check_pass`）可用 ⇒ 高 CV 任务确实不再"因 null 被排除"。
7. CLI `--high-cv-policy` 非法取值应被 argparse 拒绝。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

from al_fixtures import make_cfg, scheduler_of
from lingbotvla.auto_learning.baseline import BaselineStore, FixedBaseline, task_fingerprint
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.decision.thresholds import (
    PassCheck, PassThresholds, ThresholdsError, check_pass,
)
from lingbotvla.auto_learning.state import persistence
from lingbotvla.auto_learning.tools.build_gmean_thresholds import build_thresholds, main


# ------------------------------------------------------------------ 夹具
def _sched(*, scout_confirm_enabled=False, n=3, rescan=True):
    al = AutoLearningConfig(scout_confirm_enabled=scout_confirm_enabled,
                            rescan_every_n_task_switches=n)
    cfg = make_cfg(3, al=al)
    s = scheduler_of(cfg)
    s.al.rescan_candidates_after_transition = rescan
    return s


def _store(tmp_path, count=2, mse=0.25):
    p = tmp_path / "baseline.json"
    st = BaselineStore(path=str(p), config_fingerprint="testfp")
    for i in range(count):
        st.put(FixedBaseline(task=f"t{i}", mse=mse, mu=(0,),
                             fingerprint=task_fingerprint("testfp", f"sha{i}")))
    st.save()
    return BaselineStore.load(str(p))


# ------------------------------------------------------------------ 1) 旧 checkpoint
def test_resume_from_legacy_checkpoint_without_new_counters(tmp_path):
    """旧存档缺两个新计数器 ⇒ 从 0 开始（不报错），且第 3 次切换仍触发全池 Rescan。"""
    sched = _sched(n=3)
    sched._rescan = lambda *a, **k: []
    sched.state.transition_count = 7                      # 模拟"跑过很久"的旧运行
    p = tmp_path / "state.json"
    persistence.save_state(sched, str(p))

    raw = json.loads(Path(p).read_text(encoding="utf-8"))

    def _strip(node):
        """递归删掉两个新字段，模拟补丁前写出的 checkpoint。"""
        removed = 0
        if isinstance(node, dict):
            for k in ("task_switch_count", "full_rescan_count"):
                if k in node:
                    node.pop(k)
                    removed += 1
            for v in node.values():
                removed += _strip(v)
        elif isinstance(node, list):
            for v in node:
                removed += _strip(v)
        return removed

    assert _strip(raw) == 2, "应当在存档里找到并删掉两个新计数器"
    Path(p).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    resumed = _sched(n=3)
    calls = []
    resumed._rescan = lambda *a, **k: calls.append(1) or []
    persistence.load_state(resumed, str(p))                # 不得抛错

    assert resumed.state.task_switch_count == 0
    assert resumed.state.full_rescan_count == 0
    assert resumed.state.transition_count == 7             # 旧字段照常恢复
    for k in range(2):
        resumed._after_transition(f"t{k}", "DEFER", "test")
    assert calls == [], "第 1/2 次切换不应触发"
    resumed._after_transition("t2", "PASS", "test")
    assert len(calls) == 1 and resumed.state.full_rescan_count == 1


# ------------------------------------------------------------------ 2) N=1 / N=2 边界
@pytest.mark.parametrize("n,expected", [(1, [1, 2, 3, 4]), (2, [2, 4]), (3, [3]), (4, [4])])
def test_rescan_cadence_boundaries(n, expected):
    sched = _sched(n=n)
    hits = []
    sched._rescan = lambda *a, **k: hits.append(sched.state.task_switch_count) or []
    for k in range(1, 5):
        sched._after_transition(f"t{k}", "DEFER", "test")
    assert hits == expected, f"N={n} 应在切换 {expected} 触发，实际 {hits}"
    assert sched.state.full_rescan_count == len(expected)


# ------------------------------------------------------------------ 3) 关闭期间计数器仍推进
def test_counters_advance_while_disabled_then_enabled_resumes_schedule():
    sched = _sched(n=3, rescan=False)
    sched._rescan = lambda *a, **k: []
    calls = []
    sched._rescan = lambda *a, **k: calls.append(sched.state.task_switch_count) or []
    for _ in range(2):
        sched._after_transition("t0", "DEFER", "test")
    assert sched.state.task_switch_count == 2
    assert calls == []
    sched.al.rescan_candidates_after_transition = True     # 重新打开
    sched._after_transition("t1", "DEFER", "test")         # 第 3 次 ⇒ 触发
    assert calls == [3]


# ------------------------------------------------------------------ 4) CV 边界
def _cv(vals):
    return statistics.stdev(vals) / statistics.mean(vals)


def test_warn_cv_exactly_at_limit_is_not_flagged(tmp_path):
    vals = [0.001] * 9 + [0.05]
    cv = _cv(vals)
    table, diag = build_thresholds({"t0": vals, "t1": [0.002] * 10}, _store(tmp_path),
                                   multiplier=10, max_cv=cv, high_cv_policy="warn",
                                   reference="r")
    assert diag["t0"]["high_cv_warning"] is False          # 严格 `>` ⇒ 相等不算超标
    assert diag["t0"]["status"] in ("usable", "baseline_capped")
    assert table.tasks["t0"] is not None


def test_warn_cv_just_above_limit_is_flagged_but_numeric(tmp_path):
    vals = [0.001] * 9 + [0.05]
    cv = _cv(vals)
    table, diag = build_thresholds({"t0": vals, "t1": [0.002] * 10}, _store(tmp_path),
                                   multiplier=10, max_cv=cv * 0.999,
                                   high_cv_policy="warn", reference="r")
    assert diag["t0"]["high_cv_warning"] is True
    assert diag["t0"]["status"] == "high_cv_warning"
    assert table.tasks["t0"] is not None and table.tasks["t0"] > 0


# ------------------------------------------------------------------ 5) warn + cap 同时命中
def test_warn_and_baseline_cap_flags_are_independent(tmp_path):
    vals = [1e-4] * 9 + [1.0]                                # CV≈3（超标）；gmean×1000 > 0.99×baseline
    table, diag = build_thresholds({"t0": vals, "t1": [0.002] * 10},
                                   _store(tmp_path, mse=0.1),
                                   multiplier=1000.0, baseline_cap=0.99,
                                   max_cv=1.5, high_cv_policy="warn", reference="r")
    d = diag["t0"]
    assert d["high_cv_warning"] is True and d["baseline_capped"] is True
    assert d["status"] == "high_cv_warning"                  # status 归高 CV；cap 用独立标志
    assert table.tasks["t0"] == pytest.approx(0.99 * 0.1)


# ------------------------------------------------------------------ 6) 判定层：不再被排除
def test_warn_table_is_judgeable_at_check_pass_layer(tmp_path):
    vals = [1e-5] * 9 + [0.2]                                # CV≈3 → 旧策略会写 null
    store = _store(tmp_path)
    warn_tbl, _ = build_thresholds({"t0": vals, "t1": [0.002] * 10}, store,
                                   multiplier=220, max_cv=1.5,
                                   high_cv_policy="warn", reference="r")
    null_tbl, _ = build_thresholds({"t0": vals, "t1": [0.002] * 10}, store,
                                   multiplier=220, max_cv=1.5,
                                   high_cv_policy="null", reference="r")
    assert null_tbl.tasks["t0"] is None and warn_tbl.tasks["t0"] is not None

    def cfg_with(tbl):
        return SimpleNamespace(pass_metric="mse", pass_nmse=0.35, pass_thresholds=tbl)

    line = warn_tbl.tasks["t0"]
    assert check_pass(cfg_with(warn_tbl), "t0", nmse=0.0, mse=line * 0.5) == PassCheck.PASS
    assert check_pass(cfg_with(warn_tbl), "t0", nmse=0.0, mse=line * 2) == PassCheck.BELOW
    # 旧 null 表：同一任务在被跳过状态（NO_THRESHOLD ⇒ 不判 PASS、不耗 attempt）
    assert check_pass(cfg_with(null_tbl), "t0", nmse=0.0, mse=0.0) == PassCheck.NO_THRESHOLD


# ------------------------------------------------------------------ 7) CLI 边界
def test_cli_rejects_unknown_high_cv_policy(tmp_path):
    ref = tmp_path / "r.jsonl"
    ref.write_text("\n".join(json.dumps({"task": t, "traj": i, "mse": 0.001})
                             for t in ("t0", "t1") for i in range(10)), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--ref-per-traj", str(ref), "--baseline", str(_store(tmp_path).path),
              "--reference", "r", "--multiplier", "10",
              "--high-cv-policy", "bogus", "-o", str(tmp_path / "o.json")])


def test_warn_table_reloads_and_reports_no_null(tmp_path):
    vals = [1e-5] * 9 + [0.2]
    store = _store(tmp_path)
    out = tmp_path / "warn.json"
    ref = tmp_path / "r.jsonl"
    ref.write_text("\n".join(json.dumps({"task": t, "traj": i, "mse": v})
                             for t, vv in (("t0", vals), ("t1", [0.002] * 10))
                             for i, v in enumerate(vv)), encoding="utf-8")
    assert main(["--ref-per-traj", str(ref), "--baseline", str(store.path),
                 "--reference", "r", "--multiplier", "220",
                 "--high-cv-policy", "warn", "-o", str(out)]) == 0
    raw = json.loads(out.read_text(encoding="utf-8"))
    tbl = PassThresholds.load(str(out), expect_fingerprint="testfp", require_metric="mse")
    assert tbl.n_usable == len(tbl.tasks) == 2
    assert all(v is not None for v in tbl.tasks.values())
    assert raw["calibration"]["high_cv_policy"] == "warn"
    assert raw["calibration"]["tasks"]["t0"]["high_cv_warning"] is True
