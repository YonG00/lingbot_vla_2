"""pass_metric="mse" + 按任务阈值表（PassThresholds）的测试。

重点（用户点名）：**阈值表与任务的匹配** —— 任务名不匹配 / 覆盖不全 /
nmse↔mse 不自洽 / 阈值比 baseline 还松，全部必须 fail-fast，绝不静默产出错表。
"""

from __future__ import annotations

import json
import math
import os
import warnings

import pytest

from lingbotvla.auto_learning.baseline import BaselineStore, FixedBaseline, task_fingerprint
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.decision.metrics import forget_ratio
from lingbotvla.auto_learning.decision.thresholds import (
    PASS_METRIC_MSE,
    PASS_METRIC_NMSE,
    PassCheck,
    PassThresholds,
    ThresholdsError,
    check_pass,
    forget_code_ex,
    is_forgotten_ex,
    is_pass,
    pass_line,
)
from lingbotvla.auto_learning.tools.compute_pass_thresholds import (
    _percentile,
    _stat,
    compute_thresholds,
    main as thresholds_main,
)
from al_fixtures import local_tmpdir, make_cfg, scheduler_of

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _al(**kw) -> AutoLearningConfig:
    base = dict(seed=3, pass_nmse=0.30, pass_thresholds_file=None)
    base.update(kw)
    return AutoLearningConfig(**base)


def _attach(cfg: AutoLearningConfig, thresholds: PassThresholds) -> AutoLearningConfig:
    cfg.pass_thresholds = thresholds
    return cfg


def _make_store(tmp: str, tasks: dict, fingerprint: str = "FP-1") -> BaselineStore:
    store = BaselineStore(path=os.path.join(tmp, "task_baseline.json"),
                          config_fingerprint=fingerprint)
    for name, mse in tasks.items():
        store.put(FixedBaseline(
            task=name, mse=mse, mu=(0.0,), fingerprint=task_fingerprint(fingerprint, "sha"),
        ))
    store.save()
    return BaselineStore.load(store.path)


def _eval_rows(rows):
    """rows: {task: [(mse, nmse), ...]} → {task: [{mse,nmse}, ...]}"""
    return {t: [{"mse": m, "nmse": n} for m, n in vals] for t, vals in rows.items()}


# --------------------------------------------------------------------------- #
# 1. PassThresholds 数据结构 / 加载 / 指纹校验
# --------------------------------------------------------------------------- #
class TestPassThresholdsIO:
    def test_roundtrip(self):
        with local_tmpdir() as tmp:
            t = PassThresholds(config_fingerprint="FP", tasks={"a": 0.1, "b": None},
                               metric=PASS_METRIC_MSE, margin=0.1, stat="p75")
            p = os.path.join(tmp, "th.json")
            t.save(p)
            t2 = PassThresholds.load(p, require_metric=PASS_METRIC_MSE)
            assert t2.config_fingerprint == "FP"
            assert t2.tasks == {"a": 0.1, "b": None}
            assert t2.metric == PASS_METRIC_MSE
            assert t2.margin == 0.1 and t2.stat == "p75"

    def test_fingerprint_mismatch_fails(self):
        with local_tmpdir() as tmp:
            t = PassThresholds(config_fingerprint="OLD", tasks={"a": 0.1})
            p = os.path.join(tmp, "th.json")
            t.save(p)
            with pytest.raises(ThresholdsError):
                PassThresholds.load(p, expect_fingerprint="NEW")
            # 显式放行 ⇒ 警告不报错
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                PassThresholds.load(p, expect_fingerprint="NEW",
                                    allow_fingerprint_mismatch=True)
                assert any("fingerprint" in str(x.message) for x in w)

    def test_missing_fingerprint_rejected_when_expected(self):
        with local_tmpdir() as tmp:
            p = os.path.join(tmp, "th.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"tasks": {"a": 0.1}, "metric": "mse"}, f)
            with pytest.raises(ThresholdsError):
                PassThresholds.load(p, expect_fingerprint="X")

    def test_require_metric(self):
        with local_tmpdir() as tmp:
            t = PassThresholds(config_fingerprint="FP", tasks={"a": 0.1},
                               metric=PASS_METRIC_NMSE)
            p = os.path.join(tmp, "th.json")
            t.save(p)
            with pytest.raises(ThresholdsError):
                PassThresholds.load(p, require_metric=PASS_METRIC_MSE)

    def test_bad_value_type_rejected(self):
        with pytest.raises(ThresholdsError):
            PassThresholds.from_dict({"tasks": {"a": "not-a-number"}})

    def test_detailed_form_accepted(self):
        t = PassThresholds.from_dict({"tasks": {"a": {"pass_threshold": 0.2}}})
        assert t.tasks["a"] == 0.2


# --------------------------------------------------------------------------- #
# 2. 统一判定入口
# --------------------------------------------------------------------------- #
class TestDecisionEntry:
    def test_nmse_mode_matches_legacy(self):
        cfg = _al()  # 默认 nmse
        assert pass_line(cfg, "any") == (PASS_METRIC_NMSE, 0.30)
        assert is_pass(cfg, "any", nmse=0.25, mse=999.0)
        assert not is_pass(cfg, "any", nmse=0.35, mse=0.0)
        # NaN / None 绝不放行
        assert check_pass(cfg, "any", nmse=float("nan"), mse=0.0) == PassCheck.INVALID
        assert check_pass(cfg, "any", nmse=None, mse=None) == PassCheck.INVALID

    def test_mse_mode_uses_table(self):
        cfg = _al(pass_metric="mse", pass_thresholds_file="x.json")
        _attach(cfg, PassThresholds(tasks={"a": 0.10, "b": None}))
        assert pass_line(cfg, "a") == (PASS_METRIC_MSE, 0.10)
        # 有表但任务不在表里 ⇒ 无可用线
        assert pass_line(cfg, "zzz") == (PASS_METRIC_MSE, None)
        assert check_pass(cfg, "zzz", nmse=0.0, mse=0.0) == PassCheck.NO_THRESHOLD
        # 表里显式 null ⇒ needs_calibration，不判 PASS（哪怕 mse 极小）
        assert check_pass(cfg, "b", nmse=0.0, mse=1e-9) == PassCheck.NO_THRESHOLD
        # 判定用的是 mse，不是 nmse
        assert is_pass(cfg, "a", nmse=99.0, mse=0.05)
        assert not is_pass(cfg, "a", nmse=0.001, mse=0.50)

    def test_mse_mode_without_table_never_passes(self):
        cfg = _al(pass_metric="mse", pass_thresholds_file="x.json")
        assert check_pass(cfg, "a", nmse=0.0, mse=0.0) == PassCheck.NO_THRESHOLD

    def test_forget_relative_half_is_scale_invariant(self):
        """相对退化那半是 (cur−best)/best ⇒ 换 mse 口径数值不变。"""
        cfg = _al(pass_metric="mse", pass_thresholds_file="x.json")
        _attach(cfg, PassThresholds(tasks={"t": 0.5}))  # 线很松，不触发掉线那半
        base, cur = 0.2, 0.4  # nmse 口径退化 100%
        assert is_forgotten_ex(cfg, "t", cur_nmse=cur, cur_mse=cur * 0.25,
                               best_nmse=base)
        # mse 口径同样退化 100%（baseline_mse=0.25）
        assert forget_ratio(base, cur) == pytest.approx(
            forget_ratio(base * 0.25, cur * 0.25), abs=1e-9)

    def test_forget_below_line_uses_mse(self):
        cfg = _al(pass_metric="mse", pass_thresholds_file="x.json")
        _attach(cfg, PassThresholds(tasks={"t": 0.10}))
        assert is_forgotten_ex(cfg, "t", cur_nmse=0.001, cur_mse=0.50, best_nmse=0.001)
        code = forget_code_ex(cfg, "t", cur_nmse=0.001, cur_mse=0.50)
        assert code == "forgotten_below_pass_line"

    def test_forget_no_threshold_still_relative(self):
        cfg = _al(pass_metric="mse", pass_thresholds_file="x.json")
        _attach(cfg, PassThresholds(tasks={"t": None}))
        # 没有可用线 ⇒ 掉线那半不参与；大幅相对退化仍判遗忘
        assert is_forgotten_ex(cfg, "t", cur_nmse=0.6, cur_mse=0.6, best_nmse=0.2)
        assert not is_forgotten_ex(cfg, "t", cur_nmse=0.21, cur_mse=0.21, best_nmse=0.2)


# --------------------------------------------------------------------------- #
# 3. ⭐ 阈值表 ↔ 任务匹配（标定工具的一致性检查）
# --------------------------------------------------------------------------- #
class TestThresholdMatching:
    def _store(self, tmp):
        # baseline_mse: a=0.25, b=0.25
        return _make_store(tmp, {"a": 0.25, "b": 0.25})

    def test_stat_math(self):
        vals = [0.10, 0.20, 0.30, 0.40]
        assert _stat(vals, "max") == 0.40
        assert _stat(vals, "median") == 0.25
        assert _stat(vals, "mean") == 0.25
        assert _percentile(sorted(vals), 75) == pytest.approx(0.325)  # 0.30~0.40 线性插值

    def test_happy_path(self):
        with local_tmpdir() as tmp:
            store = self._store(tmp)
            ev = _eval_rows({"a": [(0.08, 0.32)], "b": [(0.05, 0.20)]})
            th = compute_thresholds(ev, store,
                                    stat="max", margin=0.25, reference="ref")
            assert th.tasks["a"] == pytest.approx(0.10)  # 0.08*1.25
            assert th.tasks["b"] == pytest.approx(0.0625)
            assert th.config_fingerprint == "FP-1"
            assert th.metric == PASS_METRIC_MSE

    def test_unknown_eval_task_rejected(self):
        with local_tmpdir() as tmp:
            store = self._store(tmp)
            ev = _eval_rows({"a": [(0.08, 0.32)], "ghost": [(0.01, 0.01)]})
            with pytest.raises(ThresholdsError, match="ghost"):
                compute_thresholds(ev, store, stat="max", margin=0.1,
                                   reference="")

    def test_missing_eval_task_rejected(self):
        """baseline 里有的任务没有 eval ⇒ 覆盖不全，必须拒绝（否则训练侧 fail-fast）。"""
        with local_tmpdir() as tmp:
            store = self._store(tmp)
            ev = _eval_rows({"a": [(0.08, 0.32)]})  # 缺 b
            with pytest.raises(ThresholdsError, match="b"):
                compute_thresholds(ev, store, stat="max", margin=0.1,
                                   reference="")

    def test_nmse_mse_inconsistent_rejected(self):
        """nmse 与 mse/baseline 对不上 ⇒ 两份结果不是同一配置算的。"""
        with local_tmpdir() as tmp:
            store = self._store(tmp)
            ev = _eval_rows({"a": [(0.08, 0.99)], "b": [(0.05, 0.20)]})
            with pytest.raises(ThresholdsError, match="偏差"):
                compute_thresholds(ev, store,
                                   stat="max", margin=0.1, reference="")

    def test_threshold_looser_than_baseline_rejected(self):
        with local_tmpdir() as tmp:
            store = self._store(tmp)
            ev = _eval_rows({"a": [(0.24, 0.96)], "b": [(0.05, 0.20)]})
            with pytest.raises(ThresholdsError, match="baseline"):
                compute_thresholds(ev, store,
                                   stat="max", margin=0.10, reference="")

    def test_every_task_gets_numeric_line(self):
        """每个任务都有开环分数 ⇒ 全部产出数值通过线，不产生 null。"""
        with local_tmpdir() as tmp:
            store = self._store(tmp)
            ev = _eval_rows({"a": [(0.08, 0.32)], "b": [(0.05, 0.20)]})
            th = compute_thresholds(ev, store, stat="max", margin=0.1, reference="")
            assert th.tasks["a"] is not None
            assert th.tasks["b"] is not None
            assert th.n_usable == 2

    def test_nonfinite_mse_rejected(self):
        with local_tmpdir() as tmp:
            store = self._store(tmp)
            ev = _eval_rows({"a": [(float("nan"), None)], "b": [(0.05, 0.20)]})
            with pytest.raises(ThresholdsError):
                compute_thresholds(ev, store, stat="max", margin=0.1, reference="")

    def test_main_end_to_end(self):
        with local_tmpdir() as tmp:
            store = self._store(tmp)
            eval_path = os.path.join(tmp, "eval.jsonl")
            with open(eval_path, "w", encoding="utf-8") as f:
                for mse, nmse, task in [(0.08, 0.32, "a"), (0.05, 0.20, "b")]:
                    f.write(json.dumps({"task": task, "mse": mse, "nmse": nmse}) + "\n")
            out = os.path.join(tmp, "pass_thresholds.json")
            rc = thresholds_main([
                "--eval-jsonl", eval_path, "--baseline", store.path,
                "--stat", "max", "--margin", "0.25",
                "--reference", "ref-ckpt", "-o", out,
            ])
            assert rc == 0
            th = PassThresholds.load(out, expect_fingerprint="FP-1",
                                     require_metric=PASS_METRIC_MSE)
            # a: 0.08×1.25=0.10, b: 0.05×1.25=0.0625 —— 都有数值线
            assert th.tasks == {"a": pytest.approx(0.10), "b": pytest.approx(0.0625)}


# --------------------------------------------------------------------------- #
# 4. Scheduler 端到端（mse 口径）
# --------------------------------------------------------------------------- #
class TestSchedulerMseMode:
    def test_pass_uses_per_task_mse_line(self):
        """t0 线松 ⇒ PASS；t1 线紧到 floor 都达不到 ⇒ 不 PASS。
        baseline_mse=0.25、nmse floor=0.20 ⇒ mse floor=0.05。"""
        al = AutoLearningConfig(
            seed=3, eval_interval_steps=50, min_steps_before_defer=100,
            pass_nmse=0.30, pass_metric="mse", pass_thresholds_file="fake.json",
            min_lp50=0.0, max_attempts_per_task=3, batch_size=10,
            new_slots=7, replay_slots=3,
        )
        cfg = make_cfg(2, al=al)
        cfg.auto_learning.pass_thresholds = PassThresholds(
            config_fingerprint="", tasks={"t0": 0.08, "t1": 0.03})
        sched = scheduler_of(cfg)
        sched.run(max_actions=600)
        by_name = {r.task_name: r for r in sched.registry}
        assert by_name["t0"].status == "PASS", by_name["t0"].to_state()
        assert by_name["t1"].status != "PASS"
        # mse 字段真的有被写
        assert by_name["t0"].current_val_mse is not None
        assert by_name["t0"].best_mse is not None
        # 跨任务统计仍走 nmse（口径不分叉）
        assert by_name["t0"].current_val_nmse is not None

    def test_nmse_mode_unchanged(self):
        al = AutoLearningConfig(
            seed=3, eval_interval_steps=50, min_steps_before_defer=100,
            pass_nmse=0.30, min_lp50=0.0, max_attempts_per_task=3,
            batch_size=10, new_slots=7, replay_slots=3,
        )
        cfg = make_cfg(2, al=al)
        sched = scheduler_of(cfg)
        sched.run(max_actions=600)
        assert any(r.status == "PASS" for r in sched.registry)

    def test_no_threshold_task_never_trained(self):
        """null 阈值任务（needs_calibration）必须：不进候选池、不消耗 attempt 预算。"""
        al = AutoLearningConfig(
            seed=3, eval_interval_steps=50, min_steps_before_defer=100,
            pass_nmse=0.30, pass_metric="mse", pass_thresholds_file="fake.json",
            min_lp50=0.0, max_attempts_per_task=3, batch_size=10,
            new_slots=7, replay_slots=3,
        )
        cfg = make_cfg(2, al=al)
        cfg.auto_learning.pass_thresholds = PassThresholds(
            config_fingerprint="", tasks={"t0": 0.08, "t1": None})
        sched = scheduler_of(cfg)
        sched.run(max_actions=600)
        by_name = {r.task_name: r for r in sched.registry}
        # t0 有可用线 ⇒ 正常训练并 PASS
        assert by_name["t0"].status == "PASS", by_name["t0"].to_state()
        # t1 是 null ⇒ needs_calibration，绝不训练、绝不消耗 attempt
        assert by_name["t1"].pass_line_usable is False
        assert by_name["t1"].attempt_count == 0, by_name["t1"].to_state()
        assert by_name["t1"].status == "CANDIDATE"  # 一直挂着、没被选中训练
        # 候选池确实排除了它
        assert by_name["t1"].task_name not in [
            r.task_name for r in sched.registry.candidate_records()
        ]
        # 调度器能正常收工（不会因 null 任务卡死）
        assert sched.state.finished


# --------------------------------------------------------------------------- #
# 5. 守卫：全仓不允许再出现裸的 pass_nmse 比较（防口径分叉）
# --------------------------------------------------------------------------- #
class TestNoBarePassNmse:
    #: 白名单（文件, 函数）粒度：遗留纯函数已被统一入口取代、仅兼容/测试用；
    #: 阈值模块本身是判定入口，自由。
    WHITELIST = {
        ("lingbotvla/auto_learning/decision/thresholds.py", None),
        ("lingbotvla/auto_learning/decision/metrics.py", "is_forgotten"),
        ("lingbotvla/auto_learning/decision/state_machine.py", "forget_code"),
    }

    def _iter_compares(self):
        import ast
        base = os.path.join(REPO_ROOT, "lingbotvla", "auto_learning")
        for root, _dirs, files in os.walk(base):
            if "__pycache__" in root:
                continue
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(root, fn)
                rel = os.path.relpath(path, REPO_ROOT)
                tree = ast.parse(open(path, encoding="utf-8").read(), filename=rel)
                parents = {}
                for node in ast.walk(tree):
                    for child in ast.iter_child_nodes(node):
                        parents[child] = node

                def enclosing_func(n):
                    while n in parents:
                        n = parents[n]
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            return n.name
                    return None

                for node in ast.walk(tree):
                    if not isinstance(node, ast.Compare):
                        continue
                    for expr in [node.left, *node.comparators]:
                        kind = None
                        if isinstance(expr, ast.Attribute) and expr.attr == "pass_nmse":
                            kind = "attr"
                        elif isinstance(expr, ast.Name) and expr.id == "pass_nmse":
                            kind = "name"
                        if kind:
                            yield rel, node.lineno, enclosing_func(node), kind

    def test_no_bare_pass_nmse_comparison(self):
        offenders = []
        for rel, lineno, func, kind in self._iter_compares():
            allowed = (rel, None) in self.WHITELIST or (rel, func) in self.WHITELIST
            if not allowed:
                offenders.append(f"{rel}:{lineno} 函数={func} ({kind})")
        assert not offenders, (
            "发现裸的 pass_nmse 比较（判定口径会分叉，必须走 "
            "decision.thresholds）：\n  " + "\n  ".join(offenders))
