"""随机性验收（方案 B）的 CPU 测试 + **反例测试**。

要求（用户 2026-10-09）：
- 不能只看 `0.021 < 0.031` 就判安全 ⇒ 本文件的正例都在 **reps≥5** 下给出**分布级**结论；
- 样本量不足 / 统计不明确 ⇒ **BLOCKED**；
- 必须能识别**系统性偏差**（不是"抖动大一点"，而是"整组偏移"）；
- 必须同时看**指标层**（GMean / PASS 判定稳定性），不能只比一个 max_abs_diff。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingbotvla.auto_learning import stochastic_parity as sp  # noqa: E402


#: B1/B2 共用的"真值"（真实实验里两组喂的是同一份输入 ⇒ 必须共用，否则报系统性偏差是对的）
_BASE = np.random.default_rng(999).standard_normal((50, 14)) * 0.5


def _noise_runs(reps, shape=(50, 14), sigma=3e-3, seed=0, base=None, bias=0.0):
    """模拟"同代码重复跑"：各次围绕同一真值 + 独立小抖动（可注入偏移）。"""
    rng = np.random.default_rng(seed)
    base = _BASE if base is None else base
    return [base + rng.standard_normal(shape) * sigma + bias for _ in range(reps)]


# ---------------------------------------------------------------------------
# 数值层：正例
# ---------------------------------------------------------------------------
def test_pure_noise_is_accepted_with_enough_reps():
    b1 = _noise_runs(6, seed=1)
    b2 = _noise_runs(6, seed=2)      # 同分布、无系统性偏差
    verdict = sp.decide_numeric(b1_runs=b1, b2_runs=b2, reps=6)
    assert verdict["status"] == "PASS", verdict["reasons"]
    assert verdict["cross_batch1_batch2"]["p99"] > 0          # 确实有抖动，不是"全 0 假过"
    assert verdict["bias_reference"]["exceeds_reference"] is False   # 纯噪声不得被判系统偏差
    assert verdict["bias_reference"]["n_ref"] > 0


def test_stats_report_max_mean_p95_p99():
    a = np.zeros((2, 2))
    b = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    stats = sp.diff_stats(a, b)
    assert stats["max"] == 3.0 and stats["mean"] == pytest.approx(1.5)
    assert stats["p95"] >= 2.0 and stats["p99"] >= 2.0


# ---------------------------------------------------------------------------
# 数值层：反例（必须 BLOCKED）
# ---------------------------------------------------------------------------
def test_systematic_offset_is_rejected():
    """B2 整组偏移 1e-2（比抖动大得多）⇒ 必须判系统性偏差。"""
    b1 = _noise_runs(6, sigma=1e-3, seed=3)
    b2 = [r + 1e-2 for r in _noise_runs(6, sigma=1e-3, seed=4)]
    verdict = sp.decide_numeric(b1_runs=b1, b2_runs=b2, reps=6)
    assert verdict["status"] == "BLOCKED"
    assert any(r.startswith("systematic_bias") for r in verdict["reasons"]), verdict["reasons"]
    assert verdict["bias_reference"]["obs_frac"] > verdict["bias_reference"]["ref_max"]


def test_larger_batch_variance_is_rejected():
    """B2 抖动明显更大（10×）⇒ 判 "batch2 波动放大" ⇒ BLOCKED。

    ⚠️ 注意：这种情况**不能**只靠 "cross vs within" 判——B2 自身方差占优时，B1↔B2 的差
    看起来就和 B2↔B2 的差一样大。所以要单独比 **组内**：within(B2) vs within(B1)。
    """
    b1 = _noise_runs(6, sigma=1e-3, seed=5)
    b2 = _noise_runs(6, sigma=1e-2, seed=6)
    verdict = sp.decide_numeric(b1_runs=b1, b2_runs=b2, reps=6)
    assert verdict["status"] == "BLOCKED"
    assert any(r.startswith("batch2_variance_exceeds_batch1") for r in verdict["reasons"]), \
        verdict["reasons"]


def test_too_few_reps_is_blocked():
    b1 = _noise_runs(4, seed=7)
    b2 = _noise_runs(4, seed=8)
    verdict = sp.decide_numeric(b1_runs=b1, b2_runs=b2, reps=4)
    assert verdict["status"] == "BLOCKED"
    assert any(r.startswith("reps_below_min") for r in verdict["reasons"])


def test_nonfinite_output_is_blocked():
    b1 = _noise_runs(5, seed=9)
    b2 = _noise_runs(5, seed=10)
    b2[2] = b2[2].copy()
    b2[2][0, 0] = np.nan
    verdict = sp.decide_numeric(b1_runs=b1, b2_runs=b2, reps=5)
    assert verdict["status"] == "BLOCKED"
    assert "nonfinite_output" in verdict["reasons"]


def test_rep_count_mismatch_is_blocked():
    verdict = sp.decide_numeric(b1_runs=_noise_runs(5, seed=11),
                               b2_runs=_noise_runs(6, seed=12), reps=5)
    assert verdict["status"] == "BLOCKED"
    assert "rep_count_mismatch" in verdict["reasons"]


# ---------------------------------------------------------------------------
# 指标层（GMean / PASS 稳定性）
# ---------------------------------------------------------------------------
def test_metric_stable_and_far_from_threshold_passes():
    verdict = sp.decide_metric(b1_gmeans=[0.0040, 0.0041, 0.0039, 0.0040, 0.0042],
                               b2_gmeans=[0.0041, 0.0040, 0.0042, 0.0039, 0.0041],
                               threshold=0.01)
    assert verdict["status"] == "PASS", verdict["reasons"]
    assert verdict["batch1"]["pass"] == [True] * 5
    assert verdict["batch2"]["margin"][0] > 2.0          # 离阈值有余量


def test_metric_flip_within_mode_is_blocked():
    verdict = sp.decide_metric(b1_gmeans=[0.009, 0.011, 0.0095, 0.010, 0.0099],
                               b2_gmeans=[0.0095, 0.0096, 0.0097, 0.0098, 0.0099],
                               threshold=0.01)
    assert verdict["status"] == "BLOCKED"
    assert any(r.startswith("batch1_decision_flips") for r in verdict["reasons"])


def test_metric_mode_mismatch_is_blocked():
    """B1 全过、B2 全不过 ⇒ 模式改变了判定 ⇒ BLOCKED（正是用户要防的风险）。"""
    verdict = sp.decide_metric(b1_gmeans=[0.005] * 5, b2_gmeans=[0.02] * 5, threshold=0.01)
    assert verdict["status"] == "BLOCKED"
    assert any(r.startswith("mode_decision_mismatch") for r in verdict["reasons"])


def test_metric_knife_edge_is_blocked():
    """贴线通过（抖动幅度 ≥ 到阈值的距离）⇒ 判定不可靠 ⇒ BLOCKED。"""
    verdict = sp.decide_metric(b1_gmeans=[0.0090, 0.0099, 0.0095, 0.0098, 0.0092],
                               b2_gmeans=[0.0091, 0.0097, 0.0094, 0.0099, 0.0093],
                               threshold=0.01)
    assert verdict["status"] == "BLOCKED"
    assert any(r.startswith("knife_edge") for r in verdict["reasons"])


def test_metric_missing_gmean_is_blocked():
    verdict = sp.decide_metric(b1_gmeans=[0.005, None, 0.005, 0.005, 0.005],
                               b2_gmeans=[0.005] * 5, threshold=0.01)
    assert verdict["status"] == "BLOCKED"
    assert "gmean_missing_or_nonfinite" in verdict["reasons"]

    verdict2 = sp.decide_metric(b1_gmeans=[math.inf] * 5, b2_gmeans=[0.005] * 5, threshold=0.01)
    assert verdict2["status"] == "BLOCKED"


def test_decision_margin_definition():
    assert sp.decision_margin(0.005, 0.01) == pytest.approx(2.0)
    assert sp.decision_margin(0.0, 0.01) == float("inf")
    assert sp.decision_margin(None, 0.01) is None
    assert sp.decision_margin(0.005, None) is None


# ---------------------------------------------------------------------------
# 总判定
# ---------------------------------------------------------------------------
def _numeric_pass():
    return sp.decide_numeric(b1_runs=_noise_runs(6, seed=21), b2_runs=_noise_runs(6, seed=22),
                             reps=6)


def _metric_pass():
    return sp.decide_metric(b1_gmeans=[0.004] * 5, b2_gmeans=[0.0041] * 5, threshold=0.01)


def test_overall_requires_metric_level():
    """用户要求 6：不能只比 max_abs_diff ⇒ 没跑指标层必须 BLOCKED。"""
    verdict = sp.overall_verdict(numeric=_numeric_pass(), metric=None)
    assert verdict["status"] == "BLOCKED"
    assert "metric_level_not_run" in verdict["reasons"]


def test_overall_pass_requires_both_and_reports_speedup():
    verdict = sp.overall_verdict(numeric=_numeric_pass(), metric=_metric_pass(),
                                 throughput={"batch1_seconds_per_sample": 0.5,
                                             "batch2_seconds_per_sample": 0.3})
    assert verdict["status"] == "PASS"
    assert verdict["speedup_per_sample"] == pytest.approx(0.5 / 0.3, rel=1e-6)
    assert verdict["worth_proposing_auto"] is True


def test_overall_pass_but_no_speedup_does_not_propose_auto():
    verdict = sp.overall_verdict(numeric=_numeric_pass(), metric=_metric_pass(),
                                 throughput={"batch1_seconds_per_sample": 0.3,
                                             "batch2_seconds_per_sample": 0.5})
    assert verdict["status"] == "PASS"
    assert verdict["worth_proposing_auto"] is False


def test_overall_blocks_when_numeric_fails():
    bad = sp.decide_numeric(b1_runs=_noise_runs(5, sigma=1e-3, seed=31),
                            b2_runs=[r + 1e-2 for r in _noise_runs(5, sigma=1e-3, seed=32)],
                            reps=5)
    verdict = sp.overall_verdict(numeric=bad, metric=_metric_pass())
    assert verdict["status"] == "BLOCKED"
    assert any(r.startswith("numeric:systematic_bias") for r in verdict["reasons"])


# ---------------------------------------------------------------------------
# GPU 验收入口：PLAN ONLY / 阈值读取（都不碰 GPU）
# ---------------------------------------------------------------------------
def _tool():
    from tools import eval_batch_stochastic_acceptance as tool
    return tool


def test_threshold_loader_reads_only_gmean_mse_tables(tmp_path):
    tool = _tool()
    good = tmp_path / "ok.json"
    good.write_text('{"metric":"mse","stat":"geomean","tasks":{"click_bell":0.0235}}',
                    encoding="utf-8")
    assert tool.load_threshold(str(good), "click_bell") == pytest.approx(0.0235)
    assert tool.load_threshold(str(good), "no_such_task") is None      # 缺任务 ⇒ None（fail-closed）

    for bad_doc in ('{"metric":"nmse","stat":"geomean","tasks":{"click_bell":0.1}}',
                    '{"metric":"mse","stat":"mean","tasks":{"click_bell":0.1}}',
                    '{"metric":"mse","stat":"geomean","tasks":{"click_bell":null}}',
                    '{"metric":"mse","stat":"geomean","tasks":{"click_bell":-1}}',
                    'not json'):
        bad = tmp_path / "bad.json"
        bad.write_text(bad_doc, encoding="utf-8")
        assert tool.load_threshold(str(bad), "click_bell") is None, bad_doc


def test_plan_only_is_side_effect_free(tmp_path, capsys):
    tool = _tool()
    out = tmp_path / "stoch_out"
    rc = tool.main(["--out-dir", str(out), "--ckpt", str(tmp_path / "nope"),
                    "--val-ids", str(tmp_path / "nope.json"), "--dataset", str(tmp_path / "nope"),
                    "--thresholds", str(tmp_path / "nope.json")])
    assert rc == 2                       # 路径不齐 ⇒ 明确阻塞
    assert not out.exists()              # 且**不创建任何目录**
    text = capsys.readouterr().out
    assert "PLAN ONLY" in text and "R6" in text and "BLOCKED" in text


def test_plan_only_requires_min_reps(tmp_path, capsys):
    tool = _tool()
    th = tmp_path / "th.json"
    th.write_text('{"metric":"mse","stat":"geomean","tasks":{"t":0.01}}', encoding="utf-8")
    rc = tool.main(["--out-dir", str(tmp_path / "o"), "--task", "t", "--thresholds", str(th),
                    "--reps", "3"])
    assert rc == 2
    assert not (tmp_path / "o").exists()
    assert "reps" in capsys.readouterr().out


def test_execute_refuses_non_empty_out_dir(tmp_path):
    tool = _tool()
    out = tmp_path / "busy"
    out.mkdir()
    (out / "x").write_text("y", encoding="utf-8")
    assert tool.main(["--out-dir", str(out), "--execute"]) == 2


def test_plan_documents_warmup_before_counting(tmp_path, capsys):
    """热身必须先于计数：实测第一条前向与后续腿的 max|Δ| 达 0.177（会污染统计）。"""
    tool = _tool()
    th = tmp_path / "th.json"
    th.write_text('{"metric":"mse","stat":"geomean","tasks":{"t":0.01}}', encoding="utf-8")
    tool.main(["--out-dir", str(tmp_path / "o"), "--task", "t", "--thresholds", str(th)])
    text = capsys.readouterr().out
    assert "热身" in text and "不计入统计" in text


def test_tool_runs_a_discarded_warmup_before_measured_reps():
    """热身前向必须存在且不计入统计（run g 实测：首条腿与后续差 0.17，是 autotune/惰性初始化）。"""
    src = (REPO / "tools/eval_batch_stochastic_acceptance.py").read_text(encoding="utf-8")
    assert "热身" in src and "不计入统计" in src
    assert "warmup_seconds" in src


# ---------------------------------------------------------------------------
# 随机性门（gate）：签名绑定 + fail-closed
# ---------------------------------------------------------------------------
def _payload(**over):
    base = dict(checkpoint="/x/hf_ckpt", task="click_bell", batch_size=2,
                dtype="torch.bfloat16",
                shapes=[{"images": [1, 3, 256, 1536], "state": [1, 55]}],
                grids=["tensor([[1,16,16]])"])
    base.update(over)
    return sp.gate_payload(**base)


def test_gate_roundtrip_pass(tmp_path):
    gate = tmp_path / "gate.json"
    payload = _payload()
    sp.write_gate(str(gate), payload=payload, verdict={"status": "PASS", "reasons": []})
    res = sp.load_gate(str(gate), payload=payload)
    assert res["ok"] is True and res["reasons"] == []


def test_gate_is_fail_closed_on_every_mismatch(tmp_path):
    gate = tmp_path / "gate.json"
    payload = _payload()
    sp.write_gate(str(gate), payload=payload, verdict={"status": "PASS", "reasons": []})

    assert sp.load_gate(None, payload=payload)["reasons"] == ["gate_not_configured"]
    assert sp.load_gate(str(tmp_path / "nope.json"), payload=payload)["reasons"][0].startswith(
        "gate_file_missing")
    # 任一运行条件不同 ⇒ 签名不符 ⇒ 不通过
    for over in ({"checkpoint": "/y/hf_ckpt"}, {"task": "shake_bottle"}, {"batch_size": 4},
                 {"dtype": "torch.float32"},
                 {"shapes": [{"images": [1, 3, 224, 224], "state": [1, 55]}]},
                 {"grids": ["tensor([[1,16,20]])"]}):
        assert sp.load_gate(str(gate), payload=_payload(**over))["reasons"] == [
            "gate_signature_mismatch"], over

    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert sp.load_gate(str(bad), payload=payload)["reasons"][0].startswith("gate_file_unreadable")
    bad.write_text('{"kind":"something_else"}', encoding="utf-8")
    assert sp.load_gate(str(bad), payload=payload)["reasons"] == ["gate_bad_kind"]


def test_gate_requires_verdict_pass(tmp_path):
    """验收判 BLOCKED 时，即使签名一致也**不得**开门。"""
    gate = tmp_path / "gate.json"
    payload = _payload()
    sp.write_gate(str(gate), payload=payload,
                  verdict={"status": "BLOCKED", "reasons": ["systematic_bias"]})
    res = sp.load_gate(str(gate), payload=payload)
    assert res["ok"] is False
    assert res["reasons"] == ["gate_verdict_not_pass:BLOCKED"]


def test_gate_signature_is_stable_across_json_roundtrip(tmp_path):
    """签名必须与 JSON 往返无关（否则门会假失效）。"""
    payload = _payload()
    import json
    reloaded = json.loads(json.dumps(payload))
    assert sp.gate_signature(reloaded) == sp.gate_signature(payload)


def test_restrict_starts_keeps_whole_episodes():
    ep_map = [51, 51, 51, 53, 53, 60, 60]
    starts = [0, 2, 3, 5]
    assert sp.restrict_starts_to_episodes(starts, ep_map, None) == starts
    assert sp.restrict_starts_to_episodes(starts, ep_map, 1) == [0, 2]      # 只留 ep51
    assert sp.restrict_starts_to_episodes(starts, ep_map, 2) == [0, 2, 3]   # ep51+ep53
    with pytest.raises(ValueError):
        sp.restrict_starts_to_episodes(starts, ep_map, 0)


# ---------------------------------------------------------------------------
# _infer 的返回结构（B1 必须展开 list[dict]）
# ---------------------------------------------------------------------------
class _StubValidator:
    """只实现 `_infer` 需要的两个方法，用来在 CPU 上钉住返回结构。"""

    def __init__(self):
        self.calls = []

    #: 真实 `ft.unapply()` 的返回**没有** `"actions"` 键（是物理量键）
    _PHYS = {"action.arm.position": np.zeros((2, 3), np.float32),
             "action.gripper": np.zeros((2, 1), np.float32)}

    def _infer_core(self, items, ft, *, fresh_visual_grid, noise):
        self.calls.append(("core", len(items), tuple(noise.shape)))
        return [dict(self._PHYS) for _ in items]        # 真实契约：list[dict]，物理量键

    def _infer_batch(self, items, ft, *, noise):
        self.calls.append(("batch", len(items), tuple(noise.shape)))
        return [dict(self._PHYS) for _ in items]


def test_infer_b1_flattens_list_of_dicts():
    """B1 分支必须把 `_infer_core` 的 list[dict] 展开 ⇒ 每一路都是 dict。

    2026-10-09 GPU 首跑就是死在这里（`p["actions"]` on a list）。
    """
    import torch
    tool = _tool()
    v = _StubValidator()
    items = [object(), object()]
    noise = torch.zeros(2, 2, 3)
    preds, _secs = tool._infer(v, items, None, batch=1, noise=noise)
    assert len(preds) == 2 and all(isinstance(p, dict) for p in preds)
    assert "actions" not in preds[0]                     # 真实契约里没有这个键
    arr = tool._pred_array(preds[0])                     # 打平后长度 = 2*3 + 2*1
    assert arr.shape == (8,)
    assert [c[0] for c in v.calls] == ["core", "core"]          # 单条逐次调用
    assert [c[1] for c in v.calls] == [1, 1]

    v2 = _StubValidator()
    preds2, _secs2 = tool._infer(v2, items, None, batch=2, noise=noise)
    assert len(preds2) == 2 and all(isinstance(p, dict) for p in preds2)
    assert [c[0] for c in v2.calls] == ["batch"]                # 批量一次调用
    assert v2.calls[0][1] == 2


def test_pred_array_sorts_keys_and_rejects_non_numeric():
    """打平必须**按 key 排序**（两条路径同序才可比），且无可比数组时明确报错。"""
    tool = _tool()
    a = tool._pred_array({"b": np.ones((2,), np.float32), "a": np.zeros((3,), np.float32)})
    assert a.tolist() == [0, 0, 0, 1, 1]                       # a 在前（排序）
    with pytest.raises(ValueError):
        tool._pred_array({"meta": "not-an-array"})


def test_correlated_noise_is_not_flagged_as_bias():
    """**关键回归**：元素间相关的纯噪声**不得**被判系统性偏差。

    2026-10-09 实测：i.i.d. 正态零分布会把这类噪声判成偏差（p_frac=0.000 假阳性），
    因为逐元素差值空间相关 ⇒ 零分布低估方差。改为"同代码随机二分"参考分布后应通过。
    """
    rng = np.random.default_rng(4242)
    shape = (50, 14)
    base = rng.standard_normal(shape) * 0.5
    # 每个元素有**自己的固定偏置**（相关结构）+ 每次独立抖动
    per_element_bias = rng.standard_normal(shape) * 1e-3

    def group(seed):
        r = np.random.default_rng(seed)
        return [base + per_element_bias + r.standard_normal(shape) * 5e-4 for _ in range(5)]

    b1, b2 = group(1), group(2)
    obs = sp._bias_fraction(b1, b2)
    ref = sp.bias_reference(runs_a=b1, runs_b=b2)
    assert ref["n_ref"] > 0
    assert obs <= ref["ref_max"] + 1e-12, (obs, ref)
    verdict = sp.decide_numeric(b1_runs=b1, b2_runs=b2, reps=5)
    assert verdict["status"] == "PASS", verdict["reasons"]
    # 旧的 i.i.d. 零分布判据在此仅作对照记录（真实数据上它确实误报过，合成样例未必复现）
    assert 0.0 <= verdict["bias_pvalue"]["p_frac"] <= 1.0
