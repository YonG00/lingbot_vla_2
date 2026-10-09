"""扫描加速 GPU 验收入口的 CPU 测试（PLAN ONLY + fail-closed 判定矩阵）。

对应要求：Eval Batch Probe / Hardness 批量的**自动化验收入口**必须在无卡时就可信：
没有真实 GPU 证据 ⇒ BLOCKED；有证据但数值/排序/显存不达标 ⇒ FAIL；全达标才 PASS。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/scan_accel_gpu_acceptance.py"
spec = importlib.util.spec_from_file_location("scan_accel_gpu_acceptance", SCRIPT)
assert spec and spec.loader
acc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acc)


def _probe(batch, parity=True, safe=True, peak=20.0, speedup=1.5, mode="probe"):
    return {"kind": "eval_batch_probe", "mode": mode, "batch": batch, "parity": parity,
            "safe": safe, "peak_free_gib": peak, "speedup": speedup,
            "per_traj": [{"dataset_index": 1, "max_abs_diff": 1e-6, "mean_abs_diff": 1e-7}]}


def test_plan_is_side_effect_free_and_states_the_gates(capsys, tmp_path):
    assert acc.main(["plan"]) == 0
    out = capsys.readouterr().out
    assert "AL_EVAL_BATCH_MODE=probe" in out and "AL_HARDNESS_FIXED_BATCH=1" in out
    assert "AL_EVAL_BATCH_APPROVED" in out and "禁止" in out
    assert not list(tmp_path.iterdir())


def test_probe_verdict_requires_real_evidence():
    assert acc.verdict_probe([])["status"] == "BLOCKED"


@pytest.mark.parametrize("records,status,needle", [
    ([_probe(2), _probe(4)], "PASS", "speedup_observed"),
    ([_probe(2, speedup=0.8), _probe(4, speedup=0.9)], "PASS", "no_speedup_observed_do_not_enable_auto"),
    ([_probe(2), _probe(4, parity=False)], "FAIL", "parity_failed"),
    ([_probe(2), _probe(4, safe=False)], "FAIL", "safety_or_vram_guard_failed"),
    ([_probe(2), _probe(4, peak=9.0)], "FAIL", "peak_free_below_reserve"),
    ([_probe(2), _probe(4, mode="auto")], "FAIL", "unexpected_mode"),
    ([_probe(2)], "BLOCKED", "batch4_not_reached"),
])
def test_probe_verdict_matrix(records, status, needle):
    v = acc.verdict_probe(records)
    assert v["status"] == status, v
    blob = json.dumps(v, ensure_ascii=False)
    assert needle in blob, (needle, blob)


def _hard(rows, batch=8, peak=20.0):
    return {"losses": dict(rows), "meta": {"n_records": 1, "batches": [batch],
                                          "min_peak_free_gib": peak}}


def test_hardness_verdict_passes_when_everything_holds():
    v = acc.verdict_hardness(_hard({1: 0.5, 2: 0.7, 3: 0.6}, batch=8),
                             _hard({1: 0.5, 2: 0.7, 3: 0.6}, batch=1),
                             repeat=_hard({1: 0.5, 2: 0.7, 3: 0.6}, batch=8))
    assert v["status"] == "PASS" and v["ordering_identical"] and v["rng_repeat_checked"], v


@pytest.mark.parametrize("args,needle", [
    ((_hard({1: 0.5, 2: 0.7}, 8), _hard({1: 0.5, 2: 0.9}, 1), None), "loss_mismatch"),
    ((_hard({1: 0.5, 2: 0.7}, 8), _hard({1: 0.5, 2: 0.7, 3: 0.1}, 1), None), "sample_id_mismatch"),
    ((_hard({1: 0.5, 2: 0.7}, 8), _hard({1: 0.5, 2: 0.7}, 8), None), "same_batch_size"),
    ((_hard({1: 0.5, 2: 0.7}, 8), _hard({1: 0.5, 2: 0.7}, 1, peak=9.0), None), "peak_free_below_reserve"),
])
def test_hardness_verdict_fail_closed(args, needle):
    v = acc.verdict_hardness(args[0], args[1], repeat=args[2])
    assert v["status"] == "FAIL" and needle in json.dumps(v, ensure_ascii=False), v


def test_hardness_ordering_change_and_rng_are_detected():
    a = _hard({1: 0.10, 2: 0.20}, 8)
    b = _hard({1: 0.20, 2: 0.10}, 1)
    assert "ordering_changed" in acc.verdict_hardness(a, b)["problems"]
    c = _hard({1: 0.10, 2: 0.21}, 8)
    assert "rng_not_reproducible" in acc.verdict_hardness(a, _hard({1: 0.10, 2: 0.20}, 1), repeat=c)["problems"]


def test_reports_written_by_append_json_record_are_readable(tmp_path):
    from lingbotvla.auto_learning.scan_accel import append_json_record

    p = tmp_path / "probe.json"
    append_json_record(str(p), _probe(2))
    append_json_record(str(p), _probe(4))
    assert len(acc.probe_records(str(p))) == 2
    assert acc.verdict_probe(acc.probe_records(str(p)))["status"] == "PASS"


def test_preflight_report_mentions_zero_step_risk():
    doc = (Path(__file__).resolve().parents[1] / "tools/gmean50_preflight.py").read_text(encoding="utf-8")
    assert "zero_step_risk" in doc and "零训练步" in doc and "target" in doc
