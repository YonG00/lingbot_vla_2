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


# --------------------------------------------------------------------------- #
# 单进程验收：RNG 统一 + Batch8/Batch1/Auto/复测 四阶段（不训练、无 optimizer.step）
# --------------------------------------------------------------------------- #
class _FakeDataset:
    def __getitem__(self, i):
        return {"i": i}


class _FakeScorer:
    """记录每批调用方式；loss 只由 sample_id 决定（与批大小无关）⇒ 可验并行无关性。"""

    def __init__(self):
        self.calls = []

    def score(self, items, sample_ids=None):
        self.calls.append({"n": len(items), "sample_ids": (list(sample_ids) if sample_ids else None)})
        return [0.5 + 0.001 * sid for sid in (sample_ids or range(len(items)))]


def _scorer(**env):
    from lingbotvla.auto_learning.real.backend import RealHardnessScorer
    return RealHardnessScorer(_FakeScorer(), _FakeDataset(), max_batch=8)


def test_hardness_report_mode_binds_sample_ids_but_default_does_not(monkeypatch):
    """要求5：报告模式与 auto 路径同用 per-sample-id 噪声；production 默认行为不变。"""
    s1 = _FakeScorer()
    from lingbotvla.auto_learning.real.backend import RealHardnessScorer
    RealHardnessScorer(s1, _FakeDataset(), max_batch=8).score("t", [1, 2, 3])
    assert s1.calls == [{"n": 3, "sample_ids": None}], s1.calls
    monkeypatch.setenv("AL_HARDNESS_REPORT_OUT", "/tmp/_hr_unused.json")
    s2 = _FakeScorer()
    RealHardnessScorer(s2, _FakeDataset(), max_batch=8).score("t", [1, 2, 3])
    assert s2.calls[0]["sample_ids"] == [1, 2, 3], s2.calls
    monkeypatch.setenv("AL_HARDNESS_UNIFY_RNG", "1")
    s3 = _FakeScorer()
    RealHardnessScorer(s3, _FakeDataset(), max_batch=8).score("t", [4, 5])
    assert s3.calls[0]["sample_ids"] == [4, 5], s3.calls


def test_hardness_single_process_phases_run_once_without_recursion(monkeypatch, tmp_path):
    """要求4/7：同一进程内 Batch8 → Batch1 → 复测；不得递归、不得额外占显存。"""
    import json
    from lingbotvla.auto_learning.real.backend import RealHardnessScorer
    out = tmp_path / "hr.json"
    monkeypatch.setenv("AL_HARDNESS_REPORT_OUT", str(out))
    monkeypatch.setenv("AL_HARDNESS_REPLAY_BATCH", "1")
    monkeypatch.setenv("AL_HARDNESS_REPEAT", "1")
    monkeypatch.setenv("AL_HARDNESS_FIXED_BATCH", "8")
    scorer = _FakeScorer()
    RealHardnessScorer(scorer, _FakeDataset(), max_batch=8).score("t", list(range(1, 10)))
    recs = json.loads(out.read_text())["records"]
    by_phase = {}
    for r in recs:
        by_phase.setdefault(r["phase"], set()).add(r["batch"])
    # main/repeat 用 Batch8（末批余数 1 条）；replay 全程 Batch1 —— 绝无递归放大
    assert by_phase["main"] == {8, 1} and by_phase["repeat"] == {8, 1}, by_phase
    assert by_phase["replay_batch1"] == {1}, by_phase
    assert all(r["rng_binding"] == "per_sample_id" for r in recs)
    assert len(scorer.calls) <= 30, f"调用次数异常（疑似递归）: {len(scorer.calls)}"
    # 逐样本 loss 与 sample_id 绑定 ⇒ 与批大小无关 ⇒ Batch1/8 应逐位一致
    per_sid = {}
    for r in recs:
        if r["phase"] in ("main", "replay_batch1"):
            for sid, v in r["losses"].items():
                per_sid.setdefault(int(sid), set()).add(round(float(v), 12))
    assert all(len(v) == 1 for v in per_sid.values()), per_sid


def test_run_all_covers_batch4_by_using_a_four_trajectory_acceptance_config():
    """要求1：Scout 只用 2 条轨迹时永远凑不出 Batch4 ⇒ 验收必须用 4 条轨迹的专用配置。"""
    root = Path(__file__).resolve().parents[1]
    cfg = root / acc.ACCEPTANCE_AL_CONFIG
    assert cfg.exists(), cfg
    text = cfg.read_text(encoding="utf-8")
    assert "global_scout_val_trajs: 4" in text
    cmd = acc.run_all_command("/tmp/out")
    assert "--al-config" in cmd and str(acc.ACCEPTANCE_AL_CONFIG) in cmd, cmd
    env = acc.build_run_all_env("/tmp/out")
    assert env["AL_HARDNESS_REPLAY_BATCH"] == "1" and env["AL_HARDNESS_REPEAT"] == "1"
    assert "AL_HARDNESS_BATCH_APPROVED" not in env  # 首轮不加 --include-auto


def test_zero_step_hardness_selftest_hook_exists_and_is_opt_in():
    """要求2：零 optimizer.step 时 hardness 扫描不会触发 ⇒ 需要显式钩子（默认关闭）。"""
    src = (Path(__file__).resolve().parents[1] / "lingbotvla/auto_learning/real/build.py").read_text(encoding="utf-8")
    assert "AL_HARDNESS_SELFTEST_IDS" in src
    assert "AL_HARDNESS_REPORT_OUT" in src          # 必须同时开启报告才跑，避免影响正式训练
    assert "hardness.score(" in src                 # 复用真实 scorer，不自造推理


def test_run_all_env_forces_batch_coverage_and_selftest_without_auto():
    """要求1/2/3：一次进程内强制覆盖 Batch1/2/4 + 零步 Hardness 自检；仍不设任何 APPROVED。"""
    env = acc.build_run_all_env("/tmp/out2")
    assert env["AL_EVAL_BATCH_FORCE_COVERAGE"] == "1"
    assert env["AL_HARDNESS_SELFTEST_IDS"] == "9"
    assert env["AL_HARDNESS_REPORT_OUT"].endswith("hardness_parity.json")
    assert "AL_HARDNESS_BATCH_APPROVED" not in env and "AL_EVAL_BATCH_APPROVED" not in env
    assert env["AL_HARDNESS_BATCH_MODE"] == "fixed"


def test_probe_force_coverage_keeps_guards(monkeypatch):
    """要求1：speedup 不达标不得阻止 Batch4 数值验收，但 parity/显存/失败保护不能被绕过。"""
    text = (Path(__file__).resolve().parents[1] /
            "lingbotvla/utils/open_loop_validation.py").read_text(encoding="utf-8")
    assert "AL_EVAL_BATCH_FORCE_COVERAGE" in text
    assert "if not safe or (not faster and not _force_cov):" in text, "守卫表达式必须保留 safe 优先"


def test_selftest_refuses_multirank_and_stays_opt_in():
    """要求2：默认关闭、正式训练不触发、多卡 FSDP2 明确拒绝（避免 collective 死锁）。"""
    src = (Path(__file__).resolve().parents[1] / "lingbotvla/auto_learning/real/build.py").read_text(encoding="utf-8")
    assert "AL_HARDNESS_SELFTEST_IDS" in src and "AL_HARDNESS_REPORT_OUT" in src
    assert "get_world_size() > 1" in src and "FSDP2" in src
    assert "hardness.score(" in src and "RuntimeError" in src


def test_probe_block_imports_report_helpers():
    """回归：probe 报告写入路径依赖 action_diffs/append_json_record，缺 import 会抛 NameError（实测踩过）。"""
    src = (Path(__file__).resolve().parents[1] /
           "lingbotvla/utils/open_loop_validation.py").read_text(encoding="utf-8")
    i = src.index("normalized_action_predictions)")
    seg = src[max(0, i - 300):i + 30]
    assert "action_diffs" in seg and "append_json_record" in seg, seg


def test_probe_report_error_record_forces_blocked_not_pass():
    """要求2：证据缺失（报告写入失败）必须显式 BLOCKED，不得静默当 PASS。"""
    ok = {"kind": "eval_batch_probe", "mode": "probe", "batch": 2, "parity": True,
          "safe": True, "peak_free_gib": 80.0, "speedup": 1.9, "per_traj": []}
    bad = {"kind": "eval_batch_probe_error", "mode": "probe", "batch": 2,
           "error": "NameError: name 'append_json_record' is not defined"}
    v = acc.verdict_probe([ok, ok | {"batch": 4}, bad])
    assert v["status"] == "BLOCKED", v
    assert any("probe_report_write_failed" in b for b in v["blocked"]), v


def test_parity_comparison_actually_compares_normalized_predictions():
    """根因回归（2026-10-09 GPU 诊断）：normalized_action_predictions 返回 list[dict]，
    两侧打平函数都必须支持 dict —— 否则 outputs_close 恒 False（parity 假失败）、
    action_diffs 恒 None（哨兵 -1e0），批处理被永久禁用。"""
    import numpy as np
    from lingbotvla.auto_learning.scan_accel import action_diffs, normalized_action_predictions
    from lingbotvla.auto_learning.eval_batch_policy import outputs_close

    keys = ("actions", "action_tokens")
    a = {"actions": np.zeros((2, 3), dtype=np.float32),
         "action_tokens": np.ones((2,), dtype=np.float32)}
    b = {k: v.copy() for k, v in a.items()}
    na, nb = normalized_action_predictions([a], keys), normalized_action_predictions([b], keys)
    assert isinstance(na[0], dict)                     # 契约：list[dict]
    assert outputs_close(na, nb, atol=1e-5, rtol=1e-3) is True   # 相同 ⇒ True（以前恒 False）
    assert action_diffs(na, nb) == (0.0, 0.0)                    # 差值必须真是 0，不是 None

    c = {k: v.copy() for k, v in a.items()}
    c["actions"][0, 0] += 1e-2                                   # 单元素扰动 1e-2
    nc = normalized_action_predictions([c], keys)
    assert outputs_close(na, nc, atol=1e-5, rtol=1e-3) is False  # 真的超容差 ⇒ False
    mx, mean = action_diffs(na, nc)
    assert mx is not None and abs(mx - 1e-2) < 1e-6, (mx, mean)  # 差值可见且量级正确
    assert mean is not None and 0 < mean < mx


def test_structure_mismatch_never_passes_even_with_equal_numbers():
    """用户 2026-10-09 要求的反例：key/层级不匹配不得因数值相同而判 parity=True。"""
    from lingbotvla.auto_learning.eval_batch_policy import outputs_close, same_structure
    from lingbotvla.auto_learning.scan_accel import action_diffs

    a = [{"x": [1.0, 2.0]}]
    b = [{"y": [1.0, 2.0]}]                     # key 名不同、数值相同 ⇒ 必须 False
    assert same_structure(a, b) is False
    assert outputs_close(a, b, atol=1e-5, rtol=1e-3) is False
    assert action_diffs(a, b) == (None, None)

    c = [{"x": [[1.0], [2.0]]}]                 # 层级不同、数值序列相同 ⇒ 必须 False
    assert outputs_close(a, c, atol=1e-5, rtol=1e-3) is False
    assert action_diffs(a, c) == (None, None)

    d = [{"x": [1.0, 2.0], "z": [3.0]}]         # key 集合不同 ⇒ 必须 False
    assert outputs_close(a, d, atol=1e-5, rtol=1e-3) is False

    assert outputs_close(a, [{"x": [1.0, 2.0]}], atol=1e-5, rtol=1e-3) is True   # 完全相同 ⇒ True
