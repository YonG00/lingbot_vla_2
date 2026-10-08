"""新 GMean PASS 线标定工具：独立 CPU 测试，不需要模型和 GPU。"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from lingbotvla.auto_learning.baseline import BaselineStore, FixedBaseline, task_fingerprint
from lingbotvla.auto_learning.decision.thresholds import (
    PassCheck, PassThresholds, ThresholdsError, check_pass,
)
from lingbotvla.auto_learning.tools.build_gmean_thresholds import (
    build_thresholds, load_reference, load_reference_log, main,
)


def _store(tmp_path, baseline=0.4):
    store = BaselineStore(path=str(tmp_path / "baseline.json"), config_fingerprint="config-123")
    for task in ("ok", "volatile"):
        store.put(FixedBaseline(task=task, mse=baseline, mu=(0.0,), fingerprint=task_fingerprint("config-123", "sha")))
    store.save()
    return BaselineStore.load(store.path)


def _reference():
    # ok: 一条 100 倍极端值，算术平均被拉高；几何平均仍较接近典型值
    return {
        "ok": [0.001] * 9 + [0.1],
        "volatile": [0.00001] * 9 + [0.2],
    }


def test_geomean_robust_against_outlier(tmp_path):
    table, diag = build_thresholds(_reference(), _store(tmp_path), multiplier=20, max_cv=5,
                                   reference="ref50k")
    geo = math.exp((9 * math.log(0.001) + math.log(0.1)) / 10)
    assert diag["ok"]["gmean"] == pytest.approx(geo)
    assert diag["ok"]["arithmetic_mean"] > geo * 5
    assert table.tasks["ok"] == pytest.approx(geo * 20)
    assert table.stat == "geomean"


def test_high_cv_becomes_null_not_pass(tmp_path):
    table, diag = build_thresholds(_reference(), _store(tmp_path), multiplier=220,
                                   max_cv=1.5, reference="ref50k")
    assert table.tasks["volatile"] is None
    assert table.tasks["ok"] is None  # 9+1 极端值也高 CV，安全拒绝
    assert diag["volatile"]["status"] == "needs_closed_loop_manual"
    with pytest.raises(ThresholdsError, match="CV"):
        build_thresholds(_reference(), _store(tmp_path), multiplier=220, max_cv=1.5,
                         high_cv_policy="error", reference="ref50k")


def test_baseline_cap_is_strict(tmp_path):
    per = {"ok": [0.005] * 10, "volatile": [0.003] * 10}
    table, diag = build_thresholds(per, _store(tmp_path, baseline=0.1), multiplier=220,
                                   reference="ref50k")
    assert table.tasks["ok"] == pytest.approx(0.099)
    assert table.tasks["volatile"] == pytest.approx(0.099)
    assert diag["ok"]["status"] == "baseline_capped"


def test_requires_enough_trajectories(tmp_path):
    with pytest.raises(ThresholdsError, match="只有 4 条轨迹"):
        build_thresholds({"ok": [0.001] * 4, "volatile": [0.002] * 10}, _store(tmp_path),
                         multiplier=220, reference="ref50k")


def test_rejects_missing_or_nonpositive(tmp_path):
    bad = {"ok": [0.001] * 9 + [0], "volatile": [0.002] * 10}
    with pytest.raises(ThresholdsError, match="有限正数"):
        build_thresholds(bad, _store(tmp_path), multiplier=220, reference="ref50k")
    with pytest.raises(ThresholdsError, match="任务覆盖不一致"):
        build_thresholds({"ok": [0.001] * 10}, _store(tmp_path), multiplier=220,
                         reference="ref50k")


def test_reference_rejects_duplicate_and_wrong_split(tmp_path):
    store = _store(tmp_path)
    fn = tmp_path / "ref.jsonl"
    fn.write_text('\n'.join(json.dumps(x) for x in [
        {"task": "ok", "traj": 1, "mse": 0.01},
        {"task": "ok", "traj": 1, "mse": 0.02},
    ]), encoding="utf-8")
    with pytest.raises(ThresholdsError, match="重复轨迹"):
        load_reference(str(fn), store)
    fn.write_text(json.dumps({"task": "ok", "traj": 1, "mse": 0.01, "split": "train"}), encoding="utf-8")
    with pytest.raises(ThresholdsError, match="split"):
        load_reference(str(fn), store)
    fn.write_text(json.dumps({"task": "ok", "traj": 1, "mse": 0.01, "config_fingerprint": "WRONG"}), encoding="utf-8")
    with pytest.raises(ThresholdsError, match="fingerprint"):
        load_reference(str(fn), store)


def test_end_to_end_cli(tmp_path):
    store = _store(tmp_path)
    path = tmp_path / "ref.jsonl"
    vals = {"ok": [0.001] * 10, "volatile": [0.00001] * 9 + [0.2]}
    with path.open("w", encoding="utf-8") as f:
        for task, mses in vals.items():
            for i, mse in enumerate(mses):
                f.write(json.dumps({"task": task, "traj": i, "mse": mse, "split": "active_val"}) + "\n")
    output = tmp_path / "thresholds.json"
    assert main(["--ref-per-traj", str(path), "--baseline", store.path,
                 "--reference", "ref-50k", "--multiplier", "220", "-o", str(output)]) == 0
    result = PassThresholds.load(str(output), expect_fingerprint=store.config_fingerprint,
                                 require_metric="mse")
    assert result.tasks["ok"] == pytest.approx(0.22)
    assert result.tasks["volatile"] is None
    raw = json.loads(output.read_text(encoding="utf-8"))
    assert raw["calibration"]["multiplier"] == 220
    assert raw["calibration"]["tasks"]["volatile"]["status"] == "needs_closed_loop_manual"
    assert raw["calibration"]["source_type"] == "per_trajectory_jsonl"
    assert raw["calibration"]["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def _log_fixture(tmp_path):
    store = _store(tmp_path)
    split = tmp_path / "split"
    split.mkdir()
    (split / "ok.val_ids.json").write_text(json.dumps(list(range(0, 10))), encoding="utf-8")
    (split / "volatile.val_ids.json").write_text(json.dumps(list(range(10, 20))), encoding="utf-8")
    (split / "combined.val_ids.json").write_text("[]", encoding="utf-8")
    mses = {i: (0.001 if i < 10 else 0.002) for i in range(20)}
    log = tmp_path / "open_loop_eval.log"
    log.write_text("".join(f"MSE for trajectory {i}: {v}, MAE: 0.02\n" for i, v in mses.items()), encoding="utf-8")
    return store, split, log, mses


def test_log_input_works_end_to_end_and_matches_jsonl(tmp_path):
    store, split, log, mses = _log_fixture(tmp_path)
    jsonl = tmp_path / "ref.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for i, mse in mses.items():
            f.write(json.dumps({"task": "ok" if i < 10 else "volatile", "traj": i, "mse": mse}) + "\n")
    from_log = tmp_path / "from_log.json"
    from_jsonl = tmp_path / "from_jsonl.json"
    shared = ["--baseline", store.path, "--reference", "ref50k", "--multiplier", "20"]
    assert main(["--ref-log", str(log), "--split-dir", str(split),
                 *shared, "-o", str(from_log)]) == 0
    assert main(["--ref-per-traj", str(jsonl),
                 *shared, "-o", str(from_jsonl)]) == 0
    a = PassThresholds.load(str(from_log), expect_fingerprint=store.config_fingerprint,
                            require_metric="mse")
    b = PassThresholds.load(str(from_jsonl), expect_fingerprint=store.config_fingerprint,
                            require_metric="mse")
    assert a.tasks == b.tasks == {"ok": pytest.approx(0.02), "volatile": pytest.approx(0.04)}
    assert json.loads(from_log.read_text())["calibration"]["source_type"] == "open_loop_eval_log"


@pytest.mark.parametrize("extension,problem", [
    ("MSE for trajectory 0: 0.1, MAE: 0.1\n", "重复记录"),
    ("MSE for trajectory 999: 0.1, MAE: 0.1\n", "意外轨迹"),
])
def test_log_rejects_duplicate_or_extra(tmp_path, extension, problem):
    store, split, log, _ = _log_fixture(tmp_path)
    log.write_text(log.read_text(encoding="utf-8") + extension, encoding="utf-8")
    with pytest.raises(ThresholdsError, match=problem):
        load_reference_log(str(log), str(split), store)


def test_log_rejects_missing_trajectory(tmp_path):
    store, split, log, _ = _log_fixture(tmp_path)
    log.write_text(log.read_text(encoding="utf-8").replace(
        "MSE for trajectory 8: 0.001, MAE: 0.02\n", ""), encoding="utf-8")
    with pytest.raises(ThresholdsError, match="日志缺 1 条轨迹"):
        load_reference_log(str(log), str(split), store)


def test_log_rejects_ambiguous_split_mapping(tmp_path):
    store, split, log, _ = _log_fixture(tmp_path)
    (split / "volatile.val_ids.json").write_text(json.dumps(list(range(9, 19))), encoding="utf-8")
    with pytest.raises(ThresholdsError, match="重复"):
        load_reference_log(str(log), str(split), store)


def test_log_rejects_incomplete_split(tmp_path):
    store, split, log, _ = _log_fixture(tmp_path)
    (split / "volatile.val_ids.json").unlink()
    with pytest.raises(ThresholdsError, match="split 缺任务"):
        load_reference_log(str(log), str(split), store)


@pytest.mark.parametrize("mse", ["nan", "inf", "-0.1", "0", "true"])
def test_invalid_mse_rejected_from_jsonl(tmp_path, mse):
    store = _store(tmp_path)
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"task": "ok", "traj": 1, "mse": mse}), encoding="utf-8")
    with pytest.raises(ThresholdsError):
        load_reference(str(path), store)


@pytest.mark.parametrize("traj", [[], {}, True, None, " "])
def test_invalid_trajectory_id_rejected(tmp_path, traj):
    store = _store(tmp_path)
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"task": "ok", "traj": traj, "mse": 0.1}), encoding="utf-8")
    with pytest.raises(ThresholdsError, match="traj"):
        load_reference(str(path), store)


@pytest.mark.parametrize("multiplier,max_cv,baseline_cap,min_trajectories", [
    (0, 1.5, 0.99, 10), (-1, 1.5, 0.99, 10), (float("nan"), 1.5, 0.99, 10),
    (20, 0, 0.99, 10), (20, 1.5, 1.0, 10), (20, 1.5, 0, 10),
    (20, 1.5, 0.99, 1), (20, 1.5, 0.99, 10.5),
])
def test_bad_parameters_fail_closed(tmp_path, multiplier, max_cv, baseline_cap, min_trajectories):
    with pytest.raises(ThresholdsError):
        build_thresholds(_reference(), _store(tmp_path), multiplier=multiplier,
                         max_cv=max_cv, baseline_cap=baseline_cap,
                         min_trajectories=min_trajectories, reference="ref50k")


def test_threshold_boundaries_and_null_runtime_result(tmp_path):
    table, _ = build_thresholds({"ok": [0.001] * 10,
                                 "volatile": [0.00001] * 9 + [0.2]},
                                _store(tmp_path), multiplier=20, reference="ref50k")
    cfg = SimpleNamespace(pass_metric="mse", pass_thresholds=table)
    assert check_pass(cfg, "ok", nmse=None, mse=0.02) == PassCheck.PASS
    assert check_pass(cfg, "ok", nmse=None, mse=0.020001) == PassCheck.BELOW
    assert check_pass(cfg, "volatile", nmse=0, mse=0) == PassCheck.NO_THRESHOLD
    assert check_pass(cfg, "unknown", nmse=0, mse=0) == PassCheck.NO_THRESHOLD
    assert check_pass(cfg, "ok", nmse=0, mse=float("nan")) == PassCheck.INVALID


def test_invalid_input_does_not_replace_old_output(tmp_path):
    store = _store(tmp_path)
    ref = tmp_path / "ref.jsonl"
    ref.write_text('{"task":"ok","traj":0,"mse":-1}\n', encoding="utf-8")
    output = tmp_path / "thresholds.json"
    output.write_text("DO NOT OVERWRITE", encoding="utf-8")
    with pytest.raises(ThresholdsError):
        main(["--ref-per-traj", str(ref), "--baseline", store.path,
              "--reference", "ref50k", "--multiplier", "220", "-o", str(output)])
    assert output.read_text(encoding="utf-8") == "DO NOT OVERWRITE"
