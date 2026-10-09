"""CPU-only fail-closed preflight and historical Scout inventory."""
import json

import pytest

from tools.gmean50_preflight import audit_thresholds, inventory_scout_events, main


def _inputs(tmp_path):
    base = tmp_path / "baseline.json"
    table = tmp_path / "table.json"
    baseline = {"version": 1, "config_fingerprint": "config-abc",
                "tasks": {f"t{i}": {"mse": 1.0} for i in range(3)}}
    detail = {f"t{i}": {"effective_line": float(i + 1),
                          "baseline_capped": i == 2,
                          "high_cv_warning": False} for i in range(3)}
    thresholds = {"version": 1, "config_fingerprint": "config-abc", "metric": "mse",
                  "stat": "geomean", "tasks": {f"t{i}": float(i + 1) for i in range(3)},
                  "calibration": {"method": "reference_per_trajectory_geomean_multiplier",
                                  "multiplier": 200, "tasks": detail}}
    base.write_text(json.dumps(baseline))
    table.write_text(json.dumps(thresholds))
    return base, table, baseline, thresholds


def test_valid_thresholds_report_cap_and_hash(tmp_path):
    base, table, _, _ = _inputs(tmp_path)
    report = audit_thresholds(table, base, expected_tasks=3)
    assert report["status"] == "READY"
    assert report["baseline_capped_tasks"] == ["t2"]
    assert len(report["table_sha256"]) == 64


@pytest.mark.parametrize("change", [
    lambda x: x["calibration"].update(multiplier=100),
    lambda x: x.update(stat="arithmetic"),
    lambda x: x.update(config_fingerprint="other"),
    lambda x: x["tasks"].pop("t0"),
    lambda x: x["calibration"].pop("tasks"),
    lambda x: x["calibration"]["tasks"]["t1"].update(effective_line=100),
])
def test_malformed_or_wrong_threshold_fails(tmp_path, change):
    base, table, _, threshold = _inputs(tmp_path)
    change(threshold)
    table.write_text(json.dumps(threshold))
    with pytest.raises(ValueError):
        audit_thresholds(table, base, expected_tasks=3)


def test_null_task_blocks_ready(tmp_path):
    base, table, _, threshold = _inputs(tmp_path)
    threshold["tasks"]["t0"] = None
    threshold["calibration"]["tasks"]["t0"]["effective_line"] = None
    table.write_text(json.dumps(threshold))
    assert audit_thresholds(table, base, expected_tasks=3)["status"] == "BLOCKED"


def test_history_is_only_inventory_not_reuse(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"kind":"event","action":"bootstrap","task":"t0","scout_nmse":0.1}\n')
    report = inventory_scout_events([path], ["t0", "t1"])
    assert report["tasks_with_bootstrap_event"] == 1
    assert report["missing"] == ["t1"]
    assert report["reuse_allowed"] is False


def test_cli_dry_run_validates_yaml_and_missing_paths(tmp_path, capsys):
    base, table, _, _ = _inputs(tmp_path)
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(f"pass_metric: gmean_mse\npass_thresholds_file: {table}\ntask_names: null\nscout_confirm_enabled: true\ntarget_total_passed_tasks: 4\nnew_ratio: 0.7\n")
    args = ["--config", str(cfg), "--thresholds", str(table), "--baseline", str(base), "--expected-tasks", "3"]
    assert main(args) == 0
    assert '"status": "READY"' in capsys.readouterr().out
    cfg.write_text(cfg.read_text().replace("gmean_mse", "nmse"))
    assert main(args) == 2


# --------------------------------------------------------------------------- #
# 停止条件（2026-10-09）：'至少 N 个新增非 Bootstrap PASS' 下限
# --------------------------------------------------------------------------- #
def _floor_state(newly, bootstrap):
    class S:
        newly_passed = list(newly)
        bootstrap_passed = list(bootstrap)
    return S()


def test_gmean200_train_config_declares_new_pass_floor_and_caps():
    import pathlib
    cfg = pathlib.Path(__file__).resolve().parents[1] / \
        "configs/auto_learning/experiment_50task_gmean200_train2.yaml"
    text = cfg.read_text(encoding="utf-8")
    assert "min_new_tasks_passed_this_run: 2" in text
    assert "max_new_tasks_attempted_this_run: 6" in text        # 资源上限必须保留
    assert "target_total_passed_tasks: 4" in text               # 目标未被改动
    assert "pass_metric: gmean_mse" in text


def test_new_pass_floor_semantics_bootstrap_does_not_count():
    """Bootstrap PASS 不计入新增；只有 newly_passed 增长才满足下限。"""
    for boot in (0, 2, 4, 6):
        st = _floor_state([], [f"t{i}" for i in range(boot)])
        assert len(st.newly_passed) == 0            # Bootstrap 再多也不算新增 ✓
        for n in (0, 1, 2):
            st = _floor_state([f"n{i}" for i in range(n)], st.bootstrap_passed)
            ok = len(st.newly_passed) >= 2
            assert ok == (n >= 2), (boot, n)


def test_scheduler_gates_both_early_finish_paths_on_the_floor():
    """源码级锁定：目标达成与 all_tasks_resolved 两条早退路径都必须受下限约束。"""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] /
           "lingbotvla/auto_learning/orchestration/scheduler.py").read_text(encoding="utf-8")
    assert src.count("min_new_tasks_passed_this_run") >= 2
    assert "min_new_passes_not_reached" in src
    assert "if _floor is not None and len(self.state.newly_passed) < int(_floor):" in src
