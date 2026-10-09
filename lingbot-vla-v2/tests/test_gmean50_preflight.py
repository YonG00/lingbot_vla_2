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
