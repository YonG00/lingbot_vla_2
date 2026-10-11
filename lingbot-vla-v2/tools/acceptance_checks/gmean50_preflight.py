#!/usr/bin/env python3
"""CPU-only fail-closed GMean 200x startup audit and historical Scout inventory.

No scans are skipped; old evidence cannot safely be reused unless exact model,
evaluator, seed, trajectory IDs and normalization provenance is independently
established. This tool deliberately does not load historical metrics into AL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

from lingbotvla.auto_learning.baseline import BaselineStore
from lingbotvla.auto_learning.decision.thresholds import PassThresholds, verify_threshold_stat_compatible


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_thresholds(table_path: Path, baseline_path: Path, *, required_multiplier: float = 200.0,
                     expected_tasks: int = 50) -> dict:
    """Reject missing provenance, wrong statistic/fingerprint or partial coverage."""
    if not math.isfinite(required_multiplier) or required_multiplier <= 0:
        raise ValueError("required multiplier must be finite and positive")
    if not table_path.is_file() or not baseline_path.is_file():
        raise ValueError("threshold table or baseline missing")
    raw = json.loads(table_path.read_text(encoding="utf-8"))
    calib = raw.get("calibration")
    if not isinstance(calib, dict) or calib.get("method") != "reference_per_trajectory_geomean_multiplier":
        raise ValueError("threshold calibration provenance missing; cannot certify 200x")
    mul = calib.get("multiplier")
    if type(mul) not in (int, float) or not math.isfinite(mul) or not math.isclose(mul, required_multiplier, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"threshold multiplier is not {required_multiplier}x")
    store = BaselineStore.load(str(baseline_path))
    if not store.config_fingerprint:
        raise ValueError("baseline config fingerprint missing")
    table = PassThresholds.load(str(table_path), require_metric="mse",
                                expect_fingerprint=store.config_fingerprint)
    verify_threshold_stat_compatible(table, pass_metric="gmean_mse")
    expected = set(store.tasks)
    if len(expected) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} tasks, baseline has {len(expected)}")
    if set(table.tasks) != expected:
        raise ValueError(f"threshold/baseline task mismatch: missing={sorted(expected-set(table.tasks))}, extra={sorted(set(table.tasks)-expected)}")
    excluded = sorted(k for k, v in table.tasks.items() if v is None)
    details = calib.get("tasks")
    if not isinstance(details, dict) or set(details) != expected:
        raise ValueError("calibration.tasks provenance missing or incomplete")
    capped = []
    high_cv = []
    for task, detail in details.items():
        if not isinstance(detail, dict) or "effective_line" not in detail:
            raise ValueError(f"calibration detail missing for {task}")
        observed = detail["effective_line"]
        actual = table.get(task)
        if actual is None:
            if observed is not None:
                raise ValueError(f"calibration/table mismatch for {task}")
        elif (type(observed) not in (float, int) or
              not math.isclose(float(observed), float(actual), rel_tol=1e-9, abs_tol=0)):
            raise ValueError(f"calibration/table mismatch for {task}")
        if detail.get("baseline_capped"):
            capped.append(task)
        if detail.get("high_cv_warning"):
            high_cv.append(task)
    return {"status": "READY" if not excluded else "BLOCKED", "n_tasks": len(expected),
            "usable": table.n_usable, "excluded_null": excluded, "baseline_capped_tasks": sorted(capped),
            "high_cv_warning_tasks": sorted(high_cv),
            "zero_step_risk": (
                "target_total_passed_tasks 在实验 YAML 中固定为 4；若 Bootstrap 阶段已解析出 >= 4 个 PASS，"
                "调度器会 all_tasks_resolved 零训练步收工 ⇒ 本实验不产生任何优化步（不得擅自改 target；"
                "应改为先看首轮评测的 PASS 名单再决定）"),
            "note": "priority is candidate/effective PASS threshold; baseline-cap makes it differ from candidate/(200*reference)",
            "multiplier": float(mul), "table_sha256": _sha(table_path),
            "baseline_sha256": _sha(baseline_path),
            "baseline_config_fingerprint": store.config_fingerprint}


def inventory_scout_events(paths: Iterable[Path], task_names: Iterable[str]) -> dict:
    """Read-only inventory. Historical NMSE events are NOT a GMean cache."""
    allowed = set(task_names)
    seen: Dict[str, List[str]] = {}
    source_sha: Dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"history missing: {path}")
        source_sha[str(path)] = _sha(path)
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{path}:{i} invalid JSONL: {exc}") from exc
                if item.get("kind") == "event" and item.get("action") == "bootstrap":
                    task = item.get("task")
                    if task in allowed:
                        seen.setdefault(task, []).append(str(path))
    return {"tasks_with_bootstrap_event": len(seen),
            "missing": sorted(allowed - set(seen)),
            "duplicate_event_tasks": sorted(k for k, v in seen.items() if len(v) != 1),
            "source_sha256": source_sha,
            "reuse_allowed": False,
            "reason": "event logs do not prove matching checkpoint/dtype/norm/eval IDs and full per-traj GMean; no automatic replay of Scout"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run 50-task GMean200 threshold/history audit (no GPU or writes)")
    parser.add_argument("--config", required=True, help="50-task experimental Auto Learning YAML")
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--history", action="append", default=[], help="historical auto_learning_events.jsonl; inventory only")
    parser.add_argument("--expected-tasks", type=int, default=50)
    parser.add_argument("--output", help="optional new JSON result path; refuses overwrite")
    a = parser.parse_args(argv)
    try:
        import yaml
        cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
        if not isinstance(cfg, dict) or cfg.get("pass_metric") != "gmean_mse":
            raise ValueError("experimental config must explicitly choose gmean_mse")
        if Path(str(cfg.get("pass_thresholds_file", ""))).resolve() != Path(a.thresholds).resolve():
            raise ValueError("YAML threshold file differs from audited file")
        if cfg.get("task_names") is not None or cfg.get("scout_confirm_enabled") is not True:
            raise ValueError("experiment must include all tasks and enable Scout Confirm")
        if cfg.get("target_total_passed_tasks") != 4 or cfg.get("new_ratio") != 0.7:
            raise ValueError("experiment target and NEW/Replay ratio not expected values")
        report = audit_thresholds(Path(a.thresholds), Path(a.baseline), expected_tasks=a.expected_tasks)
        baseline = BaselineStore.load(a.baseline)
        report["history"] = inventory_scout_events(map(Path, a.history), baseline.tasks)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        report = {"status": "BLOCKED", "reason": str(exc)}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if a.output:
        path = Path(a.output)
        try:
            with path.open("x", encoding="utf-8") as out:
                json.dump(report, out, ensure_ascii=False, indent=2)
        except FileExistsError:
            parser.error("output exists; refuse overwrite")
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
