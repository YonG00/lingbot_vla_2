"""GMean mode selects by candidate / per-task PASS line; old mode unchanged."""
import math

import pytest

from al_fixtures import default_al, make_cfg, scheduler_of
from lingbotvla.auto_learning.decision.thresholds import PassThresholds
from lingbotvla.auto_learning.state.registry import TaskRecord
from lingbotvla.auto_learning.types import TaskStatus


def _sched():
    al = default_al(pass_metric="gmean_mse", pass_thresholds_file="placeholder.json")
    al.pass_thresholds = PassThresholds(
        metric="mse", stat="geomean", config_fingerprint="test",
        tasks={"t0": 0.1, "t1": 10.0, "t2": 1.0})
    return scheduler_of(make_cfg(3, al=al))


def test_priority_uses_ratio_not_smallest_raw_nmse_or_mse():
    sch = _sched()
    r0, r1, r2 = [sch.registry.get(f"t{i}") for i in range(3)]
    r0.scout_nmse, r0.scout_gmean_mse = 0.05, 0.5   # ratio 5
    r1.scout_nmse, r1.scout_gmean_mse = 2.0, 11.0  # ratio 1.1 -- chosen
    r2.scout_nmse, r2.scout_gmean_mse = 0.1, 2.0   # ratio 2
    assert min([r0, r1, r2], key=sch._candidate_priority).task_name == "t1"


def test_old_nmse_ranking_unchanged():
    sch = _sched()
    sch.al.pass_metric = "nmse"
    a, b = TaskRecord("a", scout_nmse=0.1), TaskRecord("b", scout_nmse=0.2)
    assert sch._candidate_priority(a) < sch._candidate_priority(b)


@pytest.mark.parametrize("value,line", [(None, 1.0), (float("nan"), 1.0),
                                        (float("inf"), 1.0), (1.0, None),
                                        (1.0, 0.0), (1.0, -1.0)])
def test_gmean_ranking_fail_closed_missing_metric_or_bad_threshold(value, line):
    sch = _sched()
    r = TaskRecord("t0", scout_nmse=0.1, scout_gmean_mse=value)
    sch.al.pass_thresholds.tasks["t0"] = line
    with pytest.raises(ValueError):
        sch._candidate_priority(r)


def test_gmean_scout_persist_roundtrip():
    r = TaskRecord("t0", scout_nmse=3.0, scout_gmean_mse=0.33)
    rebuilt = TaskRecord.from_dict(r.to_dict())
    assert rebuilt.scout_gmean_mse == pytest.approx(0.33)
    assert r.table_row()["scout_gmean_mse"] == pytest.approx(0.33)
    assert TaskRecord.from_dict({"task_name": "old"}).scout_gmean_mse is None


def test_bootstrap_records_valid_gmean_and_rescan_updates(monkeypatch):
    sch = _sched()
    events = [sch.advance() for _ in range(3)]
    for i in range(3):
        rec = sch.registry.get(f"t{i}")
        if rec.status == TaskStatus.CANDIDATE.value:
            assert rec.scout_gmean_mse is not None
            assert math.isfinite(rec.scout_gmean_mse)
    # Re-scan must keep it coherent with evaluation history.
    sch._rescan()
    for rec in sch.registry:
        if rec.status == TaskStatus.CANDIDATE.value:
            recent = [row for row in rec.eval_history if row["kind"] == "rescan"]
            assert recent and rec.scout_gmean_mse == pytest.approx(recent[-1]["gmean_mse"])
