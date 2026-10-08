"""End-to-end semantics for explicitly enabled gmean_mse PASS, no GPU."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from al_fixtures import make_cfg, scheduler_of, default_al
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.decision.gmean import geometric_mse
from lingbotvla.auto_learning.decision.review import Reviewer
from lingbotvla.auto_learning.decision.state_machine import apply_pass, decide_after_unit
from lingbotvla.auto_learning.decision.thresholds import (
    PassCheck, PassThresholds, ThresholdsError, check_pass,
    forget_code_ex, is_forgotten_ex, verify_threshold_stat_compatible,
)
from lingbotvla.auto_learning.ports import EvalResult
from lingbotvla.auto_learning.state.registry import TaskRecord, TaskRegistry
from lingbotvla.auto_learning.types import Decision, TaskStatus, TrajectoryMetrics


def _al(threshold=0.02):
    cfg = default_al(pass_metric="gmean_mse", pass_thresholds_file="not-loaded-yet.json")
    cfg.pass_thresholds = PassThresholds(
        metric="mse", stat="geomean", tasks={"t0": threshold},
        config_fingerprint="fp",
    )
    return cfg


def _metric(task="t0", vals=(0.001, 0.1), split="active_val"):
    ids = list(range(len(vals)))
    return TrajectoryMetrics(
        task=task, split=split, episode_ids=ids,
        per_traj_mse=dict(zip(ids, vals)),
        mse=sum(vals) / len(vals), baseline_mse=0.2,
        nmse=(sum(vals) / len(vals)) / 0.2,
        r2=None, metric_valid=True, n_trajs=len(vals),
    )


def test_geometric_is_not_arithmetic_and_matches_log_reference():
    m = _metric(vals=(0.001, 0.1))
    assert m.gmean_mse == pytest.approx(0.01)
    assert m.mse == pytest.approx(0.0505)
    cfg = _al(threshold=0.02)
    assert check_pass(cfg, "t0", nmse=m.nmse, mse=m.mse, gmean_mse=m.gmean_mse) == PassCheck.PASS
    assert check_pass(cfg, "t0", nmse=m.nmse, mse=m.mse) == PassCheck.INVALID
    assert check_pass(default_al(pass_nmse=0.1), "t0", nmse=m.nmse, mse=m.mse) == PassCheck.BELOW


@pytest.mark.parametrize("vals,ids,n,expected", [
    ([0.1, 0.1], [1, 2], 2, 0.1),
    ([0.0, 0.1], [1, 2], 2, 0.0),
    ([0.1, 0.2], [1, 1], 2, None),
    ([0.1, 0.2], [1, 2], 3, None),
    ([0.1, float('nan')], [1, 2], 2, None),
    ([0.1, float('inf')], [1, 2], 2, None),
    ([-0.1, 0.2], [1, 2], 2, None),
    ([], [], 0, None),
])
def test_strict_aggregation(vals, ids, n, expected):
    result = geometric_mse(vals, expected_count=n, ids=ids)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_low_magnitude_no_product_underflow():
    assert geometric_mse([1e-200] * 16, expected_count=16) == pytest.approx(1e-200)


def test_evalresult_to_scheduler_preserves_per_traj_candidate():
    result = EvalResult(task="t0", split="val", episode_ids=[1, 2],
                        mse=0.0505, nmse=0.2525, baseline_mse=0.2,
                        per_traj_mse=[0.001, 0.1], per_traj_ids=[1, 2], n_traj=2)
    assert result.gmean_mse == pytest.approx(0.01)
    converted = result.to_trajectory_metrics()
    assert converted.gmean_mse == pytest.approx(result.gmean_mse)
    assert result.to_dict()["gmean_mse"] == pytest.approx(0.01)
    result.per_traj_ids = [1, 1]
    assert result.gmean_mse is None
    result.per_traj_ids = [1, 3]
    assert result.gmean_mse is None


def test_threshold_stat_metadata_strict():
    table = PassThresholds(metric="mse", stat="geomean", tasks={"t0": 0.02})
    verify_threshold_stat_compatible(table, pass_metric="gmean_mse")
    with pytest.raises(ThresholdsError):
        verify_threshold_stat_compatible(table, pass_metric="mse")
    with pytest.raises(ThresholdsError):
        verify_threshold_stat_compatible(PassThresholds(metric="mse", stat="p75", tasks={"t0": 0.02}),
                                         pass_metric="gmean_mse")
    with pytest.raises(ThresholdsError):
        verify_threshold_stat_compatible(PassThresholds(metric="nmse", stat="geomean", tasks={"t0": 0.02}),
                                         pass_metric="gmean_mse")
    with pytest.raises(ValueError):
        AutoLearningConfig(pass_metric="gmean_mse")


def test_unit_pass_and_registry_best_gmean_roundtrip():
    cfg = _al(0.02)
    rec = TaskRecord(task_name="t0", current_val_nmse=0.2525,
                     current_val_mse=0.0505, current_val_gmean_mse=0.01)
    outcome = decide_after_unit(rec, cfg, attempt_step=50)
    assert outcome.decision == Decision.PASS
    apply_pass(rec, cfg, "gmean passed")
    assert rec.best_gmean_mse == pytest.approx(0.01)
    assert TaskRecord.from_dict(rec.to_dict()).best_gmean_mse == pytest.approx(0.01)
    rec.current_val_gmean_mse = None
    assert decide_after_unit(rec, cfg, attempt_step=100).decision == Decision.DEFER


def test_relative_forgetting_uses_gmean_best_not_nmse_best():
    cfg = _al(0.02)
    cfg.forget_relative_threshold = 0.2
    assert is_forgotten_ex(cfg, "t0", cur_nmse=0.1, cur_mse=0.01,
                           best_nmse=0.1, cur_gmean_mse=0.015, best_gmean_mse=0.01)
    assert not is_forgotten_ex(cfg, "t0", cur_nmse=0.8, cur_mse=0.01,
                               best_nmse=0.1, cur_gmean_mse=0.01, best_gmean_mse=0.01)
    assert forget_code_ex(cfg, "t0", cur_nmse=0.1, cur_mse=0.1,
                          cur_gmean_mse=0.03).endswith("below_pass_line")


def test_bootstrap_gmean_pass_and_rescan_registry_data():
    cfg = make_cfg(1, al=_al(0.4))
    sch = scheduler_of(cfg)
    e = sch.advance()  # bootstrap scout + confirm
    assert e['action'] == 'bootstrap'
    assert e['result'] == 'pass'
    record = sch.registry.get('t0')
    assert record.status == TaskStatus.PASS.value
    assert record.current_val_gmean_mse is not None
    assert record.best_gmean_mse == record.current_val_gmean_mse
    assert any('gmean_mse' in x for x in record.eval_history)


def test_review_suspect_and_reopen_uses_gmean_not_arithmetic():
    cfg = _al(0.02)
    rec = TaskRecord(task_name='t0', status=TaskStatus.PASS.value,
                     scout_val_ids=[1, 2], confirm_val_ids=[1, 2],
                     best_nmse=0.2, best_gmean_mse=0.01,
                     current_val_gmean_mse=0.01, current_val_nmse=0.2)
    class FixedEval:
        def evaluate(self, *args):
            return _metric(vals=(0.015, 0.06), split='review')  # geo 0.03 > 0.02
    registry = TaskRegistry({'t0': rec})
    outcomes = Reviewer(cfg, FixedEval()).review(registry, global_step=10)
    assert outcomes[0].confirmed
    assert outcomes[0].forgotten
    assert rec.reopen_count == 1
    assert rec.status == TaskStatus.CANDIDATE.value


def test_review_invalid_gmean_keeps_pass_and_flags_diagnostic():
    cfg = _al(0.02)
    rec = TaskRecord(task_name='t0', status=TaskStatus.PASS.value,
                     scout_val_ids=[1, 2], confirm_val_ids=[1, 2],
                     best_nmse=0.2, best_gmean_mse=0.01,
                     current_val_gmean_mse=0.01, current_val_nmse=0.2)
    class MissingPerTraj:
        def evaluate(self, *args):
            m = _metric(vals=(0.03, 0.04))
            m.per_traj_mse = {}  # malformed/missing; no arithmetic fallback
            return m
    outcome = Reviewer(cfg, MissingPerTraj()).review(TaskRegistry({'t0': rec}), 10)[0]
    assert outcome.action == 'invalid_metric'
    assert rec.status == TaskStatus.PASS.value
    assert rec.current_val_gmean_mse == 0.01


def test_real_startup_requires_geomean_table(tmp_path):
    from lingbotvla.auto_learning.real.build import _attach_pass_thresholds
    path = tmp_path / 'thresholds.json'
    cat = SimpleNamespace(task_names=lambda: ['t0'])
    store = SimpleNamespace(config_fingerprint='fp')
    cfg = _al()
    cfg.pass_thresholds_file = str(path)
    PassThresholds(metric='mse', stat='geomean', tasks={'t0': 0.02},
                   config_fingerprint='fp').save(str(path))
    del cfg.pass_thresholds
    _attach_pass_thresholds(cfg, cat, store)
    assert check_pass(cfg, 't0', nmse=10, mse=1, gmean_mse=0.01) == PassCheck.PASS
    # Same table is forbidden if only arithmetic candidate MSE is available.
    old = default_al(pass_metric='mse', pass_thresholds_file=str(path))
    with pytest.raises(ThresholdsError, match='geomean'):
        _attach_pass_thresholds(old, cat, store)
    # Wrong candidate/reference stat and invalid thresholds must not load.
    PassThresholds(metric='mse', stat='mean', tasks={'t0': 0.02},
                   config_fingerprint='fp').save(str(path))
    with pytest.raises(ThresholdsError, match='stat'):
        _attach_pass_thresholds(cfg, cat, store)
    PassThresholds(metric='mse', stat='geomean', tasks={'t0': -1},
                   config_fingerprint='fp').save(str(path))
    with pytest.raises(ThresholdsError, match='有限正数'):
        _attach_pass_thresholds(cfg, cat, store)


def test_rescan_uses_gmean_for_auto_pass():
    cfg = make_cfg(1, al=_al(0.02))
    sch = scheduler_of(cfg)
    initial = sch.advance()
    assert initial['action'] == 'bootstrap'
    assert sch.registry.get('t0').status == TaskStatus.CANDIDATE.value

    class ImprovedEval:
        def evaluate(self, name, split, episode_ids):
            return _metric(vals=[0.001] * len(episode_ids), split=split)

    sch.evaluator = ImprovedEval()
    rows = sch._rescan()
    assert rows[0]['auto_pass'] is True
    rec = sch.registry.get('t0')
    assert rec.status == TaskStatus.PASS.value
    assert rec.current_val_gmean_mse == pytest.approx(0.001)


def test_real_scheduler_unit_gmean_pass():
    cfg = make_cfg(1, al=_al(0.13))
    sch = scheduler_of(cfg)
    sch.run(max_actions=90)
    assert any(x['kind'] == 'PASS' for x in sch.state.transitions)
    rec = sch.registry.get('t0')
    assert rec.ever_passed
    assert rec.best_gmean_mse is not None
    assert any('gmean_mse' in row for row in rec.eval_history)


def test_null_threshold_does_not_consume_attempt():
    cfg = make_cfg(1, al=_al(0.13))
    cfg.auto_learning.pass_thresholds.tasks['t0'] = None
    sch = scheduler_of(cfg)
    event = sch.advance()
    assert event['result'] == 'no_threshold'
    record = sch.registry.get('t0')
    assert record.attempt_count == 0
    assert record.pass_line_usable is False


def test_review_emits_gmean_fields_in_event():
    cfg = _al(0.02)
    rec = TaskRecord(task_name='t0', status=TaskStatus.PASS.value,
                     scout_val_ids=[1, 2], confirm_val_ids=[1, 2],
                     best_nmse=0.1, best_gmean_mse=0.01,
                     current_val_gmean_mse=0.01, current_val_nmse=0.1)
    class FixedEval:
        def evaluate(self, *args):
            return _metric(vals=(0.03, 0.04), split='review')
    out = Reviewer(cfg, FixedEval()).review(TaskRegistry({'t0': rec}), global_step=22)[0]
    row = out.to_row()
    assert row['scout_gmean_mse'] is not None
    assert row['confirm_gmean_mse'] is not None
    assert row['best_gmean_mse'] == 0.01


def test_gmean_resume_fingerprint_checks_table_contents():
    from lingbotvla.auto_learning.state.persistence import config_fingerprint, diff_config
    cfg = _al(0.02)
    a = config_fingerprint(cfg, ['t0'])
    assert a['gmean_thresholds_sha256']
    cfg.pass_thresholds.tasks['t0'] = 0.03  # same file path, different PASS meaning
    b = config_fingerprint(cfg, ['t0'])
    assert any(key == 'gmean_thresholds_sha256' for key, _, _ in diff_config(a, b))
    legacy = default_al()
    assert 'gmean_thresholds_sha256' not in config_fingerprint(legacy, ['t0'])


def test_tensorboard_emits_candidate_gmean_and_threshold_ratio():
    cfg = make_cfg(1, al=_al(0.13))
    sch = scheduler_of(cfg)
    class Collector:
        def __init__(self):
            self.metrics = []
        def log_metrics(self, step, name, value):
            self.metrics.append((name, value))
        def log_text(self, *args):
            pass
        def log_event(self, *args):
            pass
    collector = Collector()
    sch.logger = collector
    sch.run(max_actions=90)
    tags = dict(collector.metrics)
    assert 'task/t0/active_val_gmean_mse' in tags
    assert 'task/t0/pass_threshold_gmean_mse' in tags
    assert 'current_skill/val_gmean_mse' in tags
    assert 'current_skill/pass_threshold_gmean_mse' in tags
    assert 'current_skill/val_to_pass_threshold' in tags
