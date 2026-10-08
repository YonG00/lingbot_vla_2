"""Open-loop low-cost workflow: exact cadence, resume, warning-only CV, input/output safety."""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import Mock

import pytest

from al_fixtures import make_cfg, scheduler_of
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler

# 测试必须能任意 CWD 运行（2026-10-08 审查 D6）：配置路径一律锚定仓库根。
ROOT = Path(__file__).resolve().parents[1]
from lingbotvla.auto_learning.state import persistence
from lingbotvla.auto_learning.testing import fake_tasks as ft
from lingbotvla.auto_learning.tools.build_gmean_thresholds import build_thresholds, main
from lingbotvla.auto_learning.baseline import BaselineStore, FixedBaseline, task_fingerprint
from lingbotvla.auto_learning.decision.thresholds import PassThresholds, ThresholdsError
from lingbotvla.auto_learning.types import TaskStatus


def _scheduler(*, scout_confirm_enabled=False, n=3):
    al = AutoLearningConfig(
        scout_confirm_enabled=scout_confirm_enabled,
        rescan_every_n_task_switches=n,
    )
    return scheduler_of(make_cfg(3, al=al))


def _store(tmp_path, count=2):
    path = tmp_path / 'baseline.json'
    store = BaselineStore(path=str(path), config_fingerprint='testfp')
    for i in range(count):
        store.put(FixedBaseline(task=f't{i}', mse=0.25, mu=(0,), fingerprint=task_fingerprint('testfp', f'sha{i}')))
    store.save()
    return BaselineStore.load(str(path))


def _ref():
    # t1 CV ~3, t0 CV = 0
    return {'t0': [0.001] * 10, 't1': [1e-5] * 9 + [0.2]}


def test_default_legacy_behavior_is_preserved():
    cfg = AutoLearningConfig()
    assert cfg.scout_confirm_enabled is True
    assert cfg.rescan_every_n_task_switches == 1


def test_formal_profile_enables_approved_cpu_behaviors_without_enabling_uncalibrated_pass():
    import yaml
    with open(ROOT / 'configs/auto_learning/formal_50task_4pass.yaml', encoding='utf-8') as f:
        al = AutoLearningConfig.from_dict(yaml.safe_load(f))
    assert al.global_scout_val_trajs == 2
    assert al.scout_confirm_enabled is False
    assert al.rescan_every_n_task_switches == 3
    assert al.review_after_task_transitions == 2
    assert al.pass_metric == 'nmse', 'GMean must not silently become official before paired calibration'


@pytest.mark.parametrize('value', [0, -1, 1.5, True, False, '3', None])
def test_invalid_rescan_cadence_is_rejected(value):
    with pytest.raises(ValueError, match='rescan_every_n_task_switches'):
        AutoLearningConfig(rescan_every_n_task_switches=value)


@pytest.mark.parametrize('value', [0, 1, None, 'false'])
def test_invalid_scout_confirm_setting_is_rejected(value):
    with pytest.raises(ValueError, match='scout_confirm_enabled'):
        AutoLearningConfig(scout_confirm_enabled=value)


def test_bootstrap_direct_scout_pass_skips_confirm_and_preserves_pass_metadata():
    al = AutoLearningConfig(pass_nmse=0.3, scout_confirm_enabled=False)
    cfg = ft.make_cfg([
        ft._spec('a', curve=[0.45, 0.22], scout_nmse_override=0.20, confirm_nmse_override=0.42),
    ], al=al)
    sched = scheduler_of(cfg)
    ev = sched.advance()
    rec = sched.registry.get('a')
    assert ev['result'] == 'pass'
    assert ev['pass_source'] == 'scout_direct'
    assert 'confirm_nmse' not in ev
    assert rec.status == TaskStatus.PASS.value
    assert rec.best_nmse == pytest.approx(0.20)
    assert rec.current_val_mse is not None
    assert not any(x['kind'] == 'confirm' for x in rec.eval_history)
    assert 'a' in sched.replay_plan().tasks


def test_legacy_confirm_still_rejects_false_scout_pass():
    al = AutoLearningConfig(pass_nmse=0.3, scout_confirm_enabled=True)
    cfg = ft.make_cfg([
        ft._spec('a', curve=[0.45, 0.22], scout_nmse_override=0.20, confirm_nmse_override=0.42),
    ], al=al)
    sched = scheduler_of(cfg)
    ev = sched.advance()
    assert ev['result'] == 'confirm_failed'
    assert sched.registry.get('a').status == TaskStatus.CANDIDATE.value


def test_full_rescan_every_three_actual_task_switches_not_review_reopens():
    sched = _scheduler(n=3)
    sched._rescan = Mock(return_value=[])
    for k in range(2):
        sched._after_transition(f't{k}', 'DEFER', 'test')
    assert sched.state.task_switch_count == 2
    assert sched.state.full_rescan_count == 0
    assert sched._rescan.call_count == 0
    # Review also raises transition_count, but must not count as a training task switch.
    sched.state.transition_count += 2
    sched._after_transition('t2', 'PASS', 'test')
    assert sched.state.task_switch_count == 3
    assert sched.state.full_rescan_count == 1
    assert sched._rescan.call_count == 1
    for k in range(3, 6):
        sched._after_transition(f't{k%3}', 'DEFER', 'test')
    assert sched.state.full_rescan_count == 2
    assert sched._rescan.call_count == 2


def test_disabled_rescan_stays_disabled_even_on_third_switch():
    sched = _scheduler(n=3)
    sched.al.rescan_candidates_after_transition = False
    sched._rescan = Mock(return_value=[])
    for _ in range(6):
        sched._after_transition('t0', 'DEFER', 'test')
    assert sched.state.task_switch_count == 6
    assert sched.state.full_rescan_count == 0
    sched._rescan.assert_not_called()


def test_checkpoint_restores_cadence_phase(tmp_path):
    sched = _scheduler(n=3)
    sched._rescan = Mock(return_value=[])
    for _ in range(2):
        sched._after_transition('t0', 'DEFER', 'test')
    save = tmp_path / 'state.json'
    persistence.save_state(sched, str(save))
    resumed = _scheduler(n=3)
    resumed._rescan = Mock(return_value=[])
    persistence.load_state(resumed, str(save))
    assert resumed.state.task_switch_count == 2
    resumed._after_transition('t1', 'PASS', 'test')
    resumed._rescan.assert_called_once_with()
    assert resumed.state.full_rescan_count == 1


def test_rescan_auto_pass_uses_two_trajectory_scout_without_confirm():
    sched = _scheduler(n=3)
    # Bootstrap all first so candidates can be re-scanned.
    while sched.state.bootstrap_queue:
        sched.advance()
    cand = next((r for r in sched.registry if r.status == TaskStatus.CANDIDATE.value), None)
    if cand is None:
        pytest.skip('fixture accidentally auto-passed all tasks')
    # Inject a true, positive, valid MSE/NMSE passing observation and ensure confirm not called.
    original = sched.evaluator.evaluate
    seen = []
    def evaluate(task, split, ids):
        seen.append(split)
        out = original(task, split, ids)
        if task == cand.task_name and split == 'scout':
            out.nmse = 0.01
            out.mse = 0.0025
            out.metric_valid = True
        return out
    sched.evaluator.evaluate = evaluate
    rows = sched._rescan(only=[cand.task_name])
    assert rows[0]['auto_pass'] is True
    assert rows[0]['pass_source'] == 'scout_direct'
    assert sched.registry.get(cand.task_name).status == TaskStatus.PASS.value
    assert 'confirm' not in seen


def test_warn_high_cv_keeps_threshold_and_records_warning(tmp_path):
    table, diagnostics = build_thresholds(_ref(), _store(tmp_path), multiplier=20,
                                          high_cv_policy='warn', reference='50k')
    assert table.n_usable == 2
    assert all(isinstance(x, float) and math.isfinite(x) for x in table.tasks.values())
    assert diagnostics['t0']['high_cv_warning'] is False
    assert diagnostics['t1']['high_cv_warning'] is True
    assert diagnostics['t1']['status'] == 'high_cv_warning'
    assert 0 < table.tasks['t1'] < 0.25
    assert table.tasks['t1'] == pytest.approx(diagnostics['t1']['effective_line'])


def test_high_cv_legacy_policy_still_fail_closes(tmp_path):
    table, diagnostics = build_thresholds(_ref(), _store(tmp_path), multiplier=20,
                                          high_cv_policy='null', reference='50k')
    assert table.tasks['t1'] is None
    assert diagnostics['t1']['high_cv_warning'] is True
    with pytest.raises(ThresholdsError, match='CV'):
        build_thresholds(_ref(), _store(tmp_path), multiplier=20,
                         high_cv_policy='error', reference='50k')


def test_warn_cli_creates_fully_usable_and_validated_table(tmp_path):
    store = _store(tmp_path)
    src = tmp_path / 'refs.jsonl'
    with src.open('w', encoding='utf-8') as f:
        for task, mses in _ref().items():
            for traj, mse in enumerate(mses):
                f.write(json.dumps({'task': task, 'traj': traj, 'mse': mse}) + '\n')
    dest = tmp_path / 'pass.json'
    assert main(['--ref-per-traj', str(src), '--baseline', store.path,
                 '--reference', '50k', '--multiplier', '20', '--high-cv-policy', 'warn',
                 '-o', str(dest)]) == 0
    obj = PassThresholds.load(str(dest), expect_fingerprint=store.config_fingerprint,
                              require_metric='mse')
    assert obj.n_usable == 2
    raw = json.loads(dest.read_text(encoding='utf-8'))
    assert raw['calibration']['high_cv_policy'] == 'warn'
    assert raw['calibration']['tasks']['t1']['high_cv_warning'] is True


def test_cli_directory_as_output_returns_meaningful_error_and_no_tempfile(tmp_path):
    store = _store(tmp_path)
    src = tmp_path / 'refs.jsonl'
    with src.open('w', encoding='utf-8') as f:
        for task, mses in _ref().items():
            for traj, mse in enumerate(mses):
                f.write(json.dumps({'task': task, 'traj': traj, 'mse': mse}) + '\n')
    output_dir = tmp_path / 'existing_dir'
    output_dir.mkdir()
    with pytest.raises(ThresholdsError, match='输出路径是目录'):
        main(['--ref-per-traj', str(src), '--baseline', store.path,
              '--reference', '50k', '--multiplier', '20', '-o', str(output_dir)])
    assert list(output_dir.iterdir()) == []
