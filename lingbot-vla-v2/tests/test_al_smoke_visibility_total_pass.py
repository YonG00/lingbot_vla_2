"""Latest repo contract: total PASS, log transparency, and no-save smoke."""
from __future__ import annotations

import ast
import pathlib

import pytest

from al_fixtures import scheduler_of
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from lingbotvla.auto_learning.real.build import SchedulerLoggerAdapter
from lingbotvla.auto_learning.state.persistence import config_fingerprint
from lingbotvla.auto_learning.testing.fake_tasks import make_cfg, already_known, easy_pass
from lingbotvla.auto_learning.types import TaskStatus

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _al(**kw):
    cfg = dict(eval_interval_steps=5, min_steps_before_defer=5,
               defer_retry_steps=5, scout_confirm_enabled=False,
               review_after_task_transitions=0, rescan_candidates_after_transition=False)
    cfg.update(kw)
    return AutoLearningConfig(**cfg)


def test_total_pass_four_with_two_bootstrap_passes_only_trains_two_new():
    cfg = make_cfg([already_known("K1"), already_known("K2"),
                    easy_pass("L1"), easy_pass("L2"), easy_pass("L3")],
                   al=_al(target_total_passed_tasks=4))
    sched = scheduler_of(cfg)
    sched.run(max_actions=400)
    assert sched.state.finished
    assert sched.state.stop_reason == 'target_total_passed_reached(4)'
    assert len(sched.registry.by_status(TaskStatus.PASS)) == 4
    assert set(sched.state.auto_passed) == {"K1", "K2"}
    assert set(sched.state.bootstrap_passed) == {"K1", "K2"}
    assert len(sched.state.newly_passed) == 2
    assert len(sched.state.trained_tasks) == 2
    assert sched.final_report()["current_total_pass_count"] == 4
    assert sched.final_report()["bootstrap_pass_count"] == 2


def test_all_bootstrap_pass_stops_without_select_or_hardness():
    cfg = make_cfg([already_known(f"K{i}") for i in range(4)],
                   al=_al(target_total_passed_tasks=4))
    sched = scheduler_of(cfg)
    sched.run(max_actions=100)
    assert sched.state.stop_reason == 'target_total_passed_reached(4)'
    assert sched.state.units_run == 0
    assert sched.state.trained_tasks == []
    assert sched.scans == {}
    assert not any(ev.get('action') == 'select' for ev in sched.events)


def test_one_initial_pass_counts_and_target_reached_only_after_new_pass():
    cfg = make_cfg([already_known('K'), easy_pass('L1'), easy_pass('L2')],
                   al=_al(target_total_passed_tasks=2))
    sched = scheduler_of(cfg)
    sched.run(max_actions=100)
    assert sched.state.stop_reason == 'target_total_passed_reached(2)'
    assert sched.state.auto_passed == ['K']
    assert len(sched.state.newly_passed) == 1
    assert len(sched.state.trained_tasks) == 1


def test_reopen_subtracts_from_current_pass_count_not_ever_passed():
    cfg = make_cfg([already_known('K'), easy_pass('L')],
                   al=_al(target_total_passed_tasks=2))
    sched = scheduler_of(cfg)
    while sched.state.bootstrap_queue:
        sched.advance()
    assert sched._current_total_passed() == 1
    sched.registry.get('K').set_status(TaskStatus.CANDIDATE, 'simulate review reopen')
    assert sched._current_total_passed() == 0
    assert sched.next_action() == 'select'
    assert not sched._total_pass_target_reached()


def test_legacy_quota_unchanged_and_both_goals_have_or_semantics():
    cfg = make_cfg([already_known('K'), easy_pass('L1'), easy_pass('L2')],
                   al=_al(target_total_passed_tasks=3,
                          max_new_tasks_passed_this_run=1))
    sched = scheduler_of(cfg)
    sched.run(max_actions=100)
    assert sched.state.stop_reason == 'max_new_tasks_passed_reached(1)'
    assert len(sched.state.newly_passed) == 1
    assert sched._current_total_passed() == 2


@pytest.mark.parametrize('v', [0, -1, 1.5, True])
def test_target_total_pass_validation_bad_values(v):
    if v in (1.5, True):
        # defensive: type should be strictly integer, never float or bool
        with pytest.raises((ValueError, TypeError)):
            AutoLearningConfig(target_total_passed_tasks=v)
    else:
        with pytest.raises((ValueError, TypeError)):
            AutoLearningConfig(target_total_passed_tasks=v)


def test_target_total_pass_semantic_fingerprint():
    assert config_fingerprint(_al(target_total_passed_tasks=4), ['A'])['target_total_passed_tasks'] == 4
    assert 'target_total_passed_tasks' not in config_fingerprint(_al(target_total_passed_tasks=None), ['A'])


class Writer:
    def __init__(self):
        self.scalars=[]
        self.texts=[]
    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))
    def add_text(self, tag, text, step):
        self.texts.append((tag, text, step))


class RepoLogger:
    def info_rank0(self, *a, **k): pass
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


def test_current_task_text_and_per_task_unit_loss_at_absolute_steps(tmp_path):
    w = Writer()
    lg = SchedulerLoggerAdapter(RepoLogger(), writer=w, event_path=str(tmp_path/'events.jsonl'))
    lg.set_tb_step_offset(train_global_step=500, al_global_step=0)
    cfg = make_cfg([easy_pass('alpha'), easy_pass('beta')], al=_al())
    sched = scheduler_of(cfg, logger=lg)
    sched.run(max_actions=20)
    task_unit = [(tag, v, step) for tag, v, step in w.scalars if tag.endswith('/unit_loss') and tag.startswith('task/')]
    assert task_unit, w.scalars
    assert all(step >= 505 for _, _, step in task_unit)
    assert any(tag == 'curriculum/current_task_name' and task in ('alpha', 'beta') and step == 500 for tag, task, step in w.texts)
    assert any(tag == 'curriculum/passed_tasks' for tag, _, _ in w.scalars)
    assert any(tag == 'replay/available_tasks' for tag, _, _ in w.scalars)
    assert any(tag == 'sampling/total_samples' for tag, _, _ in w.scalars)
    assert any(tag == 'diagnostics/current_task_val_nmse' for tag, _, _ in w.scalars)
    assert (tmp_path/'events.jsonl').exists()
    assert '"kind": "text"' in (tmp_path/'events.jsonl').read_text()


def test_hardness_fraction_10_percent_defaults_and_files():
    import math
    assert AutoLearningConfig().hardness_probe_fraction == 0.10
    assert math.ceil(40 * 0.10) == 4
    for f in ('formal_50task_4pass.yaml','smoke_gbs4_4task_tb5.yaml'):
        text = (ROOT/'configs/auto_learning'/f).read_text()
        assert 'hardness_probe_fraction: 0.10' in text


def test_train_save_gates_are_explicit_and_stop_closes_writer():
    src = (ROOT/'tasks/vla/train_lingbotvla.py').read_text()
    ast.parse(src)
    for fragment in ('args.train.smoke_no_checkpoint',
                     'not args.train.smoke_no_checkpoint and',
                     'if args.train.smoke_no_checkpoint:',
                     'args.train.smoke_no_checkpoint or not args.train.save_hf_weights',
                     'writer.close()'):
        assert fragment in src
    # Separate period-save, terminal-save, epoch-save guards.
    assert src.count('not args.train.smoke_no_checkpoint') >= 2


def test_launcher_smoke_override_and_fail_closed_resume():
    src = (ROOT/'experiment/robotwin/al_50task_bf16.sh').read_text()
    assert 'SMOKE_NO_CHECKPOINT=${SMOKE_NO_CHECKPOINT:-0}' in src
    assert 'SMOKE_NO_CHECKPOINT=1 不支持 RESUME' in src
    assert '--train.smoke_no_checkpoint' in src
    assert '--train.save_hf_weights' in src
    assert 'SAVE_EVERY=0' in src
    assert 'PRUNE=0' in src


def test_launcher_real_dry_run_no_checkpoint_is_consistent(tmp_path):
    import json
    import os
    import subprocess
    import sys
    split = tmp_path / 'split'
    split.mkdir()
    (split/'manifest.json').write_text(json.dumps({'tasks': {'alpha': {'train_frames': 100}}}))
    (split/'combined.train_ids.json').write_text('{}')
    (split/'task_baseline.json').write_text('{}')
    qwen = tmp_path/'qwen'
    qwen.mkdir()
    yaml = ROOT/'configs/auto_learning/smoke_gbs4_4task_tb5.yaml'
    env = {**os.environ, 'PY': sys.executable, 'SPLIT_DIR': str(split),
           'QWEN3VL': str(qwen), 'AL_CFG': str(yaml),
           'SMOKE_NO_CHECKPOINT': '1', 'DRY_RUN': '1',
           'MAX_STEPS': '510', 'STEP_OFFSET': '500',
           'SAVE_EVERY': '10', 'PRUNE': '1', 'GAS':'4', 'MICRO':'1',
           'N_GPU': '1', 'TB':'0', 'TRAIN_OUT': str(tmp_path/'output')}
    proc = subprocess.run(['bash', str(ROOT/'experiment/robotwin/al_50task_bf16.sh')],
                          env=env, text=True, capture_output=True, timeout=15)
    assert proc.returncode == 0, proc.stdout + '\n' + proc.stderr
    out = proc.stdout
    assert '--train.save_steps       0' in out
    assert '--train.save_epochs      0' in out
    assert '--train.save_hf_weights  false' in out
    assert '--train.smoke_no_checkpoint true' in out
    assert '--train.disk_guard       false' in out
    assert 'NO_CHECKPOINT' in out
    assert 'PRUNE=0' in out


def test_launcher_rejects_resume_with_no_checkpoint_before_touching_disk(tmp_path):
    import os
    import subprocess
    env = {**os.environ, 'SMOKE_NO_CHECKPOINT':'1', 'RESUME':'1',
           'TRAIN_OUT':str(tmp_path/'out')}
    p = subprocess.run(['bash', str(ROOT/'experiment/robotwin/al_50task_bf16.sh')],
                       env=env, text=True, capture_output=True, timeout=10)
    assert p.returncode != 0
    assert '不支持 RESUME' in p.stderr
    assert not (tmp_path/'out').exists()


def test_bootstrap_zero_step_and_stop_after_unit_boundary_training_contract():
    """CPU AST contract: check AL finish before DataLoader iterator and step++.

    Full optimizer/GPU behavior must be verified on the remote host.
    """
    source = (ROOT/'tasks/vla/train_lingbotvla.py').read_text()
    assert '_al_initial = _al_hook.on_step_begin(0)' in source
    assert 'data_iterator = (iter(()) if _al_hook is not None' in source
    loop_start = source.index('for epoch_step in range(start_step, args.train.train_steps):')
    stop_guard = source.index('if _al_hook is not None and _al_hook.scheduler.state.finished:', loop_start)
    increment = source.index('global_step += 1', loop_start)
    assert loop_start < stop_guard < increment
    ast.parse(source)
