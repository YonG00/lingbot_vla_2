"""Prepare an isolated optional-method run from the frozen official run config."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment.paired_control.methods import validate_flags
from lingbotvla.auto_learning.sampling.target_change_window import cache_path


def prepare(source, out, *, ee_loss=False, window_sampling=False, ln_fm=False,
            ee_position=3., ee_orientation=.8, ee_gripper=1.5):
    flags = validate_flags(ee_loss, window_sampling, ln_fm)
    assert not out.exists()
    config = json.loads((source / 'trainer_config.json').read_text())
    t = config['train']
    assert 'global_step_9680' in config['model']['model_path']
    assert t['max_steps'] == 500 and t['save_steps'] == 250
    assert t['micro_batch_size'] == t['global_batch_size'] == 28 and t['gradient_accumulation_steps'] == 1
    assert t['enable_fp32'] and t['enable_mixed_precision'] and t['use_compile']
    assert config['data']['num_workers'] == 8
    t.update(output_dir=str(out), auto_learning=None, auto_learning_manifest=None,
             low_noise_fm_enabled=ln_fm, ee_fk_loss_enabled=ee_loss, ee_fk_eval_enabled=False,
             ee_fk_spec_path=str(ROOT / 'configs/kinematics/robotwin_aloha_agilex_ee.json'),
             ee_fk_position_loss_weight=ee_position if ee_loss else 0.,
             ee_fk_orientation_loss_weight=ee_orientation if ee_loss else 0.,
             ee_fk_gripper_loss_weight=ee_gripper if ee_loss else 0.)
    config['data'].update(train_path=str(out / 'datasets.txt'), episode_ids_file=str(out / 'train_ids.json'))
    out.mkdir(parents=True)
    for name in ('datasets.txt', 'train_ids.json'):
        shutil.copy2(source / name, out / name)
    (out / 'trainer_config.json').write_text(json.dumps(config, indent=2))
    (out / 'methods.json').write_text(json.dumps(dict(**flags, window_weight=2.,
        source_run=str(source), task='adjust_bottle', window_cache=str(cache_path(out / 'window_cache', 'adjust_bottle')),
        source_config_sha256=hashlib.sha256((source / 'trainer_config.json').read_bytes()).hexdigest()), indent=2))
    # Static cache preparation only; this is never passed to the AL training hook.
    cache_config = dict(enabled=False, task_names=['adjust_bottle'], learning_task_names=['adjust_bottle'],
        replay_task_names=[], train_all_episodes=True, task_scan_enabled=False, pass_transition_enabled=False,
        import_pool_state=False, review_after_task_transitions=0, rescan_candidates_after_transition=False,
        early_defer_on_overfit=False, max_global_steps=500, min_steps_before_defer=10000,
        target_change_cache_dir=str(out / 'window_cache'), sample_scan_method='target_change_window',
        target_change_dataset_root=Path((source / 'datasets.txt').read_text().split()[1]).as_posix(),
        target_change_embodiment_config=str(ROOT.parent / 'RoboTwin-lingbot/assets/embodiments/aloha-agilex/config.yml'),
        target_change_wrist_joint_names=[['fl_joint6'], ['fr_joint6']], target_change_window_weight=2., seed=t['seed'])
    (out / 'window_preparation.json').write_text(json.dumps(cache_config, indent=2))
    switches = [('--' if value else '--no-') + key.replace('_', '-') for key, value in flags.items()]
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=1', '--master_port=29541',
               str(ROOT / 'experiment/paired_control/train.py'), '--config', str(out / 'trainer_config.json'),
               '--methods', str(out / 'methods.json'), *switches]
    plan = dict(source_model=config['model']['model_path'], code_root=str(ROOT), flags=flags,
                train_command=command, expected_steps=500, save_steps=[250, 500],
                source_config_sha256=hashlib.sha256((source / 'trainer_config.json').read_bytes()).hexdigest())
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    (out / 'train.sh').write_text('#!/bin/bash\nset -euo pipefail\nexport AL_DYNAMO_CACHE_SIZE_LIMIT=0\ncd '
                                + shlex.quote(str(ROOT)) + '\nexec ' + shlex.join(command) + '\n')
    print(json.dumps(plan, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    for name in ('ee-loss', 'window-sampling', 'ln-fm'):
        p.add_argument('--' + name, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--ee-position', type=float, default=3.)
    p.add_argument('--ee-orientation', type=float, default=.8)
    p.add_argument('--ee-gripper', type=float, default=1.5)
    a = p.parse_args()
    prepare(a.source_run, a.out, ee_loss=a.ee_loss, window_sampling=a.window_sampling, ln_fm=a.ln_fm,
            ee_position=a.ee_position, ee_orientation=a.ee_orientation, ee_gripper=a.ee_gripper)


if __name__ == '__main__':
    main()
