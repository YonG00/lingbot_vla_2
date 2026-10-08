"""CPU-only checks of the 96G acceptance harness; never launches real training."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/gpu96_acceptance.py'
spec = importlib.util.spec_from_file_location('gpu96_acceptance', SCRIPT)
assert spec and spec.loader
accept = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accept)


def test_training_benchmark_has_real_optim_steps_no_ckpts(tmp_path):
    a = accept.build_parser().parse_args(['bench', '--micro', '28', '--gas', '1',
                                         '--out-root', str(tmp_path)])
    a.master_port = 40007
    a.output = tmp_path/'micro28_gas1_bf16_compileoff'
    cmd = accept.trainer_command(a)
    text = ' '.join(cmd)
    assert 'tasks/vla/train_lingbotvla.py' in cmd
    assert '--train.global_batch_size 28' in text
    assert '--train.enable_mixed_precision false' in text
    assert '--train.smoke_no_checkpoint true' in text
    assert '--train.hf_pass_interval 0' in text
    assert '--train.save_steps 0' in text
    assert '--train.auto_learning' not in cmd  # no 50-task Bootstrap during speed test
    assert '--train.max_steps 525' in text


def test_default_is_dry_plan_without_gpu(monkeypatch, tmp_path, capsys):
    def no_gpu():
        raise AssertionError('GPU must not be queried for dry-run')
    monkeypatch.setattr(accept, 'require_gpu', no_gpu)
    assert accept.main(['bench', '--out-root', str(tmp_path)]) == 0
    assert not list(tmp_path.iterdir())
    assert 'PLAN ONLY' in capsys.readouterr().out


def test_parse_steps_fails_on_short_or_invalid():
    steps = ''.join(f'Step {i}/999, Loss {1.0-i/100:.4f}, StepTime {i/10:.3f}s,\n'
                    for i in range(1, 8))
    got = accept.parse_steps(steps, warmup=2, measure=4, gbs=28)
    assert got['steps_seen'] == 7
    assert got['measured_step_range'] == [4, 7]
    assert got['samples_per_second'] == pytest.approx(4*28/(.4+.5+.6+.7), abs=.0001)
    with pytest.raises(ValueError, match='only'):
        accept.parse_steps(steps, warmup=5, measure=4, gbs=28)
    with pytest.raises(ValueError, match='nonfinite'):
        accept.parse_steps(steps+'Step 8/999, Loss nan, StepTime 0.8s\n', warmup=2, measure=4, gbs=28)


def test_sampler_ratio_gas_is_optimizer_level(capsys):
    assert accept.main(['ratio','--micro','28','--gas','2','--dp','1']) == 0
    s = capsys.readouterr().out
    assert '"new_total": 39' in s
    assert '"replay_total": 17' in s
    assert accept.main(['ratio','--micro','28','--gas','1','--dp','4']) == 0
    s = capsys.readouterr().out
    assert '"new_total": 78' in s
    assert '"replay_total": 34' in s
    assert accept.main(['ratio','--micro','28','--gas','1','--dp','1']) == 0
    assert '"new_total": 20' in capsys.readouterr().out


def test_eval_parse_fail_closed_missing_extra_duplicates_bad():
    ids = [1,2]
    good = 'MSE for trajectory 1: 0.001, MAE: 0.2\nMSE for trajectory 2: 0.004, MAE: 0.1'
    assert accept.extract_trajectory_mse(good, ids) == {1: .001, 2:.004}
    with pytest.raises(ValueError, match='ID mismatch'):
        accept.extract_trajectory_mse(good.splitlines()[0], ids)
    with pytest.raises(ValueError, match='duplicate'):
        accept.extract_trajectory_mse(good+'\n'+good.splitlines()[0], ids)
    with pytest.raises(ValueError, match='invalid'):
        accept.extract_trajectory_mse(good.replace('0.004','nan'), ids)
    with pytest.raises(ValueError, match='invalid'):
        accept.extract_trajectory_mse(good.replace('0.004','-0.2'), ids)


def test_task_list_requires_ten_matched_ids(tmp_path):
    split = tmp_path/'splits'; split.mkdir()
    (split/'task1.val_ids.json').write_text(json.dumps(list(range(10))))
    (split/'task2.val_ids.json').write_text(json.dumps(list(range(100,110))))
    got = accept.load_task_ids(split, ['task1','task2'])
    assert got['task1'] == list(range(10))
    with pytest.raises(ValueError, match='expected >='):
        accept.load_task_ids(split, ['task1'], 11)
    with pytest.raises(ValueError, match='unique'):
        accept.load_task_ids(split, ['task1','task1'])
    (split/'task2.val_ids.json').write_text(json.dumps(list(range(100,109))+[0]))
    with pytest.raises(ValueError, match='shared'):
        accept.load_task_ids(split, ['task1','task2'])


def test_paired_gmean_2_4_10_and_flip():
    ids = {'task': list(range(10))}
    bf = {i:.25 for i in range(10)}
    f32 = {i:.01 for i in range(10)}
    rows = accept.gmean_summary(ids, {'bf16':bf,'fp32':f32}, {'task':.1})
    assert [x['n_traj'] for x in rows] == [2,4,10]
    assert all(x['bf16_gmean_mse'] == pytest.approx(.25) for x in rows)
    assert all(x['fp32_gmean_mse'] == pytest.approx(.01) for x in rows)
    assert all(x['pass_flip'] is True for x in rows)
    assert rows[0]['traj_ids'] == [0,1]


def test_threshold_requires_geomean_stat(tmp_path):
    p=tmp_path/'threshold.json'
    p.write_text(json.dumps({'metric':'mse','stat':'mean','tasks':{'task':.1}}))
    with pytest.raises(ValueError, match='stat=geomean'):
        accept.load_thresholds(str(p))
    p.write_text(json.dumps({'metric':'mse','stat':'geomean','tasks':{'task':.1}}))
    assert accept.load_thresholds(str(p)) == {'task':.1}


def test_no_overwrite_run_directory(tmp_path):
    path = accept.fresh_dir(tmp_path/'case')
    with pytest.raises(FileExistsError):
        accept.fresh_dir(path)


def test_hf_plan_no_gpu(capsys):
    assert accept.main(['hf-plan']) == 0
    t = capsys.readouterr().out
    assert 'hf_direct_export_acceptance.py' in t
    assert 'skip_final_save_on_max_steps true' in t


def test_real_ratio_evidence_requires_consumed_replay_and_consistent_total():
    data = [
        {'kind':'event','action':'train_unit','step':503,'batches_built':3,
         'samples_seen':168,'old_tasks':['click_bell']},
        {'kind':'metric','name':'sampling/replay_samples_per_unit','step':503,'value':51},
    ]
    r=accept.ratio_unit_evidence(data,gbs=56,new_ratio=.7)
    assert r['status']=='PASS'
    assert r['units'][0]['actual_new']==117
    assert r['units'][0]['expected_new_when_pool_nonempty']==117
    for bad in ([], [data[0]], [data[0],{**data[1],'value':48}],
                [{**data[0],'samples_seen':167},data[1]]):
        assert accept.ratio_unit_evidence(bad,gbs=56,new_ratio=.7)['status']=='BLOCKED'


def test_hf_plan_only_cannot_query_gpu(monkeypatch,tmp_path):
    monkeypatch.setattr(accept,'require_gpu',lambda: (_ for _ in ()).throw(AssertionError('GPU touched')))
    assert accept.main(['hf','--out-root',str(tmp_path)]) == 0
    assert accept.main(['ratio-gpu','--out-root',str(tmp_path)]) == 0
    assert not list(tmp_path.iterdir())


def test_hf_acceptance_mismatch_fails_exit_code(tmp_path):
    import importlib.util
    import torch
    from safetensors.torch import save_file
    p=SCRIPT.parent/'hf_direct_export_acceptance.py'
    sp=importlib.util.spec_from_file_location('hf_acceptance_test_module',p)
    mod=importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    target=tmp_path/'hf_milestones/global_step_501/hf_ckpt'
    target.mkdir(parents=True)
    mod.CAP['snapshot']={'weight':torch.tensor([1.,2.])}
    save_file({'weight':torch.tensor([1.,2.])},str(target/'model.safetensors'))
    assert mod._verify_export(str(tmp_path))==0
    save_file({'weight':torch.tensor([1.,3.])},str(target/'model.safetensors'))
    assert mod._verify_export(str(tmp_path))==1
