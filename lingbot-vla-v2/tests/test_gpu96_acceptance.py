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


def test_ratio_target_fix_builds_isolated_config_without_mutating_source():
    raw={'auto_learning': {'pass_metric':'nmse','task_names':['click_bell','click_alarmclock'],
                           'target_total_passed_tasks':None, 'hardness_probe_fraction':.3}}
    got=accept.build_ratio_smoke_config(raw,target=2,max_global_steps=509)
    assert got['target_total_passed_tasks']==2
    assert got['max_global_steps']==509
    assert got['new_ratio']==.7
    assert got['hardness_probe_fraction']==.1
    assert raw['auto_learning']['target_total_passed_tasks'] is None
    assert 'new_ratio' not in raw['auto_learning']


@pytest.mark.parametrize('target', [0,-1,3])
def test_ratio_target_fix_rejects_invalid_targets(target):
    with pytest.raises(ValueError,match='target'):
        accept.build_ratio_smoke_config({'task_names':['a','b']},target=target,max_global_steps=509)


def test_ratio_target_dry_plan_never_queries_gpu(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(accept,'require_gpu',lambda: (_ for _ in ()).throw(AssertionError('GPU touched')))
    assert accept.main(['ratio-gpu','--micro','24','--gas','1',
                        '--target-total-passed-tasks','2','--steps','9',
                        '--out-root',str(tmp_path)])==0
    assert not list(tmp_path.iterdir())
    out=capsys.readouterr().out
    assert 'PLAN ONLY' in out and 'TARGET: 2' in out
    assert 'target2_steps9' in out


def test_ratio_target_diagnosis_is_fail_closed():
    assert accept.ratio_unit_evidence([],gbs=24,new_ratio=.7)['diagnosis']=='no_consumed_train_unit'
    unit={'kind':'event','action':'train_unit','step':503,'batches_built':1,
          'samples_seen':24,'old_tasks':[]}
    r=accept.ratio_unit_evidence([unit],gbs=24,new_ratio=.7)
    assert r['status']=='BLOCKED'
    assert r['diagnosis']=='train_unit_without_replay_pool'


# --------------------------------------------------------------------------- #
# 四任务真实 Replay 验收：显式任务数上限（minimal patch + 可复现性保证）
# --------------------------------------------------------------------------- #
FOUR = ['click_bell', 'click_alarmclock', 'turn_switch', 'put_object_cabinet']


def _four_task_raw():
    return {'auto_learning': {'pass_metric': 'nmse', 'pass_nmse': 1.66,
                              'pass_thresholds_file': None,
                              'task_names': list(FOUR), 'target_total_passed_tasks': 4,
                              'hardness_probe_fraction': 0.33, 'eval_interval_steps': 3}}


def test_ratio_probe_task_cap_is_explicit_and_fail_closed():
    """默认上限仍是 3 ⇒ 4 任务必须显式放开，避免无意中用大配置跑 GPU。"""
    with pytest.raises(ValueError, match=r'1\.\.3 explicitly named tasks'):
        accept.build_ratio_smoke_config(_four_task_raw(), target=4, max_global_steps=515)
    got = accept.build_ratio_smoke_config(_four_task_raw(), target=4, max_global_steps=515,
                                          max_named_tasks=4)
    assert got['task_names'] == FOUR


def test_ratio_probe_raised_cap_preserves_tasks_metric_and_threshold():
    """--al-config 的四任务 / NMSE 指标 / 阈值不得被工具内部生成配置覆盖。"""
    raw = _four_task_raw()
    got = accept.build_ratio_smoke_config(raw, target=4, max_global_steps=515, max_named_tasks=4)
    assert got['task_names'] == FOUR, '四任务必须原样保留且顺序不变'
    assert got['pass_metric'] == 'nmse'
    assert got['pass_nmse'] == 1.66
    assert got['pass_thresholds_file'] is None, '不得引入实验性 MSE/GMean 阈值表'
    assert got['target_total_passed_tasks'] == 4
    assert got['max_global_steps'] == 515
    # 只允许这 3 个字段被工具改写
    changed = {k for k in set(raw['auto_learning']) | set(got) if raw['auto_learning'].get(k) != got.get(k)}
    assert changed == {'hardness_probe_fraction', 'max_global_steps', 'new_ratio'}, changed
    # 源配置不得被就地修改
    assert raw['auto_learning']['target_total_passed_tasks'] == 4
    assert 'new_ratio' not in raw['auto_learning']


@pytest.mark.parametrize('raw,target,kwargs', [
    ({'task_names': list(FOUR), 'pass_metric': 'mse', 'pass_thresholds_file': '/tmp/x.json'}, 4,
     {'max_named_tasks': 4}),                      # 实验 MSE 口径 ⇒ 拒绝
    ({'task_names': list(FOUR)}, 5, {'max_named_tasks': 4}),   # target 超过任务数 ⇒ 拒绝
    ({'task_names': list(FOUR)}, 0, {'max_named_tasks': 4}),   # 非正 target ⇒ 拒绝
    ({'task_names': list(FOUR)}, 4, {'max_named_tasks': 0}),   # 非正上限 ⇒ 拒绝
])
def test_ratio_probe_rejects_bad_metric_target_or_cap(raw, target, kwargs):
    with pytest.raises(ValueError):
        accept.build_ratio_smoke_config({'auto_learning': raw}, target=target,
                                       max_global_steps=515, **kwargs)


def test_ratio_four_task_dry_plan_never_queries_gpu(monkeypatch, tmp_path, capsys):
    """四任务 PLAN ONLY：不得触碰 GPU、不得落盘，且目录名/目标/步数可见。"""
    monkeypatch.setattr(accept, 'require_gpu',
                        lambda: (_ for _ in ()).throw(AssertionError('GPU touched')))
    rc = accept.main(['ratio-gpu', '--micro', '24', '--gas', '1',
                      '--al-config', str(Path(__file__).resolve().parents[1] /
                                         'configs/auto_learning/gpu96_ratio_4task_acceptance.yaml'),
                      '--target-total-passed-tasks', '4', '--max-named-tasks', '4',
                      '--steps', '15', '--out-root', str(tmp_path)])
    assert rc == 0
    assert not list(tmp_path.iterdir()), 'PLAN ONLY 不得创建任何目录'
    out = capsys.readouterr().out
    assert 'PLAN ONLY' in out
    assert 'TARGET: 4' in out and 'MAX OPTIMIZER STEPS: 15' in out
    assert 'target4_steps15' in out
    assert '24 GBS' in out


# --------------------------------------------------------------------------- #
# HF 验收（阶段5）：必须保证里程碑判定点可达 + 判定 fail-closed
# --------------------------------------------------------------------------- #
def test_hf_smoke_config_forces_at_least_one_train_unit_and_keeps_nmse():
    base = {'task_names': ['click_bell', 'click_alarmclock'], 'batch_size': 10}
    cfg = accept.build_hf_smoke_config(base, step_offset=500, steps=3)
    # 必含历史不通过任务（否则 Bootstrap 全 PASS ⇒ 零训练步 ⇒ 判定点不可达）
    assert set(accept.HF_NON_PASSING_TASKS) <= set(cfg['task_names']), cfg['task_names']
    # target = 任务总数 ⇒ 调度器必须尝试未通过任务 ⇒ 至少 1 个 Train Unit
    assert cfg['target_total_passed_tasks'] == len(cfg['task_names'])
    # 口径显式 NMSE 且绝不引入实验性阈值表
    assert cfg['pass_metric'] == 'nmse' and cfg['pass_nmse'] == 1.66
    assert cfg['pass_thresholds_file'] is None
    assert cfg['max_global_steps'] == 503
    # 源配置不被就地修改
    assert base['task_names'] == ['click_bell', 'click_alarmclock']
    with pytest.raises(ValueError):
        accept.build_hf_smoke_config(base, step_offset=500, steps=0)


@pytest.mark.parametrize('kwargs,needle', [
    (dict(returncode=1, reason='exit', n_milestones=1, n_shards=6, size_gib=11.9, leaked_dcp=0,
          units_run=1, steps_seen=3), 'entry_nonzero_exit'),
    (dict(returncode=0, reason='exit', n_milestones=0, n_shards=0, size_gib=0.0, leaked_dcp=0,
          units_run=0, steps_seen=0), 'milestone_decision_point_never_reached'),
    (dict(returncode=0, reason='exit', n_milestones=2, n_shards=6, size_gib=11.9, leaked_dcp=0,
          units_run=1, steps_seen=3), 'exactly_one_hf_milestone'),
    (dict(returncode=0, reason='exit', n_milestones=1, n_shards=0, size_gib=11.9, leaked_dcp=0,
          units_run=1, steps_seen=3), 'no_weight_shard'),
    (dict(returncode=0, reason='exit', n_milestones=1, n_shards=6, size_gib=24.0, leaked_dcp=0,
          units_run=1, steps_seen=3), 'size_out_of_band'),
    (dict(returncode=0, reason='exit', n_milestones=1, n_shards=6, size_gib=11.9, leaked_dcp=1,
          units_run=1, steps_seen=3), 'dcp_leak'),
])
def test_hf_verdict_fail_closed_matrix(kwargs, needle):
    ok, why = accept.hf_verdict(**kwargs)
    assert ok is False and needle in why, (ok, why)


def test_hf_verdict_passes_only_when_everything_holds():
    ok, why = accept.hf_verdict(returncode=0, reason='exit', n_milestones=1, n_shards=6,
                                size_gib=11.9, leaked_dcp=0, units_run=1, steps_seen=3)
    assert ok is True and why == 'verified'


def test_hf_step_evidence_is_fail_closed(tmp_path):
    assert accept.hf_step_evidence(tmp_path) == {'units_run': 0, 'steps_seen': 0}
    (tmp_path / 'auto_learning_events.jsonl').write_text(
        '{"kind":"metric","name":"system/units_run","value":2}\n'
        '{"kind":"metric","name":"system/units_run","value":1}\n', encoding='utf-8')
    (tmp_path / 'train_hf.log').write_text('Step: 501/503 ...\nStep: 503/503 ...\n', encoding='utf-8')
    ev = accept.hf_step_evidence(tmp_path)
    assert ev == {'units_run': 2, 'steps_seen': 503}, ev
