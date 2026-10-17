"""CPU precision lifecycle, real validation wiring and observer contracts."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig, AutoConfig

from lingbotvla.losses.ee_fk import EEFKRuntime, compute_ee_fk_losses, recover_clean_action, _rotation_error
from lingbotvla.losses.ee_metrics import EEErrorAccumulator

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / 'configs/kinematics/robotwin_aloha_agilex_ee.json').read_text())


def actual_method(path, cls, method, namespace):
    tree = ast.parse((ROOT / path).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == method)
    module = ast.Module(body=[node], type_ignores=[])
    namespace = dict(namespace)
    exec(compile('from __future__ import annotations\n' + ast.unparse(module), str(path), 'exec'), namespace)
    return namespace[method]


def config_class():
    path = ROOT/'lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py'
    spec = importlib.util.spec_from_file_location('ee_test_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LingbotVLAConfig


def test_precision_lifecycle_autocast_and_weights_roundtrip(tmp_path):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(12, 12)
            self.fk = EEFKRuntime(SPEC)
    model = Tiny()
    solver = model.fk.solver('cpu')
    baseline = solver.left.origin_rpy.clone()
    model.bfloat16()
    assert model.layer.weight.dtype == torch.bfloat16
    assert solver.left.origin_rpy.dtype == torch.float32
    assert torch.equal(baseline, solver.left.origin_rpy)
    assert not any('fk' in name for name in model.state_dict())
    from safetensors.torch import save_file, load_file
    save_file(model.state_dict(), str(tmp_path/'model.safetensors'))
    restored = Tiny().bfloat16()
    restored.load_state_dict(load_file(str(tmp_path/'model.safetensors')))
    assert torch.equal(restored.fk.solver('cpu').left.origin_rpy, baseline)
    target = torch.randn(2, 4, 30)
    velocity = torch.randn(2, 4, 30, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        predicted = recover_clean_action(target, torch.randn_like(target), torch.tensor([.1, .15]), velocity)
        total, metrics = compute_ee_fk_losses(solver, predicted, target,
            action_layout={'left_arm':[0,6], 'right_arm':[6,12], 'gripper':[28,30]},
            action_norm_spec=[], position_weight=1., orientation_weight=.1, gripper_weight=.1,
            valid_time=torch.tensor([[True, False, True, False], [False]*4]))
    assert total.dtype == torch.float32 and predicted.dtype == torch.float32
    total.backward()
    assert torch.isfinite(velocity.grad).all() and (velocity.grad != 0).any()
    assert (velocity.grad[1] == 0).all()


def test_config_embedded_spec_and_invalid_values(tmp_path):
    cls = config_class()
    cfg = cls(low_noise_fm_enabled=True, ee_fk_loss_enabled=True, ee_fk_eval_enabled=True,
              ee_fk_spec=SPEC, ee_fk_position_loss_weight=1.)
    cfg.save_pretrained(tmp_path)
    restored = cls.from_pretrained(tmp_path)
    assert restored.ee_fk_spec == SPEC
    assert restored.ee_fk_loss_enabled and restored.ee_fk_eval_enabled
    for key, value in [('ee_fk_time_max',float('nan')),('ee_fk_huber_delta',0.),
                       ('ee_fk_position_loss_weight',float('inf'))]:
        with pytest.raises(ValueError):
            cls(low_noise_fm_enabled=True, ee_fk_loss_enabled=True, ee_fk_spec=SPEC, **{key:value})
    cls(ee_fk_spec_path='/missing/disabled.json')
    cls(ee_fk_eval_enabled=True, ee_fk_spec=SPEC)  # evaluation without FM loss


def test_runtime_bf16_parent_and_autocast_match_200_sapien_references():
    reference = np.load(ROOT/'tests/ee_fk/evidence/2026-10-17/sapien_reference.npz')
    joints = torch.tensor(reference['joints'], dtype=torch.float32)
    expected = torch.tensor(reference['expected'], dtype=torch.float32)
    parent = torch.nn.Linear(12, 12)
    parent.fk = EEFKRuntime(SPEC)
    # Exercise a parent conversion before first solver materialization.
    parent.bfloat16()
    solver = parent.fk.solver('cpu')
    with torch.autocast('cpu', enabled=False):
        result = solver(joints[:, :6], joints[:, 6:])
    for i, side in enumerate(('left', 'right')):
        position, orientation = result[side]
        assert float((position-expected[:,i,:3,3]).norm(dim=-1).max()) < 1e-5
        assert float(_rotation_error(orientation,expected[:,i,:3,:3]).max()) < 1e-4


def transform():
    from collections import OrderedDict
    return NS(key_mapping={
        'action.arm.position':{'origin_keys':OrderedDict([
            ('action',{'start':0,'end':6}), ('action*',{'start':7,'end':13})])},
        'action.effector.position':{'origin_keys':OrderedDict([
            ('action',{'start':6,'end':7}), ('action*',{'start':13,'end':14})])}})


def test_metrics_mask_units_equal_trajectory_weight_and_no_rng():
    accumulator = EEErrorAccumulator(SPEC)
    gt = {'action':torch.zeros(4,14)}
    pred = {'action':torch.zeros(4,14)}
    pred['action'][:2,6] = .4
    pred['action'][2:] = float('nan')  # padded garbage must not affect errors
    before = torch.random.get_rng_state().clone()
    accumulator.add(1, gt, pred, transform(), [False,False,True,True])
    accumulator.add(2, {'action':torch.zeros(1,14)}, {'action':torch.zeros(1,14)}, transform(), [False])
    result = accumulator.result()['ee_metrics']
    assert result['valid_frames'] == 3 and result['n_trajs'] == 2
    assert result['left_gripper_mae'] == pytest.approx(.2)
    assert result['gripper_mae'] == pytest.approx(.1)
    assert result['position_rmse_mm'] == pytest.approx(0.)
    assert result['orientation_mean_deg'] == pytest.approx(0.,abs=1e-5)
    assert torch.equal(before,torch.random.get_rng_state())
    with pytest.raises(ValueError,match='real action_is_pad'):
        accumulator.add(1,gt,gt,transform(),None)


def test_real_validator_reuses_predictions_and_keeps_legacy_aggregation():
    path='lingbotvla/utils/open_loop_validation.py'
    tree=ast.parse((ROOT/path).read_text())
    namespace={'np':np, 'torch':torch, 'METRIC_DTYPE_LABEL':'test'}
    def errors(pr,gt): return pr.astype(np.float64)-gt.astype(np.float64),gt.astype(np.float64)
    namespace['fp32_error_fp64_aggregation']=errors
    for name in ('_to_numpy','_as_frames','assemble_chunk','aggregate_chunks'):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec('from __future__ import annotations\n'+ast.unparse(node),namespace)
    infer=Mock(return_value=[[(0, {'action_is_pad':torch.tensor([False,True])}, {'action':torch.zeros(2,14)})]])
    fake=NS(_model_config=NS(ee_fk_loss_enabled=True,ee_fk_spec=SPEC),_phase_tracker=lambda:None,
            _prediction_groups=infer,_shape_dumped=True,_chunk_dumped=True,
            _pick_action_keys=lambda *args:['action'],dump_dir=None,logger=Mock())
    ft=transform();ft.unapply=lambda item:{'action':torch.zeros(2,14)}
    fn=actual_method(path,'OpenLoopValidator','_evaluate_ids',namespace)
    result=fn(fake,[3],'active_val',prep={'ds':[None]*2,'ft':ft,'ep_map':np.array([3,3]),
        'starts':[0],'stride':50,'preflight_done':True})
    assert infer.call_count==1
    assert result['mse']==0. and result['frames']==2
    assert result['ee_metrics']['valid_frames']==1
    fake._model_config=NS()
    result_off=fn(fake,[3],'active_val',prep={'ds':[None]*2,'ft':ft,'ep_map':np.array([3,3]),
        'starts':[0],'stride':50,'preflight_done':True})
    assert 'ee_metrics' not in result_off
    assert all(result_off[k]==result[k] for k in result_off if not isinstance(result_off[k],float) or not np.isnan(result_off[k]))


def test_dto_scheduler_tensorboard_and_panel(tmp_path):
    from lingbotvla.auto_learning.ports import EvalResult
    from lingbotvla.auto_learning.real.build import SchedulerLoggerAdapter
    from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
    from lingbotvla.auto_learning.state.registry import TaskRecord
    from lingbotvla.auto_learning.config import AutoLearningConfig
    values={'position_rmse_mm':2.,'orientation_mean_deg':3.,'gripper_mae':.1}
    result=EvalResult(task='stand',split='val',episode_ids=[1],mse=.1,baseline_mse=1.,
        per_traj_ids=[1],per_traj_mse=[.1],n_traj=1,ee_metrics=values)
    metrics=result.to_trajectory_metrics()
    assert metrics.ee_metrics==values
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(tmp_path/'tb'))
    logger=SchedulerLoggerAdapter(Mock(),writer=writer,event_path=str(tmp_path/'events.jsonl'),write_events=True)
    rec=TaskRecord('stand')
    fake=NS(state=NS(eval_events=0,global_step=100),al=AutoLearningConfig(),logger=logger,heatmap_rows=[])
    Scheduler._record_eval(fake,rec,metrics,'active_val')
    writer.close()
    assert rec.eval_history[-1]['ee_metrics']==values
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=EventAccumulator(str(tmp_path/'tb')).Reload()
    assert events.Scalars('task/stand/active_val_ee_position_rmse_mm')[-1].value==2.
    spec=importlib.util.spec_from_file_location('ee_panel',ROOT/'tools/auto_learning_pipeline/al_panel.py')
    panel=importlib.util.module_from_spec(spec);spec.loader.exec_module(panel)
    text=panel.ee_val_text(panel.read_series(tmp_path/'events.jsonl'),'stand')
    assert '2.00 mm' in text and '3.00°' in text and 'Step 100' in text
    assert panel.ee_val_text({},'old_run') is None
    assert panel.ee_val_text({'active_val_ee_position_rmse_mm':{'stand':[(200,5.)]}},'stand') is None


def test_policy_preserves_fp32_gt_and_adds_loss_without_changing_interface():
    path='lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py'
    forward=actual_method(path,'LingbotVlaV2Policy','forward',{'torch':torch})
    class Flow:
        def forward(self,*args,**kwargs):
            self.actions=args[5]
            self.gold=kwargs['ee_fk_target_actions']
            losses=torch.ones_like(self.actions)
            return (losses,0,0,0,None,0,0,{'_ee_fk_total':self.gold.sum()*0+2.,
                'ee_fk/weighted_loss':torch.tensor(2.)},None,None,None)
    flow=Flow()
    fake=NS(config=NS(ee_fk_loss_enabled=True,action_fp32=False,loss_type='fm',action_dim=30),
            model=flow,parameters=lambda:iter([torch.zeros(1,dtype=torch.bfloat16)]))
    actions=torch.randn(2,4,30)
    output=forward(fake,None,None,torch.zeros(2,30),None,None,actions)
    assert flow.gold is actions and flow.gold.dtype==torch.float32
    assert flow.actions.dtype==torch.bfloat16
    assert len(output)==11 and float(output[0])==3.
    assert '_ee_fk_total' not in output[6]


def test_shard_merge_deduplicates_ee_and_rejects_partial_coverage():
    path='lingbotvla/utils/open_loop_validation.py'
    tree=ast.parse((ROOT/path).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_merge_shard_raw_payloads')
    namespace={'_SHARD_PER_TRAJ_KEYS':('per_traj_mse','per_traj_ids','per_traj_frames')}
    exec('from __future__ import annotations\n'+ast.unparse(node),namespace)
    merge=namespace[node.name]
    def part(eid):
        row={'frames':1,'left_position_sq':.000004,'right_position_sq':.000004,
             'left_angle':0.,'right_angle':0.,'left_gripper_abs':0.,'right_gripper_abs':0.}
        return {'per_traj_ids':[eid],'per_traj_mse':[.1],'per_traj_frames':[1],
                'ee_per_traj':{str(eid):row},'n_chunks':1}
    result=merge([part(1),part(2),part(1)],{})
    assert result['ee_metrics']['n_trajs']==2
    assert result['ee_metrics']['position_rmse_mm']==pytest.approx(2.)
    missing=part(2);missing.pop('ee_per_traj')
    with pytest.raises(ValueError,match='do not cover'):
        merge([part(1),missing],{})


def test_real_launcher_ee_flags_and_resume_propagation(tmp_path):
    spec=importlib.util.spec_from_file_location('ee_launcher',ROOT/'experiment/robotwin/launch_skill_mix_task.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args=NS(ee_fk_loss=True,ee_fk_eval=True,ee_position_weight=1.,ee_orientation_weight=.1,ee_gripper_weight=.1)
    options=module.ee_training_options(args,{},dict(low_noise_fm_enabled=True))
    assert options['train.ee_fk_loss_enabled'] and options['train.ee_fk_eval_enabled']
    assert Path(options['train.ee_fk_spec_path']).is_file()
    with pytest.raises(ValueError,match='low_noise'):
        module.ee_training_options(args,{},dict(low_noise_fm_enabled=False))
    assert module.ee_training_options(NS(),{}, {})=={
        'train.ee_fk_loss_enabled':False,'train.ee_fk_eval_enabled':False}
