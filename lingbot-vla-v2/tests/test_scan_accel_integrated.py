"""CPU contract tests: Scout cache, Hardness adaptive policy, real model batch glue."""
from __future__ import annotations
import dataclasses
import os
import time
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lingbotvla.auto_learning.scan_accel import HardnessAutoBatch, identical_tensor_shapes
from lingbotvla.auto_learning.scout_cache import BootstrapScoutCache, provenance
from lingbotvla.auto_learning.types import TrajectoryMetrics
from lingbotvla.auto_learning.real.backend import RealEvaluator, RealHardnessScorer
# Avoid importing the full training DataLoader stack (torchdata) in CPU-only CI.
# Compile the exact production methods from their source AST, not copied logic.
import ast
from pathlib import Path

def _validator_class():
    src=Path(__file__).parents[1]/'lingbotvla/utils/open_loop_validation.py'
    tree=ast.parse(src.read_text(encoding='utf-8'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='OpenLoopValidator')
    methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in
             ('_infer_one','_infer_batch','_infer_core','_noise_generator',
              '_prediction_groups','_probe_capture','_probe_identity_for',
              '_infer_serial_group')]
    cls.body=methods
    # 模块级诊断小工具（默认路径无副作用）也按源码编译，避免"替身与生产不一致"
    mod_funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef)
               and x.name in ('_probe_warn_legacy_dump','_probe_repeat_env',
                              '_eval_batch_size')]
    ns={'torch':torch,'np':np,'Dict':dict,'Any':object,'Sequence':list,'List':list,
        'EVAL_SEED':1234,'os':os,'time':time,'_world_size':lambda:1,'_visual_grid_cache_clear':lambda model:None,
        '_visual_grid_cache_restore':lambda model,saved:None}
    ast.fix_missing_locations(tree)
    exec(compile(ast.Module(body=mod_funcs+[cls],type_ignores=[]),str(src),'exec'),ns)
    return ns['OpenLoopValidator']

OpenLoopValidator=_validator_class()


def _metrics(ids=(51,52)):
    return TrajectoryMetrics('click_bell','scout',list(ids), {int(i):.1 for i in ids},
                             .1,1.,.1,None,True,len(ids),wall_time_s=1.5)


def test_cache_provenance_content_not_mtime(tmp_path):
    w=tmp_path/'weight.safetensors'; w.write_bytes(b'A')
    s=tmp_path/'norm.json'; s.write_bytes(b'{}')
    fp=provenance(weight_files=[w],sources={'norm':s},options={'inference_dtype':'bf16'})
    w.write_bytes(b'B')
    fp2=provenance(weight_files=[w],sources={'norm':s},options={'inference_dtype':'bf16'})
    assert fp!=fp2 and len(fp)==64


def test_cache_writes_reads_strict_trajectory_ids(tmp_path):
    cache=BootstrapScoutCache(tmp_path,fingerprint='f'*64)
    x=dataclasses.asdict(_metrics())
    cache.store('click_bell',[51,52],x)
    assert cache.load('click_bell',[51,52])['mse']==.1
    assert cache.load('click_bell',[51,52,56]) is None
    assert cache.load('click_alarmclock',[51,52]) is None
    assert BootstrapScoutCache(tmp_path,fingerprint='e'*64).load('click_bell',[51,52]) is None
    f=next(cache.path.glob('*.json'))
    payload=json.loads(f.read_text());payload['metrics']['per_traj_mse']['52']='NaN'
    f.write_text(json.dumps(payload))
    assert cache.load('click_bell',[51,52]) is None


def test_cache_bad_shard_names(tmp_path):
    p=tmp_path/'one';p.mkdir(); q=tmp_path/'two';q.mkdir()
    (p/'same.safetensors').write_bytes(b'a');(q/'same.safetensors').write_bytes(b'b')
    with pytest.raises(ValueError, match='ambiguous'):
        provenance(weight_files=[p/'same.safetensors',q/'same.safetensors'],
                   sources={'a':p/'same.safetensors'},options={'inference_dtype':'bf16'})


def test_cache_rejects_invalid_records(tmp_path):
    cache=BootstrapScoutCache(tmp_path,fingerprint='1'*64)
    with pytest.raises(ValueError):
        cache.store('t',[1,2],dataclasses.asdict(_metrics()))
    with pytest.raises(ValueError):
        cache.store('click_bell',[51,51],dataclasses.asdict(_metrics((51,51))))


def test_bootstrap_uses_cache_only_when_explicit(tmp_path):
    class Adapter:
        def __init__(self): self.calls=0
        def evaluate_task(self,t,sp,ids):
            self.calls+=1
            class R:
                def to_trajectory_metrics(self): return _metrics()
            return R()
    adapter=Adapter()
    cache=BootstrapScoutCache(tmp_path,fingerprint='f'*64)
    ev=RealEvaluator(adapter, object(), scout_cache=cache)
    assert ev.evaluate_bootstrap_scout('click_bell','scout',[51,52]).gmean_mse==pytest.approx(.1)
    assert ev.evaluate_bootstrap_scout('click_bell','scout',[51,52]).gmean_mse==pytest.approx(.1)
    assert adapter.calls==1
    ev.evaluate('click_bell','scout',[51,52])  # Rescan is never cached
    assert adapter.calls==2


def test_hardness_uses_existing_batch_eight_when_auto_off(monkeypatch):
    monkeypatch.delenv('AL_HARDNESS_BATCH_MODE',raising=False)
    class Scorer:
        def score(self, xs):return np.asarray([float(x) for x in xs])
    r=RealHardnessScorer(Scorer(),list(range(19)))
    assert r.score('t',list(range(19)))=={i:float(i) for i in range(19)}
    assert r.last_timing['batches']==3
    assert r.last_timing['auto_batch_mode'] is False


def test_hardness_runtime_policy_fail_closed():
    p=HardnessAutoBatch(current=8,maximum=16,reserve_gib=10)
    assert p.observe(free_before_gib=30,peak_free_gib=9,free_after_gib=20,processed=8)==4
    assert p.observe(free_before_gib=30,peak_free_gib=4,free_after_gib=20,processed=4)==2
    assert p.observe(free_before_gib=float('nan'),peak_free_gib=20,free_after_gib=20,processed=2)==1


def test_hardness_auto_mode_does_not_mutate_scores(monkeypatch):
    monkeypatch.setenv('AL_HARDNESS_BATCH_MODE','auto')
    monkeypatch.setenv('AL_HARDNESS_BATCH_APPROVED','1')
    class Scorer:
        def score(self,xs, *,sample_ids):
            assert len(xs)==len(sample_ids)
            return [float(x*x) for x in xs]
    r=RealHardnessScorer(Scorer(),list(range(10)))
    assert r.score('t',list(range(10)))=={i:float(i*i) for i in range(10)}
    assert r.last_timing['auto_batch_mode']


def _item(s):
    return {'images':torch.zeros(2,3,2,2),'img_masks':torch.ones(2),
            'lang_tokens':torch.tensor([1,2]),'lang_masks':torch.ones(2),
            'state':torch.tensor([s]),'actions':torch.zeros(2,3)}


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.p=torch.nn.Parameter(torch.zeros(1))
    def sample_actions(self,images,img_masks,lang_tokens,lang_masks,state,noise,image_grid_thw=None):
        assert images.shape[0]==state.shape[0]==noise.shape[0]
        return noise + state.reshape(-1,1,1)


class _Transform:
    def unapply(self,d):return {'actions':d['actions']}


def _validator():
    v=object.__new__(OpenLoopValidator)
    v.model=_Policy()
    v.device='cpu'
    v._model_config=SimpleNamespace(n_action_steps=2,max_action_dim=3,action_fp32=False)
    v.args=SimpleNamespace(train=SimpleNamespace(eval_inference_dtype='auto',use_bf16=False))
    v._noise_gen=None
    v._pick_action_keys=lambda ft, gt, pred:['actions']
    v._precision_logged=True
    v.dump_dir=None
    v._dump_prefix=None
    return v


def test_batched_noise_follows_sequential_chunk_stream():
    v=_validator();ft=_Transform();items=[_item(i) for i in (0.,1.,2.,3.)]
    serial=[v._infer_one(it,ft)['actions'].clone() for it in items]
    v._noise_gen=None
    batched=v._infer_batch(items,ft)
    assert all(torch.equal(s,b['actions']) for s,b in zip(serial,batched))
    assert all(b['actions'].shape==(2,3) for b in batched)


def test_batched_shape_mismatch_fails_without_call():
    v=_validator();a=_item(0);b=_item(1);b['lang_tokens']=torch.tensor([1,2,3])
    assert not identical_tensor_shapes([a,b],('images','img_masks','lang_tokens','lang_masks','state'))
    with pytest.raises(ValueError,match='homogeneous'):
        v._infer_batch([a,b],_Transform())


def test_batched_no_noise_after_ambiguous_prebatch():
    v=_validator();a=_item(0);a['state']=a['state'].unsqueeze(0)
    with pytest.raises(ValueError,match='pre-batched'):
        v._infer_batch([a,a],_Transform())


def test_probe_executes_real_batch_but_yields_serial(monkeypatch):
    monkeypatch.setenv('AL_EVAL_BATCH_MODE','probe')
    monkeypatch.setenv('AL_EVAL_BATCH_MAX','4')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(80 * 1024**3,96*1024**3))
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:14*1024**3)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    v=_validator();v.logger=SimpleNamespace(info_rank0=lambda *_:None)
    ft=_Transform(); items=[_item(float(i)) for i in range(6)]
    ref=[v._infer_one(it,ft)['actions'].clone() for it in items]
    v._noise_gen=None
    got=[pred['actions'] for group in v._prediction_groups(items,list(range(6)),ft,'t')
         for _,_,pred in group]
    assert len(got)==len(ref)
    assert all(torch.equal(a,b) for a,b in zip(got,ref))


def test_auto_without_approval_is_blocked_before_inference(monkeypatch):
    monkeypatch.setenv('AL_EVAL_BATCH_MODE','auto')
    monkeypatch.delenv('AL_EVAL_BATCH_APPROVED',raising=False)
    v=_validator()
    with pytest.raises(RuntimeError,match='APPROVED'):
        list(v._prediction_groups([_item(1),_item(2)],[0,1],_Transform(),'t'))


def _cuda_stub(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(80 * 1024**3,96*1024**3))
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:14*1024**3)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)


def test_auto_uses_batched_outputs_directly(monkeypatch):
    """**2026-10-09 定案**：auto 模式**不再做数值判定**（门与探测已退出判定链）——
    每组直接采用批量结果，且该组**不再调用逐条串行**。"""
    monkeypatch.setenv('AL_EVAL_BATCH_MODE','auto')
    monkeypatch.setenv('AL_EVAL_BATCH_APPROVED','1')
    monkeypatch.setenv('AL_EVAL_BATCH_MAX','2')
    _cuda_stub(monkeypatch)
    v=_validator();v.logger=SimpleNamespace(info_rank0=lambda *_:None)
    seen={'batched':[], 'serial':0}
    original_batch=v._infer_batch
    original_one=v._infer_one
    def spy_batch(items,ft):
        out=original_batch(items,ft); seen['batched'].append([dict(d) for d in out]); return out
    def spy_one(item,ft):
        seen['serial']+=1; return original_one(item,ft)
    v._infer_batch=spy_batch; v._infer_one=spy_one
    inputs=[_item(float(i)) for i in range(4)]
    v._noise_gen=None
    got=[pr for group in v._prediction_groups(inputs,list(range(4)),_Transform(),'t')
         for _,_,pr in group]
    assert seen['batched'], 'auto 模式必须真的走批量'
    assert seen['serial']==0, f'auto 组不应再跑逐条串行（实测 {seen["serial"]} 次）'
    flat=[d for batch in seen['batched'] for d in batch]
    assert len(got)==4==len(flat)
    assert all(torch.equal(a['actions'],b['actions']) for a,b in zip(got,flat)), \
        'yield 的结果必须就是批量结果本身'


def test_auto_group_fallback_is_per_group_not_global(monkeypatch):
    """成组条件不足（形状不一致 / 只剩 1 条）⇒ **该组**单条，不整体退回串行。"""
    monkeypatch.setenv('AL_EVAL_BATCH_MODE','auto')
    monkeypatch.setenv('AL_EVAL_BATCH_APPROVED','1')
    monkeypatch.setenv('AL_EVAL_BATCH_MAX','2')
    _cuda_stub(monkeypatch)
    v=_validator();v.logger=SimpleNamespace(info_rank0=lambda *_:None)
    calls={'batch':0,'serial':0}
    ob=v._infer_batch; oo=v._infer_one
    v._infer_batch=lambda items,ft:(calls.__setitem__('batch',calls['batch']+1), ob(items,ft))[1]
    v._infer_one=lambda item,ft:(calls.__setitem__('serial',calls['serial']+1), oo(item,ft))[1]
    # 3 条 + 批大小 2 ⇒ 前两条成组批处理；**剩下 1 条**该组单条（且不是整体退回串行）
    inputs=[_item(float(i)) for i in range(3)]
    v._noise_gen=None
    got=[pr['actions'] for group in v._prediction_groups(inputs,list(range(3)),_Transform(),'t')
         for _,_,pr in group]
    assert len(got)==3
    assert calls['batch']==1, '前两条应成组批处理'
    assert calls['serial']==1, '剩余 1 条应单条（per-group 回退，不是整体退回串行）'


def test_hardness_auto_missing_approval_blocked(monkeypatch):
    monkeypatch.setenv('AL_HARDNESS_BATCH_MODE','auto')
    monkeypatch.delenv('AL_HARDNESS_BATCH_APPROVED',raising=False)
    class Fake:
        def score(self,xs):return xs
    with pytest.raises(RuntimeError,match='approval'):
        RealHardnessScorer(Fake(),[0,1])


def test_hardness_per_sample_noise_invariant_to_grouping():
    from lingbotvla.auto_learning.hardness import HardnessScorer
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param=torch.nn.Parameter(torch.ones(1))
            self.config=SimpleNamespace(loss_type='L1_fm',align_params={},
                                        n_action_steps=2,max_action_dim=3,action_fp32=False)
        def forward(self,**kw):
            noise=kw['noise']
            return {'batch_mean_losses':noise.mean(dim=(1,2))}
    sc=HardnessScorer(Model(),device='cpu')
    items=[{'actions':torch.zeros(2,3),'joint_mask':torch.ones(3)} for _ in range(3)]
    both=sc.score(items,sample_ids=[15,99,30])
    solo=sc.score(items[1:2],sample_ids=[99])
    reordered=sc.score([items[1],items[0]],sample_ids=[99,15])
    assert np.array_equal(both[1:2],solo)
    assert np.array_equal(both[[1,0]],reordered)
    with pytest.raises(ValueError,match='uniquely'):
        sc.score(items,sample_ids=[15,15,30])


def test_preflight_default_is_read_only_and_missing_sources_blocked(tmp_path):
    import importlib.util
    file=Path(__file__).parents[1]/'tools/scan_accel_preflight.py'
    spec=importlib.util.spec_from_file_location('scan_accel_preflight',file)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    a=SimpleNamespace(verify=False,checkpoint=None,eval_mode='serial',hardness_mode='fixed',cache_root=None)
    assert m.plan(a)['status']=='PLAN_ONLY'
    a.verify=True
    assert m.verify(a)['status']=='BLOCKED'
    a.checkpoint=tmp_path
    a.manifest=a.norm=a.thresholds=a.baseline=None
    assert m.verify(a)['status']=='BLOCKED'
    assert not list(tmp_path.iterdir())


def _eval_batch_size_fn():
    """按源码编译 `_eval_batch_size`（模块级纯函数，只需 os）。"""
    src = Path(__file__).parents[1] / 'lingbotvla/utils/open_loop_validation.py'
    tree = ast.parse(src.read_text(encoding='utf-8'))
    fn = next(x for x in tree.body
              if isinstance(x, ast.FunctionDef) and x.name == '_eval_batch_size')
    ns = {'os': os}
    ast.fix_missing_locations(tree)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), 'exec'), ns)
    return ns['_eval_batch_size']


class _Args:
    def __init__(self, micro):
        self.train = SimpleNamespace(micro_batch_size=micro)


def test_eval_batch_size_follows_training_micro(monkeypatch):
    """**核心契约**：评测批大小默认 = 训练的前向批大小（本次生产 = 24）。"""
    fn = _eval_batch_size_fn()
    monkeypatch.delenv('AL_EVAL_BATCH_MAX', raising=False)
    assert fn(SimpleNamespace(args=_Args(24))) == 24
    assert fn(SimpleNamespace(args=_Args(1))) == 1
    assert fn(SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace()))) == 8   # 无训练配置即回退
    assert fn(SimpleNamespace()) == 8


def test_eval_batch_size_env_override_and_validation(monkeypatch):
    """`AL_EVAL_BATCH_MAX` 由"硬上限"改为**覆盖值**（保留旧名字），且做范围校验。"""
    fn = _eval_batch_size_fn()
    monkeypatch.setenv('AL_EVAL_BATCH_MAX', '4')
    assert fn(SimpleNamespace(args=_Args(24))) == 4
    monkeypatch.setenv('AL_EVAL_BATCH_MAX', '0')
    with pytest.raises(ValueError):
        fn(SimpleNamespace(args=_Args(24)))
    monkeypatch.setenv('AL_EVAL_BATCH_MAX', '999')
    with pytest.raises(ValueError):
        fn(SimpleNamespace(args=_Args(24)))
