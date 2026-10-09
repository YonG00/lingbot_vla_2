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
             ('_infer_one','_infer_batch','_noise_generator','_prediction_groups')]
    cls.body=methods
    ns={'torch':torch,'np':np,'Dict':dict,'Any':object,'Sequence':list,'List':list,
        'EVAL_SEED':1234,'os':os,'time':time,'_world_size':lambda:1,'_visual_grid_cache_clear':lambda model:None,
        '_visual_grid_cache_restore':lambda model,saved:None}
    ast.fix_missing_locations(tree)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(src),'exec'),ns)
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


def test_auto_parity_mismatch_disables_batched_outputs(monkeypatch):
    monkeypatch.setenv('AL_EVAL_BATCH_MODE','auto')
    monkeypatch.setenv('AL_EVAL_BATCH_APPROVED','1')
    monkeypatch.setenv('AL_EVAL_BATCH_MAX','2')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(80 * 1024**3,96*1024**3))
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:14*1024**3)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    v=_validator();v.logger=SimpleNamespace(info_rank0=lambda *_:None)
    original=v._infer_batch
    def wrong(items,ft):
        p=original(items,ft)
        for it in p:it['actions']=it['actions']+1
        return p
    v._infer_batch=wrong
    inputs=[_item(float(i)) for i in range(4)]
    ref=[v._infer_one(it,_Transform())['actions'].clone() for it in inputs]
    v._noise_gen=None
    got=[pr['actions'] for group in v._prediction_groups(inputs,list(range(4)),_Transform(),'t')
         for _,_,pr in group]
    assert all(torch.equal(a,b) for a,b in zip(got,ref))


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
