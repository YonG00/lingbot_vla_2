"""CPU checks: dynamic 70:30 GBS/DP allocation, sampling and Resume guards."""
from __future__ import annotations

import random
import sys
import types

import pytest

from lingbotvla.auto_learning.batch_ratio import ratio_plan
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.ports import ReplayPlan, TaskEntry, TrainRequest
from lingbotvla.auto_learning.real.sampler import AutoLearnSampler
from lingbotvla.auto_learning.real.build import validate_batch_alignment
from lingbotvla.auto_learning.sampling.sampler import BatchSampler
from lingbotvla.auto_learning.types import SampleRef


@pytest.mark.parametrize('gbs,dp,total_new', [(4,1,3),(10,1,7),(20,1,14),
                                             (28,1,20),(32,1,22),(112,4,78),
                                             (4,4,3),(3,3,2),(100,5,70)])
def test_total_exact_allocation(gbs,dp,total_new):
    plans = [ratio_plan(global_batch_size=gbs,dp_size=dp,dp_rank=r,new_ratio=.7)
             for r in range(dp)]
    assert sum(p.local_new for p in plans) == total_new
    assert sum(p.local_replay for p in plans) == gbs - total_new
    assert all(p.local_new + p.local_replay == gbs // dp for p in plans)
    assert max(p.local_new for p in plans)-min(p.local_new for p in plans) <= 1


def test_gbs112_differs_from_per_rank_rounding():
    plans = [ratio_plan(global_batch_size=112,dp_size=4,dp_rank=r,new_ratio=.7)
             for r in range(4)]
    assert [p.local_new for p in plans] == [19,20,19,20]
    assert [p.local_replay for p in plans] == [9,8,9,8]


@pytest.mark.parametrize('kwargs', [
    dict(global_batch_size=11,dp_size=4,dp_rank=0,new_ratio=.7),
    dict(global_batch_size=10,dp_size=0,dp_rank=0,new_ratio=.7),
    dict(global_batch_size=10,dp_size=1,dp_rank=1,new_ratio=.7),
    dict(global_batch_size=10,dp_size=1,dp_rank=0,new_ratio=-.1),
    dict(global_batch_size=10,dp_size=1,dp_rank=0,new_ratio=float('nan')),
    dict(global_batch_size=10,dp_size=1,dp_rank=0,new_ratio=True),
])
def test_invalid_ratio_plan_fails(kwargs):
    with pytest.raises(ValueError):
        ratio_plan(**kwargs)


def test_static_defaults_unchanged():
    c = AutoLearningConfig()
    assert (c.new_ratio,c.batch_size,c.new_slots,c.replay_slots) == (None,10,7,3)
    with pytest.raises(ValueError, match='new_slots'):
        AutoLearningConfig(batch_size=12)


def test_dynamic_stage_a_batch_derivation():
    c=AutoLearningConfig(batch_size=28,new_ratio=.7)
    assert (c.batch_size,c.new_slots,c.replay_slots) == (28,20,8)
    c.batch_size=32
    c.validate()
    assert (c.new_slots,c.replay_slots) == (22,10)


class _Log:
    def info_rank0(self, *args, **kwargs): pass


def _args(gbs, dp, rank, micro=1, gas=None, dls=None):
    dls = gbs//dp if dls is None else dls
    gas = dls//micro if gas is None else gas
    return types.SimpleNamespace(train=types.SimpleNamespace(
        micro_batch_size=micro,global_batch_size=gbs,
        dataloader_batch_size=dls,gradient_accumulation_steps=gas,
        data_parallel_size=dp))


@pytest.mark.parametrize('rank,slots', [(0,(19,9)),(1,(20,8)),(2,(19,9)),(3,(20,8))])
def test_real_runtime_binding(monkeypatch, rank, slots):
    # Mock only DP topology; no torch process required.
    import lingbotvla.distributed.parallel_state as ps
    monkeypatch.setattr(ps, 'get_parallel_state',
                        lambda: types.SimpleNamespace(dp_size=4,dp_rank=rank))
    cfg=AutoLearningConfig(new_ratio=.7)
    validate_batch_alignment(cfg,_args(112,4,rank,micro=28,gas=1),_Log())
    assert (cfg.batch_size,cfg.new_slots,cfg.replay_slots)==(28,*slots)
    cfg.validate()  # subsequent validation cannot reset rank-specific slots
    assert (cfg.new_slots,cfg.replay_slots)==slots


def test_runtime_mismatch_fails(monkeypatch):
    import lingbotvla.distributed.parallel_state as ps
    monkeypatch.setattr(ps,'get_parallel_state',
                        lambda:types.SimpleNamespace(dp_size=4,dp_rank=0))
    cfg=AutoLearningConfig(new_ratio=.7)
    with pytest.raises(ValueError,match='local_batch'):
        validate_batch_alignment(cfg,_args(112,4,0,micro=28,gas=2),_Log())


class _Catalog:
    def __init__(self, entries): self.entries=entries
    def entry(self, task): return self.entries[task]


class _Resolver:
    def __init__(self, owner): self.owner=owner
    def resolve(self,task,sid):
        assert self.owner[sid]==task
        return SampleRef(task=task,sample_id=sid,episode_id=sid//100,frame_id=sid%100)


def _sampler(rank,with_pass=True):
    cfg=AutoLearningConfig(new_ratio=.7,seed=19)
    plan=ratio_plan(global_batch_size=112,dp_size=4,dp_rank=rank,new_ratio=.7)
    cfg.batch_size,cfg.new_slots,cfg.replay_slots=(28,plan.local_new,plan.local_replay)
    cfg._ratio_dp_rank=rank;cfg._ratio_dp_size=4;cfg._ratio_global_batch_size=112
    e={};owners={}
    for name,start in [('new',0),('old',200)]:
        ids=list(range(start,start+160))
        e[name]=TaskEntry(name=name,train_sample_ids=ids)
        owners.update({i:name for i in ids})
    resolver=_Resolver(owners);catalog=_Catalog(e)
    underlying=BatchSampler(cfg,resolver,catalog,random.Random(999))
    sam=AutoLearnSampler(underlying,batch_size=28)
    replay=(ReplayPlan(tasks=['old'],sample_ids={'old':e['old'].train_sample_ids})
            if with_pass else ReplayPlan())
    req=TrainRequest(task='new',probs={i:1.0 for i in e['new'].train_sample_ids},
                     replay=replay,start_step=14,batch_size=28,
                     new_slots=cfg.new_slots,replay_slots=cfg.replay_slots)
    sam.set_request(req)
    return sam


def _draw_two(sam):
    it=iter(sam)
    return [[next(it) for _ in range(28)] for _ in range(2)]


def test_distributed_sampler_actual_global_ratio_and_provenance():
    all_sam=[_sampler(r) for r in range(4)]
    batches=[_draw_two(s) for s in all_sam]
    for step in range(2):
        all_comps=[s.compositions[step] for s in all_sam]
        assert sum(c.n_new for c in all_comps)==78
        assert sum(c.n_old for c in all_comps)==34
        assert all(len(c.refs)==28 for c in all_comps)
        assert all(set(r.task for r in c.new)=={'new'} for c in all_comps)
        assert all(set(r.task for r in c.old)=={'old'} for c in all_comps)
    # Independent per-rank seeds: never duplicate the entire batch.
    assert len({tuple(bs[0]) for bs in batches})==4


def test_no_replay_pool_falls_back_to_all_new():
    s=_sampler(2,with_pass=False)
    _draw_two(s)
    assert all(c.n_old==0 and c.n_new==28 for c in s.compositions)


def test_sampler_deterministic_across_resume_and_prefetch():
    a=_sampler(1); expected=_draw_two(a)
    b=_sampler(1)
    # RNG used by Scheduler is deliberately not consumed in ratio mode.
    before=b.batch_sampler.rng.getstate()
    got=_draw_two(b)
    assert b.batch_sampler.rng.getstate()==before
    assert got==expected
    snap=b.state_dict()
    c=_sampler(1);c.load_state_dict(snap)
    assert c.state_dict()['ratio_layout']==snap['ratio_layout']
    # At a fresh unit boundary request/start_step resets the deterministic stream.
    assert _draw_two(_sampler(1))==expected


def test_resume_changed_gbs_or_ratio_fails():
    a=_sampler(1)
    state=a.state_dict()
    bad={**state, 'ratio_layout': {**state['ratio_layout'],'global_batch_size':224}}
    with pytest.raises(ValueError,match='ratio/GBS/DP'):
        a.load_state_dict(bad)
    with pytest.raises(ValueError,match='batch size'):
        a.load_state_dict({**state,'batch_size':56})
    with pytest.raises(ValueError,match='ratio/GBS/DP'):
        a.load_state_dict({'version':1,'batch_size':28})


def test_hook_aggregates_real_dp_counts(monkeypatch):
    """Behavior-level hook test: rank-local count must become global TrainResult."""
    from collections import Counter
    import torch.distributed as dist
    from lingbotvla.auto_learning.real.hook import AutoLearnLoopHook
    import lingbotvla.distributed.parallel_state as ps

    group=object()
    monkeypatch.setattr(ps,'get_parallel_state',
                        lambda:types.SimpleNamespace(dp_group=group))
    monkeypatch.setattr(dist,'is_initialized',lambda:True)
    monkeypatch.setattr(dist,'get_world_size',lambda group=None:4)
    def fake_gather(out,payload,group=None):
        assert group is not None
        for r,new in enumerate([19,20,19,20]):
            out[r]=(1,28,{'new':new},{'old':28-new})
    monkeypatch.setattr(dist,'all_gather_object',fake_gather)

    class _Sampler:
        def stats_upto(self,steps):
            assert steps==1
            from lingbotvla.auto_learning.real.sampler import UnitStats
            return UnitStats(steps=1,n_new=19,n_old=9,samples_seen=28,
                             new_slot_counts_by_task=Counter({'new':19}),
                             old_slot_counts_by_task=Counter({'old':9}),unique_batches=1)
    class _Scheduler:
        pending_train_request=object()
        def complete_train_unit(self,result,allow_partial=False):
            self.result=result
            self.pending_train_request=None
            return None
    cfg=AutoLearningConfig(new_ratio=.7)
    cfg._ratio_dp_size=4;cfg._ratio_global_batch_size=112
    sch=_Scheduler()
    hook=AutoLearnLoopHook(scheduler=sch,sampler=_Sampler(),cfg=cfg)
    hook._step_in_unit=1;hook._unit_steps=1;hook._unit_losses=[.2]
    hook._drain_pending_unit(1)
    assert sch.result.samples_seen==112
    assert sch.result.new_slot_counts=={'new':78}
    assert sch.result.old_slot_counts=={'old':34}
    assert sch.result.batches_built==sch.result.steps==1
