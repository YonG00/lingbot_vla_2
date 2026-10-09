from lingbotvla.auto_learning.eval_batch_policy import (
    EvalBatchSettings, outputs_close, profile_batch_candidates, allowed_runtime_batch)
import pytest


def test_serial_default_does_not_call_model():
    r=profile_batch_candidates([1,2],serial_infer=lambda _:pytest.fail('called'),
      batch_infer=lambda _:pytest.fail('called'),available_gib=lambda:90.,settings=EvalBatchSettings())
    assert r['status']=='SERIAL' and r['selected_batch']==1


def test_auto_measured_chooses_safe_parity_batch(monkeypatch):
    import lingbotvla.auto_learning.eval_batch_policy as m
    time_values=iter(range(20))
    monkeypatch.setattr(m.time,'perf_counter',lambda:next(time_values))
    r=profile_batch_candidates([1,2,3,4],serial_infer=lambda x:[x*2],
      batch_infer=lambda xs:[[x*2] for x in xs], available_gib=lambda:50.,peak_free_gib=lambda:40.,
      settings=EvalBatchSettings(mode='auto',candidates=(1,2,4)))
    assert r['status']=='CALIBRATED' and r['selected_batch']==4
    assert len(r['observations'])==3


def test_parity_mismatch_cannot_enable_batch2():
    r=profile_batch_candidates([1,2,3,4],serial_infer=lambda x:x,
      batch_infer=lambda xs:[999]*len(xs),available_gib=lambda:50.,peak_free_gib=lambda:40.,
      settings=EvalBatchSettings(mode='auto',candidates=(1,2,4)))
    assert r['selected_batch']==1
    assert r['observations'][-1]['parity'] is False
    assert len(r['observations'])==2


def test_multi_rank_fail_closed():
    r=profile_batch_candidates([1,2],serial_infer=lambda x:x,
      batch_infer=lambda x:x, available_gib=lambda:90.,
      world_size=4,settings=EvalBatchSettings(mode='auto'))
    assert r['status']=='BLOCKED'


def test_headroom_changes_force_serial_fallback():
    assert allowed_runtime_batch(4,20.,reserve_gib=10.,calibrated_peak_extra_gib=5.)==4
    assert allowed_runtime_batch(4,12.,reserve_gib=10.,calibrated_peak_extra_gib=5.)==1
    assert allowed_runtime_batch(4,90.,reserve_gib=10.,calibrated_peak_extra_gib=5.,world_size=4)==1


def test_numeric_parity_finite_only():
    assert outputs_close([[.01]],[[.01000001]],atol=1e-6,rtol=0)
    assert not outputs_close([[.01]],[[float('nan')]],atol=1e-6,rtol=0)
    assert not outputs_close([[.01]],[[.02]],atol=1e-6,rtol=0)


@pytest.mark.parametrize('kw', [{'candidates':(2,4)},{'candidates':(1,2,2)},
                              {'reserve_gib':-1},{'mode':'dynamic'}])
def test_invalid_policy(kw):
    with pytest.raises(ValueError):
        EvalBatchSettings(**kw)


def test_auto_requires_peak_sensor():
    r=profile_batch_candidates([1,2],serial_infer=lambda x:x,
      batch_infer=lambda x:x,available_gib=lambda:50.,settings=EvalBatchSettings(mode='auto'))
    assert r['status']=='BLOCKED' and r['selected_batch']==1

def test_peak_sensor_denies_unsafe_batch():
    calls=iter([50.,8.])
    r=profile_batch_candidates([1,2,3,4],serial_infer=lambda x:x,
      batch_infer=lambda x:x,available_gib=lambda:50.,peak_free_gib=lambda:next(calls),
      settings=EvalBatchSettings(mode='auto',candidates=(1,2,4)))
    assert r['selected_batch']==1
    assert r['observations'][-1]['safe'] is False
