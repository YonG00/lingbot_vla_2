"""Contract tests use independent event/metric rows; never infer consumption from a plan."""
import importlib.util
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location('offline_verify',Path(__file__).resolve().parents[1]/'tools/replay_offline_verify.py')
mod=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(mod)


def fixtures():
    rows=[]
    for i in range(5):
        k=i*3
        rows.append({'kind':'event','action':'train_unit','step':k,'steps':3,
                     'request':"TrainRequest(task='turn_switch', replay=ReplayPlan(tasks=['click_bell', 'click_alarmclock']))"})
        rows.append({'kind':'metric','name':'sampling/replay_samples_per_unit','step':500+k,'value':21})
    for name,value in [('sampling/total_samples',360),('system/global_samples_seen',360),
                       ('system/units_run',5),('curriculum/units_completed',5)]:
        rows.append({'kind':'metric','name':name,'step':515,'value':value})
    return rows


def verify(rows):
    return mod.validate(rows,gbs=24,new_per_step=17,replay_per_step=7,step_offset=500)


def test_actual_unit_replay_5_x_3_17_7_passes():
    res=verify(fixtures())
    assert res['status']=='PASS'
    assert res['unit_count']==5 and res['optimizer_steps']==15
    assert res['observed_replay_samples']==105
    assert res['expected_new_samples']==255
    assert res['observed_total_samples_expected']==360
    assert res['step_relation']=={'unit_metric_step_relation':'start','metric_step_shift':500}


def test_replay_tasks_safe_string_parse():
    r={'request':"TrainRequest(replay=ReplayPlan(tasks=['click_bell', 'click_alarmclock'], ratios=1))"}
    assert mod.replay_tasks(r)==['click_bell','click_alarmclock']
    bad={'request':"ReplayPlan(tasks=__import__('os').system('touch /tmp/evil'))"}
    assert mod.replay_tasks(bad)==[]


@pytest.mark.parametrize('change', ['missing_metric','wrong_replay','missing_pool',
    'duplicate_unit','gap','wrong_total','missing_counter','duplicate_metric','metric_step_wrong','nonfinite'])
def test_fail_closed(change):
    rows=fixtures()
    if change=='missing_metric': rows=[r for r in rows if not (r.get('name')=='sampling/replay_samples_per_unit' and r.get('step')==506)]
    if change=='wrong_replay': next(r for r in rows if r.get('name')=='sampling/replay_samples_per_unit')['value']=20
    if change=='missing_pool': rows[0]['request']='TrainRequest(replay=None)'
    if change=='duplicate_unit': rows.append(dict(rows[0]))
    if change=='gap': rows[2]['step']=4
    if change=='wrong_total': next(r for r in rows if r.get('name')=='sampling/total_samples')['value']=359
    if change=='missing_counter': rows=[r for r in rows if r.get('name')!='system/global_samples_seen']
    if change=='duplicate_metric': rows.append(dict(rows[1]))
    if change=='metric_step_wrong': rows[1]['step']=800
    if change=='nonfinite': rows[1]['value']='nan'
    assert verify(rows)['status']=='BLOCKED'


def test_metric_end_steps_with_offset_accepted():
    rows=fixtures()
    for r in rows:
        if r.get('name')=='sampling/replay_samples_per_unit':
            r['step']+=3
    got=verify(rows)
    assert got['status']=='PASS'
    assert got['step_relation']['unit_metric_step_relation']=='end'


def test_no_unit_cannot_pass_even_with_metrics():
    rows=[r for r in fixtures() if r.get('action')!='train_unit']
    assert verify(rows)['status']=='BLOCKED'


def test_reject_bad_global_ratio():
    with pytest.raises(ValueError,match='GBS'):
        mod.validate(fixtures(),gbs=24,new_per_step=17,replay_per_step=6,step_offset=500)


def test_cli_preserves_historical_result(tmp_path,capsys):
    import json
    audit=tmp_path/'audit';audit.mkdir()
    original=audit/'result.json';original.write_text('{"status":"BLOCKED"}')
    (audit/'auto_learning_events.jsonl').write_text('\n'.join(json.dumps(r) for r in fixtures()))
    new=tmp_path/'offline_verification.json'
    assert mod.main(['--audit-dir',str(audit),'--output',str(new)])==0
    assert original.read_text()=='{"status":"BLOCKED"}'
    assert json.loads(new.read_text())['status']=='PASS'
    with pytest.raises(SystemExit):
        mod.main(['--audit-dir',str(audit),'--output',str(new)])


def test_real_long_train_request_parses_bounded_replay_plan():
    # Matches the real audit's 161,609-byte TrainRequest without shipping private logs.
    head = "TrainRequest(dataset=" + "x" * 161_400
    tail = ", replay=ReplayPlan(tasks=['click_bell', 'click_alarmclock'], batch_size=24, new_slots=17, replay_slots=7))"
    req = (head + tail).ljust(161_609, ' ')
    assert len(req) == 161_609
    assert mod.replay_tasks({'request': req}) == ['click_bell', 'click_alarmclock']


def test_real_end_of_unit_steps_and_long_request_passes():
    rows = fixtures()
    req = ('TrainRequest(meta=' + 'x' * 161_400 +
           ", replay=ReplayPlan(tasks=['click_bell', 'click_alarmclock'], new_slots=17, replay_slots=7))")
    for r in rows:
        if r.get('kind') == 'event':
            r['request'] = req
        if r.get('name') == 'sampling/replay_samples_per_unit':
            r['step'] += 3  # TB offsets: unit ends 503/506/509/512/515
    result = verify(rows)
    assert result['status'] == 'PASS', result['errors']
    assert result['step_relation'] == {'unit_metric_step_relation':'end', 'metric_step_shift':500}
    assert result['observed_replay_samples'] == 105


@pytest.mark.parametrize('mutation', [
    'two_plans','too_long','non_literal','missing_tasks','nested_other_field',
    'metric_end_missing','metric_end_extra','metric_end_duplicate',
    'metric_end_wrong_count','unit_length_incorrect',
])
def test_long_request_and_end_metrics_fail_closed(mutation):
    rows = fixtures()
    req = ('TrainRequest(metadata=' + 'x' * 161_400 +
           ", replay=ReplayPlan(tasks=['click_bell', 'click_alarmclock'], new_slots=17))")
    for r in rows:
        if r.get('kind') == 'event': r['request'] = req
        if r.get('name') == 'sampling/replay_samples_per_unit': r['step'] += 3
    first = next(r for r in rows if r.get('kind') == 'event')
    if mutation == 'two_plans':
        first['request'] += " ReplayPlan(tasks=['fake'])"
    elif mutation == 'too_long':
        first['request'] = 'x' * (mod.MAX_REQUEST_CHARS + 1) + req
    elif mutation == 'non_literal':
        first['request'] = req.replace("['click_bell', 'click_alarmclock']", "eval('bad')")
    elif mutation == 'missing_tasks':
        first['request'] = req.replace('tasks=', 'missing=')
    elif mutation == 'nested_other_field':
        first['request'] = req.replace('tasks=', "metadata=[], tasks=")
    elif mutation == 'metric_end_missing':
        rows = [r for r in rows if not (r.get('name') == 'sampling/replay_samples_per_unit' and r.get('step') == 509)]
    elif mutation == 'metric_end_extra':
        rows.append({'kind':'metric','name':'sampling/replay_samples_per_unit','step':518,'value':21})
    elif mutation == 'metric_end_duplicate':
        rows.append(next(r.copy() for r in rows if r.get('name') == 'sampling/replay_samples_per_unit'))
    elif mutation == 'metric_end_wrong_count':
        next(r for r in rows if r.get('name') == 'sampling/replay_samples_per_unit')['value'] = 20
    elif mutation == 'unit_length_incorrect':
        first['steps'] = 2
    result = verify(rows)
    assert result['status'] == 'BLOCKED', mutation
