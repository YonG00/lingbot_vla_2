#!/usr/bin/env python3
"""Strict, CPU-only re-check of observed AL Replay consumption.

Never modifies historical result.json. No GPU, checkpoint or training imports.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import sys

REPLAY_RX = re.compile(r'\bReplayPlan\s*\(')
# Bound parser work, not the entire TrainRequest repr. Long reprs may contain
# legitimate ReplayPlans after 100k characters of unrelated task metadata.
MAX_REQUEST_CHARS = 2_000_000
MAX_PLAN_FIELD_CHARS = 8_192
TAGS = ('sampling/replay_samples_per_unit', 'sampling/total_samples',
        'system/global_samples_seen', 'system/units_run', 'curriculum/units_completed')


def _safe_int(x, context):
    if isinstance(x, bool):
        raise ValueError(f'{context}: boolean is not a count')
    v = float(x)
    if not math.isfinite(v) or v < 0 or not v.is_integer():
        raise ValueError(f'{context}: invalid count {x!r}')
    return int(v)


def _literal_after(text: str, pos: int):
    """Extract balanced Python list/tuple literal without executing text."""
    at = next((i for i in range(pos, len(text)) if text[i] in '[('), -1)
    if at < 0:
        return None
    stack = []
    quote = None
    escaped = False
    for i in range(at, len(text)):
        c = text[i]
        if quote:
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == quote:
                quote = None
        elif c in ('\'', '"'):
            quote = c
        elif c in '[(':
            stack.append(']' if c == '[' else ')')
        elif c in '])':
            if not stack or c != stack.pop():
                return None
            if not stack:
                try:
                    return ast.literal_eval(text[at:i+1])
                except (SyntaxError, ValueError, TypeError, MemoryError):
                    return None
    return None


def replay_tasks(row: dict) -> list[str]:
    direct = row.get('old_tasks')
    if isinstance(direct, (list, tuple)):
        return [x for x in direct if isinstance(x, str) and x]
    req = row.get('request')
    if isinstance(req, dict):
        for k in ('replay_plan', 'replay', 'old_tasks'):
            value = req.get(k)
            if isinstance(value, dict) and isinstance(value.get('tasks'), list):
                return [x for x in value['tasks'] if isinstance(x, str) and x]
            if isinstance(value, list):
                return [x for x in value if isinstance(x, str) and x]
        return []
    if not isinstance(req, str) or len(req) > MAX_REQUEST_CHARS:
        return []
    found = REPLAY_RX.finditer(req)
    match = next(found, None)
    # A second ReplayPlan makes provenance ambiguous: never guess which one
    # was the actual TrainRequest replay allocation. Avoid collecting all matches.
    if match is None or next(found, None) is not None:
        return []
    # Read only the bounded ReplayPlan field, never parse the giant repr.
    # The current production repr starts ReplayPlan(tasks=[...], ...).
    tail = req[match.end():match.end() + MAX_PLAN_FIELD_CHARS]
    tm = re.match(r'\s*tasks\s*=\s*', tail)
    if not tm:
        return []
    lit = _literal_after(tail, tm.end())
    if not isinstance(lit, (list, tuple)) or any(not isinstance(t, str) or not t for t in lit):
        return []
    return list(lit)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{i}: JSON object required')
            out.append(row)
    return out


def sha(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tb_scalars(path: Path) -> list[dict]:
    """Read actual TensorBoard scalar events, not console plans.

    TensorBoard is an optional dependency; fail closed when unavailable.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    found = []
    files = [path] if path.is_file() else sorted(path.rglob('events.out.tfevents.*'))
    for f in files:
        acc = EventAccumulator(str(f), size_guidance={'scalars':0,'tensors':0})
        acc.Reload()
        for name in acc.Tags().get('scalars', []):
            if name in TAGS:
                for e in acc.Scalars(name):
                    found.append({'kind':'metric','name':name,'step':e.step,'value':e.value,
                                  'source_file':str(f)})
        # Modern SummaryWriter may put scalars in tensor entries.
        for name in acc.Tags().get('tensors', []):
            if name not in TAGS:
                continue
            from tensorboard.util import tensor_util
            for e in acc.Tensors(name):
                scalar = tensor_util.make_ndarray(e.tensor_proto)
                if scalar.size == 1:
                    found.append({'kind':'metric','name':name,'step':e.step,
                                  'value':float(scalar.item()),'source_file':str(f)})
    return found


def _metrics_by_tag(rows):
    by = {}
    for r in rows:
        if r.get('kind') != 'metric' or r.get('name') not in TAGS:
            continue
        name = r['name']
        try:
            step = _safe_int(r['step'], f'{name} step')
            value = _safe_int(r['value'], name)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f'malformed observed metric {r}: {e}') from e
        bucket = by.setdefault(name, {})
        # Multiple identical scalar tags/steps from different sources can hide conflicts.
        if step in bucket and bucket[step] != value:
            raise ValueError(f'conflicting {name} at {step}')
        if step in bucket:
            raise ValueError(f'duplicate {name} at {step} (choose ONE authoritative metric source)')
        bucket[step] = value
    return by


def _unit_match_modes(units, metrics, offset):
    """Select ONE unambiguous step relation for all units, never nearest-step join."""
    modes = []
    for relation in ('start', 'end'):
        for shift in (0, offset):
            keys = [u['start'] + (u['steps'] if relation == 'end' else 0) + shift
                    for u in units]
            if len(set(keys)) != len(keys):
                continue
            if set(keys) == set(metrics):
                modes.append((relation, shift, keys))
    unique = {(tuple(x[2])):x for x in modes}
    if len(unique) != 1:
        raise ValueError(f'unit-to-metric step mapping ambiguous/missing: '
                         f'unit starts {[x["start"] for x in units]}, '
                         f'metric steps {sorted(metrics)}, candidates {[(x[0],x[1]) for x in modes]}')
    return next(iter(unique.values()))


def validate(rows: list[dict], *, gbs: int, new_per_step: int, replay_per_step: int,
             step_offset: int, require_all_units: bool=True) -> dict:
    errors = []
    units = []
    if gbs < 1 or new_per_step < 1 or replay_per_step < 1 or new_per_step+replay_per_step != gbs:
        raise ValueError('invalid GBS/NEW/Replay plan')
    for r in rows:
        if r.get('kind') != 'event' or r.get('action') != 'train_unit':
            continue
        try:
            start = _safe_int(r['step'],'train_unit start')
            steps = _safe_int(r.get('steps',r.get('batches_built')),'train_unit steps')
        except (ValueError, TypeError) as e:
            errors.append(str(e)); continue
        if steps == 0:
            errors.append(f'unit@{start}: zero optimizer steps'); continue
        tasks = replay_tasks(r)
        if not tasks or len(set(tasks)) != len(tasks):
            errors.append(f'unit@{start}: no/duplicate observed ReplayPlan tasks')
        units.append({'start':start,'steps':steps,'replay_tasks':tasks})
    units.sort(key=lambda x:x['start'])
    if not units:
        errors.append('no consumed Train Unit')
    for i, u in enumerate(units):
        if i and units[i-1]['start']+units[i-1]['steps'] != u['start']:
            errors.append('Train Unit step sequence has gap/overlap/duplicates')
    try:
        by = _metrics_by_tag(rows)
    except ValueError as e:
        errors.append(str(e)); by = {}
    by_replay = by.get('sampling/replay_samples_per_unit', {})
    relation = None
    if units:
        try:
            mode, shift, keys = _unit_match_modes(units,by_replay,step_offset)
            relation = {'unit_metric_step_relation':mode,'metric_step_shift':shift}
            for u, key in zip(units, keys):
                u['metric_step'] = key
                u['actual_replay'] = by_replay[key]
                u['expected_replay'] = u['steps']*replay_per_step
                u['expected_samples'] = u['steps']*gbs
                u['expected_new'] = u['steps']*new_per_step
                if u['actual_replay'] != u['expected_replay']:
                    errors.append(f'unit@{u["start"]}: observed Replay={u["actual_replay"]}, expected={u["expected_replay"]}')
        except ValueError as e:
            errors.append(str(e))
    steps = sum(u['steps'] for u in units)
    replay_actual = sum(u.get('actual_replay',0) for u in units)
    total_expected = steps*gbs
    for name, expected in [('sampling/total_samples',total_expected),
                           ('system/global_samples_seen',total_expected),
                           ('system/units_run',len(units)),
                           ('curriculum/units_completed',len(units))]:
        vals = by.get(name,{})
        if not vals:
            errors.append(f'missing independent metric {name}'); continue
        # Cumulative metrics may be reported at each Unit; require final = expected.
        last_step = max(vals)
        if vals[last_step] != expected:
            errors.append(f'{name} last={vals[last_step]} != {expected}')
    # Replay totals must be observed per Unit, not reconstructed from a global plan.
    status = 'PASS' if not errors else 'BLOCKED'
    return {'kind':'offline_replay_audit','status':status,'diagnosis':'verified' if not errors else 'evidence_incomplete_or_inconsistent',
            'errors':errors, 'unit_count':len(units), 'optimizer_steps':steps, 'gbs':gbs,
            'observed_total_samples_expected':total_expected,
            'observed_replay_samples':replay_actual if relation else None,
            'expected_replay_samples':steps*replay_per_step,'expected_new_samples':steps*new_per_step,
            'step_relation':relation,'units':units,
            'note':'Independent per-unit Replay metrics and cumulative sample counters REQUIRED for PASS; original verdict unchanged.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--tb-logdir',type=Path,help='original unmodified TensorBoard event directory')
    p.add_argument('--metric-jsonl',type=Path,help='observed scalar export, not a hand-authored ratio plan')
    p.add_argument('--gbs',type=int,default=24)
    p.add_argument('--new',type=int,default=17)
    p.add_argument('--replay',type=int,default=7)
    p.add_argument('--step-offset',type=int,default=500)
    p.add_argument('--output',type=Path,help='new output JSON; refuses overwrite')
    a=p.parse_args(argv)
    evidence=a.audit_dir/'auto_learning_events.jsonl'
    rows=load_jsonl(evidence)
    sources={str(evidence):sha(evidence)}
    if a.tb_logdir:
        trows=tb_scalars(a.tb_logdir)
        rows.extend(trows)
        sources.update({str(f):sha(f) for f in sorted(a.tb_logdir.rglob('events.out.tfevents.*'))})
    if a.metric_jsonl:
        rows.extend(load_jsonl(a.metric_jsonl))
        sources[str(a.metric_jsonl)]=sha(a.metric_jsonl)
    result=validate(rows,gbs=a.gbs,new_per_step=a.new,replay_per_step=a.replay,step_offset=a.step_offset)
    result['sources_sha256']=sources
    result['historical_result_unchanged']=True
    result['source_count']=len(sources)
    if a.output:
        if a.output.exists():
            p.error('output exists: refusal to overwrite evidence')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['status']=='PASS' else 2

if __name__=='__main__':
    sys.exit(main())
