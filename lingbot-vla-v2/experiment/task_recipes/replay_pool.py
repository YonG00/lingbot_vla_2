"""Build an auditable replay pool from historical successful closed-loop observations."""
from copy import deepcopy
import json
from pathlib import Path


def eligible_tasks(snapshot):
    if snapshot.get('version')!=1:raise ValueError('invalid replay pool version')
    selected=[];seen=set()
    for task in snapshot['tasks']:
        if task['task'] in seen:raise ValueError('duplicate replay task')
        seen.add(task['task'])
        observations=task['observations']
        for row in observations:
            s,n=row['successes'],row['episodes']
            if type(s) is not int or type(n) is not int or n<=0 or not 0<=s<=n or not row.get('source'):
                raise ValueError('invalid replay qualification evidence')
        if any(row['successes']*3>=row['episodes'] for row in observations):
            if task['task'] in selected:raise ValueError('duplicate qualifying task')
            selected.append(task['task'])
    return selected


def merge_results(snapshot,results,source):
    """Import complete closed-loop reports; a later failure never removes prior eligibility."""
    updated=deepcopy(snapshot);eligible_tasks(updated)
    if not isinstance(results,list):results=[results]
    for result in results:
        if result.get('state')!='complete':raise ValueError('cannot import incomplete evaluation')
        row=dict(successes=result['successes'],episodes=result['episodes'],source=source,evidence_kind='run_summary')
        name=result['task']
        eligible_tasks(dict(version=1,tasks=[dict(task=name,observations=[row])]))
        task=next((t for t in updated['tasks'] if t['task']==name),None)
        if task is None:task=dict(task=name,observations=[]);updated['tasks'].append(task)
        existing=[x for x in task['observations'] if x['source']==source]
        if existing and existing!=[row]:raise ValueError('conflicting evaluation evidence for '+name)
        if not existing:task['observations'].append(row)
    return updated


def resolve_pool(raw,root):
    c=deepcopy(raw);r=c['train'].get('replay',{})
    if 'pool_file' not in r:return c
    if 'pool_snapshot' not in r:
        if r.get('tasks'):raise ValueError('choose either explicit replay tasks or pool_file')
        path=Path(r['pool_file']);path=path if path.is_absolute() else root/path
        r['pool_file']=str(path.resolve())
        r['pool_snapshot']=json.loads(path.read_text())
    names=eligible_tasks(r['pool_snapshot'])
    expected={name:dict(weight=1) for name in names}
    if r.get('tasks') and r['tasks']!=expected:raise ValueError('frozen replay pool/tasks mismatch')
    r['tasks']=expected
    return c


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Import completed evaluation summaries into a cumulative replay registry.')
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--summary',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True,help='New registry file; never overwrite evidence in place.')
    args=parser.parse_args();snapshot=json.loads(args.registry.read_text())
    for summary in args.summary:snapshot=merge_results(snapshot,json.loads(summary.read_text()),str(summary.resolve()))
    with args.output.open('x') as stream:stream.write(json.dumps(snapshot,indent=2)+'\n')
    print(json.dumps(dict(eligible_tasks=eligible_tasks(snapshot),output=str(args.output)),indent=2))
