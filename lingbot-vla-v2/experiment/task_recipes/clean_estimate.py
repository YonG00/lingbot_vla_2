"""Task-balanced historical estimate; not a full current-model benchmark."""
import random

def estimate(tasks,historical,current):
    tasks=list(tasks)
    if len(tasks)!=len(set(tasks)):raise ValueError('duplicate catalog')
    latest={r['task']:dict(r,current_model=False) for r in historical};fresh={}
    for r in current:
        if r['task'] in fresh:raise ValueError('duplicate current result')
        fresh[r['task']]=dict(r,current_model=True)
    latest.update(fresh);rows=[];missing=[]
    for task in tasks:
        if task not in latest:missing.append(task);continue
        r=latest[task];s,n=r['successes'],r['episodes']
        if type(s)is not int or type(n)is not int or n<=0 or not 0<=s<=n:raise ValueError('invalid counts')
        rows.append(dict(r,success_rate=s/n))
    measured=[r for r in rows if r['current_model']]
    return dict(kind='mixed_checkpoint_latest_per_task_estimate',not_full_current_model_benchmark=True,
        task_count=len(tasks),covered_tasks=len(rows),missing_tasks=missing,
        macro_success_rate=sum(r['success_rate'] for r in rows)/len(rows) if rows else None,
        current_model_tasks=len(measured),historical_tasks=len(rows)-len(measured),
        current_model_successes=sum(r['successes'] for r in measured),current_model_episodes=sum(r['episodes'] for r in measured),per_task=rows)

def choose_retention(snapshot,excluded,seed,count=3):
    eligible=sorted(t['task'] for t in snapshot['tasks'] if t['task'] not in excluded and any(r['successes']>0 for r in t['observations']))
    if len(eligible)<count:raise ValueError('not enough successful tasks')
    return random.Random(seed).sample(eligible,count)
