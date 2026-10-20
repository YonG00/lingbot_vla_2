"""确定性的主任务步数配额与复习样本比例，不把混合样本算作独立优化器步。"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import random


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class FairChoice:
    def __init__(self,weights):
        self.weights={k:Fraction(str(v)) for k,v in weights.items()}
        self.counts=Counter()
    def next(self,eligible=None):
        names=sorted(self.weights) if eligible is None else sorted(eligible)
        if not names:raise ValueError('empty eligible task pool')
        name=min(names,key=lambda n:self.counts[n]/self.weights[n])
        self.counts[name]+=1
        return name


def build_plan(c):
    t=c['train'];main=t['main_tasks'];r=t['replay']
    batch=t['micro_batch_size']*t['gradient_accumulation_steps']
    primary=FairChoice({k:v['weight'] for k,v in main.items()})
    frame_choice=FairChoice({k:v['weight'] for k,v in main.items()})
    replay=FairChoice({k:v['weight'] for k,v in r['tasks'].items()})
    fraction=Fraction(str(r['fraction']))
    ordered=[name for name in t.get('task_order',[]) for _ in range(main[name]['steps'])]
    rows=[]
    for step in range(t['steps']):
        review_n=int((step+1)*batch*fraction)-int(step*batch*fraction)
        counts=Counter()
        chosen=None
        if t['mode']=='sequential':
            chosen=ordered[step]
            counts[chosen]=batch-review_n
        elif t['mode']=='task_units':
            eligible=[name for name in main if primary.counts[name]<main[name]['steps']]
            chosen=primary.next(eligible)
            counts[chosen]=batch-review_n
        else:
            for _ in range(batch-review_n):counts[frame_choice.next()]+=1
        for _ in range(review_n):counts[replay.next()]+=1
        rows.append(dict(step=step+1,main_task=chosen,samples_by_task=dict(counts),replay_samples=review_n))
    return dict(mode=t['mode'],global_batch_size=batch,steps=rows,recipe_sha256=digest(c))


def task_groups(c,manifest):
    tasks=list(c['train']['main_tasks'])+list(c['train']['replay']['tasks'])
    groups={};seen=set();frames=0
    for task in tasks:
        entry=manifest['tasks'][task]
        ids=entry['train_ids']+(entry['val_ids'] if c['train']['episode_split']=='all' else [])
        if not ids or len(ids)!=len(set(ids)) or seen&set(ids):raise ValueError('empty/overlapping episode groups')
        groups[task]=sorted(ids);seen.update(ids)
        frames+=entry['train_frames']+(entry['val_frames'] if c['train']['episode_split']=='all' else 0)
    for task in c['evaluation']['tasks']:
        if task not in manifest['tasks']:raise ValueError('unknown evaluation task '+task)
    return groups,frames


def index_pools(groups,episodes):
    owner={e:t for t,ids in groups.items() for e in ids}
    if len(owner)!=sum(map(len,groups.values())) or set(episodes)!=set(owner):raise ValueError('episode/index mismatch')
    pools={t:[] for t in groups}
    for i,e in enumerate(episodes):pools[owner[e]].append(i)
    if any(not p for p in pools.values()):raise ValueError('empty task frame pool')
    return pools


class PlannedSampler:
    """Raw filtered-dataset indices. Cursor is not an optimizer completion counter."""
    def __init__(self,plan,pools,seed):
        self.plan,self.pools,self.seed=plan,pools,seed
        self.cursor=0
        self.signature=digest(dict(plan=plan,pools=pools,seed=seed))
    def __len__(self):return len(self.plan['steps'])*self.plan['global_batch_size']
    def set_epoch(self,epoch):
        if epoch!=0:raise ValueError('recipe must fit one virtual epoch; refusing duplicate budget')
    def __iter__(self):
        for row in self.plan['steps']:
            rng=random.Random(f'{self.seed}:{row["step"]}')
            ids=[]
            for task,count in row['samples_by_task'].items():ids.extend(rng.choice(self.pools[task]) for _ in range(count))
            rng.shuffle(ids)
            base=(row['step']-1)*self.plan['global_batch_size']
            for i,index in enumerate(ids):
                position=base+i
                if position<self.cursor:continue
                self.cursor=position+1
                yield index
    def state_dict(self):return dict(cursor=self.cursor,signature=self.signature)
    def load_state_dict(self,state):
        if state['signature']!=self.signature or type(state['cursor']) is not int or not 0<=state['cursor']<=len(self):raise ValueError('incompatible sampler state')
        self.cursor=state['cursor']


def confirmed_progress(plan,step):
    if not 0<=step<=len(plan['steps']):raise ValueError('step outside plan')
    samples=Counter();main=Counter();review=0
    for row in plan['steps'][:step]:
        samples.update(row['samples_by_task']);review+=row['replay_samples']
        if row['main_task'] is not None:main[row['main_task']]+=1
    return dict(confirmed_optimizer_steps=step,main_steps=dict(main),samples_by_task=dict(samples),replay_samples=review)
