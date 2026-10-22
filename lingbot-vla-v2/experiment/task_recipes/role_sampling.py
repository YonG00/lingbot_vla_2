"""Role quotas and occurrence-level loss weights, with deterministic resume."""
from collections import Counter
from fractions import Fraction
import random
import math
from .planning import PlannedSampler,digest

class RoleSampler(PlannedSampler):
    def __init__(self,plan,pools,seed,task,events,background,ratios,loss_weights):
        super().__init__(plan,pools,seed)
        self.task,self.events,self.background=task,events,background
        self.ratios={k:Fraction(str(v)) for k,v in ratios.items()};self.loss_weights=loss_weights
        if set(ratios)!=set(events)|{'background'} or set(ratios)!=set(loss_weights):raise ValueError('role identity mismatch')
        if any(v<=0 for v in self.ratios.values()) or any(isinstance(v,bool)or not math.isfinite(v)or v<=0 for v in loss_weights.values()):raise ValueError('invalid role weight')
        if not background or any(not groups or any(not event for event in groups) for groups in events.values()):raise ValueError('empty role pool')
        union={i for groups in events.values() for event in groups for i in event}
        if union&set(background) or union|set(background)!=set(pools[task]):raise ValueError('role pools must cover task')
        self.signature=digest(dict(base=self.signature,events=events,background=background,ratios=ratios,loss_weights=loss_weights,task=task))
    def draw_row(self,row,rng,counters):
        ids=[];roles=[]
        for task,count in row['samples_by_task'].items():
            if task!=self.task:raise ValueError('role sampler supports one task only')
            before=counters['task'];after=before+count;total=sum(self.ratios.values())
            names=list(self.events)+['background'];counts={k:int(after*self.ratios[k]/total)-int(before*self.ratios[k]/total) for k in names[:-1]}
            counts['background']=count-sum(counts.values())
            if min(counts.values())<0:raise ValueError('batch too small for cumulative role quotas')
            for kind,n in counts.items():
                for _ in range(n):
                    i=rng.choice(self.background) if kind=='background' else rng.choice(rng.choice(self.events[kind]))
                    ids.append((i,float(self.loss_weights[kind]),kind));roles.append(kind)
            counters['task']=after
        pairs=list(zip(ids,roles));rng.shuffle(pairs)
        return [p[0] for p in pairs],[p[1] for p in pairs]
    def __iter__(self):
        counters=Counter()
        for row in self.plan['steps']:
            ids,_=self.draw_row(row,random.Random(f'{self.seed}:{row["step"]}'),counters)
            base=(row['step']-1)*self.plan['global_batch_size']
            for slot,index in enumerate(ids):
                position=base+slot
                if position<self.cursor:continue
                self.cursor=position+1;yield index

def override_role_weight(item,weight):
    import torch
    if isinstance(weight,bool)or not isinstance(weight,(int,float))or not math.isfinite(weight)or weight<=0:raise ValueError('invalid occurrence loss weight')
    item=dict(item);item['sample_loss_weight']=torch.tensor(weight,dtype=torch.float32)
    return item
