"""Exact cumulative target/background and target-kind quotas; stable resume."""
from collections import Counter
from fractions import Fraction
import random
from .planning import PlannedSampler,digest

class SegmentSampler(PlannedSampler):
    def __init__(self,plan,pools,seed,task,events,background,fraction,ratios):
        super().__init__(plan,pools,seed)
        self.task,self.events,self.background=task,events,background
        self.fraction=Fraction(str(fraction));self.ratios={k:Fraction(str(v)) for k,v in ratios.items()}
        if not 0<self.fraction<1 or set(events)!=set(ratios) or len(events)!=2:raise ValueError('two target kinds and nonzero background required')
        if any(v<=0 for v in self.ratios.values()) or any(not groups or any(not group for group in groups) for groups in events.values()) or not background:raise ValueError('empty segment sampling pool')
        union={i for groups in events.values() for group in groups for i in group}
        if union & set(background) or union|set(background)!=set(pools[task]):raise ValueError('segment/background must partition task')
        self.signature=digest(dict(base=self.signature,task=task,events=events,background=background,fraction=str(self.fraction),ratios={k:str(v) for k,v in self.ratios.items()}))
    def draw_row(self,row,rng,counters):
        ids=[];roles=[]
        for task,count in row['samples_by_task'].items():
            if task!=self.task:
                ids.extend(rng.choice(self.pools[task]) for _ in range(count));roles.extend(['other_task']*count);continue
            before=counters['task'];after=before+count
            target_before=int(before*self.fraction);target_after=int(after*self.fraction)
            names=list(self.ratios);a,b=names;ratio=self.ratios[a]/sum(self.ratios.values())
            first=int(target_after*ratio)-int(target_before*ratio)
            counts={a:first,b:target_after-target_before-first,'background':count-(target_after-target_before)}
            for kind,n in counts.items():
                for _ in range(n):
                    index=rng.choice(self.background) if kind=='background' else rng.choice(rng.choice(self.events[kind]))
                    ids.append(index);roles.append(kind)
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
