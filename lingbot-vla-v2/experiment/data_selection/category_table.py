"""One category/weight contract for trajectory phases and overlapping target labels."""
import math
from .full_stages import checked_json,StageTable,STAGES
ARMS=('left','right')
BRANCHES=('fm','current_depth','future_depth','current_video','future_video')
def arm_values(value):
    values=value if isinstance(value,dict)else {a:value for a in ARMS}
    if set(values)!=set(ARMS)or any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or v<=0 for v in values.values()):raise ValueError('category weight must be positive scalar or left/right map')
    return {a:float(values[a])for a in ARMS}
class CategoryTable:
    sample=StageTable.sample
    def __init__(self,config,base):
        self.table=checked_json(config,'category_table',base);self.policy=checked_json(config,'category_weights',base)
        t,p=self.table,self.policy
        if type(t.get('schema_version'))is not int or t.get('schema_version')!=1 or t.get('coordinate_space')!='original_episode_frame' or t.get('label_source')not in ('trajectory_stages','target_classes'):raise ValueError('category table schema/source')
        fps=t.get('fps')
        if isinstance(fps,bool)or not isinstance(fps,(int,float))or not math.isfinite(fps)or fps<=0:raise ValueError('invalid category FPS')
        if not isinstance(t.get('source_action_channels'),dict)or not t['source_action_channels']:raise ValueError('category action channel mapping required')
        if type(p.get('version'))is not int or p['version']!=1:raise ValueError('category weight version')
        if config['task']!=t.get('task')or config['dataset_name']!=t.get('dataset_name'):raise ValueError('category task/dataset identity')
        if config.get('weighting_unit')!='action_timestep'or p.get('weighting_unit')!='action_timestep'or p.get('normalization')!='weight_sum'or p.get('overlap_rule')!='max':raise ValueError('category requires per-action weight_sum/max')
        if p.get('future_video_rule')!='target_endpoint':raise ValueError('category visual weights require actual target endpoint')
        if not isinstance(p.get('classes'),dict)or not p['classes']or 'background'not in p['classes']:raise ValueError('explicit category definitions/background required')
        self.coefficients={b:{}for b in BRANCHES}
        for name,entry in p['classes'].items():
            if not isinstance(name,str)or not name or not isinstance(entry,dict)or set(entry)-{'fm','visual','visual_branches'}:raise ValueError('invalid category definition')
            self.coefficients['fm'][name]=arm_values(entry['fm']);visual=arm_values(entry['visual']);overrides=entry.get('visual_branches',{})
            if not isinstance(overrides,dict)or set(overrides)-set(BRANCHES[1:]):raise ValueError('unknown category visual branch')
            for branch in BRANCHES[1:]:self.coefficients[branch][name]=arm_values(overrides[branch])if branch in overrides else visual
        if any(self.coefficients[b][name][a]!=1 for b in BRANCHES for a in ARMS for name in p['classes']if name=='background'or name=='unknown'or name.startswith('unknown/')):raise ValueError('background/unknown must remain1')
        self.episodes={};self.labels={};self.sub_labels={};self.category_labels={};self.frame_weights={b:{}for b in BRANCHES}
        if not t.get('episodes'):raise ValueError('empty category episodes')
        for ep in t['episodes']:
            idx,n=ep['episode_index'],ep['length']
            if type(idx)is not int or idx<0 or idx in self.episodes or type(n)is not int or n<1 or set(ep['arms'])!=set(ARMS):raise ValueError('category episode identity')
            self.episodes[idx]=ep;self.labels[idx]={};self.sub_labels[idx]={};self.category_labels[idx]={}
            for b in BRANCHES:self.frame_weights[b][idx]={}
            for arm,segs in ep['arms'].items():
                cursor=0;labels=[];oldids=[]
                for seg in segs:
                    if type(seg['start'])is not int or type(seg['end'])is not int or seg['start']!=cursor or not cursor<seg['end']<=n:raise ValueError('category gap/overlap/bounds')
                    names=seg['classes']
                    if not isinstance(names,list)or not names or len(set(names))!=len(names)or set(names)-set(p['classes']):raise ValueError('unknown/duplicate frame category')
                    legacy=seg.get('legacy_stage');sid=STAGES.index(legacy)if legacy in STAGES else -1
                    labels.extend([tuple(names)]*(seg['end']-cursor));oldids.extend([sid]*(seg['end']-cursor));cursor=seg['end']
                if cursor!=n:raise ValueError('category coverage incomplete')
                self.labels[idx][arm]=oldids;self.category_labels[idx][arm]=labels;self.sub_labels[idx][arm]=['|'.join(v)for v in labels]
                for b in BRANCHES:self.frame_weights[b][idx][arm]=[max(self.coefficients[b][name][arm]for name in names)for names in labels]
