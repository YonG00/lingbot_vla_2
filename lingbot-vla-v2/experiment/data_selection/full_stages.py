"""Versioned full-episode labels, independent policies and exact chunk lookups (stdlib only)."""
import hashlib,json,math
from pathlib import Path
STAGES=('idle_or_prepare','reach_aim','fine_align','pre_contact','gripper_close','grasp_lift','transport','target_align','target_hold','unknown')

def checked_json(config,field,base):
    if field not in config or field+'_sha256' not in config:raise ValueError('missing stage artifact reference: '+field)
    p=Path(config[field]);p=p if p.is_absolute() else Path(base)/p
    payload=p.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=config[field+'_sha256']:raise ValueError('stage artifact fingerprint mismatch: '+field)
    return json.loads(payload)

def validate_table(table):
    if type(table.get('schema_version'))is not int or table.get('schema_version')!=1 or table.get('coordinate_space')!='original_episode_frame':raise ValueError('stage schema/coordinates')
    if not isinstance(table.get('task'),str)or not table.get('episodes'):raise ValueError('stage task/episodes')
    fps=table.get('fps')
    if isinstance(fps,bool)or not isinstance(fps,(int,float))or not math.isfinite(fps)or fps<=0:raise ValueError('stage fps')
    ids=set()
    for ep in table['episodes']:
        idx=ep['episode_index'];n=ep['length']
        if type(idx)is not int or idx<0 or idx in ids or type(n)is not int or n<1:raise ValueError('duplicate/invalid stage episode')
        ids.add(idx)
        if set(ep['arms'])!={'left','right'}:raise ValueError('both arms need complete labels')
        for arm,segments in ep['arms'].items():
            cursor=0
            for seg in segments:
                if type(seg['start'])is not int or type(seg['end'])is not int or seg['start']!=cursor or not cursor<seg['end']<=n:raise ValueError('stage gap/overlap/bounds')
                if seg['stage']not in STAGES or not seg.get('substage')or not seg.get('event_id')or not seg.get('object_id'):raise ValueError('stage identity')
                if seg.get('proxy_only')is not True:raise ValueError('kinematic stages must declare proxy_only')
                cursor=seg['end']
            if cursor!=n:raise ValueError('stage coverage incomplete')
    return table

class StageTable:
    def __init__(self,config,base):
        self.table=validate_table(checked_json(config,'stage_table',base));self.policy=checked_json(config,'stage_policy',base)
        if config['task']!=self.table['task']or config['dataset_name']!=self.table['dataset_name']:raise ValueError('stage task/dataset identity')
        if type(self.policy.get('version'))is not int or self.policy.get('version')!=1 or self.policy.get('normalization')!='weight_sum':raise ValueError('stage policy must use weight_sum')
        if self.policy.get('future_video_rule')not in ('target_endpoint','endpoint_max'):raise ValueError('explicit future-video endpoint rule required')
        self.coefficients={}
        for branch in ['fm','visual']:
            values=self.policy[branch]
            if set(values)!=set(STAGES):raise ValueError('all stages require explicit policy weights')
            if any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or v<=0 for v in values.values()):raise ValueError('invalid stage multiplier')
            if values['unknown']!=1:raise ValueError('unknown stage cannot receive high weight')
            self.coefficients[branch]=[float(values[name])for name in STAGES]
        overrides=self.policy.get('visual_branches',{})
        allowed={'current_depth','future_depth','current_video','future_video'}
        if not isinstance(overrides,dict)or set(overrides)-allowed:raise ValueError('unknown visual branch policy')
        for branch in allowed:
            values=overrides.get(branch,self.policy['visual'])
            if set(values)!=set(STAGES)or values['unknown']!=1 or any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or v<=0 for v in values.values()):raise ValueError('invalid independent visual stage weights')
            self.coefficients[branch]=[float(values[name])for name in STAGES]
        self.episodes={};self.labels={};self.sub_labels={}
        for ep in self.table['episodes']:
            idx=ep['episode_index'];self.episodes[idx]=ep
            self.labels[idx]={};self.sub_labels[idx]={}
            for arm,segments in ep['arms'].items():
                labels=[];sub=[]
                for seg in segments:
                    labels.extend([STAGES.index(seg['stage'])]*(seg['end']-seg['start']));sub.extend([seg['stage']+'/'+seg['substage']]*(seg['end']-seg['start']))
                self.labels[idx][arm]=labels;self.sub_labels[idx][arm]=sub
        overrides=self.policy.get('overrides',{})
        if not isinstance(overrides,dict)or set(overrides)-set(self.coefficients):raise ValueError('unknown override branch')
        known={name for ep in self.sub_labels.values()for labels in ep.values()for name in labels}
        for branch,arms in overrides.items():
            if not isinstance(arms,dict)or set(arms)-{'left','right'}:raise ValueError('override arm identity')
            for arm,values in arms.items():
                if not isinstance(values,dict)or set(values)-known:raise ValueError('unknown substage override')
                for name,value in values.items():
                    if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or value<=0 or (name.startswith('unknown/')and value!=1):raise ValueError('invalid substage multiplier')
        self.frame_weights={}
        for branch,coeff in self.coefficients.items():
            self.frame_weights[branch]={}
            for ep,arms in self.labels.items():
                self.frame_weights[branch][ep]={}
                for arm,labels in arms.items():
                    base=overrides.get('visual',{}).get(arm,{})if branch not in ('fm','visual')else {}
                    own=overrides.get(branch,{}).get(arm,{})
                    self.frame_weights[branch][ep][arm]=[float(own.get(self.sub_labels[ep][arm][f],base.get(self.sub_labels[ep][arm][f],coeff[sid])))for f,sid in enumerate(labels)]

    def sample(self,ep,frame,horizon,width,layout,action_offset=0):
        if ep not in self.episodes or type(frame)is not int or not 0<=frame<self.episodes[ep]['length']:raise ValueError('stage anchor identity/bounds')
        if type(horizon)is not int or horizon<1 or action_offset not in (0,1):raise ValueError('stage horizon/offset')
        if set(layout)!={'left','right'}or not all(layout.values()):raise ValueError('stage arm layout')
        dims=layout['left']+layout['right']
        if len(set(dims))!=len(dims)or any(type(d)is not int or not 0<=d<width for d in dims):raise ValueError('stage dimension layout')
        n=self.episodes[ep]['length'];fm=[];ids=[];valid=[];arm_index=[-1]*width
        for arm_id,arm in enumerate(['left','right']):
            for d in layout[arm]:arm_index[d]=arm_id
        for k in range(horizon):
            actual=frame+action_offset+k;ok=actual<n;valid.append(ok);row=[1. if ok else 0.]*width;label=[]
            for arm in ['left','right']:
                stage=self.labels[ep][arm][actual]if ok else -1;label.append(stage)
                for d in layout[arm]:row[d]=self.frame_weights['fm'][ep][arm][actual]if ok else 0.
            fm.append(row);ids.append(label)
        def visual(t,branch):return max(self.frame_weights[branch][ep][arm][t]for arm in ['left','right'])
        future_frame=frame+horizon-1;future_valid=future_frame<n
        future_video=(visual(future_frame,'future_video')if self.policy['future_video_rule']=='target_endpoint' else max(visual(frame,'future_video'),visual(future_frame,'future_video')))if future_valid else 0.
        return dict(fm=fm,ids=ids,arm_index=arm_index,valid=valid,current_depth=visual(frame,'current_depth'),future_depth=visual(future_frame,'future_depth')if future_valid else 0.,current_video=visual(frame,'current_video'),future_video=future_video,future_target_frame=future_frame if future_valid else -1)

def derive_arm_layout(feature_config,key_mapping,active_dimensions,source_channels):
    """Map named raw channels through the actual transform's concatenation and padding."""
    layout={'left':[],'right':[]};offset=0
    for feature in feature_config.joints:
        capacity=feature_config.joints_max_dim[feature];info=key_mapping.get('action.'+feature)
        if info is not None:
            origins=info['origin_keys'];items=origins.items()if isinstance(origins,dict)else [(next(iter(v)),next(iter(v.values())))for v in origins]
            used=0
            for source,span in items:
                source=source.split('*')[0]
                if source not in source_channels:raise ValueError('unregistered source action channels')
                for raw in range(span['start'],span['end']):
                    owners=[arm for arm in ['left','right']if raw in source_channels[source][arm]]
                    if len(owners)!=1:raise ValueError('ambiguous source action channel')
                    layout[owners[0]].append(offset+used);used+=1
            if used>capacity:raise ValueError('transform capacity mismatch')
        offset+=capacity
    if set(layout['left']+layout['right'])!=set(active_dimensions):raise ValueError('derived arm layout differs from actual joint mask')
    return layout


def policy_coverage(stage,horizon=50):
    from collections import Counter
    raw={arm:Counter()for arm in ['left','right']};chunk={arm:Counter()for arm in raw};visual={name:Counter()for name in ['current_depth','future_depth','current_video','future_video']};padding=0
    for ep,e in stage.episodes.items():
        n=e['length']
        for t in range(n):
            for arm in raw:
                value=stage.frame_weights['fm'][ep][arm][t];raw[arm][value]+=1;chunk[arm][value]+=min(horizon,t+1)
            def maximum(branch,at):return max(stage.frame_weights[branch][ep][arm][at]for arm in raw)
            for branch in ['current_depth','current_video']:visual[branch][maximum(branch,t)]+=1
            future=t+horizon-1;ok=future<n
            visual['future_depth'][maximum('future_depth',future)if ok else 0.]+=1
            value=maximum('future_video',future)if ok else 0.
            if ok and stage.policy['future_video_rule']=='endpoint_max':value=max(value,maximum('future_video',t))
            visual['future_video'][value]+=1;padding+=max(0,t+horizon-n)
    return dict(raw_fm_weight_frames={a:dict(v)for a,v in raw.items()},all_anchor_chunk_fm_arm_steps={a:dict(v)for a,v in chunk.items()},all_anchor_padded_time_steps=padding,visual_weight_samples={a:dict(v)for a,v in visual.items()},coordinate_offset=0,horizon=horizon,statistics_scope='all original anchors uniformly; not sampled training progress')
