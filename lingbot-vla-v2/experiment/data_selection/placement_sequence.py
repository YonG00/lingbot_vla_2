"""Task-specific offline placement proxy; no image labels, contact claim or torch."""
from dataclasses import dataclass, asdict
from collections import Counter
import math
import numpy as np
from .grasp_sequence import forward_kinematics, select_rows, Settings as GraspSettings

@dataclass(frozen=True)
class PlacementSettings:
    placement_ordinal: int = 2
    approach_search_seconds: float = 2.
    aiming_search_seconds: float = 4.
    approach_distance_min_m: float = .08
    approach_distance_max_m: float = .11
    approach_downward_degrees: float = 45.
    minimum_descent_m: float = .04
    perpendicular_tolerance_m: float = .02
    forward_fraction: float = .7
    reverse_tolerance_m: float = .002
    closed_threshold: float = .2
    open_threshold: float = .95
    release_drift_m: float = .015
    settle_distance_m: float = .01
    settle_hold_seconds: float = .13
    retreat_seconds: float = 3.
    retreat_height_m: float = .03
    selected_stages: tuple = ('place_aim','place_approach','place_settle','place_release','place_retreat')

    def validate(self):
        if type(self.placement_ordinal) is not int or self.placement_ordinal not in (1,2):
            raise ValueError('placement ordinal must be 1 or 2')
        for key,value in asdict(self).items():
            if key in ('placement_ordinal','selected_stages'):continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError('invalid '+key)
        if not self.approach_distance_min_m<self.approach_distance_max_m:raise ValueError('distance range')
        if not self.closed_threshold<.5<self.open_threshold<=1 or self.forward_fraction>1 or self.approach_downward_degrees>=90:raise ValueError('invalid threshold')
        allowed={'place_aim','place_approach','place_settle','place_release','place_retreat'}
        if not self.selected_stages or set(self.selected_stages)-allowed:raise ValueError('invalid stages')


def select_from_arrays(x, grasp_events, fps, settings=PlacementSettings()):
    settings.validate()
    if not math.isfinite(fps) or fps<=0:raise ValueError('invalid fps')
    raw=[]
    for ep in np.unique(x['episode']):
        ids=np.flatnonzero(x['episode']==ep);s=x['state'][ids];local=[]
        if list(x['frame'][ids])!=list(range(len(ids))):raise ValueError('missing/duplicate frame indices')
        for arm,j in [('left',6),('right',13)]:
            g=s[:,j];p=x[arm+'_tcp'][ids]
            closes=np.flatnonzero((g[1:]<=.5)&(g[:-1]>.5))+1
            opens=np.flatnonzero((g[1:]>=.5)&(g[:-1]<.5))+1
            for close in closes:
                next_closes=closes[closes>close]
                upper=int(next_closes[0]) if len(next_closes) else len(g)
                match=opens[(opens>close)&(opens<upper)]
                if not len(match):continue
                op=int(match[0]);before=np.flatnonzero(g[close:op]<=settings.closed_threshold);after=np.flatnonzero(g[op:upper]>=settings.open_threshold)
                if not len(before) or not len(after):continue
                start=int(close+before[-1]+1);end=int(op+after[0]);target=np.median(p[start:end+1],axis=0)
                local.append(dict(episode_index=int(ep),arm=arm,closure=int(close),release_start=start,release_end=end,release_drift_m=float(np.linalg.norm(p[start:end+1]-target,axis=1).max())))
        local.sort(key=lambda e:e['release_start'])
        for i,e in enumerate(local):e.update(placement_ordinal=i+1,episode_placements=len(local));raw.append(e)
    events=[];selected={}
    for row in raw:
        e=dict(row);ids=np.flatnonzero(x['episode']==e['episode_index']);p=x[e['arm']+'_tcp'][ids];g=x['state'][ids,6 if e['arm']=='left' else 13]
        s=e['release_start'];end=e['release_end'];target=np.median(p[s:end+1],axis=0);dist=np.linalg.norm(p-target,axis=1)
        lower=max(e['closure'],s-math.ceil(settings.approach_search_seconds*fps))
        candidates=np.flatnonzero((dist[lower:s]>=settings.approach_distance_min_m)&(dist[lower:s]<=settings.approach_distance_max_m)&(p[lower:s,2]>target[2])&(g[lower:s]<=settings.closed_threshold))+lower
        reasons=[]
        if e['episode_placements']!=2:reasons.append('ambiguous_placement_count')
        if not len(candidates):
            e.update(accepted=False,rejection=reasons+['no_final_approach']);events.append(e);continue
        a=int(candidates[-1]);axis=target-p[a];axis/=np.linalg.norm(axis)
        angle=float(np.degrees(np.arccos(np.clip(-axis[2],-1,1))))
        delta=p[a:s+1]-target;perp=np.linalg.norm(delta-np.outer(delta@axis,axis),axis=1)
        progress=np.diff(p[a:s+1],axis=0)@axis
        retreat=np.flatnonzero(p[end:min(len(p),end+math.ceil(settings.retreat_seconds*fps)+1),2]-target[2]>=settings.retreat_height_m)
        e.update(approach_start=a,approach_angle_deg=angle,downward_drop_m=float(p[a,2]-target[2]),path_perpendicular_max_m=float(perp.max()),forward_fraction=float(np.mean(progress>=-settings.reverse_tolerance_m)))
        if angle>settings.approach_downward_degrees:reasons.append('approach_not_downward')
        if e['downward_drop_m']<settings.minimum_descent_m:reasons.append('insufficient_descent')
        if e['path_perpendicular_max_m']>settings.perpendicular_tolerance_m:reasons.append('curved_final_approach')
        if e['forward_fraction']<settings.forward_fraction:reasons.append('nonprogressive_approach')
        if np.any(g[a:s]>settings.closed_threshold):reasons.append('early_opening')
        if e['release_drift_m']>settings.release_drift_m:reasons.append('moving_during_release')
        if not len(retreat):reasons.append('no_retreat')
        stable=np.linalg.norm(p-target,axis=1)<=settings.settle_distance_m
        settle=s-1
        while settle>a and stable[settle-1] and g[settle-1]<=settings.closed_threshold:settle-=1
        if s-settle<max(2,math.ceil(settings.settle_hold_seconds*fps)):reasons.append('no_stable_settle')
        matches=[v for v in grasp_events if v['accepted'] and v['episode_index']==e['episode_index'] and v['arm']==e['arm'] and v['closure_crossing_frame']==e['closure']]
        if len(matches)!=1:reasons.append('no_confirmed_grasp_proxy')
        lift=matches[0]['lift_confirmation_frame'] if len(matches)==1 else e['closure']
        aim=max(lift,a-math.ceil(settings.aiming_search_seconds*fps))
        ret=end+int(retreat[0]) if len(retreat) else end
        if len(retreat) and np.min(g[end:ret+1])<settings.open_threshold:reasons.append('reclosing_during_retreat')
        stages=dict(place_aim=[int(aim),a-1],place_approach=[a,int(settle-1)],place_settle=[int(settle),s-1],place_release=[s,end],place_retreat=[end+1,ret])
        e.update(accepted=not reasons,rejection=reasons,stages=stages);events.append(e)
        if reasons or e['placement_ordinal']!=settings.placement_ordinal:continue
        for stage in settings.selected_stages:
            lo,hi=stages[stage]
            for frame in range(lo,hi+1):
                key=(e['episode_index'],frame)
                selected.setdefault(key,dict(episode_index=key[0],frame_index=key[1],labels=[]))['labels'].append(dict(stage=stage,arm=e['arm'],placement_ordinal=settings.placement_ordinal))
    summary=dict(task='stack_bowls_two',target_placement_ordinal=settings.placement_ordinal,episodes=len(np.unique(x['episode'])),total_frames=len(x['frame']),placement_pairs=len(events),accepted_target_sequences=sum(e['accepted'] and e['placement_ordinal']==settings.placement_ordinal for e in events),selected_frames=len(selected),selected_fraction=len(selected)/len(x['frame']),rejections=dict(Counter(r for e in events for r in e['rejection'])),stage_frames=dict(Counter(label['stage']for e in selected.values()for label in e['labels'])),kinematic_proxy_only=True,collision_verified=False,object_pose_verified=False)
    return summary,events,[selected[k] for k in sorted(selected)]


def select_placements(rows,spec,fps,settings=PlacementSettings()):
    # Grasp selector validates frame continuity, timestamps, 14D state and finite FK.
    _,grasp_events,_=select_rows(rows,spec,fps,GraspSettings())
    rows=sorted(rows,key=lambda r:(r['episode_index'],r['frame_index']))
    state=np.asarray([r['observation.state']for r in rows],float)
    x=dict(episode=np.array([r['episode_index']for r in rows]),frame=np.array([r['frame_index']for r in rows]),state=state)
    for arm,joints in [('left',slice(0,6)),('right',slice(7,13))]:x[arm+'_tcp']=forward_kinematics(state[:,joints],spec[arm])[0]
    return select_from_arrays(x,grasp_events,fps,settings)
